"""
Hydra Baseline for HMSPAR
Reference: Hydra: Competing Convolutional Kernels for Fast and Accurate Time Series Classification (DMKD 2023)
Supports: Merchant, CDNOW, Retail, Instacart, sales_weekly, tafeng

Hydra uses competing groups of random convolutional kernels (like ROCKET/MiniRocket)
but adds a competition mechanism between kernels, achieving SOTA on UCR/UEA benchmarks.
"""
import os
import sys
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import RidgeClassifierCV
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, average_precision_score
)
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.append(str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))
from configs.config import Config
from data_utils import load_dataset, prepare_data_splits, flatten_time_series, print_dataset_info, print_split_info


class HydraTransform(nn.Module):
    """
    Hydra feature extractor (DMKD 2023).

    Key idea: K groups of g competing random convolutional kernels.
    For each group, only the kernel with the max response "wins" at each
    position (soft competition via softmax). Features = PPV (proportion
    of positive values) per kernel across the sequence.

    This differs from ROCKET/MiniRocket by adding inter-kernel competition,
    which improves discriminability without increasing inference cost.
    """
    def __init__(self, input_length, k=8, g=64, max_kernel_size=9, seed=42):
        """
        Args:
            input_length: time series length
            k: number of competing groups
            g: number of kernels per group
            max_kernel_size: maximum kernel size (odd values 3..max)
            seed: random seed for kernel initialisation
        """
        super().__init__()
        torch.manual_seed(seed)
        self.k = k
        self.g = g
        self.input_length = input_length

        # Sample random kernel sizes (odd, 3 to max_kernel_size)
        odd_sizes = list(range(3, max_kernel_size + 1, 2))
        sizes = torch.tensor(
            [odd_sizes[i % len(odd_sizes)] for i in range(k * g)]
        )
        self.register_buffer('sizes', sizes)

        # Build kernels: one Conv1d per unique size to allow batching
        unique_sizes = sorted(set(odd_sizes))
        self.convs = nn.ModuleDict()
        self.kernel_indices = {}  # size -> list of (group_idx, kernel_idx)

        # For simplicity: one large Conv1d with padding per unique size
        # Weights are random normal, NOT trained (frozen feature extractor)
        for s in unique_sizes:
            n_kernels_of_size = int((sizes == s).sum().item())
            if n_kernels_of_size == 0:
                continue
            conv = nn.Conv1d(
                in_channels=1,
                out_channels=n_kernels_of_size,
                kernel_size=s,
                padding=s // 2,
                bias=False
            )
            # Random normal init, frozen
            nn.init.normal_(conv.weight)
            for p in conv.parameters():
                p.requires_grad = False
            self.convs[str(s)] = conv

        # Output feature dim: k groups * g kernels * 2 (PPV + mean)
        self.n_features = k * g * 2

    @torch.no_grad()
    def forward(self, x):
        """
        x: [B, L] or [B, L, C] (only first channel used for 1D case)
        Returns: [B, k*g*2] feature vector
        """
        if x.ndim == 3:
            x = x[:, :, 0]  # use first channel
        B = x.size(0)
        x = x.unsqueeze(1)  # [B, 1, L]

        # Collect all kernel responses
        all_responses = []  # will be [B, k*g, L]
        for s, conv in self.convs.items():
            resp = conv(x)  # [B, n_k, L]
            all_responses.append(resp)
        all_responses = torch.cat(all_responses, dim=1)  # [B, k*g, L]

        # Reshape into groups: [B, k, g, L]
        all_responses = all_responses.view(B, self.k, self.g, -1)

        # Competition: softmax over g kernels within each group
        weights = F.softmax(all_responses.max(dim=-1, keepdim=True).values, dim=2)
        # Weighted responses
        weighted = all_responses * weights  # [B, k, g, L]

        # PPV: proportion of positive values
        ppv = (weighted > 0).float().mean(dim=-1)   # [B, k, g]
        # Mean of positive values
        pos_mean = weighted.clamp(min=0).mean(dim=-1)  # [B, k, g]

        features = torch.cat([ppv, pos_mean], dim=-1)  # [B, k, 2g]
        return features.view(B, -1).cpu().numpy()       # [B, k*2g]


def compute_metrics(y_true, y_pred, y_proba):
    return {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall': recall_score(y_true, y_pred, zero_division=0),
        'f1': f1_score(y_true, y_pred, zero_division=0),
        'auc': roc_auc_score(y_true, y_proba) if len(np.unique(y_true)) > 1 else 0.0,
        'auprc': average_precision_score(y_true, y_proba) if len(np.unique(y_true)) > 1 else 0.0,
    }


def run_experiment(args, seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    X, y, info = load_dataset(args.dataset, args.industry, args.task)
    print_dataset_info(X, y, info)

    X_train, X_val, X_test, y_train, y_val, y_test = prepare_data_splits(X, y)
    print_split_info(X_train, X_val, X_test, y_train, y_val, y_test)

    # For Hydra we flatten to 2D and use the first channel only
    X_train_2d = flatten_time_series(X_train)
    X_val_2d = flatten_time_series(X_val)
    X_test_2d = flatten_time_series(X_test)

    scaler = StandardScaler()
    X_train_2d = scaler.fit_transform(X_train_2d)
    X_val_2d = scaler.transform(X_val_2d)
    X_test_2d = scaler.transform(X_test_2d)

    seq_len = X_train_2d.shape[1]

    # Build Hydra transform (no training, random frozen kernels)
    hydra = HydraTransform(
        input_length=seq_len,
        k=args.k,
        g=args.g,
        max_kernel_size=args.max_kernel_size,
        seed=seed
    ).to(device)

    print(f"Hydra feature dim: {hydra.n_features}")
    print("Extracting features...")

    def extract(X_2d):
        X_t = torch.FloatTensor(X_2d).to(device)
        feats = []
        bs = 512
        for i in range(0, len(X_t), bs):
            feats.append(hydra(X_t[i:i+bs]))
        return np.concatenate(feats, axis=0)

    F_train = extract(X_train_2d)
    F_val = extract(X_val_2d)
    F_test = extract(X_test_2d)

    # Combine train+val for final classifier (standard Hydra protocol)
    F_trainval = np.concatenate([F_train, F_val], axis=0)
    y_trainval = np.concatenate([y_train, y_val], axis=0)

    # Ridge classifier (standard for ROCKET-family)
    print("Fitting RidgeClassifierCV...")
    clf = RidgeClassifierCV(alphas=np.logspace(-3, 3, 10))
    clf.fit(F_trainval, y_trainval)

    y_test_pred = clf.predict(F_test)
    # Decision function as proxy for probability
    df = clf.decision_function(F_test)
    # Normalise to [0,1] via sigmoid
    y_test_proba = 1 / (1 + np.exp(-df))

    return compute_metrics(y_test, y_test_pred, y_test_proba)


def main():
    parser = argparse.ArgumentParser(description='Hydra baseline (DMKD 2023)')
    parser.add_argument('--dataset', type=str, default='merchant',
                        choices=['merchant', 'cdnow', 'retail', 'instacart', 'sales_weekly', 'tafeng'])
    parser.add_argument('--industry', type=str, default=None,
                        choices=['Industry-0', 'Industry-1', 'Industry-2', 'Industry-3'])
    parser.add_argument('--task', type=str, default=None,
                        choices=['churn', 'seasonality', 'repurchase'])
    parser.add_argument('--k', type=int, default=8,
                        help='Number of competing groups')
    parser.add_argument('--g', type=int, default=64,
                        help='Number of kernels per group')
    parser.add_argument('--max_kernel_size', type=int, default=9,
                        help='Maximum kernel size (odd)')
    parser.add_argument('--multi-seed', action='store_true')
    args = parser.parse_args()

    if args.multi_seed:
        seeds = [42, 123, 456]
        all_metrics = [run_experiment(args, seed=s) for s in seeds]
        keys = list(all_metrics[0].keys())
        print("\n" + "=" * 80)
        print(" Multi-Seed Results (mean +/- std) ".center(80, "="))
        print("=" * 80)
        for k in keys:
            vals = [m[k] for m in all_metrics]
            print(f"  {k}: {np.mean(vals):.4f} +/- {np.std(vals):.4f}")
        metrics = {k: np.mean([m[k] for m in all_metrics]) for k in keys}
        metrics_std = {k: np.std([m[k] for m in all_metrics], ddof=1) for k in keys}

        result_name = args.industry.lower().replace('-', '') if args.dataset == 'merchant' else args.dataset
        if args.task:
            result_name += f'_{args.task}'
        results = pd.DataFrame([{
            'Dataset': result_name, 'Model': 'Hydra',
            'Accuracy': metrics['accuracy'], 'Accuracy_std': metrics_std['accuracy'],
            'Precision': metrics['precision'], 'Precision_std': metrics_std['precision'],
            'Recall': metrics['recall'], 'Recall_std': metrics_std['recall'],
            'F1-Score': metrics['f1'], 'F1-Score_std': metrics_std['f1'],
            'AUC-ROC': metrics['auc'], 'AUC-ROC_std': metrics_std['auc'],
            'AUPRC': metrics['auprc'], 'AUPRC_std': metrics_std['auprc'],
            'Seeds': '42,123,456'
        }])
        os.makedirs(Config.RESULTS_DIR, exist_ok=True)
        result_path = Config.RESULTS_DIR / f'hydra_{result_name}_multiseed_results.csv'
        results.to_csv(result_path, index=False)
        print(f"\nResults saved to: {result_path}")
        return
    else:
        metrics = run_experiment(args, seed=42)
        print("\n" + "=" * 80)
        print(" Test Results ".center(80, "="))
        print("=" * 80)
        for k, v in metrics.items():
            print(f"  {k}: {v:.4f}")

    result_name = args.industry.lower().replace('-', '') if args.dataset == 'merchant' else args.dataset
    if args.task:
        result_name += f'_{args.task}'
    results = pd.DataFrame([{
        'Dataset': result_name,
        'Model': 'Hydra',
        'Accuracy': metrics['accuracy'],
        'Precision': metrics['precision'],
        'Recall': metrics['recall'],
        'F1-Score': metrics['f1'],
        'AUC-ROC': metrics['auc'],
        'AUPRC': metrics['auprc'],
    }])
    os.makedirs(Config.RESULTS_DIR, exist_ok=True)
    result_path = Config.RESULTS_DIR / f'hydra_{result_name}{suffix}_results.csv'
    results.to_csv(result_path, index=False)
    print(f"\nResults saved to: {result_path}")


if __name__ == '__main__':
    main()
