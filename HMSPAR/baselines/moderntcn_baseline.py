"""
ModernTCN Baseline for HMSPAR
Reference: ModernTCN: A Modern Pure Convolution Structure for General Time Series Analysis (ICLR 2024)
Supports: Merchant (anomaly detection), CDNOW, Retail, Instacart (classification)
"""
import os
import sys
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, average_precision_score
)
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))
from configs.config import Config
from utils.seed import set_seed
from data_utils import load_dataset, prepare_data_splits, print_dataset_info, print_split_info


class TSDataset(Dataset):
    def __init__(self, X, y):
        if X.ndim == 2:
            self.X = torch.FloatTensor(X).unsqueeze(-1)  # [N, L, 1]
        else:
            self.X = torch.FloatTensor(X)  # [N, L, C]
        self.y = torch.FloatTensor(y)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class ModernTCNBlock(nn.Module):
    """
    ModernTCN block: large-kernel depthwise conv + LayerNorm + FFN.
    Key insight from ICLR 2024: large receptive field via large kernel
    instead of deep stacking, combined with modern training techniques.
    """
    def __init__(self, d_model, kernel_size=51, ffn_ratio=4, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        # Large-kernel depthwise conv (core of ModernTCN)
        self.dw_conv = nn.Conv1d(
            d_model, d_model,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            groups=d_model  # depthwise
        )
        self.norm2 = nn.LayerNorm(d_model)
        ffn_dim = int(d_model * ffn_ratio)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, ffn_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, d_model),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        # x: [B, L, C]
        residual = x
        x = self.norm1(x)
        x = self.dw_conv(x.permute(0, 2, 1)).permute(0, 2, 1)
        x = x + residual
        residual = x
        x = self.norm2(x)
        x = self.ffn(x)
        return x + residual


class ModernTCN(nn.Module):
    """
    ModernTCN for time series classification (ICLR 2024).
    Architecture: patch-embedding stem + stacked large-kernel blocks + GAP head.
    """
    def __init__(self, input_dim=1, seq_len=50,
                 d_model=64, n_layers=3, kernel_size=51,
                 ffn_ratio=4, dropout=0.1,
                 patch_size=8, patch_stride=4):
        super().__init__()
        # Patch embedding stem
        self.patch_embed = nn.Sequential(
            nn.Conv1d(input_dim, d_model,
                      kernel_size=patch_size, stride=patch_stride,
                      padding=patch_size // 2),
            nn.GELU()
        )
        self.norm_stem = nn.LayerNorm(d_model)
        self.blocks = nn.ModuleList([
            ModernTCNBlock(d_model, kernel_size=kernel_size,
                           ffn_ratio=ffn_ratio, dropout=dropout)
            for _ in range(n_layers)
        ])
        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1)
        )

    def forward(self, x):
        # x: [B, L, C] -> patch embed expects [B, C, L]
        x = self.patch_embed(x.permute(0, 2, 1)).permute(0, 2, 1)
        x = self.norm_stem(x)
        for block in self.blocks:
            x = block(x)
        x = x.mean(dim=1)  # Global average pooling
        return self.head(x).squeeze(-1)


def compute_metrics(y_true, y_pred, y_proba):
    return {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall': recall_score(y_true, y_pred, zero_division=0),
        'f1': f1_score(y_true, y_pred, zero_division=0),
        'auc': roc_auc_score(y_true, y_proba) if len(np.unique(y_true)) > 1 else 0.0,
        'auprc': average_precision_score(y_true, y_proba) if len(np.unique(y_true)) > 1 else 0.0,
    }


def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    for X_batch, y_batch in loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()
        loss = criterion(model(X_batch), y_batch)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    all_proba, all_true = [], []
    for X_batch, y_batch in loader:
        proba = torch.sigmoid(model(X_batch.to(device))).cpu().numpy()
        all_proba.append(proba)
        all_true.append(y_batch.numpy())
    return np.concatenate(all_true), np.concatenate(all_proba)


def run_experiment(args, seed=42):
    set_seed(seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    X, y, info = load_dataset(args.dataset, args.industry, args.task)
    print_dataset_info(X, y, info)

    X_train, X_val, X_test, y_train, y_val, y_test = prepare_data_splits(X, y)
    print_split_info(X_train, X_val, X_test, y_train, y_val, y_test)

    # Normalize
    scaler = StandardScaler()
    if X_train.ndim == 2:
        X_train = scaler.fit_transform(X_train)
        X_val = scaler.transform(X_val)
        X_test = scaler.transform(X_test)
        seq_len, input_dim = X_train.shape[1], 1
    else:
        shape = X_train.shape
        X_train = scaler.fit_transform(X_train.reshape(-1, shape[-1])).reshape(shape)
        X_val = scaler.transform(X_val.reshape(-1, shape[-1])).reshape(X_val.shape)
        X_test = scaler.transform(X_test.reshape(-1, shape[-1])).reshape(X_test.shape)
        seq_len, input_dim = shape[1], shape[2]

    train_loader = DataLoader(TSDataset(X_train, y_train), batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(TSDataset(X_val, y_val), batch_size=args.batch_size)
    test_loader = DataLoader(TSDataset(X_test, y_test), batch_size=args.batch_size)

    model = ModernTCN(
        input_dim=input_dim, seq_len=seq_len,
        d_model=args.d_model, n_layers=args.n_layers,
        kernel_size=args.kernel_size, ffn_ratio=args.ffn_ratio,
        dropout=args.dropout, patch_size=args.patch_size,
        patch_stride=args.patch_stride
    ).to(device)

    pos_weight = torch.tensor(
        [(y_train == 0).sum() / max((y_train == 1).sum(), 1)]
    ).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_val_f1, patience_counter = 0.0, 0
    ds_tag = (args.industry or args.dataset).replace('-', '').replace('_', '')
    ckpt = f'best_moderntcn_{ds_tag}_{seed}.pth'

    for epoch in range(args.epochs):
        loss = train_epoch(model, train_loader, criterion, optimizer, device)
        scheduler.step()
        y_val_true, y_val_proba = evaluate(model, val_loader, device)
        val_f1 = f1_score(y_val_true, (y_val_proba > 0.5).astype(int), zero_division=0)
        print(f"Epoch {epoch+1:3d}/{args.epochs} | Loss: {loss:.4f} | Val F1: {val_f1:.4f}")
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            patience_counter = 0
            torch.save(model.state_dict(), ckpt)
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"Early stopping at epoch {epoch+1}")
                break

    if os.path.exists(ckpt):
        model.load_state_dict(torch.load(ckpt))
        os.remove(ckpt)

    y_test_true, y_test_proba = evaluate(model, test_loader, device)
    return compute_metrics(y_test_true, (y_test_proba > 0.5).astype(int), y_test_proba)


def main():
    parser = argparse.ArgumentParser(description='ModernTCN baseline (ICLR 2024)')
    parser.add_argument('--dataset', type=str, default='merchant',
                        choices=['merchant', 'cdnow', 'retail', 'instacart', 'sales_weekly', 'tafeng'])
    parser.add_argument('--industry', type=str, default=None,
                        choices=['Industry-0', 'Industry-1', 'Industry-2', 'Industry-3'])
    parser.add_argument('--task', type=str, default=None,
                        choices=['churn', 'seasonality', 'repurchase'])
    parser.add_argument('--d_model', type=int, default=64)
    parser.add_argument('--n_layers', type=int, default=3)
    parser.add_argument('--kernel_size', type=int, default=51)
    parser.add_argument('--patch_size', type=int, default=8)
    parser.add_argument('--patch_stride', type=int, default=4)
    parser.add_argument('--ffn_ratio', type=float, default=4.0)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--patience', type=int, default=10)
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
            'Dataset': result_name, 'Model': 'ModernTCN',
            'Accuracy': metrics['accuracy'], 'Accuracy_std': metrics_std['accuracy'],
            'Precision': metrics['precision'], 'Precision_std': metrics_std['precision'],
            'Recall': metrics['recall'], 'Recall_std': metrics_std['recall'],
            'F1-Score': metrics['f1'], 'F1-Score_std': metrics_std['f1'],
            'AUC-ROC': metrics['auc'], 'AUC-ROC_std': metrics_std['auc'],
            'AUPRC': metrics['auprc'], 'AUPRC_std': metrics_std['auprc'],
            'Seeds': '42,123,456'
        }])
        os.makedirs(Config.RESULTS_DIR, exist_ok=True)
        result_path = Config.RESULTS_DIR / f'moderntcn_{result_name}_multiseed_results.csv'
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
        'Model': 'ModernTCN',
        'Accuracy': metrics['accuracy'],
        'Precision': metrics['precision'],
        'Recall': metrics['recall'],
        'F1-Score': metrics['f1'],
        'AUC-ROC': metrics['auc'],
        'AUPRC': metrics['auprc'],
    }])
    os.makedirs(Config.RESULTS_DIR, exist_ok=True)
    result_path = Config.RESULTS_DIR / f'moderntcn_{result_name}{suffix}_results.csv'
    results.to_csv(result_path, index=False)
    print(f"\nResults saved to: {result_path}")


if __name__ == '__main__':
    main()
