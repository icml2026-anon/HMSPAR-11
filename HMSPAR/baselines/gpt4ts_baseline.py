"""
GPT4TS Baseline for HMSPAR
Reference: One Fits All: Power General Time Series Analysis by Pretrained LM (NeurIPS 2023)
Key idea: Freeze GPT-2 backbone, only train input embedding + output head.
Supports: Merchant, CDNOW, Retail, Instacart, Sales_Weekly, TaFeng
"""
import os
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

import sys
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, average_precision_score
)
from sklearn.preprocessing import StandardScaler
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))
from configs.config import Config
from utils.seed import set_seed
from data_utils import load_dataset, prepare_data_splits, print_dataset_info, print_split_info

# GPT-2 cache dir
GPT2_CACHE = str(Path('/root/pretrained_models/gpt2'))


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


class GPT4TSClassifier(nn.Module):
    """
    GPT4TS for classification (One Fits All, NeurIPS 2023).
    Architecture:
      1. Input projection: (C) -> d_model (patch-level, applied per time step)
      2. Frozen GPT-2 Transformer layers
      3. Global average pool over sequence
      4. Linear classification head
    The key insight: GPT-2's pretrained weights contain rich temporal patterns.
    Only the input/output adapters are trained.
    """
    def __init__(self, input_dim, seq_len, d_model=768, patch_len=8, stride=4,
                 gpt_layers=6, dropout=0.1, cache_dir=None):
        super().__init__()
        from transformers import GPT2Model, GPT2Config

        self.patch_len = patch_len
        self.stride = stride
        self.seq_len = seq_len
        self.input_dim = input_dim

        # Number of patches
        n_patches = max(1, (seq_len - patch_len) // stride + 1)
        self.n_patches = n_patches
        patch_dim = patch_len * input_dim

        # Input embedding: project patches to GPT-2 d_model
        self.patch_embedding = nn.Sequential(
            nn.Linear(patch_dim, d_model),
            nn.LayerNorm(d_model),
            nn.Dropout(dropout)
        )

        # Load pretrained GPT-2, use only first `gpt_layers` layers
        gpt2 = GPT2Model.from_pretrained('gpt2', cache_dir=cache_dir)
        self.gpt2_layers = nn.ModuleList(gpt2.h[:gpt_layers])
        self.gpt2_ln = gpt2.ln_f  # final layer norm

        # Freeze all GPT-2 parameters
        for param in self.gpt2_layers.parameters():
            param.requires_grad = False
        for param in self.gpt2_ln.parameters():
            param.requires_grad = False

        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1)
        )

    def _patchify(self, x):
        """x: [B, L, C] -> patches: [B, n_patches, patch_len*C]"""
        B, L, C = x.shape
        patches = []
        for i in range(self.n_patches):
            start = i * self.stride
            end = start + self.patch_len
            if end > L:
                # Pad last patch
                p = x[:, start:L, :]
                pad = torch.zeros(B, end - L, C, device=x.device)
                p = torch.cat([p, pad], dim=1)
            else:
                p = x[:, start:end, :]
            patches.append(p.reshape(B, -1))  # [B, patch_len*C]
        return torch.stack(patches, dim=1)  # [B, n_patches, patch_len*C]

    def forward(self, x):
        # Patchify: [B, L, C] -> [B, n_patches, patch_len*C]
        patches = self._patchify(x)

        # Project patches to d_model
        h = self.patch_embedding(patches)  # [B, n_patches, d_model]

        # Pass through frozen GPT-2 transformer blocks
        for layer in self.gpt2_layers:
            h = layer(h)[0]
        h = self.gpt2_ln(h)  # [B, n_patches, d_model]

        # Global average pooling over patches
        h = h.mean(dim=1)  # [B, d_model]

        # Classification
        return self.classifier(h).squeeze(-1)  # [B]


def compute_metrics(y_true, y_pred, y_proba):
    return {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall': recall_score(y_true, y_pred, zero_division=0),
        'f1': f1_score(y_true, y_pred, zero_division=0),
        'auc': roc_auc_score(y_true, y_proba) if len(np.unique(y_true)) > 1 else 0.0,
        'auprc': average_precision_score(y_true, y_proba) if len(np.unique(y_true)) > 1 else 0.0
    }


def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    for X_batch, y_batch in loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()
        out = model(X_batch)
        loss = criterion(out, y_batch)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    all_true, all_proba = [], []
    for X_batch, y_batch in loader:
        X_batch = X_batch.to(device)
        out = model(X_batch)
        proba = torch.sigmoid(out).cpu().numpy()
        all_true.extend(y_batch.numpy())
        all_proba.extend(proba)
    return np.array(all_true), np.array(all_proba)


def run_experiment(args, seed=42):
    set_seed(seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    X, y, info = load_dataset(args.dataset, args.industry, getattr(args, 'task', None))
    print_dataset_info(X, y, info)

    X_train, X_val, X_test, y_train, y_val, y_test = prepare_data_splits(X, y)
    print_split_info(X_train, X_val, X_test, y_train, y_val, y_test)

    # Normalize
    if X_train.ndim == 2:
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_val = scaler.transform(X_val)
        X_test = scaler.transform(X_test)
    else:
        orig_shape = X_train.shape
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train.reshape(-1, orig_shape[-1])).reshape(orig_shape)
        X_val = scaler.transform(X_val.reshape(-1, orig_shape[-1])).reshape(X_val.shape)
        X_test = scaler.transform(X_test.reshape(-1, orig_shape[-1])).reshape(X_test.shape)

    seq_len = X_train.shape[1]
    input_dim = X_train.shape[2] if X_train.ndim == 3 else 1

    train_loader = DataLoader(TSDataset(X_train, y_train), batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(TSDataset(X_val, y_val), batch_size=64, shuffle=False)
    test_loader = DataLoader(TSDataset(X_test, y_test), batch_size=64, shuffle=False)

    model = GPT4TSClassifier(
        input_dim=input_dim,
        seq_len=seq_len,
        d_model=768,
        patch_len=args.patch_len,
        stride=args.stride,
        gpt_layers=args.gpt_layers,
        dropout=args.dropout,
        cache_dir=GPT2_CACHE
    ).to(device)

    pos_weight = torch.tensor([(y_train == 0).sum() / max((y_train == 1).sum(), 1)]).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    # Only train non-frozen parameters
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.AdamW(trainable, lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_val_f1, patience_counter = 0.0, 0
    dataset_tag = f"{args.dataset}_{args.industry or 'none'}"
    ckpt = f'best_gpt4ts_{dataset_tag}_{seed}.pth'

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
        model.load_state_dict(torch.load(ckpt, weights_only=True))
        os.remove(ckpt)

    y_test_true, y_test_proba = evaluate(model, test_loader, device)
    return compute_metrics(y_test_true, (y_test_proba > 0.5).astype(int), y_test_proba)


def main():
    parser = argparse.ArgumentParser(description='GPT4TS baseline (NeurIPS 2023)')
    parser.add_argument('--dataset', type=str, default='merchant',
                        choices=['merchant', 'cdnow', 'retail', 'instacart', 'sales_weekly', 'tafeng'])
    parser.add_argument('--industry', type=str, default=None,
                        choices=['Industry-0', 'Industry-1', 'Industry-2', 'Industry-3'])
    parser.add_argument('--task', type=str, default=None,
                        choices=['churn', 'seasonality', 'repurchase'])
    parser.add_argument('--multi-seed', action='store_true')
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--patience', type=int, default=7)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--weight-decay', type=float, default=1e-4)
    parser.add_argument('--patch-len', type=int, default=8)
    parser.add_argument('--stride', type=int, default=4)
    parser.add_argument('--gpt-layers', type=int, default=6)
    parser.add_argument('--dropout', type=float, default=0.1)
    args = parser.parse_args()

    if args.dataset == 'merchant' and args.industry is None:
        raise ValueError('--industry required for merchant dataset')

    keys = ['accuracy', 'precision', 'recall', 'f1', 'auc', 'auprc']

    if args.multi_seed:
        seeds = [42, 123, 456]
        all_metrics = [run_experiment(args, seed=s) for s in seeds]
        print('\n' + '=' * 80)
        print(' Multi-Seed Results (mean +/- std) '.center(80, '='))
        print('=' * 80)
        for k in keys:
            vals = [m[k] for m in all_metrics]
            print(f'  {k}: {np.mean(vals):.4f} +/- {np.std(vals):.4f}')
        metrics = {k: np.mean([m[k] for m in all_metrics]) for k in keys}
        metrics_std = {k: np.std([m[k] for m in all_metrics], ddof=1) for k in keys}

        result_name = args.industry.lower().replace('-', '') if args.dataset == 'merchant' else args.dataset
        if args.task:
            result_name += f'_{args.task}'
        results = pd.DataFrame([{
            'Dataset': result_name, 'Model': 'GPT4TS',
            'Accuracy': metrics['accuracy'], 'Accuracy_std': metrics_std['accuracy'],
            'Precision': metrics['precision'], 'Precision_std': metrics_std['precision'],
            'Recall': metrics['recall'], 'Recall_std': metrics_std['recall'],
            'F1-Score': metrics['f1'], 'F1-Score_std': metrics_std['f1'],
            'AUC-ROC': metrics['auc'], 'AUC-ROC_std': metrics_std['auc'],
            'AUPRC': metrics['auprc'], 'AUPRC_std': metrics_std['auprc'],
            'Seeds': '42,123,456'
        }])
        os.makedirs(Config.RESULTS_DIR, exist_ok=True)
        result_path = Config.RESULTS_DIR / f'gpt4ts_{result_name}_multiseed_results.csv'
        results.to_csv(result_path, index=False)
        print(f'\nResults saved to: {result_path}')
        return
    else:
        metrics = run_experiment(args, seed=42)
        print('\n' + '=' * 80)
        print(' Test Results '.center(80, '='))
        print('=' * 80)
        for k, v in metrics.items():
            print(f'  {k}: {v:.4f}')

    result_name = args.industry.lower().replace('-', '') if args.dataset == 'merchant' else args.dataset
    if args.task:
        result_name += f'_{args.task}'
    results = pd.DataFrame([{
        'Dataset': result_name, 'Model': 'GPT4TS',
        'Accuracy': metrics['accuracy'], 'Precision': metrics['precision'],
        'Recall': metrics['recall'], 'F1-Score': metrics['f1'],
        'AUC-ROC': metrics['auc'], 'AUPRC': metrics['auprc'],
    }])
    os.makedirs(Config.RESULTS_DIR, exist_ok=True)
    result_path = Config.RESULTS_DIR / f'gpt4ts_{result_name}{suffix}_results.csv'
    results.to_csv(result_path, index=False)
    print(f'\nResults saved to: {result_path}')


if __name__ == '__main__':
    main()
