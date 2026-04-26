"""
DSN Baseline for HMSPAR
Dynamic Sparse Network (NeurIPS 2022)
Xiao et al., "Dynamic Sparse Network for Time Series Classification: Learning What to 'See'"

Core idea: multi-layer 1D CNN with large but *sparse* kernels. During training the binary 
mask on each kernel is periodically updated — lowest-magnitude weights are pruned and new 
positions are randomly re-grown — giving adaptive effective receptive fields at low cost.

Supports: Merchant, CDNOW, Retail, Instacart, sales_weekly, tafeng
"""
import os
import sys
import argparse
import math
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
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
            self.X = torch.FloatTensor(X).unsqueeze(1)
        else:
            self.X = torch.FloatTensor(X).transpose(1, 2)
        self.y = torch.FloatTensor(y)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class SparseCNNLayer(nn.Module):
    """1D convolution with a binary sparse mask on weights."""

    def __init__(self, in_ch, out_ch, kernel_size, density=0.2):
        super().__init__()
        self.conv = nn.Conv1d(in_ch, out_ch, kernel_size, padding=kernel_size // 2)
        self.register_buffer("mask", torch.ones_like(self.conv.weight))
        self.density = density
        self._init_sparse()

    def _init_sparse(self):
        """Keep top-`density` fraction of weights by magnitude."""
        with torch.no_grad():
            w = self.conv.weight.abs().flatten()
            k = max(int(w.numel() * self.density), 1)
            thr = torch.topk(w, k).values[-1]
            self.mask.copy_((self.conv.weight.abs() >= thr).float())
            self.conv.weight.mul_(self.mask)

    def forward(self, x):
        self.conv.weight.data.mul_(self.mask)
        return self.conv(x)


class SparseCNNModule(nn.Module):
    """sparse conv → BN → ReLU → 1×1 conv → BN → ReLU (one DSN module)."""

    def __init__(self, in_ch, out_ch, kernel_size, density):
        super().__init__()
        self.sparse_conv = SparseCNNLayer(in_ch, out_ch, kernel_size, density)
        self.bn1 = nn.BatchNorm1d(out_ch)
        self.conv1x1 = nn.Conv1d(out_ch, out_ch, 1)
        self.bn2 = nn.BatchNorm1d(out_ch)

    def forward(self, x):
        x = F.relu(self.bn1(self.sparse_conv(x)))
        x = F.relu(self.bn2(self.conv1x1(x)))
        return x


class DSNNet(nn.Module):
    """DSN architecture: N sparse-CNN modules + 1 sparse conv + dual pooling + linear classifier."""

    def __init__(self, in_channels, seq_len, ch_size=47, kernel_size=39, depth=4, density=0.2):
        super().__init__()
        self.depth = depth
        self.density = density

        modules = []
        c_in = in_channels
        for _ in range(min(depth - 1, 3)):
            modules.append(SparseCNNModule(c_in, ch_size, kernel_size, density))
            c_in = ch_size
        self.sparse_modules = nn.ModuleList(modules)

        self.final_sparse = SparseCNNLayer(ch_size, ch_size, kernel_size, density)
        self.final_bn = nn.BatchNorm1d(ch_size)

        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.max_pool = nn.AdaptiveMaxPool1d(1)
        self.classifier = nn.Linear(ch_size * 2, 1)

    def forward(self, x):
        for mod in self.sparse_modules:
            x = mod(x)
        x = F.relu(self.final_bn(self.final_sparse(x)))
        x_avg = self.avg_pool(x).squeeze(-1)
        x_max = self.max_pool(x).squeeze(-1)
        x = torch.cat([x_avg, x_max], dim=1)
        return self.classifier(x).squeeze(-1)

    def get_sparse_layers(self):
        """Collect all SparseCNNLayer modules for dynamic sparse training."""
        layers = []
        for mod in self.sparse_modules:
            layers.append(mod.sparse_conv)
        layers.append(self.final_sparse)
        return layers


def dsn_prune_and_regrow(sparse_layers, iteration, total_iters, alpha=0.5):
    """Dynamic sparse training: prune lowest magnitude, regrow random."""
    update_frac = alpha / 2 * (1 + math.cos(math.pi * iteration / max(total_iters, 1)))
    if update_frac < 1e-6:
        return

    for layer in sparse_layers:
        with torch.no_grad():
            mask = layer.mask
            w = layer.conv.weight * mask
            active_idx = mask.flatten().nonzero(as_tuple=True)[0]
            n_active = len(active_idx)
            n_update = max(int(n_active * update_frac), 1)
            if n_active <= n_update:
                continue

            active_vals = w.flatten()[active_idx].abs()
            _, prune_local = torch.topk(active_vals, n_update, largest=False)
            prune_idx = active_idx[prune_local]

            flat_mask = mask.flatten()
            flat_mask[prune_idx] = 0.0

            inactive_idx = (flat_mask == 0).nonzero(as_tuple=True)[0]
            if len(inactive_idx) > 0:
                n_regrow = min(n_update, len(inactive_idx))
                perm = torch.randperm(len(inactive_idx), device=mask.device)[:n_regrow]
                flat_mask[inactive_idx[perm]] = 1.0

            mask.copy_(flat_mask.view_as(mask))
            layer.conv.weight.data.mul_(mask)


def compute_metrics(y_true, y_pred, y_proba):
    return {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall': recall_score(y_true, y_pred, zero_division=0),
        'f1': f1_score(y_true, y_pred, zero_division=0),
        'auc': roc_auc_score(y_true, y_proba) if len(np.unique(y_true)) > 1 else 0.0,
        'auprc': average_precision_score(y_true, y_proba) if len(np.unique(y_true)) > 1 else 0.0,
    }


def train_epoch(model, loader, criterion, optimizer, device, global_step, total_steps):
    """Training with dynamic sparse prune-and-regrow every 100 steps."""
    model.train()
    sparse_layers = model.get_sparse_layers()
    total_loss = 0.0
    delta_t = 100

    for X_batch, y_batch in loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()
        logits = model(X_batch)
        loss = criterion(logits, y_batch)
        loss.backward()

        for layer in sparse_layers:
            if layer.conv.weight.grad is not None:
                layer.conv.weight.grad.mul_(layer.mask)

        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item()

        global_step += 1
        if global_step % delta_t == 0:
            dsn_prune_and_regrow(sparse_layers, global_step, total_steps)

    return total_loss / len(loader), global_step


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

    kernel_size = min(args.kernel_size, seq_len // 2 * 2 + 1)
    if kernel_size % 2 == 0:
        kernel_size -= 1
    kernel_size = max(kernel_size, 3)

    model = DSNNet(
        input_dim, seq_len, ch_size=args.ch_size,
        kernel_size=kernel_size, depth=args.depth, density=args.density
    ).to(device)

    pos_weight = torch.tensor([(y_train == 0).sum() / max((y_train == 1).sum(), 1)]).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    n_batches = max(len(X_train) // args.batch_size, 1)
    total_steps = n_batches * args.epochs
    global_step = 0

    best_val_f1, patience_counter = 0.0, 0
    dataset_tag = f"{args.dataset}_{args.industry or 'none'}"
    ckpt = f'best_dsn_{dataset_tag}_{seed}.pth'

    for epoch in range(args.epochs):
        loss, global_step = train_epoch(model, train_loader, criterion, optimizer, device, global_step, total_steps)
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
    parser = argparse.ArgumentParser(description='DSN baseline (NeurIPS 2022)')
    parser.add_argument('--dataset', type=str, default='merchant',
                        choices=['merchant', 'cdnow', 'retail', 'instacart', 'sales_weekly', 'tafeng'])
    parser.add_argument('--industry', type=str, default=None,
                        choices=['Industry-0', 'Industry-1', 'Industry-2', 'Industry-3'])
    parser.add_argument('--task', type=str, default=None,
                        choices=['churn', 'seasonality', 'repurchase'])
    parser.add_argument('--density', type=float, default=0.2)
    parser.add_argument('--ch_size', type=int, default=47)
    parser.add_argument('--kernel_size', type=int, default=39)
    parser.add_argument('--depth', type=int, default=4)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--patience', type=int, default=15)
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
            'Dataset': result_name, 'Model': 'DSN',
            'Accuracy': metrics['accuracy'], 'Accuracy_std': metrics_std['accuracy'],
            'Precision': metrics['precision'], 'Precision_std': metrics_std['precision'],
            'Recall': metrics['recall'], 'Recall_std': metrics_std['recall'],
            'F1-Score': metrics['f1'], 'F1-Score_std': metrics_std['f1'],
            'AUC-ROC': metrics['auc'], 'AUC-ROC_std': metrics_std['auc'],
            'AUPRC': metrics['auprc'], 'AUPRC_std': metrics_std['auprc'],
            'Seeds': '42,123,456'
        }])
        os.makedirs(Config.RESULTS_DIR, exist_ok=True)
        result_path = Config.RESULTS_DIR / f'dsn_{result_name}_multiseed_results.csv'
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
        'Dataset': result_name, 'Model': 'DSN',
        'Accuracy': metrics['accuracy'], 'Precision': metrics['precision'],
        'Recall': metrics['recall'], 'F1-Score': metrics['f1'],
        'AUC-ROC': metrics['auc'], 'AUPRC': metrics['auprc'],
    }])
    os.makedirs(Config.RESULTS_DIR, exist_ok=True)
    result_path = Config.RESULTS_DIR / f'dsn_{result_name}{suffix}_results.csv'
    results.to_csv(result_path, index=False)
    print(f"\nResults saved to: {result_path}")


if __name__ == '__main__':
    main()
