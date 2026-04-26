"""
SoftShape Baseline for HMSPAR
Learning Soft Sparse Shapes (ICML 2025 Spotlight)
Liu et al., "Learning Soft Sparse Shapes for Efficient Time-Series Classification"

Core idea: extract shapelet patches via Conv1d, score each patch by classification contribution,
soft-sparsify (keep top-k, merge rest into a single token), then route through a Mixture-of-Experts
block + Inception module for intra/inter-shape pattern learning.

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


class RMSNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.scale = dim ** 0.5
        self.gamma = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        return F.normalize(x, dim=-1) * self.gamma * self.scale


class MoEBlock(nn.Module):
    """Mixture-of-Experts block with top-1 gating (per-patch routing)."""

    def __init__(self, dim, num_experts, hidden_dim=None):
        super().__init__()
        hidden_dim = hidden_dim or dim
        self.num_experts = num_experts
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(dim, hidden_dim), nn.GELU(), nn.Dropout(0.15),
                nn.Linear(hidden_dim, dim)
            ) for _ in range(num_experts)
        ])
        self.gate = nn.Linear(dim, num_experts)
        self.norm = RMSNorm(dim)

    def forward(self, x):
        B, N, D = x.shape
        x_flat = x.reshape(B * N, D)

        logits = F.softmax(self.gate(x_flat), dim=-1)
        top_val, top_idx = logits.topk(1, dim=-1)

        out = torch.zeros_like(x_flat)
        for i in range(self.num_experts):
            mask_i = (top_idx.squeeze(-1) == i)
            if mask_i.any():
                out[mask_i] = self.experts[i](x_flat[mask_i])

        y = x_flat + out * top_val
        y = self.norm(y.view(B, N, D))

        importance = logits.sum(0)
        eps = 1e-10
        load_loss = importance.float().var() / (importance.float().mean() ** 2 + eps)

        return F.gelu(y), load_loss


class InceptionModule1D(nn.Module):
    """InceptionTime-style module for inter-shape temporal patterns."""

    NF = 32

    def __init__(self, in_dim, ks=40):
        super().__init__()
        nf = self.NF
        ks_list = [max(ks // (2 ** i), 1) for i in range(3)]
        ks_list = [k if k % 2 != 0 else k - 1 for k in ks_list]
        ks_list = [max(k, 1) for k in ks_list]

        self.bottleneck = (nn.Conv1d(in_dim, nf, 1, bias=False)
                           if in_dim > 1 else nn.Identity())
        bn_out = nf if in_dim > 1 else in_dim
        self.convs = nn.ModuleList([
            nn.Conv1d(bn_out, nf, k, padding=k // 2, bias=False)
            for k in ks_list
        ])
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool1d(3, stride=1, padding=1),
            nn.Conv1d(in_dim, nf, 1, bias=False)
        )
        self.bn = nn.BatchNorm1d(nf * 4)
        self.act = nn.GELU()

    def forward(self, x):
        T = x.shape[2]
        x_bn = self.bottleneck(x)
        outs = [conv(x_bn) for conv in self.convs] + [self.maxpool_conv(x)]
        out = torch.cat(outs, dim=1)
        if T > 1:
            out = self.bn(out)
        return self.act(out)


class SoftShapeLayer(nn.Module):
    """Attention scoring → soft sparsification → MoE + Inception."""

    def __init__(self, dim, moe, attn_head):
        super().__init__()
        self.norm1 = RMSNorm(dim)
        self.norm2 = RMSNorm(dim)
        self.attn_head = attn_head
        self.moe = moe
        self.inception = InceptionModule1D(dim, ks=min(40, dim))
        incep_out_dim = InceptionModule1D.NF * 4
        self.incep_proj = nn.Linear(incep_out_dim, dim)
        self.drop = nn.Dropout(0.15)

    def forward(self, x, remain_ratio=1.0, is_last=False):
        B, N, D = x.shape
        x_n = self.norm1(x)
        scores = self.attn_head(x_n)

        if remain_ratio < 1.0 and N > 2:
            left_k = max(math.ceil(remain_ratio * N), 1)
            _, top_idx = torch.topk(scores.squeeze(-1), left_k, dim=1)
            sorted_idx, _ = torch.sort(top_idx, dim=1)

            left_index = sorted_idx.unsqueeze(-1).expand(-1, -1, D)
            left_x = torch.gather(x_n * scores, 1, left_index)

            comp_mask = torch.ones(B, N, dtype=torch.bool, device=x.device)
            comp_mask.scatter_(1, sorted_idx, False)
            all_idx = torch.arange(N, device=x.device).unsqueeze(0).expand(B, -1)
            comp_idx = all_idx[comp_mask].view(B, N - left_k)
            comp_index = comp_idx.unsqueeze(-1).expand(-1, -1, D)
            non_topk = torch.gather(x_n * scores, 1, comp_index)
            extra_token = non_topk.sum(dim=1, keepdim=True)

            x = torch.cat([left_x, extra_token], dim=1)

            x_n2 = self.norm2(x)
            moe_out, moe_loss = self.moe(x_n2)
            incep_out = self.inception(x_n2.permute(0, 2, 1))
            incep_out = self.incep_proj(incep_out.permute(0, 2, 1))
            x = x + moe_out + incep_out
        else:
            x = x_n * scores
            moe_loss = 0.0
            incep_out = self.inception(self.norm2(x).permute(0, 2, 1))
            incep_out = self.incep_proj(incep_out.permute(0, 2, 1))
            x = x + incep_out

        end_scores = None
        if is_last:
            x = self.drop(x)
            end_scores = self.attn_head(x)

        return F.gelu(x), moe_loss, end_scores


class SoftShapeNet(nn.Module):
    """SoftShape architecture (ICML 2025)."""

    def __init__(self, in_channels, seq_len, emb_dim=128, depth=2,
                 sparse_rate=0.5, shape_size=8, stride=4, num_experts=2):
        super().__init__()
        self.emb_dim = emb_dim
        self.depth = depth
        self.sparse_rate = sparse_rate

        shape_size = min(shape_size, seq_len)
        stride = min(stride, shape_size)
        num_patches = max((seq_len - shape_size) // stride + 1, 1)

        self.shape_embed = nn.Conv1d(in_channels, emb_dim,
                                     kernel_size=shape_size, stride=stride)
        self.num_patches = num_patches
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, emb_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        self.pos_drop = nn.Dropout(0.15)

        self.attn_head = nn.Sequential(
            nn.Linear(emb_dim, 8), nn.Tanh(),
            nn.Linear(8, 1), nn.Sigmoid(),
        )
        self.moe = MoEBlock(emb_dim, num_experts=num_experts)

        self.sparse_schedule = [
            x.item() for x in torch.linspace(0, sparse_rate, depth)
        ]
        self.blocks = nn.ModuleList([
            SoftShapeLayer(emb_dim, self.moe, self.attn_head)
            for _ in range(depth)
        ])

        self.head = nn.Linear(emb_dim, 1)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x, epoch=100, warm_up=50):
        x = self.shape_embed(x)
        x = x.transpose(1, 2)

        N = x.shape[1]
        if N != self.pos_embed.shape[1]:
            pos = F.interpolate(
                self.pos_embed.transpose(1, 2), size=N,
                mode='linear', align_corners=False
            ).transpose(1, 2)
        else:
            pos = self.pos_embed

        x = x + pos
        x = self.pos_drop(x)

        total_moe_loss = 0.0
        end_scores = None

        for i, blk in enumerate(self.blocks):
            ratio = 1.0 - self.sparse_schedule[i]
            if epoch < warm_up:
                ratio = 1.0
            is_last = (i == self.depth - 1)
            x, moe_loss, end_scores = blk(x, remain_ratio=ratio, is_last=is_last)
            total_moe_loss = total_moe_loss + moe_loss

        if end_scores is not None:
            logits = self.head(x)
            weighted = logits * end_scores
            cls_logits = weighted.mean(dim=1)
        else:
            cls_logits = self.head(x.mean(dim=1))

        return cls_logits.squeeze(-1), total_moe_loss


def compute_metrics(y_true, y_pred, y_proba):
    return {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall': recall_score(y_true, y_pred, zero_division=0),
        'f1': f1_score(y_true, y_pred, zero_division=0),
        'auc': roc_auc_score(y_true, y_proba) if len(np.unique(y_true)) > 1 else 0.0,
        'auprc': average_precision_score(y_true, y_proba) if len(np.unique(y_true)) > 1 else 0.0,
    }


def train_epoch(model, loader, criterion, optimizer, device, current_epoch, warm_up, moe_loss_rate):
    """SoftShape training with MoE auxiliary loss."""
    model.train()
    total_loss = 0.0
    for X_batch, y_batch in loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()
        logits, moe_loss = model(X_batch, epoch=current_epoch, warm_up=warm_up)
        loss = criterion(logits, y_batch)
        if isinstance(moe_loss, torch.Tensor):
            loss = loss + moe_loss_rate * moe_loss
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)


@torch.no_grad()
def evaluate(model, loader, device, current_epoch, warm_up):
    model.eval()
    all_proba, all_true = [], []
    for X_batch, y_batch in loader:
        logits, _ = model(X_batch.to(device), epoch=current_epoch, warm_up=warm_up)
        proba = torch.sigmoid(logits).cpu().numpy()
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

    model = SoftShapeNet(
        input_dim, seq_len, emb_dim=args.emb_dim, depth=args.depth,
        sparse_rate=args.sparse_rate, shape_size=args.shape_size,
        stride=args.stride, num_experts=2
    ).to(device)

    pos_weight = torch.tensor([(y_train == 0).sum() / max((y_train == 1).sum(), 1)]).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_val_f1, patience_counter = 0.0, 0
    dataset_tag = f"{args.dataset}_{args.industry or 'none'}"
    ckpt = f'best_softshape_{dataset_tag}_{seed}.pth'

    for epoch in range(args.epochs):
        loss = train_epoch(model, train_loader, criterion, optimizer, device, 
                          epoch, args.warm_up_epoch, args.moe_loss_rate)
        scheduler.step()
        y_val_true, y_val_proba = evaluate(model, val_loader, device, epoch, args.warm_up_epoch)
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

    y_test_true, y_test_proba = evaluate(model, test_loader, device, 999, args.warm_up_epoch)
    return compute_metrics(y_test_true, (y_test_proba > 0.5).astype(int), y_test_proba)


def main():
    parser = argparse.ArgumentParser(description='SoftShape baseline (ICML 2025)')
    parser.add_argument('--dataset', type=str, default='merchant',
                        choices=['merchant', 'cdnow', 'retail', 'instacart', 'sales_weekly', 'tafeng'])
    parser.add_argument('--industry', type=str, default=None,
                        choices=['Industry-0', 'Industry-1', 'Industry-2', 'Industry-3'])
    parser.add_argument('--task', type=str, default=None,
                        choices=['churn', 'seasonality', 'repurchase'])
    parser.add_argument('--emb_dim', type=int, default=128)
    parser.add_argument('--depth', type=int, default=2)
    parser.add_argument('--sparse_rate', type=float, default=0.5)
    parser.add_argument('--shape_size', type=int, default=8)
    parser.add_argument('--stride', type=int, default=4)
    parser.add_argument('--warm_up_epoch', type=int, default=50)
    parser.add_argument('--moe_loss_rate', type=float, default=0.001)
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
            'Dataset': result_name, 'Model': 'SoftShape',
            'Accuracy': metrics['accuracy'], 'Accuracy_std': metrics_std['accuracy'],
            'Precision': metrics['precision'], 'Precision_std': metrics_std['precision'],
            'Recall': metrics['recall'], 'Recall_std': metrics_std['recall'],
            'F1-Score': metrics['f1'], 'F1-Score_std': metrics_std['f1'],
            'AUC-ROC': metrics['auc'], 'AUC-ROC_std': metrics_std['auc'],
            'AUPRC': metrics['auprc'], 'AUPRC_std': metrics_std['auprc'],
            'Seeds': '42,123,456'
        }])
        os.makedirs(Config.RESULTS_DIR, exist_ok=True)
        result_path = Config.RESULTS_DIR / f'softshape_{result_name}_multiseed_results.csv'
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
        'Dataset': result_name, 'Model': 'SoftShape',
        'Accuracy': metrics['accuracy'], 'Precision': metrics['precision'],
        'Recall': metrics['recall'], 'F1-Score': metrics['f1'],
        'AUC-ROC': metrics['auc'], 'AUPRC': metrics['auprc'],
    }])
    os.makedirs(Config.RESULTS_DIR, exist_ok=True)
    result_path = Config.RESULTS_DIR / f'softshape_{result_name}{suffix}_results.csv'
    results.to_csv(result_path, index=False)
    print(f"\nResults saved to: {result_path}")


if __name__ == '__main__':
    main()
