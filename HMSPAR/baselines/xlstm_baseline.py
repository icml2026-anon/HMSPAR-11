"""
xLSTM Baseline for HMSPAR
Reference: xLSTM: Extended Long Short-Term Memory (NeurIPS 2024)
Supports: Merchant, CDNOW, Retail, Instacart, sales_weekly, tafeng
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
            self.X = torch.FloatTensor(X).unsqueeze(-1)
        else:
            self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class sLSTMCell(nn.Module):
    """
    sLSTM cell: scalar memory with exponential gating (NeurIPS 2024).
    Key innovations: exponential forget/input gates + normalizer state
    to prevent gradient vanishing.
    """
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size
        # Combined projection for i, f, z, o gates
        self.Wx = nn.Linear(input_size, 4 * hidden_size, bias=False)
        self.Wh = nn.Linear(hidden_size, 4 * hidden_size, bias=True)

    def forward(self, x, state):
        # x: [B, input_size]
        h, c, n, m = state  # hidden, cell, normalizer, stabilizer
        gates = self.Wx(x) + self.Wh(h)  # [B, 4*H]
        i_tilde, f_tilde, z, o = gates.chunk(4, dim=-1)

        # Exponential gating with log-sum-exp stabilization
        m_new = torch.maximum(f_tilde + m, i_tilde)  # stabilizer
        f = torch.exp(f_tilde + m - m_new)           # stabilized forget gate
        i = torch.exp(i_tilde - m_new)               # stabilized input gate

        z = torch.tanh(z)
        o = torch.sigmoid(o)

        c_new = f * c + i * z
        n_new = f * n + i
        # Normalized hidden state
        h_new = o * (c_new / torch.clamp(torch.abs(n_new), min=1.0))
        return h_new, c_new, n_new, m_new


class mLSTMCell(nn.Module):
    """
    mLSTM cell: matrix memory with covariance update rule (NeurIPS 2024).
    Replaces scalar cell state with a matrix for increased expressivity.
    """
    def __init__(self, input_size, hidden_size, head_dim=16):
        super().__init__()
        self.hidden_size = hidden_size
        self.head_dim = head_dim
        self.n_heads = hidden_size // head_dim

        self.q_proj = nn.Linear(input_size, hidden_size)
        self.k_proj = nn.Linear(input_size, hidden_size)
        self.v_proj = nn.Linear(input_size, hidden_size)
        self.o_proj = nn.Linear(hidden_size, hidden_size)

        self.i_gate = nn.Linear(input_size, self.n_heads)
        self.f_gate = nn.Linear(input_size, self.n_heads)

    def forward(self, x, state):
        # x: [B, input_size]
        C, n, m = state  # C: [B, n_heads, head_dim, head_dim]
        B = x.size(0)
        H, D = self.n_heads, self.head_dim

        q = self.q_proj(x).view(B, H, D)
        k = self.k_proj(x).view(B, H, D) / (D ** 0.5)
        v = self.v_proj(x).view(B, H, D)

        i_tilde = self.i_gate(x)  # [B, H]
        f_tilde = self.f_gate(x)  # [B, H]

        m_new = torch.maximum(f_tilde + m, i_tilde)
        f = torch.exp(f_tilde + m - m_new).unsqueeze(-1).unsqueeze(-1)  # [B,H,1,1]
        i = torch.exp(i_tilde - m_new).unsqueeze(-1).unsqueeze(-1)

        # Matrix memory update: C = f*C + i * (v outer k)
        vk = torch.einsum('bhd,bhe->bhde', v, k)  # [B,H,D,D]
        C_new = f * C + i * vk
        n_new = f.squeeze(-1) * n + i.squeeze(-1) * k  # [B,H,D]

        # Retrieve: h = o * C*q / max(|n^T q|, 1)
        Cq = torch.einsum('bhde,bhd->bhe', C_new, q)  # [B,H,D]
        denom = torch.clamp(
            torch.abs((n_new * q).sum(dim=-1, keepdim=True)), min=1.0
        )  # [B,H,1]
        h = (Cq / denom).reshape(B, -1)  # [B, H*D]
        h = self.o_proj(h)
        return h, C_new, n_new, m_new


class xLSTMBlock(nn.Module):
    """Single xLSTM block with pre-norm, residual, and optional mLSTM or sLSTM."""
    def __init__(self, d_model, use_mlstm=True, head_dim=16, dropout=0.1):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.use_mlstm = use_mlstm
        if use_mlstm:
            self.cell = mLSTMCell(d_model, d_model, head_dim=head_dim)
        else:
            self.cell = sLSTMCell(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x: [B, L, d_model]
        B, L, D = x.shape
        out = []
        if self.use_mlstm:
            C = torch.zeros(B, D // self.cell.head_dim,
                            self.cell.head_dim, self.cell.head_dim, device=x.device)
            n = torch.zeros(B, D // self.cell.head_dim,
                            self.cell.head_dim, device=x.device)
            m = torch.zeros(B, D // self.cell.head_dim, device=x.device)
            state = (C, n, m)
            for t in range(L):
                xt = self.norm(x[:, t, :])
                h, *state = self.cell(xt, state)
                out.append(h)
        else:
            h = torch.zeros(B, D, device=x.device)
            c = torch.zeros(B, D, device=x.device)
            n = torch.zeros(B, D, device=x.device)
            m = torch.zeros(B, D, device=x.device)
            state = (h, c, n, m)
            for t in range(L):
                xt = self.norm(x[:, t, :])
                h, *state_rest = self.cell(xt, state)
                state = (h, *state_rest)
                out.append(h)
        out = torch.stack(out, dim=1)  # [B, L, D]
        return x + self.dropout(out)


class xLSTMClassifier(nn.Module):
    """
    xLSTM for time series classification (NeurIPS 2024).
    Interleaves mLSTM and sLSTM blocks as per original paper.
    """
    def __init__(self, input_dim=1, d_model=64, n_blocks=4,
                 head_dim=16, dropout=0.1):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
        # Interleave: even=mLSTM, odd=sLSTM
        self.blocks = nn.ModuleList([
            xLSTMBlock(d_model, use_mlstm=(i % 2 == 0),
                       head_dim=head_dim, dropout=dropout)
            for i in range(n_blocks)
        ])
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1)
        )

    def forward(self, x):
        # x: [B, L, C]
        x = self.input_proj(x)
        for block in self.blocks:
            x = block(x)
        x = self.norm(x)
        x = x[:, -1, :]  # Take last timestep
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

    model = xLSTMClassifier(
        input_dim=input_dim, d_model=args.d_model,
        n_blocks=args.n_blocks, head_dim=args.head_dim,
        dropout=args.dropout
    ).to(device)

    pos_weight = torch.tensor(
        [(y_train == 0).sum() / max((y_train == 1).sum(), 1)]
    ).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_val_f1, patience_counter = 0.0, 0
    dataset_tag = f"{args.dataset}_{args.industry or 'none'}"
    ckpt = f'best_xlstm_{dataset_tag}_{seed}.pth'

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
    parser = argparse.ArgumentParser(description='xLSTM baseline (NeurIPS 2024)')
    parser.add_argument('--dataset', type=str, default='merchant',
                        choices=['merchant', 'cdnow', 'retail', 'instacart', 'sales_weekly', 'tafeng'])
    parser.add_argument('--industry', type=str, default=None,
                        choices=['Industry-0', 'Industry-1', 'Industry-2', 'Industry-3'])
    parser.add_argument('--task', type=str, default=None,
                        choices=['churn', 'seasonality', 'repurchase'])
    parser.add_argument('--d_model', type=int, default=64)
    parser.add_argument('--n_blocks', type=int, default=4)
    parser.add_argument('--head_dim', type=int, default=16)
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
            'Dataset': result_name, 'Model': 'xLSTM',
            'Accuracy': metrics['accuracy'], 'Accuracy_std': metrics_std['accuracy'],
            'Precision': metrics['precision'], 'Precision_std': metrics_std['precision'],
            'Recall': metrics['recall'], 'Recall_std': metrics_std['recall'],
            'F1-Score': metrics['f1'], 'F1-Score_std': metrics_std['f1'],
            'AUC-ROC': metrics['auc'], 'AUC-ROC_std': metrics_std['auc'],
            'AUPRC': metrics['auprc'], 'AUPRC_std': metrics_std['auprc'],
            'Seeds': '42,123,456'
        }])
        os.makedirs(Config.RESULTS_DIR, exist_ok=True)
        result_path = Config.RESULTS_DIR / f'xlstm_{result_name}_multiseed_results.csv'
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
        'Model': 'xLSTM',
        'Accuracy': metrics['accuracy'],
        'Precision': metrics['precision'],
        'Recall': metrics['recall'],
        'F1-Score': metrics['f1'],
        'AUC-ROC': metrics['auc'],
        'AUPRC': metrics['auprc'],
    }])
    os.makedirs(Config.RESULTS_DIR, exist_ok=True)
    result_path = Config.RESULTS_DIR / f'xlstm_{result_name}{suffix}_results.csv'
    results.to_csv(result_path, index=False)
    print(f"\nResults saved to: {result_path}")


if __name__ == '__main__':
    main()
