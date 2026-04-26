import os
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

class GAFCNNDataset(Dataset):
    def __init__(self, X, y, target_size=32):
        self.y = torch.FloatTensor(y)
        self.target_size = target_size

        if X.ndim == 2:
            X_t = torch.FloatTensor(X).unsqueeze(1).unsqueeze(2)
        else:
            X_t = torch.FloatTensor(X).permute(0, 2, 1).unsqueeze(2)

        self.X = nn.functional.interpolate(
            X_t, size=(target_size, target_size),
            mode='bilinear', align_corners=False
        ).detach()

    def __len__(self): return len(self.X)
    def __getitem__(self, idx): return self.X[idx], self.y[idx]

class GAFCNNClassifier(nn.Module):
    def __init__(self, in_channels=1, dropout=0.3):
        super().__init__()
        import torchvision.models as tv_models

        resnet = tv_models.resnet18(weights=None)

        resnet.conv1 = nn.Conv2d(
            in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False
        )

        n_features = resnet.fc.in_features
        resnet.fc = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(n_features, n_features // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(n_features // 2, 1)
        )
        self.model = resnet

    def forward(self, x):
        return self.model(x).squeeze(-1)

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
        loss = criterion(model(X_batch), y_batch)
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
        proba = torch.sigmoid(model(X_batch.to(device))).cpu().numpy()
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

    if X_train.ndim == 2:
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_val   = scaler.transform(X_val)
        X_test  = scaler.transform(X_test)
        in_channels = 1
    else:
        orig_shape = X_train.shape
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train.reshape(-1, orig_shape[-1])).reshape(orig_shape)
        X_val   = scaler.transform(X_val.reshape(-1, orig_shape[-1])).reshape(X_val.shape)
        X_test  = scaler.transform(X_test.reshape(-1, orig_shape[-1])).reshape(X_test.shape)
        in_channels = X_train.shape[2]

    train_ds = GAFCNNDataset(X_train, y_train, target_size=args.image_size)
    val_ds   = GAFCNNDataset(X_val,   y_val,   target_size=args.image_size)
    test_ds  = GAFCNNDataset(X_test,  y_test,  target_size=args.image_size)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=64, shuffle=False)
    test_loader  = DataLoader(test_ds,  batch_size=64, shuffle=False)

    model = GAFCNNClassifier(in_channels=in_channels, dropout=args.dropout).to(device)

    pos_weight = torch.tensor([(y_train == 0).sum() / max((y_train == 1).sum(), 1)]).to(device)
    criterion  = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer  = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler  = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_val_f1, patience_counter = 0.0, 0
    dataset_tag = f"{args.dataset}_{args.industry or 'none'}"
    ckpt = f'best_gafcnn_{dataset_tag}_{seed}.pth'

    for epoch in range(args.epochs):
        loss = train_epoch(model, train_loader, criterion, optimizer, device)
        scheduler.step()
        y_val_true, y_val_proba = evaluate(model, val_loader, device)
        val_f1 = f1_score(y_val_true, (y_val_proba > 0.5).astype(int), zero_division=0)
        print(f"Epoch {epoch+1:3d}/{args.epochs} | Loss: {loss:.4f} | Val F1: {val_f1:.4f}")
        if val_f1 > best_val_f1:
            best_val_f1, patience_counter = val_f1, 0
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
    parser = argparse.ArgumentParser(description='GAF-CNN baseline')
    parser.add_argument('--dataset', type=str, default='merchant',
                        choices=['merchant', 'cdnow', 'retail', 'instacart', 'sales_weekly', 'tafeng'])
    parser.add_argument('--industry', type=str, default=None,
                        choices=['Industry-0', 'Industry-1', 'Industry-2', 'Industry-3'])
    parser.add_argument('--task', type=str, default=None,
                        choices=['churn', 'seasonality', 'repurchase'])
    parser.add_argument('--multi-seed',   action='store_true')
    parser.add_argument('--epochs',       type=int,   default=30)
    parser.add_argument('--patience',     type=int,   default=7)
    parser.add_argument('--batch-size',   type=int,   default=32)
    parser.add_argument('--lr',           type=float, default=1e-3)
    parser.add_argument('--weight-decay', type=float, default=1e-4)
    parser.add_argument('--image-size',   type=int,   default=32)
    parser.add_argument('--dropout',      type=float, default=0.3)
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
            print(f'  {k}: {np.mean(vals):.4f} +/- {np.std(vals, ddof=1):.4f}')
        metrics = {k: np.mean([m[k] for m in all_metrics]) for k in keys}
        metrics_std = {k: np.std([m[k] for m in all_metrics], ddof=1) for k in keys}

        result_name = args.industry.lower().replace('-', '') if args.dataset == 'merchant' else args.dataset
        if args.task:
            result_name += f'_{args.task}'
        results = pd.DataFrame([{
            'Dataset': result_name, 'Model': 'GAF-CNN',
            'Accuracy': metrics['accuracy'], 'Accuracy_std': metrics_std['accuracy'],
            'Precision': metrics['precision'], 'Precision_std': metrics_std['precision'],
            'Recall': metrics['recall'], 'Recall_std': metrics_std['recall'],
            'F1-Score': metrics['f1'], 'F1-Score_std': metrics_std['f1'],
            'AUC-ROC': metrics['auc'], 'AUC-ROC_std': metrics_std['auc'],
            'AUPRC': metrics['auprc'], 'AUPRC_std': metrics_std['auprc'],
            'Seeds': '42,123,456'
        }])
        os.makedirs(Config.RESULTS_DIR, exist_ok=True)
        result_path = Config.RESULTS_DIR / f'gafcnn_{result_name}_multiseed_results.csv'
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
        'Dataset': result_name, 'Model': 'GAF-CNN',
        'Accuracy': metrics['accuracy'], 'Precision': metrics['precision'],
        'Recall': metrics['recall'], 'F1-Score': metrics['f1'],
        'AUC-ROC': metrics['auc'], 'AUPRC': metrics['auprc'],
    }])
    os.makedirs(Config.RESULTS_DIR, exist_ok=True)
    result_path = Config.RESULTS_DIR / f'gafcnn_{result_name}_results.csv'
    results.to_csv(result_path, index=False)
    print(f'\nResults saved to: {result_path}')

if __name__ == '__main__':
    main()
