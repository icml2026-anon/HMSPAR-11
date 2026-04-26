"""
TabM Baseline for HMSPAR
Supports: Merchant (anomaly detection), CDNOW, Retail, Instacart (classification)
"""
import sys
import os
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, average_precision_score
from tabm import TabM  # type: ignore
from configs.config import Config
from utils.seed import set_seed
from data_utils import load_dataset, prepare_data_splits, flatten_time_series, print_dataset_info, print_split_info
import argparse


def compute_metrics(y_true, y_pred, y_proba):
    return {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall': recall_score(y_true, y_pred, zero_division=0),
        'f1': f1_score(y_true, y_pred, zero_division=0),
        'auc': roc_auc_score(y_true, y_proba) if len(np.unique(y_true)) > 1 else 0,
        'auprc': average_precision_score(y_true, y_proba) if len(np.unique(y_true)) > 1 else 0
    }


def train_epoch(model, loader, optimizer, device, n_ensembles):
    model.train()
    total_loss = 0
    criterion = nn.BCEWithLogitsLoss()
    
    for X_batch, y_batch in loader:
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)
        
        optimizer.zero_grad()
        logits = model(X_batch, None)
        logits_mean = logits.mean(dim=1).squeeze(-1)
        
        loss = criterion(logits_mean, y_batch.float())
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    return total_loss / len(loader)


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    all_preds = []
    all_probs = []
    all_labels = []
    
    for X_batch, y_batch in loader:
        X_batch = X_batch.to(device)
        logits = model(X_batch, None)
        logits_mean = logits.mean(dim=1).squeeze(-1)
        probs = torch.sigmoid(logits_mean)
        
        all_probs.extend(probs.cpu().numpy())
        all_preds.extend((probs > 0.5).cpu().numpy().astype(int))
        all_labels.extend(y_batch.numpy())
    
    return np.array(all_labels), np.array(all_preds), np.array(all_probs)


def main(args, seed=None):
    print("=" * 80)
    print(" TabM Baseline ".center(80, "="))
    print("=" * 80)
    
    if seed is not None:
        set_seed(seed)
    else:
        set_seed(Config.SEED)
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    
    # Load data using unified loader
    X, y, info = load_dataset(args.dataset, args.industry, getattr(args, 'task', None))
    print_dataset_info(X, y, info)
    
    # Flatten for tabular model
    X = flatten_time_series(X)
    
    # Split data
    X_train, X_val, X_test, y_train, y_val, y_test = prepare_data_splits(X, y)
    print_split_info(X_train, X_val, X_test, y_train, y_val, y_test)
    
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)
    
    X_train_tensor = torch.FloatTensor(X_train)
    y_train_tensor = torch.LongTensor(y_train)
    X_val_tensor = torch.FloatTensor(X_val)
    y_val_tensor = torch.LongTensor(y_val)
    X_test_tensor = torch.FloatTensor(X_test)
    y_test_tensor = torch.LongTensor(y_test)
    
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
    
    train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=256, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False)
    
    print(f"\nDataLoader sizes:")
    print(f"  Train: {len(train_dataset)}")
    print(f"  Val:   {len(val_dataset)}")
    print(f"  Test:  {len(test_dataset)}")
    
    print("\n" + "=" * 80)
    print(" Training TabM ".center(80, "="))
    print("=" * 80)
    
    n_ensembles = 8
    model = TabM.make(
        n_num_features=X_train.shape[1],
        cat_cardinalities=None,
        d_out=1,
        k=n_ensembles
    ).to(device)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=10, factor=0.5)
    
    best_val_f1 = 0.0
    patience_counter = 0
    max_patience = 30
    
    for epoch in range(200):
        train_loss = train_epoch(model, train_loader, optimizer, device, n_ensembles)
        
        y_val_true, y_val_pred, y_val_proba = evaluate(model, val_loader, device)
        val_metrics = compute_metrics(y_val_true, y_val_pred, y_val_proba)
        
        scheduler.step(train_loss)
        
        if (epoch + 1) % 20 == 0:
            print(f"Epoch {epoch+1:3d} | Train Loss: {train_loss:.4f} | "
                  f"Val F1: {val_metrics['f1']:.4f} | Val AUC: {val_metrics['auc']:.4f}")
        
        if val_metrics['f1'] > best_val_f1:
            best_val_f1 = val_metrics['f1']
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= max_patience:
                print(f"\nEarly stopping at epoch {epoch+1}")
                break
    
    print("\n" + "=" * 80)
    print(" Evaluating on Test Set ".center(80, "="))
    print("=" * 80)
    
    y_true, y_pred, y_proba = evaluate(model, test_loader, device)
    test_metrics = compute_metrics(y_true, y_pred, y_proba)
    
    print(f"\n  Accuracy:  {test_metrics['accuracy']:.4f}")
    print(f"  Precision: {test_metrics['precision']:.4f}")
    print(f"  Recall:    {test_metrics['recall']:.4f}")
    print(f"  F1-Score:  {test_metrics['f1']:.4f}")
    print(f"  AUC-ROC:   {test_metrics['auc']:.4f}")
    print(f"  AUPRC:     {test_metrics['auprc']:.4f}")
    
    # Determine result name
    if args.dataset == 'merchant':
        result_name = args.industry.lower().replace("-", "")
    else:
        result_name = args.dataset
    
    results_df = pd.DataFrame([test_metrics])
    results_df['dataset'] = info['name']
    results_path = Config.RESULTS_DIR / f'tabm_{result_name}_results.csv'
    results_df.to_csv(results_path, index=False)
    print(f"\nResults saved to: {results_path}")
    
    print("\n" + "=" * 80)
    print(" Complete ".center(80, "="))
    print("=" * 80)
    
    return test_metrics

def run_tabm_single_seed(args, seed):
    """Run TabM with a specific seed and return results"""
    # Call main with the specific seed and return actual test metrics
    return main(args, seed)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, default=None,
                       choices=['churn', 'seasonality', 'repurchase'],
                       help='Task type for appendix validation (churn, seasonality, repurchase)')
    parser.add_argument('--dataset', type=str, default='merchant',
                       choices=['merchant', 'cdnow', 'retail', 'instacart', 'sales_weekly', 'tafeng'],
                       help='Dataset to use')
    parser.add_argument('--industry', type=str, default=None,
                       choices=['Industry-0', 'Industry-1', 'Industry-2', 'Industry-3'],
                       help='Industry for merchant dataset')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use')
    parser.add_argument('--multi-seed', action='store_true', help='Run with 3 different random seeds and report statistics')
    args = parser.parse_args()
    
    if args.dataset == 'merchant' and args.industry is None:
        parser.error('--industry is required when --dataset=merchant')
        
    # Run with multiple seeds for statistical reporting
    if args.multi_seed:
        seeds = [42, 123, 456]
        all_results = []
        
        print("="*80)
        print(f"MULTI-SEED STATISTICAL ANALYSIS - TABM {args.dataset.upper()}")
        print("="*80)
        
        for i, seed in enumerate(seeds, 1):
            print(f"\n[SEED {i}/3] Running with seed = {seed}")
            print("-" * 50)
            
            result = run_tabm_single_seed(args, seed)
            all_results.append(result)
            
        # Calculate statistics
        print("\n" + "="*80)
        print("STATISTICAL SUMMARY (3 SEEDS)")
        print("="*80)
        
        metrics = ['accuracy', 'precision', 'recall', 'f1', 'auc', 'auprc']
        
        print(f"Dataset: {args.dataset.upper()}")
        print(f"Seeds: {seeds}")
        print("\nTest Results (Mean ± Std):")
        print("-" * 40)
        
        import numpy as np
        for metric in metrics:
            values = [result[metric] for result in all_results]
            mean_val = np.mean(values)
            std_val = np.std(values, ddof=1)
            
            if metric == 'f1':
                print(f"**{metric.upper():>10}: {mean_val:.4f} ± {std_val:.4f}**")
            else:
                print(f"{metric.upper():>12}: {mean_val:.4f} ± {std_val:.4f}")
        
        print(f"\nIndividual Results:")
        print("-" * 40)
        for i, (seed, result) in enumerate(zip(seeds, all_results)):
            print(f"Seed {seed}: F1 = {result['f1']:.4f}")
    else:
        # Single seed run
        main(args)

