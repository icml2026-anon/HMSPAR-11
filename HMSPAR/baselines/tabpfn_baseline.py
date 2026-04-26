"""
TabPFN Baseline for HMSPAR
Supports: Merchant (anomaly detection), CDNOW, Retail, Instacart (classification)
"""
import sys
import os
from pathlib import Path

os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
os.environ['TABPFN_TELEMETRY_ENABLED'] = '0'

sys.path.append(str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, average_precision_score
import torch
from tabpfn import TabPFNClassifier  # type: ignore
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


def main(args, seed=None):
    print("=" * 80)
    print(" TabPFN Baseline ".center(80, "="))
    print("=" * 80)
    
    if seed is not None:
        set_seed(seed)
    else:
        set_seed(Config.SEED)
    
    # Load data using unified loader
    X, y, info = load_dataset(args.dataset, args.industry, getattr(args, 'task', None))
    print_dataset_info(X, y, info)
    
    # Flatten for tabular model
    X = flatten_time_series(X)
    
    # Split data
    X_train, X_val, X_test, y_train, y_val, y_test = prepare_data_splits(X, y)
    print_split_info(X_train, X_val, X_test, y_train, y_val, y_test)
    
    print("\n" + "=" * 80)
    print(" Training TabPFN ".center(80, "="))
    print("=" * 80)
    
    requested_device = getattr(args, 'device', 'auto')
    if requested_device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = requested_device
    
    model = TabPFNClassifier(device=device, n_estimators=4, ignore_pretraining_limits=True)
    
    try:
        model.fit(X_train, y_train)
    except Exception as e:
        err_str = str(e)
        if device == 'cuda' and (
            'invalid configuration argument' in err_str
            or 'torch.AcceleratorError' in err_str
            or 'CUDA error' in err_str
        ):
            print("\n[TabPFN] CUDA execution failed during fit; falling back to CPU. Error was:")
            print(f"  {e}")
            device = 'cpu'
            model = TabPFNClassifier(device=device, n_estimators=4, ignore_pretraining_limits=True)
            model.fit(X_train, y_train)
        else:
            raise
    
    print("\n" + "=" * 80)
    print(" Evaluating on Test Set ".center(80, "="))
    print("=" * 80)
    
    try:
        y_test_proba = model.predict_proba(X_test)[:, 1]
    except Exception as e:
        err_str = str(e)
        if device == 'cuda' and (
            'invalid configuration argument' in err_str
            or 'torch.AcceleratorError' in err_str
            or 'CUDA error' in err_str
        ):
            print("\n[TabPFN] CUDA execution failed during predict_proba; falling back to CPU. Error was:")
            print(f"  {e}")
            device = 'cpu'
            model = TabPFNClassifier(device=device, n_estimators=4, ignore_pretraining_limits=True)
            model.fit(X_train, y_train)
            y_test_proba = model.predict_proba(X_test)[:, 1]
        else:
            raise
    
    y_test_pred = (y_test_proba > 0.5).astype(int)
    test_metrics = compute_metrics(y_test, y_test_pred, y_test_proba)
    
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
    results_path = Config.RESULTS_DIR / f"tabpfn_{result_name}{suffix}_results.csv"
    results_df.to_csv(results_path, index=False)
    print(f"\nResults saved to: {results_path}")
    
    print("\n" + "=" * 80)
    print(" Complete ".center(80, "="))
    print("=" * 80)
    
    return test_metrics

def run_tabpfn_single_seed(args, seed):
    """Run TabPFN with a specific seed and return results"""
    # Call main with the specific seed and return actual test metrics
    return main(args, seed)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='merchant',
                       choices=['merchant', 'cdnow', 'retail', 'instacart', 'sales_weekly', 'tafeng'],
                       help='Dataset to use')
    parser.add_argument('--industry', type=str, default=None,
                       choices=['Industry-0', 'Industry-1', 'Industry-2', 'Industry-3'],
                       help='Industry for merchant dataset')
    parser.add_argument('--task', type=str, default=None,
                       choices=['churn', 'seasonality', 'repurchase'],
                       help='Task type for appendix validation (churn, seasonality, repurchase)')
    parser.add_argument('--device', type=str, default='auto', choices=['auto', 'cuda', 'cpu'],
                       help='Device for TabPFN (auto will try cuda first then fallback to cpu)')
    parser.add_argument('--multi-seed', action='store_true', help='Run with 3 different random seeds and report statistics')
    args = parser.parse_args()
    
    if args.dataset == 'merchant' and args.industry is None:
        parser.error('--industry is required when --dataset=merchant')
        
    # Run with multiple seeds for statistical reporting
    if args.multi_seed:
        seeds = [42, 123, 456]
        all_results = []
        
        print("="*80)
        print(f"MULTI-SEED STATISTICAL ANALYSIS - TABPFN {args.dataset.upper()}")
        print("="*80)
        
        for i, seed in enumerate(seeds, 1):
            print(f"\n[SEED {i}/3] Running with seed = {seed}")
            print("-" * 50)
            
            result = run_tabpfn_single_seed(args, seed)
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
