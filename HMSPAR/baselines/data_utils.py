

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
import sys

sys.path.append(str(Path(__file__).parent.parent))
from configs.config import Config


# Dataset configuration
DATASET_INFO = {
    'merchant': {
        'data_dir': Config.DATA_DIR,
        'task': 'anomaly_detection',
        'label_names': ['Normal', 'Anomalous'],
        'has_industries': True,
        'industries': ['Industry-0', 'Industry-1', 'Industry-2', 'Industry-3']
    },
    'cdnow': {
        'data_dir': Config.CDNOW_PROCESSED_DIR,
        'task': 'customer_classification',
        'label_names': ['Low-Value', 'High-Value'],
        'has_industries': False
    },
    'retail': {
        'data_dir': Config.RETAIL_PROCESSED_DIR,
        'task': 'customer_classification', 
        'label_names': ['Low-Value', 'High-Value'],
        'has_industries': False
    },
    'instacart': {
        'data_dir': Config.INSTACART_PROCESSED_DIR,
        'task': 'user_classification',
        'label_names': ['Low-Activity', 'High-Activity'],
        'has_industries': False
    },
    'sales_weekly': {
        'data_dir': Config.SALES_WEEKLY_PROCESSED_DIR,
        'task': 'sales_risk_classification',
        'label_names': ['Low-Risk', 'High-Risk'],
        'has_industries': False
    },
    'tafeng': {
        'data_dir': Config.TAFENG_PROCESSED_DIR,
        'task': 'customer_risk_classification',
        'label_names': ['Low-Risk', 'High-Risk'],
        'has_industries': False
    }
}


def get_dataset_info(dataset):
    """Get dataset information"""
    if dataset not in DATASET_INFO:
        raise ValueError(f"Unknown dataset: {dataset}. Available: {list(DATASET_INFO.keys())}")
    return DATASET_INFO[dataset]


def load_dataset(dataset, industry=None, task=None):
    """
    Load dataset and return X (features), y (labels), and metadata
    """
    info = get_dataset_info(dataset).copy()
    
    if dataset == 'merchant':
        if industry is None:
            raise ValueError("Industry is required for merchant dataset")
        
        df = pd.read_csv(info['data_dir'] / 'merchant_data.csv')
        industry_df = df[df['Industry'] == industry].copy()
        
        ts_cols = [col for col in df.columns if col.startswith('txn_')]
        X = industry_df[ts_cols].fillna(0).values
        y = industry_df['is_anomalous'].values
        
        info['name'] = industry
        info['n_samples'] = len(X)
        info['seq_len'] = X.shape[1]
        info['n_features'] = 1
        
    else:
        data_dir = info['data_dir']
        
        # Check for task-specific subdirectories
        if task:
            if task == 'churn' and dataset in ['retail', 'cdnow', 'instacart']:
                task_dir = data_dir / 'churn_task'
                if task_dir.exists():
                    data_dir = task_dir
                    info['task'] = 'churn_prediction'
                    info['label_names'] = ['No Churn', 'Churn']
            elif task == 'seasonality' and dataset == 'sales_weekly':
                task_dir = data_dir / 'seasonality_task'
                if task_dir.exists():
                    data_dir = task_dir
                    info['task'] = 'seasonality_detection'
                    info['label_names'] = ['Non-Seasonal', 'Seasonal']
            elif task == 'repurchase' and dataset == 'tafeng':
                task_dir = Config.TAFENG_PROCESSED_DIR
                if task_dir.exists():
                    data_dir = task_dir
                    info['task'] = 'repurchase_prediction'
                    info['label_names'] = ['Infrequent', 'Frequent']
        
        # Load time series data based on dataset type
        if dataset == 'tafeng':
            if task == 'repurchase' and (data_dir / 'amount_series.npy').exists():
                # Repurchase task format
                amount_series = np.load(data_dir / 'amount_series.npy')
                trans_series = np.load(data_dir / 'trans_series.npy')
                X = np.stack([amount_series, trans_series], axis=-1)
                labels = np.load(data_dir / 'labels.npy')
            else:
                amount_series = np.load(data_dir / 'amount_series.npy')
                trans_series = np.load(data_dir / 'trans_series.npy')
                X = np.stack([amount_series, trans_series], axis=-1)
                labels = np.load(data_dir / 'labels.npy')
        elif dataset == 'sales_weekly':
            # Sales Weekly format - single time series
            sales_series = np.load(data_dir / 'sales_series.npy')
            X = sales_series[:, :, np.newaxis]  # Add feature dimension
            labels = np.load(data_dir / 'labels.npy')
        elif (data_dir / 'amount_series.npy').exists():
            # CDNOW / Retail format (including churn task)
            amount_series = np.load(data_dir / 'amount_series.npy')
            trans_series = np.load(data_dir / 'trans_series.npy')
            X = np.stack([amount_series, trans_series], axis=-1)
            labels = np.load(data_dir / 'labels.npy')
        elif (data_dir / 'order_count_series.npy').exists():
            # Instacart format (including churn task)
            order_series = np.load(data_dir / 'order_count_series.npy')
            item_series = np.load(data_dir / 'item_count_series.npy')
            X = np.stack([order_series, item_series], axis=-1)
            labels = np.load(data_dir / 'labels.npy')
        else:
            ts_trend = np.load(data_dir / 'ts_trend.npy')
            ts_seasonal = np.load(data_dir / 'ts_seasonal.npy')
            X = np.stack([ts_trend, ts_seasonal], axis=-1)
            labels = np.load(data_dir / 'labels.npy')
        
        y = labels
        
        info['name'] = dataset.upper()
        info['n_samples'] = len(X)
        info['seq_len'] = X.shape[1]
        info['n_features'] = X.shape[2]
    
    return X, y, info


def prepare_data_splits(X, y, test_ratio=None, val_ratio=None, seed=None):
    """Split data into train/val/test sets"""
    test_ratio = test_ratio or Config.TEST_RATIO
    val_ratio = val_ratio or Config.VAL_RATIO
    train_ratio = 1 - test_ratio - val_ratio
    seed = seed or Config.SEED
    
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y, test_size=test_ratio, random_state=seed, stratify=y
    )
    
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val,
        test_size=val_ratio / (train_ratio + val_ratio),
        random_state=seed, stratify=y_train_val
    )
    
    return X_train, X_val, X_test, y_train, y_val, y_test


def flatten_time_series(X):
    """Flatten time series for tree-based models (use raw values)"""
    if X.ndim == 2:
        return X
    elif X.ndim == 3:
        # (n_samples, seq_len, n_features) -> (n_samples, seq_len * n_features)
        return X.reshape(X.shape[0], -1)
    else:
        raise ValueError(f"Unexpected X shape: {X.shape}")


def print_dataset_info(X, y, info):
    """Print dataset information"""
    print(f"\nDataset: {info['name']}")
    print(f"Task: {info['task']}")
    print(f"Labels: {info['label_names']}")
    print(f"Total samples: {len(X)}")
    print(f"Sequence shape: {X.shape}")
    print(f"Positive class ratio: {y.mean():.2%}")


def print_split_info(X_train, X_val, X_test, y_train, y_val, y_test):
    """Print data split information"""
    print(f"\nData split:")
    print(f"  Train: {len(X_train)} ({y_train.mean():.2%} positive)")
    print(f"  Val:   {len(X_val)} ({y_val.mean():.2%} positive)")
    print(f"  Test:  {len(X_test)} ({y_test.mean():.2%} positive)")


def get_all_dataset_configs():
    """Get all dataset configurations for running experiments"""
    configs = []
    for industry in DATASET_INFO['merchant']['industries']:
        configs.append(('merchant', industry))
    for dataset in ['cdnow', 'retail', 'instacart', 'sales_weekly', 'tafeng']:
        configs.append((dataset, None))
    return configs
