"""Configuration for HMSPAR."""
import torch
from pathlib import Path

class Config:
    SEED = 42
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    PROJECT_ROOT = Path(__file__).parent.parent
    DATA_DIR     = PROJECT_ROOT / "data"
    PRETRAINED_DIR = PROJECT_ROOT / "pretrained_models"
    CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"
    RESULTS_DIR    = PROJECT_ROOT / "results"

    
    N_MERCHANTS    = 15000
    K_INDUSTRIES   = 4
    START_DATE     = '2021-01'
    END_DATE       = '2023-11'
    ANOMALOUS_RATIO = 0.175

    IMAGE_SIZE      = 32
    TEXT_MODEL_NAME = 'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2'

    TRAIN_RATIO = 0.7
    VAL_RATIO   = 0.15
    TEST_RATIO  = 0.15

    # ── Dataset paths (resolved relative to DATA_DIR) ────────────────
    RETAIL_DATA_DIR      = DATA_DIR / 'retail'
    RETAIL_PROCESSED_DIR = RETAIL_DATA_DIR / 'processed'
    RETAIL_AMOUNT_CSV    = RETAIL_DATA_DIR / 'online_retail_amount.csv'
    RETAIL_TRANS_CSV     = RETAIL_DATA_DIR / 'online_retail_transactions.csv'
    RETAIL_IMAGE_SIZE    = 13
    RETAIL_HYPERPARAMS   = {
        'learning_rate': 1e-3,
        'weight_decay':  1e-4,
        'epochs':        20,
        'batch_size':    64,
        'ts_hidden_dim': 128,
        'fusion_dim':    256,
        'dropout_rate':  0.3,
        'input_dim':     2,
        'seq_len':       13,
    }

    
    CDNOW_DATA_DIR      = DATA_DIR / 'cdnow'
    CDNOW_PROCESSED_DIR = CDNOW_DATA_DIR / 'processed'
    CDNOW_AMOUNT_CSV    = CDNOW_DATA_DIR / 'cdnow_amount.csv'
    CDNOW_TRANS_CSV     = CDNOW_DATA_DIR / 'cdnow_transactions.csv'
    CDNOW_IMAGE_SIZE    = 13
    CDNOW_HYPERPARAMS   = {
        'learning_rate': 1e-3,
        'weight_decay':  1e-4,
        'epochs':        10,
        'batch_size':    64,
        'ts_hidden_dim': 128,
        'fusion_dim':    256,
        'dropout_rate':  0.3,
        'input_dim':     2,
        'seq_len':       13,
    }

    INSTACART_DATA_DIR      = DATA_DIR / 'instacart'
    INSTACART_PROCESSED_DIR = INSTACART_DATA_DIR / 'processed'
    INSTACART_IMAGE_SIZE    = 20
    INSTACART_HYPERPARAMS   = {
        'learning_rate': 1e-3,
        'weight_decay':  1e-4,
        'epochs':        20,
        'batch_size':    64,
        'ts_hidden_dim': 128,
        'fusion_dim':    256,
        'dropout_rate':  0.3,
        'input_dim':     2,
        'seq_len':       20,
        'focal_alpha':   0.1,
        'focal_gamma':   3.0,
    }

    SALES_WEEKLY_DATA_DIR      = DATA_DIR / 'sales_weekly'
    SALES_WEEKLY_PROCESSED_DIR = SALES_WEEKLY_DATA_DIR / 'processed'
    SALES_WEEKLY_CSV           = SALES_WEEKLY_DATA_DIR / 'sales_weekly.csv'
    SALES_WEEKLY_IMAGE_SIZE    = 52
    SALES_WEEKLY_HYPERPARAMS   = {
        'learning_rate': 1e-3,
        'weight_decay':  1e-4,
        'epochs':        20,
        'batch_size':    32,
        'ts_hidden_dim': 128,
        'fusion_dim':    256,
        'dropout_rate':  0.3,
        'input_dim':     1,
        'seq_len':       52,
    }

    TAFENG_DATA_DIR      = DATA_DIR / 'tafeng'
    TAFENG_PROCESSED_DIR = TAFENG_DATA_DIR / 'repurchase_task'
    TAFENG_CSV           = DATA_DIR / 'ta_feng_all_months_merged.csv'
    TAFENG_IMAGE_SIZE    = 4
    TAFENG_HYPERPARAMS   = {
        'learning_rate': 1e-3,
        'weight_decay':  1e-4,
        'epochs':        20,
        'batch_size':    64,
        'ts_hidden_dim': 128,
        'fusion_dim':    256,
        'dropout_rate':  0.3,
        'input_dim':     2,
        'seq_len':       4,
    }

    # Per-industry hyperparameters
    INDUSTRY_HYPERPARAMS = {
        'Industry-0': {
            'learning_rate': 1e-3,
            'weight_decay':  1e-4,
            'epochs':        10,
            'batch_size':    64,
            'ts_hidden_dim': 256,
            'fusion_dim':    256,
            'dropout_rate':  0.3,
        },
        'Industry-1': {
            'learning_rate': 1e-3,
            'weight_decay':  1e-4,
            'epochs':        10,
            'batch_size':    64,
            'ts_hidden_dim': 64,
            'fusion_dim':    256,
            'dropout_rate':  0.4,
        },
        'Industry-2': {
            'learning_rate': 5e-4,
            'weight_decay':  1e-4,
            'epochs':        10,
            'batch_size':    32,
            'ts_hidden_dim': 256,
            'fusion_dim':    128,
            'dropout_rate':  0.0,
        },
        'Industry-3': {
            'learning_rate': 1e-3,
            'weight_decay':  1e-4,
            'epochs':        10,
            'batch_size':    32,
            'ts_hidden_dim': 256,
            'fusion_dim':    128,
            'dropout_rate':  0.1,
            'order':         4,
            'n_experts':     4,
            'n_heads':       2,
        },
    }

    @classmethod
    def get_industry_params(cls, industry):
        """Return hyperparameters for a given industry segment."""
        return cls.INDUSTRY_HYPERPARAMS.get(industry, cls.INDUSTRY_HYPERPARAMS['Industry-0'])
