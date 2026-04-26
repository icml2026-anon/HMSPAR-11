"""
Time-MoE Baseline for HMSPAR
A mixture-of-experts transformer model for time series classification
Supports: Merchant (anomaly detection), CDNOW, Retail, Instacart (classification)
"""
import os
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, Dataset
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, average_precision_score
from configs.config import Config
from utils.seed import set_seed
from data_utils import load_dataset, prepare_data_splits, print_dataset_info, print_split_info
import argparse
import math


class TimeMoEDataset(Dataset):
    def __init__(self, X, y):
        # X can be [N, seq_len] or [N, seq_len, features]
        if X.ndim == 2:
            # Merchant data: [N, seq_len] -> [N, seq_len, 1]
            self.X = torch.FloatTensor(X).unsqueeze(-1)
        else:
            self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        return x + self.pe[:x.size(0), :]


class Expert(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.GELU()
    
    def forward(self, x):
        return self.linear2(self.dropout(self.activation(self.linear1(x))))


class MoELayer(nn.Module):
    def __init__(self, d_model, num_experts=8, expert_capacity_factor=1.0, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.num_experts = num_experts
        self.expert_capacity_factor = expert_capacity_factor
        
        # Router/Gate network
        self.gate = nn.Linear(d_model, num_experts, bias=False)
        
        # Experts
        self.experts = nn.ModuleList([
            Expert(d_model, d_model * 4, dropout) for _ in range(num_experts)
        ])
        
        # Noise for load balancing during training
        self.noise_std = 0.1
        
    def forward(self, x):
        batch_size, seq_len, d_model = x.shape
        x_flat = x.view(-1, d_model)  # [batch_size * seq_len, d_model]
        
        # Compute gate scores
        gate_logits = self.gate(x_flat)
        
        # Add noise during training for load balancing
        if self.training:
            noise = torch.randn_like(gate_logits) * self.noise_std
            gate_logits += noise
        
        # Compute routing probabilities
        gate_probs = F.softmax(gate_logits, dim=-1)
        
        # Top-2 gating: select top 2 experts for each token
        top_k = min(2, self.num_experts)
        top_k_probs, top_k_indices = torch.topk(gate_probs, top_k, dim=-1)
        
        # Normalize top-k probabilities
        top_k_probs = top_k_probs / (top_k_probs.sum(dim=-1, keepdim=True) + 1e-8)
        
        # Initialize output
        output = torch.zeros_like(x_flat)
        
        # Route through experts
        for i in range(top_k):
            expert_mask = top_k_indices[:, i]
            expert_probs = top_k_probs[:, i:i+1]
            
            for expert_idx in range(self.num_experts):
                mask = (expert_mask == expert_idx)
                if mask.any():
                    expert_input = x_flat[mask]
                    expert_output = self.experts[expert_idx](expert_input)
                    output[mask] += expert_probs[mask] * expert_output
        
        return output.view(batch_size, seq_len, d_model)


class TimeMoEBlock(nn.Module):
    def __init__(self, d_model, n_heads, num_experts=8, dropout=0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.moe = MoELayer(d_model, num_experts, dropout=dropout)
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, src_mask=None):
        # Self-attention with residual connection
        attn_out, _ = self.self_attn(x, x, x, attn_mask=src_mask)
        x = self.norm1(x + self.dropout(attn_out))
        
        # MoE with residual connection
        moe_out = self.moe(x)
        x = self.norm2(x + self.dropout(moe_out))
        
        return x


class TimeMoEModel(nn.Module):
    def __init__(self, input_dim, d_model=128, n_heads=8, n_layers=6, num_experts=8, 
                 max_seq_len=1000, dropout=0.1, num_classes=1):
        super().__init__()
        self.d_model = d_model
        self.input_dim = input_dim
        
        # Input projection
        self.input_projection = nn.Linear(input_dim, d_model)
        
        # Positional encoding
        self.pos_encoding = PositionalEncoding(d_model, max_seq_len)
        
        # Transformer blocks with MoE
        self.blocks = nn.ModuleList([
            TimeMoEBlock(d_model, n_heads, num_experts, dropout)
            for _ in range(n_layers)
        ])
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, num_classes)
        )
        
        # Global average pooling
        self.pooling = nn.AdaptiveAvgPool1d(1)
        
    def forward(self, x):
        # x: [batch_size, seq_len, input_dim]
        batch_size, seq_len, _ = x.shape
        
        # Project input to model dimension
        x = self.input_projection(x)  # [batch_size, seq_len, d_model]
        
        # Add positional encoding
        x = x.transpose(0, 1)  # [seq_len, batch_size, d_model]
        x = self.pos_encoding(x)
        x = x.transpose(0, 1)  # [batch_size, seq_len, d_model]
        
        # Pass through transformer blocks
        for block in self.blocks:
            x = block(x)
        
        # Global average pooling over sequence dimension
        x = x.transpose(1, 2)  # [batch_size, d_model, seq_len]
        x = self.pooling(x)  # [batch_size, d_model, 1]
        x = x.squeeze(-1)  # [batch_size, d_model]
        
        # Classification
        output = self.classifier(x)  # [batch_size, num_classes]
        
        return output.squeeze(-1) if output.shape[-1] == 1 else output


def compute_metrics(y_true, y_pred, y_proba):
    return {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall': recall_score(y_true, y_pred, zero_division=0),
        'f1': f1_score(y_true, y_pred, zero_division=0),
        'auc': roc_auc_score(y_true, y_proba) if len(np.unique(y_true)) > 1 else 0,
        'auprc': average_precision_score(y_true, y_proba) if len(np.unique(y_true)) > 1 else 0
    }


def train_epoch(model, loader, criterion, optimizer, device, clip_grad_norm=1.0):
    model.train()
    total_loss = 0
    for X_batch, y_batch in loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        
        optimizer.zero_grad()
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)
        loss.backward()
        
        # Gradient clipping for stability
        if clip_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad_norm)
        
        optimizer.step()
        total_loss += loss.item()
    
    return total_loss / len(loader)


def evaluate(model, loader, device):
    model.eval()
    all_probs = []
    all_labels = []
    with torch.no_grad():
        for X_batch, y_batch in loader:
            X_batch = X_batch.to(device)
            outputs = model(X_batch)
            probs = torch.sigmoid(outputs)
            all_probs.extend(probs.cpu().numpy())
            all_labels.extend(y_batch.numpy())
    return np.array(all_labels), np.array(all_probs)


def main(args):
    print("=" * 80)
    print(" Time-MoE Baseline ".center(80, "="))
    print("=" * 80)
    
    set_seed(Config.SEED)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load data using unified loader
    X, y, info = load_dataset(args.dataset, args.industry, getattr(args, 'task', None))
    print_dataset_info(X, y, info)
    print(f"Device: {device}")
    
    # Split data
    X_train, X_val, X_test, y_train, y_val, y_test = prepare_data_splits(X, y)
    print_split_info(X_train, X_val, X_test, y_train, y_val, y_test)
    
    train_dataset = TimeMoEDataset(X_train, y_train)
    val_dataset = TimeMoEDataset(X_val, y_val)
    test_dataset = TimeMoEDataset(X_test, y_test)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size * 2, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size * 2, shuffle=False)
    
    print("\n" + "=" * 80)
    print(" Training Time-MoE ".center(80, "="))
    print("=" * 80)
    
    # Get input_dim from data shape
    if X_train.ndim == 2:
        input_dim = 1  # Merchant: single feature per timestep
        seq_len = X_train.shape[1]
    else:
        input_dim = X_train.shape[2]  # Multiple features per timestep
        seq_len = X_train.shape[1]
    
    print(f"Model config: input_dim={input_dim}, seq_len={seq_len}, d_model={args.d_model}")
    print(f"              n_heads={args.n_heads}, n_layers={args.n_layers}, num_experts={args.num_experts}")
    
    model = TimeMoEModel(
        input_dim=input_dim,
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        num_experts=args.num_experts,
        max_seq_len=seq_len,
        dropout=args.dropout
    ).to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    
    best_val_f1 = -1
    patience = args.patience
    patience_counter = 0
    
    for epoch in range(args.epochs):
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        y_val_true, y_val_proba = evaluate(model, val_loader, device)
        y_val_pred = (y_val_proba > 0.5).astype(int)
        val_f1 = f1_score(y_val_true, y_val_pred, zero_division=0)
        
        scheduler.step()
        
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_model = model.state_dict()
            patience_counter = 0
        else:
            patience_counter += 1
        
        if (epoch + 1) % 10 == 0 or epoch == 0:
            current_lr = optimizer.param_groups[0]['lr']
            print(f"Epoch {epoch+1:3d}: Loss={train_loss:.4f}, Val F1={val_f1:.4f}, LR={current_lr:.2e}")
        
        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch+1}")
            break
    
    model.load_state_dict(best_model)
    
    print("\n" + "=" * 80)
    print(" Evaluation ".center(80, "="))
    print("=" * 80)
    
    y_test_true, y_test_proba = evaluate(model, test_loader, device)
    y_test_pred = (y_test_proba > 0.5).astype(int)
    test_metrics = compute_metrics(y_test_true, y_test_pred, y_test_proba)
    
    print("\nTest Metrics:")
    for k, v in test_metrics.items():
        print(f"  {k}: {v:.4f}")
    
    # Determine result name
    if args.dataset == 'merchant':
        result_name = args.industry.lower().replace("-", "")
    else:
        result_name = args.dataset
    
    results_df = pd.DataFrame([test_metrics])
    results_df['dataset'] = info['name']
    results_df['model'] = 'Time-MoE'
    results_df['d_model'] = args.d_model
    results_df['n_layers'] = args.n_layers
    results_df['num_experts'] = args.num_experts
    
    results_path = Config.RESULTS_DIR / f"timemoe_{result_name}{suffix}_results.csv"
    results_df.to_csv(results_path, index=False)
    print(f"\nResults saved to: {results_path}")
    
    print("\n" + "=" * 80)
    print(" Complete ".center(80, "="))
    print("=" * 80)

def run_timemoe_single_seed(args, seed):
    """Run Time-MoE with a specific seed and return results"""
    print("=" * 80)
    print(" Time-MoE Baseline ".center(80, "="))
    print("=" * 80)
    
    set_seed(seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load data
    X, y, info = load_dataset(args.dataset, args.industry, getattr(args, 'task', None))
    print_dataset_info(X, y, info)
    print(f"Device: {device}")
    
    # Split data  
    X_train, X_val, X_test, y_train, y_val, y_test = prepare_data_splits(X, y)
    print_split_info(X_train, X_val, X_test, y_train, y_val, y_test)
    
    # Get dimensions
    if X_train.ndim == 2:
        seq_len, input_dim = X_train.shape[1], 1
        X_train = X_train.reshape(X_train.shape[0], seq_len, 1)
        X_val = X_val.reshape(X_val.shape[0], seq_len, 1)
        X_test = X_test.reshape(X_test.shape[0], seq_len, 1)
    else:
        seq_len, input_dim = X_train.shape[1], X_train.shape[2]
    
    print(f"\nModel config:")
    print(f"  Input dim: {input_dim}")
    print(f"  Sequence length: {seq_len}")
    print(f"  Model dimension: {args.d_model}")
    print(f"  Number of experts: {args.num_experts}")
    
    # Create datasets
    train_dataset = TensorDataset(torch.FloatTensor(X_train), torch.LongTensor(y_train))
    val_dataset = TensorDataset(torch.FloatTensor(X_val), torch.LongTensor(y_val))
    test_dataset = TensorDataset(torch.FloatTensor(X_test), torch.LongTensor(y_test))
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size * 2, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size * 2, shuffle=False)
    
    # Create model
    model = TimeMoEModel(
        input_dim=input_dim,
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        num_experts=args.num_experts,
        dropout=args.dropout,
        max_seq_len=seq_len,
        num_classes=2
    ).to(device)
    
    # Training setup
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    
    print("\n" + "=" * 80)
    print(" Training Time-MoE ".center(80, "="))
    print("=" * 80)
    
    best_val_f1 = -1
    patience_counter = 0
    
    for epoch in range(args.epochs):
        # Training
        model.train()
        train_loss = 0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            
            optimizer.zero_grad()
            logits = model(X_batch)
            loss = criterion(logits, y_batch)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        # Validation
        model.eval()
        val_probs = []
        val_labels = []
        
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch = X_batch.to(device)
                logits = model(X_batch)
                probs = torch.softmax(logits, dim=-1)[:, 1]
                val_probs.extend(probs.cpu().numpy())
                val_labels.extend(y_batch.numpy())
        
        val_probs = np.array(val_probs)
        val_labels = np.array(val_labels)
        val_preds = (val_probs > 0.5).astype(int)
        val_metrics = compute_metrics(val_labels, val_preds, val_probs)
        val_f1 = val_metrics['f1']
        
        print(f"Epoch {epoch+1:3d}/{args.epochs} | Loss: {train_loss/len(train_loader):.4f} | Val F1: {val_f1:.4f}")
        
        # Early stopping
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            patience_counter = 0
            print(f"Saving model with val_f1: {val_f1:.4f}")
            torch.save(model.state_dict(), 'best_timemoe_model.pth')
        else:
            patience_counter += 1
            
        if patience_counter >= args.patience:
            print(f"Early stopping at epoch {epoch+1}")
            break
    
    # Load best model and test
    import os
    if os.path.exists('best_timemoe_model.pth'):
        try:
            model.load_state_dict(torch.load('best_timemoe_model.pth'))
            print("Loaded best model successfully")
        except RuntimeError as e:
            print(f"Model loading failed due to architecture mismatch: {e}")
            print("Using current model instead")
    else:
        print("Warning: No best model found, using current model")
    model.eval()
    test_probs = []
    test_labels = []
    
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch = X_batch.to(device)
            logits = model(X_batch)
            probs = torch.softmax(logits, dim=-1)[:, 1]
            test_probs.extend(probs.cpu().numpy())
            test_labels.extend(y_batch.numpy())
    
    test_probs = np.array(test_probs)
    test_labels = np.array(test_labels)
    test_preds = (test_probs > 0.5).astype(int)
    test_metrics = compute_metrics(test_labels, test_preds, test_probs)
    
    print("\n" + "=" * 80)
    print(" Test Results ".center(80, "="))
    print("=" * 80)
    for metric, value in test_metrics.items():
        print(f"  {metric}: {value:.4f}")
    
    # Cleanup
    if os.path.exists('best_timemoe_model.pth'):
        os.remove('best_timemoe_model.pth')
    
    return test_metrics

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Time-MoE Baseline for Time Series Classification')
    parser.add_argument('--dataset', type=str, default='merchant',
                       choices=['merchant', 'cdnow', 'retail', 'instacart', 'sales_weekly', 'tafeng'],
                       help='Dataset to use')
    parser.add_argument('--industry', type=str, default=None,
                       choices=['Industry-0', 'Industry-1', 'Industry-2', 'Industry-3'],
                       help='Industry for merchant dataset')
    parser.add_argument('--task', type=str, default=None,
                       choices=['churn', 'seasonality', 'repurchase'],
                       help='Task type for appendix validation')
    
    # Model hyperparameters
    parser.add_argument('--d-model', type=int, default=128, help='Model dimension')
    parser.add_argument('--n-heads', type=int, default=8, help='Number of attention heads')
    parser.add_argument('--n-layers', type=int, default=4, help='Number of transformer layers')
    parser.add_argument('--num-experts', type=int, default=8, help='Number of experts in MoE')
    parser.add_argument('--dropout', type=float, default=0.1, help='Dropout rate')
    
    # Training hyperparameters
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--weight-decay', type=float, default=1e-5, help='Weight decay')
    parser.add_argument('--patience', type=int, default=15, help='Early stopping patience')
    parser.add_argument('--multi-seed', action='store_true', help='Run with 3 different random seeds and report statistics')
    
    args = parser.parse_args()
    
    # Validate arguments
    if args.dataset == 'merchant' and args.industry is None:
        parser.error('--industry is required when --dataset=merchant')
        
    # Run with multiple seeds for statistical reporting
    if args.multi_seed:
        seeds = [42, 123, 456]
        all_results = []
        
        print("="*80)
        print(f"MULTI-SEED STATISTICAL ANALYSIS - TIMEMOE {args.dataset.upper()}")
        print("="*80)
        
        for i, seed in enumerate(seeds, 1):
            print(f"\n[SEED {i}/3] Running with seed = {seed}")
            print("-" * 50)
            
            result = run_timemoe_single_seed(args, seed)
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

