#!/usr/bin/env python3
"""
Cross-Region Validation (Split C) for DLR Paper - Version 2

This script implements the critical cross-region validation experiments:
- Split C.1: Train on Vietnam → Test on US
- Split C.2: Train on US → Test on Vietnam  
- Split C.3: Within-region baselines (Vietnam-only and US-only)

Simplified architecture using a proven MLP design.

Author: Code Assistant
Copyright (C) 2025 Gelavizh Ahmadi / Invariant Research
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import json
import argparse
import sys
import time
from typing import Dict, Tuple, Optional, List
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))


# =============================================================================
# Simple MLP Model (Proven Architecture)
# =============================================================================

class SimpleDLRModel(nn.Module):
    """
    Simple MLP for Dynamic Line Rating prediction.
    Proven to work well for tabular regression tasks.
    """
    
    def __init__(self, input_dim=3, hidden_dims=[128, 128, 64], dropout=0.1):
        super().__init__()
        
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.LayerNorm(hidden_dim))
            layers.append(nn.GELU())
            layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim
        
        self.encoder = nn.Sequential(*layers)
        
        # Single output head for ampacity
        self.ampacity_head = nn.Sequential(
            nn.Linear(hidden_dims[-1], 32),
            nn.GELU(),
            nn.Linear(32, 1)
        )
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(self, x):
        """
        Forward pass.
        Args:
            x: [batch, input_dim] - normalized input features
        Returns:
            ampacity: [batch, 1] - predicted ampacity
        """
        h = self.encoder(x)
        ampacity = self.ampacity_head(h)
        return ampacity


# =============================================================================
# Configuration
# =============================================================================

class CrossRegionConfig:
    """Configuration for cross-region validation experiments"""
    
    UNIFIED_DATA_PATH = "data/processed/unified_dlr_training.h5"
    OUTPUT_DIR = "cross_region_results"
    
    # Model architecture
    HIDDEN_DIMS = [128, 128, 64]
    DROPOUT = 0.1
    
    # Training hyperparameters
    EPOCHS = 100
    BATCH_SIZE = 256
    LEARNING_RATE = 1e-3
    WEIGHT_DECAY = 1e-4
    
    # Device
    DEVICE = torch.device("mps" if torch.backends.mps.is_available() 
                         else "cuda" if torch.cuda.is_available() 
                         else "cpu")
    
    # Random seeds
    SEED = 42
    
    # Test set sizes
    US_TEST_SAMPLES = 10000


# =============================================================================
# Data Loading and Preprocessing
# =============================================================================

class CrossRegionDataset(Dataset):
    """Dataset for cross-region DLR experiments."""
    
    def __init__(
        self,
        df: pd.DataFrame,
        region: str,
        feature_cols: List[str],
        target_col: str,
        normalizer: Optional['CrossRegionNormalizer'] = None,
        is_training: bool = True
    ):
        self.region = region
        self.feature_cols = feature_cols
        self.target_col = target_col
        
        # Filter to region and drop NaN
        region_df = df[df['region'] == region].copy()
        region_df = region_df.dropna(subset=feature_cols + [target_col])
        self.df = region_df
        
        # Extract features and target
        self.X = self.df[feature_cols].values.astype(np.float32)
        self.y = self.df[target_col].values.astype(np.float32).reshape(-1, 1)
        
        # Setup or use normalizer
        if normalizer is None and is_training:
            self.normalizer = CrossRegionNormalizer()
            self.normalizer.fit(self.X)
        elif normalizer is not None:
            self.normalizer = normalizer
        else:
            raise ValueError("Must provide normalizer for validation/test data")
        
        # Normalize features
        self.X_normalized = self.normalizer.transform(self.X)
        
        print(f"  {region} dataset: {len(self)} samples, {len(feature_cols)} features")
        print(f"    Target: {target_col}, mean={self.y.mean():.1f}, std={self.y.std():.1f}")
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return (
            torch.tensor(self.X_normalized[idx], dtype=torch.float32),
            torch.tensor(self.y[idx], dtype=torch.float32)
        )


class CrossRegionNormalizer:
    """Normalizer for cross-region experiments"""
    
    def __init__(self):
        self.mean = None
        self.std = None
    
    def fit(self, X: np.ndarray):
        self.mean = X.mean(axis=0)
        self.std = X.std(axis=0)
        self.std[self.std < 1e-8] = 1.0
        return self
    
    def transform(self, X: np.ndarray) -> np.ndarray:
        return (X - self.mean) / self.std
    
    def inverse_transform(self, X: np.ndarray) -> np.ndarray:
        return X * self.std + self.mean


def load_unified_data(config: CrossRegionConfig) -> pd.DataFrame:
    """Load unified dataset from HDF5"""
    print(f"\n📊 Loading unified data from {config.UNIFIED_DATA_PATH}...")
    df = pd.read_hdf(config.UNIFIED_DATA_PATH, key='data')
    print(f"   Total samples: {len(df):,}")
    print(f"   Region distribution:")
    for region, count in df['region'].value_counts().items():
        print(f"     {region}: {count:,}")
    return df


def prepare_data_splits(df: pd.DataFrame, config: CrossRegionConfig, split_type: str):
    """Prepare data splits for cross-region experiments."""
    
    print(f"\n{'='*60}")
    print(f"Preparing data for: {split_type}")
    print(f"{'='*60}")
    
    # Use features that exist in both regions
    # Vietnam has: temp, Wind1, WinDir, GHI, Ampacity
    # US has: temperature, wind_speed, wind_direction, solar_irradiance, actual
    # Common features: temperature, wind_speed, solar_irradiance
    feature_cols = ['temperature', 'wind_speed', 'solar_irradiance']
    
    if split_type == 'VN_to_US':
        # Vietnam: 80/20 train/val
        vn_df = df[df['region'] == 'VN'].copy()
        vn_train, vn_val = train_test_split(vn_df, test_size=0.2, random_state=config.SEED)
        
        # US: Random 10k samples for test
        us_df = df[df['region'] == 'US'].copy()
        us_test = us_df.sample(n=min(config.US_TEST_SAMPLES, len(us_df)), random_state=config.SEED)
        
        train_dataset = CrossRegionDataset(vn_train, 'VN', feature_cols, 'Ampacity', 
                                           normalizer=None, is_training=True)
        val_dataset = CrossRegionDataset(vn_val, 'VN', feature_cols, 'Ampacity',
                                         normalizer=train_dataset.normalizer, is_training=False)
        test_dataset = CrossRegionDataset(us_test, 'US', feature_cols, 'actual',
                                          normalizer=train_dataset.normalizer, is_training=False)
        
    elif split_type == 'US_to_VN':
        # US: 80/20 train/val (sample for efficiency)
        us_df = df[df['region'] == 'US'].copy()
        if len(us_df) > 50000:
            us_df = us_df.sample(n=50000, random_state=config.SEED)
        us_train, us_val = train_test_split(us_df, test_size=0.2, random_state=config.SEED)
        
        # Vietnam: All as test
        vn_test = df[df['region'] == 'VN'].copy()
        
        train_dataset = CrossRegionDataset(us_train, 'US', feature_cols, 'actual',
                                           normalizer=None, is_training=True)
        val_dataset = CrossRegionDataset(us_val, 'US', feature_cols, 'actual',
                                         normalizer=train_dataset.normalizer, is_training=False)
        test_dataset = CrossRegionDataset(vn_test, 'VN', feature_cols, 'Ampacity',
                                          normalizer=train_dataset.normalizer, is_training=False)
        
    elif split_type == 'VN_only':
        vn_df = df[df['region'] == 'VN'].copy()
        vn_train, vn_temp = train_test_split(vn_df, test_size=0.2, random_state=config.SEED)
        vn_val, vn_test = train_test_split(vn_temp, test_size=0.5, random_state=config.SEED)
        
        train_dataset = CrossRegionDataset(vn_train, 'VN', feature_cols, 'Ampacity',
                                           normalizer=None, is_training=True)
        val_dataset = CrossRegionDataset(vn_val, 'VN', feature_cols, 'Ampacity',
                                         normalizer=train_dataset.normalizer, is_training=False)
        test_dataset = CrossRegionDataset(vn_test, 'VN', feature_cols, 'Ampacity',
                                          normalizer=train_dataset.normalizer, is_training=False)
        
    elif split_type == 'US_only':
        us_df = df[df['region'] == 'US'].copy()
        # Sample 50k for faster training (still representative)
        if len(us_df) > 50000:
            us_df = us_df.sample(n=50000, random_state=config.SEED)
        us_train, us_temp = train_test_split(us_df, test_size=0.2, random_state=config.SEED)
        us_val, us_test = train_test_split(us_temp, test_size=0.5, random_state=config.SEED)
        
        train_dataset = CrossRegionDataset(us_train, 'US', feature_cols, 'actual',
                                           normalizer=None, is_training=True)
        val_dataset = CrossRegionDataset(us_val, 'US', feature_cols, 'actual',
                                         normalizer=train_dataset.normalizer, is_training=False)
        test_dataset = CrossRegionDataset(us_test, 'US', feature_cols, 'actual',
                                          normalizer=train_dataset.normalizer, is_training=False)
    else:
        raise ValueError(f"Unknown split_type: {split_type}")
    
    return train_dataset, val_dataset, test_dataset


# =============================================================================
# Training
# =============================================================================

def train_model(train_dataset, val_dataset, config, experiment_name):
    """Train the model."""
    
    print(f"\n🚀 Training model for {experiment_name}...")
    print(f"   Device: {config.DEVICE}")
    print(f"   Epochs: {config.EPOCHS}")
    
    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, 
                              shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE, 
                            shuffle=False, num_workers=0)
    
    # Create model (3 input features: temperature, wind_speed, solar_irradiance)
    model = SimpleDLRModel(input_dim=3, hidden_dims=config.HIDDEN_DIMS, 
                          dropout=config.DROPOUT)
    model = model.to(config.DEVICE)
    print(f"   Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Optimizer and scheduler
    optimizer = optim.AdamW(model.parameters(), lr=config.LEARNING_RATE, 
                           weight_decay=config.WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', 
                                                     factor=0.5, patience=10)
    criterion = nn.MSELoss()
    
    # Training loop
    best_val_loss = float('inf')
    best_model_state = None
    history = {'train_loss': [], 'val_loss': [], 'val_mae': []}
    start_time = time.time()
    
    for epoch in range(config.EPOCHS):
        # Training
        model.train()
        train_losses = []
        
        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(config.DEVICE)
            batch_y = batch_y.to(config.DEVICE)
            
            optimizer.zero_grad()
            pred = model(batch_x)
            loss = criterion(pred, batch_y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            train_losses.append(loss.item())
        
        # Validation
        model.eval()
        val_losses = []
        val_preds = []
        val_targets = []
        
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x = batch_x.to(config.DEVICE)
                batch_y = batch_y.to(config.DEVICE)
                
                pred = model(batch_x)
                loss = criterion(pred, batch_y)
                
                val_losses.append(loss.item())
                val_preds.extend(pred.cpu().numpy())
                val_targets.extend(batch_y.cpu().numpy())
        
        avg_train_loss = np.mean(train_losses)
        avg_val_loss = np.mean(val_losses)
        val_mae = mean_absolute_error(val_targets, val_preds)
        
        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(avg_val_loss)
        history['val_mae'].append(val_mae)
        
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_model_state = model.state_dict().copy()
        
        if (epoch + 1) % 10 == 0 or epoch == 0:
            elapsed = time.time() - start_time
            print(f"  Epoch {epoch+1:3d}/{config.EPOCHS} | "
                  f"Train: {avg_train_loss:.2e} | Val: {avg_val_loss:.2e} | "
                  f"MAE: {val_mae:.1f}A | {elapsed:.1f}s")
        
        scheduler.step(avg_val_loss)
    
    # Load best model
    if best_model_state:
        model.load_state_dict(best_model_state)
    
    print(f"\n✅ Training complete. Best val MAE: {min(history['val_mae']):.1f}A")
    return model, history


# =============================================================================
# Evaluation
# =============================================================================

def evaluate_model(model, test_dataset, config, experiment_name):
    """Evaluate model on test set."""
    
    print(f"\n📊 Evaluating {experiment_name}...")
    
    test_loader = DataLoader(test_dataset, batch_size=config.BATCH_SIZE, 
                            shuffle=False, num_workers=0)
    
    model.eval()
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            batch_x = batch_x.to(config.DEVICE)
            pred = model(batch_x)
            all_preds.extend(pred.cpu().numpy())
            all_targets.extend(batch_y.numpy())
    
    predictions = np.array(all_preds).flatten()
    targets = np.array(all_targets).flatten()
    
    mae = mean_absolute_error(targets, predictions)
    rmse = np.sqrt(mean_squared_error(targets, predictions))
    r2 = r2_score(targets, predictions)
    bias = np.mean(predictions - targets)
    
    print(f"   MAE:  {mae:.2f} A")
    print(f"   RMSE: {rmse:.2f} A")
    print(f"   Bias: {bias:.2f} A")
    print(f"   R²:   {r2:.4f}")
    
    return {
        'experiment': experiment_name,
        'n_samples': len(predictions),
        'mae': float(mae),
        'rmse': float(rmse),
        'bias': float(bias),
        'r2': float(r2),
        'predictions': predictions,
        'targets': targets
    }


# =============================================================================
# Visualization
# =============================================================================

def normalize_exp_name(exp_name):
    """Convert experiment names to standardized format"""
    name_map = {'VN_only': 'VN_to_VN', 'US_only': 'US_to_US'}
    return name_map.get(exp_name, exp_name)


def create_results_table(all_metrics):
    """Create markdown results table"""
    
    vn_baseline_mae = None
    us_baseline_mae = None
    
    for m in all_metrics:
        norm_name = normalize_exp_name(m['experiment'])
        if norm_name == 'VN_to_VN':
            vn_baseline_mae = m['mae']
        elif norm_name == 'US_to_US':
            us_baseline_mae = m['mae']
    
    table = "## Cross-Region Transfer Learning Results\n\n"
    table += "| Train Region | Test Region | MAE (A) | RMSE (A) | Bias (A) | R² | vs. Within-Region |\n"
    table += "|--------------|-------------|---------|----------|----------|-----|-------------------|\n"
    
    for m in all_metrics:
        norm_name = normalize_exp_name(m['experiment'])
        parts = norm_name.split('_to_')
        if len(parts) != 2:
            continue
        train_region, test_region = parts
        
        if train_region == test_region:
            vs_str = "Baseline"
        elif test_region == 'VN' and vn_baseline_mae:
            vs_str = f"+{(m['mae'] / vn_baseline_mae - 1) * 100:.1f}%"
        elif test_region == 'US' and us_baseline_mae:
            vs_str = f"+{(m['mae'] / us_baseline_mae - 1) * 100:.1f}%"
        else:
            vs_str = "N/A"
        
        table += f"| {train_region} | {test_region} | {m['mae']:.1f} | {m['rmse']:.1f} | {m['bias']:.1f} | {m['r2']:.3f} | {vs_str} |\n"
    
    return table


def create_visualizations(all_metrics, output_dir):
    """Create publication-quality visualizations"""
    
    print("\n📈 Creating visualizations...")
    output_dir = Path(output_dir)
    
    # Figure A: Scatter plots
    fig, axes = plt.subplots(2, 2, figsize=(12, 12))
    fig.suptitle('Cross-Region Prediction Scatter Plots', fontsize=14, fontweight='bold')
    
    experiments = ['VN_to_VN', 'VN_to_US', 'US_to_US', 'US_to_VN']
    titles = ['Vietnam → Vietnam (Within-Region)', 'Vietnam → US (Cross-Region)',
              'US → US (Within-Region)', 'US → Vietnam (Cross-Region)']
    
    for idx, (exp, title) in enumerate(zip(experiments, titles)):
        ax = axes[idx // 2, idx % 2]
        metrics = next((m for m in all_metrics if normalize_exp_name(m['experiment']) == exp), None)
        
        if metrics is None:
            ax.text(0.5, 0.5, 'No data', ha='center', va='center')
            ax.set_title(title)
            continue
        
        preds, targets = metrics['predictions'], metrics['targets']
        ax.scatter(targets, preds, alpha=0.5, s=10)
        
        min_val = min(targets.min(), preds.min())
        max_val = max(targets.max(), preds.max())
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2)
        ax.annotate(f'R² = {metrics["r2"]:.3f}', xy=(0.05, 0.95), 
                   xycoords='axes fraction', fontsize=12, verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        ax.set_xlabel('True Ampacity (A)')
        ax.set_ylabel('Predicted Ampacity (A)')
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'figure_A_cross_region_scatter.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"   Saved: figure_A_cross_region_scatter.png")
    
    # Figure B: Error distributions
    fig, ax = plt.subplots(figsize=(10, 6))
    
    error_data, labels = [], []
    for exp in experiments:
        metrics = next((m for m in all_metrics if normalize_exp_name(m['experiment']) == exp), None)
        if metrics:
            error_data.append(metrics['predictions'] - metrics['targets'])
            labels.append(exp.replace('_to_', '→'))
    
    if error_data:
        bp = ax.boxplot(error_data, labels=labels, patch_artist=True)
        colors = ['lightblue', 'lightcoral', 'lightblue', 'lightcoral']
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
        ax.axhline(y=0, color='red', linestyle='--', linewidth=2)
        ax.set_ylabel('Prediction Error (A)')
        ax.set_title('Error Distribution Comparison (Blue=Within-Region, Red=Cross-Region)')
        ax.grid(True, alpha=0.3, axis='y')
        plt.tight_layout()
        plt.savefig(output_dir / 'figure_B_error_distributions.png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"   Saved: figure_B_error_distributions.png")


# =============================================================================
# Main
# =============================================================================

def run_experiment(split_type, config, df):
    """Run a single cross-region experiment"""
    
    print(f"\n{'='*70}")
    print(f"EXPERIMENT: {split_type}")
    print(f"{'='*70}")
    
    train_dataset, val_dataset, test_dataset = prepare_data_splits(df, config, split_type)
    model, history = train_model(train_dataset, val_dataset, config, split_type)
    metrics = evaluate_model(model, test_dataset, config, split_type)
    
    # Save model
    output_dir = Path(config.OUTPUT_DIR)
    output_dir.mkdir(exist_ok=True)
    torch.save({
        'model_state_dict': model.state_dict(),
        'normalizer_mean': train_dataset.normalizer.mean,
        'normalizer_std': train_dataset.normalizer.std,
        'history': history,
        'metrics': {k: v for k, v in metrics.items() if k not in ['predictions', 'targets']}
    }, output_dir / f"model_{split_type}.pt")
    
    return metrics


def main():
    parser = argparse.ArgumentParser(description='Cross-Region Validation for DLR Paper (Split C)')
    parser.add_argument('--experiments', nargs='+', 
                       choices=['VN_to_US', 'US_to_VN', 'VN_only', 'US_only', 'all'],
                       default=['all'], help='Which experiments to run')
    parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs')
    parser.add_argument('--quick', action='store_true', help='Quick mode: fewer epochs')
    parser.add_argument('--output-dir', type=str, default='cross_region_results')
    
    args = parser.parse_args()
    
    config = CrossRegionConfig()
    config.EPOCHS = 20 if args.quick else args.epochs
    config.OUTPUT_DIR = args.output_dir
    
    print("="*70)
    print("CROSS-REGION VALIDATION (SPLIT C) FOR DLR PAPER")
    print("="*70)
    print(f"Device: {config.DEVICE}")
    print(f"Epochs: {config.EPOCHS}")
    
    torch.manual_seed(config.SEED)
    np.random.seed(config.SEED)
    
    df = load_unified_data(config)
    
    experiments = ['VN_only', 'US_only', 'VN_to_US', 'US_to_VN'] if 'all' in args.experiments else args.experiments
    
    all_metrics = []
    for exp in experiments:
        try:
            metrics = run_experiment(exp, config, df)
            all_metrics.append(metrics)
        except Exception as e:
            print(f"\n❌ Error in experiment {exp}: {e}")
            import traceback
            traceback.print_exc()
    
    if all_metrics:
        output_dir = Path(config.OUTPUT_DIR)
        output_dir.mkdir(exist_ok=True)
        
        table = create_results_table(all_metrics)
        print("\n" + "="*70)
        print(table)
        print("="*70)
        
        with open(output_dir / 'results_table.md', 'w') as f:
            f.write(table)
        
        create_visualizations(all_metrics, output_dir)
        
        metrics_dict = {m['experiment']: {k: v for k, v in m.items() 
                                         if k not in ['predictions', 'targets']} 
                       for m in all_metrics}
        with open(output_dir / 'metrics.json', 'w') as f:
            json.dump(metrics_dict, f, indent=2)
        
        # Analysis
        print("\n" + "="*70)
        print("ANALYSIS")
        print("="*70)
        
        vn_to_vn = next((m for m in all_metrics if normalize_exp_name(m['experiment']) == 'VN_to_VN'), None)
        vn_to_us = next((m for m in all_metrics if normalize_exp_name(m['experiment']) == 'VN_to_US'), None)
        us_to_us = next((m for m in all_metrics if normalize_exp_name(m['experiment']) == 'US_to_US'), None)
        us_to_vn = next((m for m in all_metrics if normalize_exp_name(m['experiment']) == 'US_to_VN'), None)
        
        if vn_to_us and us_to_us:
            degradation = (vn_to_us['mae'] / us_to_us['mae'] - 1) * 100
            print(f"\n1. Vietnam → US Transfer:")
            print(f"   Within-US MAE: {us_to_us['mae']:.1f} A, Vietnam→US MAE: {vn_to_us['mae']:.1f} A")
            print(f"   Degradation: +{degradation:.1f}%")
            print(f"   {'⚠️ Transfer FAILED' if vn_to_us['mae'] > 400 else '✅ Transfer reasonably successful'}")
        
        if us_to_vn and vn_to_vn:
            degradation = (us_to_vn['mae'] / vn_to_vn['mae'] - 1) * 100
            print(f"\n2. US → Vietnam Transfer:")
            print(f"   Within-VN MAE: {vn_to_vn['mae']:.1f} A, US→VN MAE: {us_to_vn['mae']:.1f} A")
            print(f"   Degradation: +{degradation:.1f}%")
            print(f"   {'✅ Transfer successful' if us_to_vn['mae'] < 300 else '⚠️ Room for improvement'}")
        
        print("\n" + "="*70)
        print("✅ CROSS-REGION VALIDATION COMPLETE")
        print("="*70)
        print(f"\nResults saved to: {output_dir}")


if __name__ == "__main__":
    main()
