#!/usr/bin/env python3
"""
Cross-Region Validation (Split C) using HWF-PIKAN (InvariantPIKAN v2)

Uses the proper InvariantPIKAN v2 architecture with LinePhysicsParams.
For unseen region lines, uses global mean μ only (zero deviation vectors).

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

sys.path.append(str(Path(__file__).parent.parent))

from models.invariant_pikan_v2 import InvariantPIKANV2
from models.line_physics import LinePhysicsParams, LinePhysicsConfig


# =============================================================================
# HWF-PIKAN Model with LinePhysics
# =============================================================================

class HWFPIKANWithLinePhysics(nn.Module):
    """
    HWF-PIKAN (InvariantPIKAN v2) with LinePhysicsParams for per-line physics.
    """
    
    def __init__(self, model_config: dict, line_physics_config: LinePhysicsConfig):
        super().__init__()
        
        # Base HWF-PIKAN v2 model
        self.pikan = InvariantPIKANV2(**model_config)
        
        # Line-specific physics parameters
        self.line_physics = LinePhysicsParams(line_physics_config)
        
        # For mapping line_id to index
        self.line_id_to_idx = {}
        self.next_idx = 0
    
    def register_lines(self, line_ids: List):
        """Register new line IDs and assign indices"""
        for line_id in line_ids:
            line_id_int = int(line_id)
            if line_id_int not in self.line_id_to_idx:
                self.line_id_to_idx[line_id_int] = self.next_idx
                self.next_idx += 1
                if self.next_idx >= self.line_physics.num_lines:
                    # Use index 0 if we exceed capacity
                    self.line_id_to_idx[line_id_int] = 0
    
    def get_line_indices(self, line_ids: torch.Tensor) -> torch.Tensor:
        """Convert line IDs to indices, using 0 for unseen lines"""
        indices = []
        for lid in line_ids.cpu().numpy():
            idx = self.line_id_to_idx.get(int(lid), 0)
            indices.append(idx)
        return torch.tensor(indices, dtype=torch.long, device=line_ids.device)
    
    def forward(self, x, weather_dict, line_ids=None, use_global_mean=False):
        """
        Forward pass.
        
        Args:
            x: Input features [batch, input_dim]
            weather_dict: Dict with weather tensors
            line_ids: Line identifiers [batch] (optional)
            use_global_mean: If True, use global mean μ for all lines
        """
        # Get base predictions from PIKAN
        outputs = self.pikan(x, weather_dict)
        
        # Get physics parameters
        if use_global_mean or line_ids is None:
            # Use global mean only (zero deviations) for unseen regions
            device = x.device
            dummy_ids = torch.zeros(x.shape[0], dtype=torch.long, device=device)
            params = self.line_physics(dummy_ids)
        else:
            # Get per-line indices
            line_indices = self.get_line_indices(line_ids)
            params = self.line_physics(line_indices)
        
        outputs['physics_params'] = params
        return outputs
    
    def get_global_physics_params(self):
        """Get global mean physics parameters"""
        return self.line_physics.get_global_params()


# =============================================================================
# Configuration
# =============================================================================

class CrossRegionConfig:
    UNIFIED_DATA_PATH = "data/processed/unified_dlr_training.h5"
    OUTPUT_DIR = "cross_region_results/hwf_pikan_results"
    
    # HWF-PIKAN v2 architecture
    MODEL_CONFIG = {
        'input_dim': 3,
        'hidden_dim': 64,
        'output_dim': 2,
        'fourier_bands': 16,
        'wavelet_scales': 8,
        'kan_grid': 5,
        'kan_k': 3
    }
    
    # LinePhysics config
    LINE_PHYSICS_CONFIG = LinePhysicsConfig(
        num_lines=100,
        reg_weight=0.01,
        init_resistance_factor_mean=0.0,
        init_emissivity_mean=0.8,
        init_absorptivity_mean=0.8
    )
    
    # Training hyperparameters
    EPOCHS = 50
    BATCH_SIZE = 256
    LEARNING_RATE = 1e-4
    WEIGHT_DECAY = 1e-4
    PHYSICS_WEIGHT = 0.05
    
    DEVICE = torch.device("mps" if torch.backends.mps.is_available() 
                         else "cuda" if torch.cuda.is_available() 
                         else "cpu")
    
    SEED = 42
    US_TEST_SAMPLES = 10000


# =============================================================================
# Data Loading
# =============================================================================

class CrossRegionDataset(Dataset):
    def __init__(self, df, region, feature_cols, target_col, 
                 normalizer=None, is_training=True):
        self.region = region
        self.feature_cols = feature_cols
        self.target_col = target_col
        
        region_df = df[df['region'] == region].copy()
        region_df = region_df.dropna(subset=feature_cols + [target_col])
        self.df = region_df
        
        self.X = self.df[feature_cols].values.astype(np.float32)
        self.y = self.df[target_col].values.astype(np.float32).reshape(-1, 1)
        
        if 'line_id' in self.df.columns:
            line_ids_raw = self.df['line_id'].fillna(0).values
            self.line_ids = np.array([int(float(x)) if pd.notna(x) else 0 
                                      for x in line_ids_raw])
        else:
            self.line_ids = np.zeros(len(self.df), dtype=int)
        
        self.unique_lines = np.unique(self.line_ids)
        
        if normalizer is None and is_training:
            self.normalizer = CrossRegionNormalizer()
            self.normalizer.fit(self.X)
        elif normalizer is not None:
            self.normalizer = normalizer
        else:
            raise ValueError("Must provide normalizer for validation/test data")
        
        self.X_normalized = self.normalizer.transform(self.X)
        
        print(f"  {region}: {len(self)} samples, {len(self.unique_lines)} lines, "
              f"target mean={self.y.mean():.1f}")
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return (
            torch.tensor(self.X_normalized[idx], dtype=torch.float32),
            torch.tensor(self.y[idx], dtype=torch.float32),
            torch.tensor(self.line_ids[idx], dtype=torch.long)
        )


class CrossRegionNormalizer:
    def __init__(self):
        self.mean = None
        self.std = None
    
    def fit(self, X):
        self.mean = X.mean(axis=0)
        self.std = X.std(axis=0)
        self.std[self.std < 1e-8] = 1.0
        return self
    
    def transform(self, X):
        return (X - self.mean) / self.std


def load_unified_data(config):
    print(f"\n📊 Loading unified data...")
    df = pd.read_hdf(config.UNIFIED_DATA_PATH, key='data')
    print(f"   Total: {len(df):,} samples")
    for region, count in df['region'].value_counts().items():
        print(f"     {region}: {count:,}")
    
    # Print temperature statistics for investigation
    print("\n📊 Temperature column statistics (before normalization):")
    for region in ['VN', 'US']:
        temps = df[df['region'] == region]['temperature'].dropna()
        print(f"   {region}: min={temps.min():.1f}°C, max={temps.max():.1f}°C, "
              f"mean={temps.mean():.1f}°C")
    
    return df


def prepare_data_splits(df, config, split_type):
    print(f"\n{'='*60}")
    print(f"Preparing: {split_type}")
    print(f"{'='*60}")
    
    feature_cols = ['temperature', 'wind_speed', 'solar_irradiance']
    
    if split_type == 'VN_to_US':
        vn_df = df[df['region'] == 'VN'].copy()
        vn_train, vn_val = train_test_split(vn_df, test_size=0.2, random_state=config.SEED)
        
        us_df = df[df['region'] == 'US'].copy()
        us_test = us_df.sample(n=min(config.US_TEST_SAMPLES, len(us_df)), 
                               random_state=config.SEED)
        
        train_dataset = CrossRegionDataset(vn_train, 'VN', feature_cols, 'Ampacity', 
                                           normalizer=None, is_training=True)
        val_dataset = CrossRegionDataset(vn_val, 'VN', feature_cols, 'Ampacity',
                                         normalizer=train_dataset.normalizer, is_training=False)
        test_dataset = CrossRegionDataset(us_test, 'US', feature_cols, 'actual',
                                          normalizer=train_dataset.normalizer, is_training=False)
        
    elif split_type == 'US_to_VN':
        us_df = df[df['region'] == 'US'].copy()
        if len(us_df) > 50000:
            us_df = us_df.sample(n=50000, random_state=config.SEED)
        us_train, us_val = train_test_split(us_df, test_size=0.2, random_state=config.SEED)
        
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
    print(f"\n🚀 Training {experiment_name}...")
    print(f"   Device: {config.DEVICE}, Epochs: {config.EPOCHS}")
    
    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, 
                              shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE, 
                            shuffle=False, num_workers=0)
    
    # Create HWF-PIKAN with LinePhysics
    model = HWFPIKANWithLinePhysics(config.MODEL_CONFIG, config.LINE_PHYSICS_CONFIG)
    model.register_lines(train_dataset.unique_lines.tolist())
    model = model.to(config.DEVICE)
    
    print(f"   Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    optimizer = optim.AdamW(model.parameters(), lr=config.LEARNING_RATE, 
                           weight_decay=config.WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', 
                                                     factor=0.5, patience=10)
    criterion = nn.MSELoss()
    
    best_val_loss = float('inf')
    best_model_state = None
    history = {'train_loss': [], 'val_loss': [], 'val_mae': []}
    start_time = time.time()
    
    for epoch in range(config.EPOCHS):
        # Training
        model.train()
        train_losses = []
        
        for batch_x, batch_y, batch_line_ids in train_loader:
            batch_x = batch_x.to(config.DEVICE)
            batch_y = batch_y.to(config.DEVICE)
            batch_line_ids = batch_line_ids.to(config.DEVICE)
            
            # Prepare weather dict
            batch_x_np = batch_x.cpu().numpy()
            T_amb = batch_x_np[:, 0] * train_dataset.normalizer.std[0] + train_dataset.normalizer.mean[0]
            wind_speed = batch_x_np[:, 1] * train_dataset.normalizer.std[1] + train_dataset.normalizer.mean[1]
            solar = batch_x_np[:, 2] * train_dataset.normalizer.std[2] + train_dataset.normalizer.mean[2]
            
            weather_dict = {
                'T_amb': torch.tensor(T_amb, dtype=torch.float32, device=config.DEVICE),
                'wind_speed': torch.tensor(wind_speed, dtype=torch.float32, device=config.DEVICE),
                'solar': torch.tensor(solar, dtype=torch.float32, device=config.DEVICE)
            }
            
            optimizer.zero_grad()
            outputs = model(batch_x, weather_dict, batch_line_ids, use_global_mean=False)
            pred_amp = outputs['ampacity']
            
            loss = criterion(pred_amp, batch_y)
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
            for batch_x, batch_y, batch_line_ids in val_loader:
                batch_x = batch_x.to(config.DEVICE)
                batch_y = batch_y.to(config.DEVICE)
                batch_line_ids = batch_line_ids.to(config.DEVICE)
                
                batch_x_np = batch_x.cpu().numpy()
                T_amb = batch_x_np[:, 0] * val_dataset.normalizer.std[0] + val_dataset.normalizer.mean[0]
                wind_speed = batch_x_np[:, 1] * val_dataset.normalizer.std[1] + val_dataset.normalizer.mean[1]
                solar = batch_x_np[:, 2] * val_dataset.normalizer.std[2] + val_dataset.normalizer.mean[2]
                
                weather_dict = {
                    'T_amb': torch.tensor(T_amb, dtype=torch.float32, device=config.DEVICE),
                    'wind_speed': torch.tensor(wind_speed, dtype=torch.float32, device=config.DEVICE),
                    'solar': torch.tensor(solar, dtype=torch.float32, device=config.DEVICE)
                }
                
                outputs = model(batch_x, weather_dict, batch_line_ids, use_global_mean=False)
                pred_amp = outputs['ampacity']
                
                loss = criterion(pred_amp, batch_y)
                val_losses.append(loss.item())
                val_preds.extend(pred_amp.cpu().numpy())
                val_targets.extend(batch_y.cpu().numpy())
        
        avg_train_loss = np.mean(train_losses)
        avg_val_loss = np.mean(val_losses)
        val_mae = mean_absolute_error(val_targets, val_preds)
        
        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(avg_val_loss)
        history['val_mae'].append(val_mae)
        
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        
        if (epoch + 1) % 10 == 0 or epoch == 0:
            elapsed = time.time() - start_time
            print(f"  Epoch {epoch+1:3d}/{config.EPOCHS} | "
                  f"Train: {avg_train_loss:.2e} | Val: {avg_val_loss:.2e} | "
                  f"MAE: {val_mae:.1f}A | {elapsed:.1f}s")
        
        scheduler.step(avg_val_loss)
    
    # Load best model
    if best_model_state:
        model.load_state_dict(best_model_state)
    
    # Get learned physics parameters
    physics_params = model.get_global_physics_params()
    print(f"\n📊 Learned Physics μ:")
    print(f"   Resistance: {physics_params['resistance_factor']:.4f}")
    print(f"   Emissivity: {physics_params['emissivity']:.4f}")
    print(f"   Absorptivity: {physics_params['absorptivity']:.4f}")
    
    print(f"\n✅ Best val MAE: {min(history['val_mae']):.1f}A")
    return model, history, physics_params


# =============================================================================
# Evaluation
# =============================================================================

def evaluate_model(model, test_dataset, config, experiment_name, use_global_mean):
    """Evaluate model. use_global_mean=True for cross-region (unseen lines)."""
    print(f"\n📊 Evaluating {experiment_name} (global_mean={use_global_mean})...")
    
    test_loader = DataLoader(test_dataset, batch_size=config.BATCH_SIZE, 
                            shuffle=False, num_workers=0)
    
    model.eval()
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for batch_x, batch_y, batch_line_ids in test_loader:
            batch_x = batch_x.to(config.DEVICE)
            batch_y = batch_y.to(config.DEVICE)
            batch_line_ids = batch_line_ids.to(config.DEVICE)
            
            batch_x_np = batch_x.cpu().numpy()
            T_amb = batch_x_np[:, 0] * test_dataset.normalizer.std[0] + test_dataset.normalizer.mean[0]
            wind_speed = batch_x_np[:, 1] * test_dataset.normalizer.std[1] + test_dataset.normalizer.mean[1]
            solar = batch_x_np[:, 2] * test_dataset.normalizer.std[2] + test_dataset.normalizer.mean[2]
            
            weather_dict = {
                'T_amb': torch.tensor(T_amb, dtype=torch.float32, device=config.DEVICE),
                'wind_speed': torch.tensor(wind_speed, dtype=torch.float32, device=config.DEVICE),
                'solar': torch.tensor(solar, dtype=torch.float32, device=config.DEVICE)
            }
            
            outputs = model(batch_x, weather_dict, batch_line_ids, use_global_mean=use_global_mean)
            pred_amp = outputs['ampacity']
            
            all_preds.extend(pred_amp.cpu().numpy())
            all_targets.extend(batch_y.cpu().numpy())
    
    predictions = np.array(all_preds).flatten()
    targets = np.array(all_targets).flatten()
    
    mae = mean_absolute_error(targets, predictions)
    rmse = np.sqrt(mean_squared_error(targets, predictions))
    r2 = r2_score(targets, predictions)
    bias = np.mean(predictions - targets)
    
    print(f"   MAE:  {mae:.2f} A")
    print(f"   RMSE: {rmse:.2f} A")
    print(f"   Bias: {bias:+.2f} A")
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
# Main
# =============================================================================

def run_experiment(split_type, config, df):
    print(f"\n{'='*70}")
    print(f"EXPERIMENT: {split_type}")
    print(f"{'='*70}")
    
    train_dataset, val_dataset, test_dataset = prepare_data_splits(df, config, split_type)
    
    model, history, physics_params = train_model(train_dataset, val_dataset, config, split_type)
    
    # Use global mean for cross-region, per-line for within-region
    is_cross_region = '_to_' in split_type and split_type not in ['VN_only', 'US_only']
    use_global_mean = is_cross_region
    
    metrics = evaluate_model(model, test_dataset, config, split_type, use_global_mean=use_global_mean)
    metrics['physics_params'] = physics_params
    
    # Save model
    output_dir = Path(config.OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save({
        'model_state_dict': model.state_dict(),
        'line_id_to_idx': model.line_id_to_idx,
        'normalizer_mean': train_dataset.normalizer.mean,
        'normalizer_std': train_dataset.normalizer.std,
        'physics_params': physics_params,
        'history': history,
        'metrics': {k: v for k, v in metrics.items() 
                   if k not in ['predictions', 'targets']}
    }, output_dir / f"model_{split_type}.pt")
    
    return metrics


def create_results_table(all_metrics):
    def normalize_name(n):
        return {'VN_only': 'VN_to_VN', 'US_only': 'US_to_US'}.get(n, n)
    
    vn_baseline = None
    us_baseline = None
    for m in all_metrics:
        norm = normalize_name(m['experiment'])
        if norm == 'VN_to_VN':
            vn_baseline = m['mae']
        elif norm == 'US_to_US':
            us_baseline = m['mae']
    
    table = "## Cross-Region Results (HWF-PIKAN v2 with LinePhysics)\n\n"
    table += "| Train | Test | MAE (A) | RMSE (A) | Bias (A) | R² | vs. Baseline |\n"
    table += "|-------|------|---------|----------|----------|-----|--------------|\n"
    
    for m in all_metrics:
        norm = normalize_name(m['experiment'])
        parts = norm.split('_to_')
        if len(parts) != 2:
            continue
        train, test = parts
        
        if train == test:
            vs_str = "Baseline"
        elif test == 'VN' and vn_baseline:
            vs_str = f"+{(m['mae'] / vn_baseline - 1) * 100:.1f}%"
        elif test == 'US' and us_baseline:
            vs_str = f"+{(m['mae'] / us_baseline - 1) * 100:.1f}%"
        else:
            vs_str = "N/A"
        
        table += f"| {train} | {test} | {m['mae']:.1f} | {m['rmse']:.1f} | {m['bias']:+.1f} | {m['r2']:.3f} | {vs_str} |\n"
    
    return table


def main():
    parser = argparse.ArgumentParser(description='Cross-Region HWF-PIKAN v2')
    parser.add_argument('--experiments', nargs='+', 
                       choices=['VN_to_US', 'US_to_VN', 'VN_only', 'US_only', 'all'],
                       default=['all'])
    parser.add_argument('--epochs', type=int, default=50)
    
    args = parser.parse_args()
    
    config = CrossRegionConfig()
    config.EPOCHS = args.epochs
    
    print("="*70)
    print("CROSS-REGION VALIDATION - HWF-PIKAN v2 WITH LINE PHYSICS")
    print("="*70)
    print(f"Device: {config.DEVICE}")
    print(f"Epochs: {config.EPOCHS}")
    print(f"Output: {config.OUTPUT_DIR}")
    
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
            print(f"\n❌ Error in {exp}: {e}")
            import traceback
            traceback.print_exc()
    
    if all_metrics:
        output_dir = Path(config.OUTPUT_DIR)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Results table
        table = create_results_table(all_metrics)
        print("\n" + "="*70)
        print(table)
        print("="*70)
        
        with open(output_dir / 'results_table.md', 'w') as f:
            f.write(table)
        
        # Physics params comparison
        print("\n📊 PHYSICS PARAMETERS COMPARISON:")
        print("-" * 70)
        print(f"{'Experiment':<20} {'Resistance μ':<15} {'Emissivity μ':<15} {'Absorptivity μ'}")
        print("-" * 70)
        for m in all_metrics:
            params = m.get('physics_params', {})
            print(f"{m['experiment']:<20} "
                  f"{params.get('resistance_factor', 0):<15.4f} "
                  f"{params.get('emissivity', 0):<15.4f} "
                  f"{params.get('absorptivity', 0):.4f}")
        print("-" * 70)
        
        # Save metrics
        metrics_dict = {}
        for m in all_metrics:
            metrics_dict[m['experiment']] = {
                k: v for k, v in m.items() 
                if k not in ['predictions', 'targets']
            }
        with open(output_dir / 'metrics.json', 'w') as f:
            json.dump(metrics_dict, f, indent=2)
        
        print("\n✅ All experiments complete!")
        print(f"Results saved to: {output_dir}")


if __name__ == "__main__":
    main()
