#!/usr/bin/env python3
"""
Cross-Region Validation (Split C) for DLR Paper

This script implements the critical cross-region validation experiments:
- Split C.1: Train on Vietnam → Test on US
- Split C.2: Train on US → Test on Vietnam  
- Split C.3: Within-region baselines (Vietnam-only and US-only)

The goal is to validate whether the HWF-PIKAN model can generalize across
different grid regions (tropical Vietnam vs continental US) without data leakage.

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
import seaborn as sns
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

from models.invariant_pikan_v2 import create_invariant_pikan_v2


# =============================================================================
# Configuration
# =============================================================================

class CrossRegionConfig:
    """Configuration for cross-region validation experiments"""
    
    # Data paths
    UNIFIED_DATA_PATH = "data/processed/unified_dlr_training.h5"
    OUTPUT_DIR = "cross_region_results"
    
    # Model architecture (HWF-PIKAN v2)
    MODEL_CONFIG = {
        'input_dim': 4,            # T_amb, wind_speed, wind_angle, solar
        'hidden_dim': 64,
        'output_dim': 2,           # temperature, ampacity
        'fourier_bands': 16,
        'wavelet_scales': 8,
        'kan_grid': 5,
        'kan_k': 3
    }
    
    # Training hyperparameters
    EPOCHS = 100
    BATCH_SIZE = 256
    LEARNING_RATE = 1e-4
    WEIGHT_DECAY = 1e-4
    PHYSICS_WEIGHT = 0.05
    REGULARIZATION_WEIGHT = 0.01
    
    # Device
    DEVICE = torch.device("mps" if torch.backends.mps.is_available() 
                         else "cuda" if torch.cuda.is_available() 
                         else "cpu")
    
    # Random seeds for reproducibility
    SEED = 42
    
    # Test set sizes
    US_TEST_SAMPLES = 10000  # Random sample from US for Vietnam→US test


# =============================================================================
# Data Loading and Preprocessing
# =============================================================================

class CrossRegionDataset(Dataset):
    """
    Dataset for cross-region DLR experiments.
    
    Handles the unified data format with both US and Vietnam data.
    """
    
    def __init__(
        self,
        df: pd.DataFrame,
        region: str,  # 'VN' or 'US'
        feature_cols: List[str],
        target_col: str,
        normalizer: Optional['CrossRegionNormalizer'] = None,
        is_training: bool = True
    ):
        """
        Initialize dataset for a specific region.
        
        Args:
            df: DataFrame with unified data
            region: 'VN' or 'US'
            feature_cols: List of feature column names
            target_col: Target column name (ampacity)
            normalizer: Optional normalizer (if None, will fit on this data)
            is_training: Whether this is training data (for normalizer fitting)
        """
        self.region = region
        self.feature_cols = feature_cols
        self.target_col = target_col
        
        # Filter to region
        self.df = df[df['region'] == region].copy()
        
        # Extract features and target
        self.X = self.df[feature_cols].values.astype(np.float32)
        self.y = self.df[target_col].values.astype(np.float32)
        
        # Store line_ids if available (handle NaN)
        if 'line_id' in self.df.columns:
            line_ids_raw = self.df['line_id'].values
            # Replace NaN with 0, handle both numeric and object types
            self.line_ids = np.array([
                int(float(x)) if pd.notna(x) and str(x).replace('.','',1).isdigit() 
                else 0 for x in line_ids_raw
            ], dtype=int)
        else:
            self.line_ids = np.zeros(len(self.df), dtype=int)
        
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
        """Get normalized input and target"""
        return (
            torch.tensor(self.X_normalized[idx], dtype=torch.float32),
            torch.tensor(self.y[idx], dtype=torch.float32),
            torch.tensor(int(self.line_ids[idx]), dtype=torch.long)
        )
    
    def get_raw_features(self, idx):
        """Get unnormalized features"""
        return self.X[idx]
    
    def get_statistics(self) -> Dict:
        """Get dataset statistics"""
        return {
            'n_samples': len(self),
            'n_features': len(self.feature_cols),
            'target_mean': float(self.y.mean()),
            'target_std': float(self.y.std()),
            'target_min': float(self.y.min()),
            'target_max': float(self.y.max()),
            'feature_means': {col: float(self.X[:, i].mean()) 
                             for i, col in enumerate(self.feature_cols)},
        }


class CrossRegionNormalizer:
    """Normalizer for cross-region experiments"""
    
    def __init__(self):
        self.mean = None
        self.std = None
        self.feature_names = None
    
    def fit(self, X: np.ndarray):
        """Fit normalizer on training data"""
        self.mean = X.mean(axis=0)
        self.std = X.std(axis=0)
        self.std[self.std < 1e-8] = 1.0  # Prevent division by zero
        return self
    
    def transform(self, X: np.ndarray) -> np.ndarray:
        """Normalize data"""
        return (X - self.mean) / self.std
    
    def inverse_transform(self, X: np.ndarray) -> np.ndarray:
        """Denormalize data"""
        return X * self.std + self.mean
    
    def save(self, path: str):
        """Save normalizer"""
        np.savez(path, mean=self.mean, std=self.std)
    
    def load(self, path: str):
        """Load normalizer"""
        data = np.load(path)
        self.mean = data['mean']
        self.std = data['std']
        return self


def load_unified_data(config: CrossRegionConfig) -> pd.DataFrame:
    """Load unified dataset from HDF5"""
    print(f"\n📊 Loading unified data from {config.UNIFIED_DATA_PATH}...")
    df = pd.read_hdf(config.UNIFIED_DATA_PATH, key='data')
    print(f"   Total samples: {len(df):,}")
    print(f"   Region distribution:")
    for region, count in df['region'].value_counts().items():
        print(f"     {region}: {count:,}")
    return df


def prepare_data_splits(
    df: pd.DataFrame,
    config: CrossRegionConfig,
    split_type: str  # 'VN_to_US', 'US_to_VN', 'VN_only', 'US_only'
) -> Tuple:
    """
    Prepare data splits for cross-region experiments.
    
    Returns:
        train_dataset, val_dataset, test_dataset, normalizer
    """
    print(f"\n{'='*60}")
    print(f"Preparing data for: {split_type}")
    print(f"{'='*60}")
    
    # Define common features (must exist in both regions)
    feature_cols = ['temperature', 'wind_speed', 'WinDir', 'solar_irradiance']
    
    if split_type == 'VN_to_US':
        # Split C.1: Train on Vietnam, test on US
        
        # Vietnam: 80/20 train/val
        vn_df = df[df['region'] == 'VN'].copy()
        vn_train, vn_val = train_test_split(
            vn_df, test_size=0.2, random_state=config.SEED
        )
        
        # US: Random 10k samples for test
        us_df = df[df['region'] == 'US'].copy()
        us_test = us_df.sample(n=min(config.US_TEST_SAMPLES, len(us_df)), 
                               random_state=config.SEED)
        
        # Create datasets with Vietnam normalizer
        train_dataset = CrossRegionDataset(
            vn_train, 'VN', feature_cols, 'Ampacity', 
            normalizer=None, is_training=True
        )
        normalizer = train_dataset.normalizer
        
        val_dataset = CrossRegionDataset(
            vn_val, 'VN', feature_cols, 'Ampacity',
            normalizer=normalizer, is_training=False
        )
        
        test_dataset = CrossRegionDataset(
            us_test, 'US', feature_cols, 'actual',
            normalizer=normalizer, is_training=False
        )
        
    elif split_type == 'US_to_VN':
        # Split C.2: Train on US, test on Vietnam
        
        # US: 80/20 train/val
        us_df = df[df['region'] == 'US'].copy()
        us_train, us_val = train_test_split(
            us_df, test_size=0.2, random_state=config.SEED
        )
        
        # Vietnam: All as test
        vn_test = df[df['region'] == 'VN'].copy()
        
        # Create datasets with US normalizer
        train_dataset = CrossRegionDataset(
            us_train, 'US', feature_cols, 'actual',
            normalizer=None, is_training=True
        )
        normalizer = train_dataset.normalizer
        
        val_dataset = CrossRegionDataset(
            us_val, 'US', feature_cols, 'actual',
            normalizer=normalizer, is_training=False
        )
        
        test_dataset = CrossRegionDataset(
            vn_test, 'VN', feature_cols, 'Ampacity',
            normalizer=normalizer, is_training=False
        )
        
    elif split_type == 'VN_only':
        # Split C.3a: Within-region Vietnam baseline
        
        vn_df = df[df['region'] == 'VN'].copy()
        
        # 80/10/10 split
        vn_train, vn_temp = train_test_split(
            vn_df, test_size=0.2, random_state=config.SEED
        )
        vn_val, vn_test = train_test_split(
            vn_temp, test_size=0.5, random_state=config.SEED
        )
        
        train_dataset = CrossRegionDataset(
            vn_train, 'VN', feature_cols, 'Ampacity',
            normalizer=None, is_training=True
        )
        normalizer = train_dataset.normalizer
        
        val_dataset = CrossRegionDataset(
            vn_val, 'VN', feature_cols, 'Ampacity',
            normalizer=normalizer, is_training=False
        )
        
        test_dataset = CrossRegionDataset(
            vn_test, 'VN', feature_cols, 'Ampacity',
            normalizer=normalizer, is_training=False
        )
        
    elif split_type == 'US_only':
        # Split C.3b: Within-region US baseline
        
        us_df = df[df['region'] == 'US'].copy()
        
        # 80/10/10 split
        us_train, us_temp = train_test_split(
            us_df, test_size=0.2, random_state=config.SEED
        )
        us_val, us_test = train_test_split(
            us_temp, test_size=0.5, random_state=config.SEED
        )
        
        train_dataset = CrossRegionDataset(
            us_train, 'US', feature_cols, 'actual',
            normalizer=None, is_training=True
        )
        normalizer = train_dataset.normalizer
        
        val_dataset = CrossRegionDataset(
            us_val, 'US', feature_cols, 'actual',
            normalizer=normalizer, is_training=False
        )
        
        test_dataset = CrossRegionDataset(
            us_test, 'US', feature_cols, 'actual',
            normalizer=normalizer, is_training=False
        )
    
    else:
        raise ValueError(f"Unknown split_type: {split_type}")
    
    return train_dataset, val_dataset, test_dataset, normalizer


# =============================================================================
# Model Training
# =============================================================================

class PhysicsInformedLoss(nn.Module):
    """Physics-informed loss function for DLR"""
    
    def __init__(self, physics_weight: float = 0.05, reg_weight: float = 0.01):
        super().__init__()
        self.physics_weight = physics_weight
        self.reg_weight = reg_weight
        self.mse = nn.MSELoss()
    
    def forward(
        self,
        pred_temp: torch.Tensor,
        pred_amp: torch.Tensor,
        true_temp: torch.Tensor,
        true_amp: torch.Tensor,
        physics_residual: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Dict]:
        """
        Compute physics-informed loss.
        
        Args:
            pred_temp: Predicted temperature
            pred_amp: Predicted ampacity
            true_temp: True temperature
            true_amp: True ampacity
            physics_residual: Physics constraint residual (optional)
        
        Returns:
            total_loss, metrics_dict
        """
        # Data loss (MSE on ampacity only - that's what we care about)
        amp_loss = self.mse(pred_amp, true_amp)
        temp_loss = self.mse(pred_temp, true_temp)
        data_loss = amp_loss + 0.1 * temp_loss
        
        # Physics loss
        if physics_residual is not None:
            physics_loss = physics_residual.mean()
        else:
            physics_loss = torch.tensor(0.0, device=pred_amp.device)
        
        # Regularization (L2 on predictions for smoothness)
        reg_loss = torch.mean(pred_amp ** 2) * 0.001
        
        # Total loss
        total_loss = (
            data_loss + 
            self.physics_weight * physics_loss +
            self.reg_weight * reg_loss
        )
        
        metrics = {
            'total_loss': total_loss.item(),
            'data_loss': data_loss.item(),
            'amp_loss': amp_loss.item(),
            'temp_loss': temp_loss.item(),
            'physics_loss': physics_loss.item(),
            'reg_loss': reg_loss.item(),
        }
        
        return total_loss, metrics


def train_model(
    train_dataset: CrossRegionDataset,
    val_dataset: CrossRegionDataset,
    config: CrossRegionConfig,
    experiment_name: str
) -> Tuple[nn.Module, Dict]:
    """
    Train HWF-PIKAN model.
    
    Returns:
        model, training_history
    """
    print(f"\n🚀 Training model for {experiment_name}...")
    print(f"   Device: {config.DEVICE}")
    print(f"   Epochs: {config.EPOCHS}")
    print(f"   Batch size: {config.BATCH_SIZE}")
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=True
    )
    
    # Create model
    model = create_invariant_pikan_v2(config=config.MODEL_CONFIG)
    model = model.to(config.DEVICE)
    
    # Optimizer
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config.LEARNING_RATE,
        weight_decay=config.WEIGHT_DECAY
    )
    
    # Scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=config.EPOCHS,
        eta_min=config.LEARNING_RATE * 0.01
    )
    
    # Loss function
    loss_fn = PhysicsInformedLoss(
        physics_weight=config.PHYSICS_WEIGHT,
        reg_weight=config.REGULARIZATION_WEIGHT
    )
    
    # Training history
    history = {
        'train_loss': [],
        'val_loss': [],
        'val_mae': [],
        'best_epoch': 0,
        'best_val_loss': float('inf')
    }
    
    # Training loop
    best_model_state = None
    start_time = time.time()
    
    for epoch in range(config.EPOCHS):
        # Training
        model.train()
        train_losses = []
        
        for batch_x, batch_y, _ in train_loader:
            batch_x = batch_x.to(config.DEVICE)
            batch_y = batch_y.to(config.DEVICE)
            
            # Prepare weather dict for physics
            # Denormalize to get actual weather values
            batch_x_np = batch_x.cpu().numpy()
            T_amb = batch_x_np[:, 0] * train_dataset.normalizer.std[0] + train_dataset.normalizer.mean[0]
            wind_speed = batch_x_np[:, 1] * train_dataset.normalizer.std[1] + train_dataset.normalizer.mean[1]
            solar = batch_x_np[:, 3] * train_dataset.normalizer.std[3] + train_dataset.normalizer.mean[3]
            
            weather_dict = {
                'T_amb': torch.tensor(T_amb, dtype=torch.float32, device=config.DEVICE),
                'wind_speed': torch.tensor(wind_speed, dtype=torch.float32, device=config.DEVICE),
                'solar': torch.tensor(solar, dtype=torch.float32, device=config.DEVICE)
            }
            
            # Forward pass (only first 4 features for embedding)
            outputs = model(batch_x[:, :4], weather_dict)
            
            pred_temp = outputs['temperature']
            pred_amp = outputs['ampacity']
            
            # Targets: we'll use batch_y as ampacity target, estimate temperature
            true_amp = batch_y
            true_temp = torch.full_like(pred_temp, 50.0)  # Placeholder
            
            # Compute loss
            loss, metrics = loss_fn(
                pred_temp, pred_amp,
                true_temp, true_amp,
                outputs.get('physics_residual')
            )
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            train_losses.append(loss.item())
        
        # Validation
        model.eval()
        val_losses = []
        val_predictions = []
        val_targets = []
        
        with torch.no_grad():
            for batch_x, batch_y, _ in val_loader:
                batch_x = batch_x.to(config.DEVICE)
                batch_y = batch_y.to(config.DEVICE)
                
                # Denormalize for physics
                batch_x_np = batch_x.cpu().numpy()
                T_amb = batch_x_np[:, 0] * val_dataset.normalizer.std[0] + val_dataset.normalizer.mean[0]
                wind_speed = batch_x_np[:, 1] * val_dataset.normalizer.std[1] + val_dataset.normalizer.mean[1]
                solar = batch_x_np[:, 3] * val_dataset.normalizer.std[3] + val_dataset.normalizer.mean[3]
                
                weather_dict = {
                    'T_amb': torch.tensor(T_amb, dtype=torch.float32, device=config.DEVICE),
                    'wind_speed': torch.tensor(wind_speed, dtype=torch.float32, device=config.DEVICE),
                    'solar': torch.tensor(solar, dtype=torch.float32, device=config.DEVICE)
                }
                
                outputs = model(batch_x[:, :4], weather_dict)
                pred_amp = outputs['ampacity']
                
                val_predictions.extend(pred_amp.cpu().numpy())
                val_targets.extend(batch_y.cpu().numpy())
                
                loss = nn.MSELoss()(pred_amp, batch_y)
                val_losses.append(loss.item())
        
        # Calculate metrics
        avg_train_loss = np.mean(train_losses)
        avg_val_loss = np.mean(val_losses)
        val_mae = mean_absolute_error(val_targets, val_predictions)
        
        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(avg_val_loss)
        history['val_mae'].append(val_mae)
        
        # Save best model
        if avg_val_loss < history['best_val_loss']:
            history['best_val_loss'] = avg_val_loss
            history['best_epoch'] = epoch
            best_model_state = model.state_dict().copy()
        
        # Logging
        if (epoch + 1) % 10 == 0 or epoch == 0:
            elapsed = time.time() - start_time
            print(f"  Epoch {epoch+1:3d}/{config.EPOCHS} | "
                  f"Train: {avg_train_loss:.4f} | "
                  f"Val: {avg_val_loss:.4f} | "
                  f"MAE: {val_mae:.1f}A | "
                  f"LR: {scheduler.get_last_lr()[0]:.2e} | "
                  f"{elapsed:.1f}s")
        
        scheduler.step()
    
    # Load best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    
    print(f"\n✅ Training complete. Best epoch: {history['best_epoch']+1}")
    print(f"   Best val loss: {history['best_val_loss']:.4f}")
    print(f"   Best val MAE: {min(history['val_mae']):.1f}A")
    
    return model, history


# =============================================================================
# Evaluation
# =============================================================================

def evaluate_model(
    model: nn.Module,
    test_dataset: CrossRegionDataset,
    config: CrossRegionConfig,
    experiment_name: str
) -> Dict:
    """
    Evaluate model on test set.
    
    Returns:
        Dictionary of metrics
    """
    print(f"\n📊 Evaluating {experiment_name}...")
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=0
    )
    
    model.eval()
    all_predictions = []
    all_targets = []
    all_line_ids = []
    
    with torch.no_grad():
        for batch_x, batch_y, batch_line_ids in test_loader:
            batch_x = batch_x.to(config.DEVICE)
            
            # Denormalize for physics
            batch_x_np = batch_x.cpu().numpy()
            T_amb = batch_x_np[:, 0] * test_dataset.normalizer.std[0] + test_dataset.normalizer.mean[0]
            wind_speed = batch_x_np[:, 1] * test_dataset.normalizer.std[1] + test_dataset.normalizer.mean[1]
            solar = batch_x_np[:, 3] * test_dataset.normalizer.std[3] + test_dataset.normalizer.mean[3]
            
            weather_dict = {
                'T_amb': torch.tensor(T_amb, dtype=torch.float32, device=config.DEVICE),
                'wind_speed': torch.tensor(wind_speed, dtype=torch.float32, device=config.DEVICE),
                'solar': torch.tensor(solar, dtype=torch.float32, device=config.DEVICE)
            }
            
            outputs = model(batch_x[:, :4], weather_dict)
            pred_amp = outputs['ampacity']
            
            all_predictions.extend(pred_amp.cpu().numpy())
            all_targets.extend(batch_y.numpy())
            all_line_ids.extend(batch_line_ids.numpy())
    
    # Convert to arrays
    predictions = np.array(all_predictions)
    targets = np.array(all_targets)
    line_ids = np.array(all_line_ids)
    
    # Calculate metrics
    mae = mean_absolute_error(targets, predictions)
    rmse = np.sqrt(mean_squared_error(targets, predictions))
    r2 = r2_score(targets, predictions)
    bias = np.mean(predictions - targets)
    
    # Per-line metrics (if multiple lines)
    unique_lines = np.unique(line_ids)
    per_line_mae = {}
    for line_id in unique_lines:
        mask = line_ids == line_id
        if mask.sum() > 10:  # Only if enough samples
            line_mae = mean_absolute_error(targets[mask], predictions[mask])
            per_line_mae[str(line_id)] = float(line_mae)
    
    metrics = {
        'experiment': experiment_name,
        'n_samples': len(predictions),
        'mae': float(mae),
        'rmse': float(rmse),
        'bias': float(bias),
        'r2': float(r2),
        'per_line_mae': per_line_mae,
        'predictions': predictions,
        'targets': targets,
    }
    
    print(f"   MAE:  {mae:.2f} A")
    print(f"   RMSE: {rmse:.2f} A")
    print(f"   Bias: {bias:.2f} A")
    print(f"   R²:   {r2:.4f}")
    
    return metrics


# =============================================================================
# Visualization
# =============================================================================

def normalize_exp_name(exp_name: str) -> str:
    """Convert experiment names to standardized format"""
    name_map = {
        'VN_only': 'VN_to_VN',
        'US_only': 'US_to_US',
    }
    return name_map.get(exp_name, exp_name)


def create_results_table(all_metrics: List[Dict]) -> str:
    """Create markdown results table"""
    
    # Extract within-region baselines (using normalized names)
    vn_baseline_mae = None
    us_baseline_mae = None
    
    for m in all_metrics:
        norm_name = normalize_exp_name(m['experiment'])
        if norm_name == 'VN_to_VN':
            vn_baseline_mae = m['mae']
        elif norm_name == 'US_to_US':
            us_baseline_mae = m['mae']
    
    # Build table
    table = "## Cross-Region Transfer Learning Results\n\n"
    table += "| Train Region | Test Region | MAE (A) | RMSE (A) | Bias (A) | R² | vs. Within-Region |\n"
    table += "|--------------|-------------|---------|----------|----------|-----|-------------------|\n"
    
    for m in all_metrics:
        norm_name = normalize_exp_name(m['experiment'])
        parts = norm_name.split('_to_')
        if len(parts) != 2:
            continue
        train_region = parts[0]
        test_region = parts[1]
        
        # Calculate vs. within-region
        if train_region == test_region:
            vs_str = "Baseline"
        elif test_region == 'VN' and vn_baseline_mae:
            pct = (m['mae'] / vn_baseline_mae - 1) * 100
            vs_str = f"+{pct:.1f}%"
        elif test_region == 'US' and us_baseline_mae:
            pct = (m['mae'] / us_baseline_mae - 1) * 100
            vs_str = f"+{pct:.1f}%"
        else:
            vs_str = "N/A"
        
        table += f"| {train_region} | {test_region} | "
        table += f"{m['mae']:.1f} | {m['rmse']:.1f} | {m['bias']:.1f} | "
        table += f"{m['r2']:.3f} | {vs_str} |\n"
    
    return table


def create_visualizations(all_metrics: List[Dict], output_dir: Path):
    """Create publication-quality visualizations"""
    
    print("\n📈 Creating visualizations...")
    
    # Figure A: Cross-Region Prediction Scatter Plots
    fig, axes = plt.subplots(2, 2, figsize=(12, 12))
    fig.suptitle('Cross-Region Prediction Scatter Plots', fontsize=14, fontweight='bold')
    
    experiments = ['VN_to_VN', 'VN_to_US', 'US_to_US', 'US_to_VN']
    titles = [
        'Vietnam → Vietnam (Within-Region)',
        'Vietnam → US (Cross-Region)',
        'US → US (Within-Region)',
        'US → Vietnam (Cross-Region)'
    ]
    
    for idx, (exp, title) in enumerate(zip(experiments, titles)):
        ax = axes[idx // 2, idx % 2]
        
        # Find metrics for this experiment (check both naming conventions)
        metrics = next((m for m in all_metrics 
                       if normalize_exp_name(m['experiment']) == exp), None)
        if metrics is None:
            ax.text(0.5, 0.5, 'No data', ha='center', va='center')
            ax.set_title(title)
            continue
        
        preds = metrics['predictions']
        targets = metrics['targets']
        
        # Scatter plot
        ax.scatter(targets, preds, alpha=0.5, s=10)
        
        # Perfect prediction line
        min_val = min(targets.min(), preds.min())
        max_val = max(targets.max(), preds.max())
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect prediction')
        
        # R² annotation
        r2 = metrics['r2']
        ax.annotate(f'R² = {r2:.3f}', xy=(0.05, 0.95), xycoords='axes fraction',
                   fontsize=12, verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        ax.set_xlabel('True Ampacity (A)')
        ax.set_ylabel('Predicted Ampacity (A)')
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        ax.legend()
    
    plt.tight_layout()
    plt.savefig(output_dir / 'figure_A_cross_region_scatter.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"   Saved: figure_A_cross_region_scatter.png")
    
    # Figure B: Error Distribution Comparison
    fig, ax = plt.subplots(figsize=(10, 6))
    
    error_data = []
    labels = []
    
    for exp in experiments:
        metrics = next((m for m in all_metrics 
                       if normalize_exp_name(m['experiment']) == exp), None)
        if metrics:
            errors = metrics['predictions'] - metrics['targets']
            error_data.append(errors)
            labels.append(exp.replace('_to_', '→'))
    
    bp = ax.boxplot(error_data, labels=labels, patch_artist=True)
    
    # Color within-region vs cross-region differently
    colors = ['lightblue', 'lightcoral', 'lightblue', 'lightcoral']
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
    
    ax.axhline(y=0, color='red', linestyle='--', linewidth=2)
    ax.set_ylabel('Prediction Error (A)')
    ax.set_title('Error Distribution Comparison\n(Blue=Within-Region, Red=Cross-Region)')
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'figure_B_error_distributions.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"   Saved: figure_B_error_distributions.png")
    
    # Figure C: Per-Line Performance (US only)
    us_metrics = next((m for m in all_metrics if m['experiment'] == 'US_to_US'), None)
    if us_metrics and len(us_metrics['per_line_mae']) > 1:
        fig, ax = plt.subplots(figsize=(10, 6))
        
        line_ids = list(us_metrics['per_line_mae'].keys())
        maes = list(us_metrics['per_line_mae'].values())
        
        ax.scatter(range(len(line_ids)), maes, s=100, alpha=0.6)
        ax.set_xlabel('Line Index')
        ax.set_ylabel('Per-Line MAE (A)')
        ax.set_title('Per-Line Performance (US→US)\nShowing Generalization Across Lines')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_dir / 'figure_C_per_line_performance.png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"   Saved: figure_C_per_line_performance.png")


# =============================================================================
# Main Execution
# =============================================================================

def run_experiment(split_type: str, config: CrossRegionConfig, df: pd.DataFrame) -> Dict:
    """Run a single cross-region experiment"""
    
    print(f"\n{'='*70}")
    print(f"EXPERIMENT: {split_type}")
    print(f"{'='*70}")
    
    # Prepare data
    train_dataset, val_dataset, test_dataset, normalizer = prepare_data_splits(
        df, config, split_type
    )
    
    # Train model
    model, history = train_model(train_dataset, val_dataset, config, split_type)
    
    # Evaluate
    metrics = evaluate_model(model, test_dataset, config, split_type)
    
    # Save model
    output_dir = Path(config.OUTPUT_DIR)
    output_dir.mkdir(exist_ok=True)
    model_path = output_dir / f"model_{split_type}.pt"
    torch.save({
        'model_state_dict': model.state_dict(),
        'normalizer_mean': train_dataset.normalizer.mean,
        'normalizer_std': train_dataset.normalizer.std,
        'history': history,
        'metrics': {k: v for k, v in metrics.items() if k not in ['predictions', 'targets']}
    }, model_path)
    print(f"   Model saved: {model_path}")
    
    return metrics


def main():
    parser = argparse.ArgumentParser(
        description='Cross-Region Validation for DLR Paper (Split C)'
    )
    parser.add_argument(
        '--experiments',
        nargs='+',
        choices=['VN_to_US', 'US_to_VN', 'VN_only', 'US_only', 'all'],
        default=['all'],
        help='Which experiments to run'
    )
    parser.add_argument(
        '--epochs',
        type=int,
        default=100,
        help='Number of training epochs'
    )
    parser.add_argument(
        '--quick',
        action='store_true',
        help='Quick mode: fewer epochs for testing'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='cross_region_results',
        help='Output directory for results'
    )
    
    args = parser.parse_args()
    
    # Setup
    config = CrossRegionConfig()
    config.EPOCHS = 20 if args.quick else args.epochs
    config.OUTPUT_DIR = args.output_dir
    
    print("="*70)
    print("CROSS-REGION VALIDATION (SPLIT C) FOR DLR PAPER")
    print("="*70)
    print(f"Device: {config.DEVICE}")
    print(f"Epochs: {config.EPOCHS}")
    print(f"Output: {config.OUTPUT_DIR}")
    
    # Set random seeds
    torch.manual_seed(config.SEED)
    np.random.seed(config.SEED)
    
    # Load data
    df = load_unified_data(config)
    
    # Determine which experiments to run
    if 'all' in args.experiments:
        experiments = ['VN_only', 'US_only', 'VN_to_US', 'US_to_VN']
    else:
        experiments = args.experiments
    
    # Run experiments
    all_metrics = []
    for exp in experiments:
        try:
            metrics = run_experiment(exp, config, df)
            all_metrics.append(metrics)
        except Exception as e:
            print(f"\n❌ Error in experiment {exp}: {e}")
            import traceback
            traceback.print_exc()
    
    # Generate outputs
    if all_metrics:
        output_dir = Path(config.OUTPUT_DIR)
        output_dir.mkdir(exist_ok=True)
        
        # Results table
        table = create_results_table(all_metrics)
        print("\n" + "="*70)
        print(table)
        print("="*70)
        
        # Save table
        with open(output_dir / 'results_table.md', 'w') as f:
            f.write(table)
        print(f"\n💾 Results table saved: {output_dir / 'results_table.md'}")
        
        # Visualizations
        create_visualizations(all_metrics, output_dir)
        
        # Save metrics
        metrics_dict = {
            m['experiment']: {k: v for k, v in m.items() 
                            if k not in ['predictions', 'targets', 'per_line_mae']}
            for m in all_metrics
        }
        with open(output_dir / 'metrics.json', 'w') as f:
            json.dump(metrics_dict, f, indent=2)
        print(f"💾 Metrics saved: {output_dir / 'metrics.json'}")
        
        # Analysis
        print("\n" + "="*70)
        print("ANALYSIS")
        print("="*70)
        
        vn_to_vn = next((m for m in all_metrics if normalize_exp_name(m['experiment']) == 'VN_to_VN'), None)
        vn_to_us = next((m for m in all_metrics if normalize_exp_name(m['experiment']) == 'VN_to_US'), None)
        us_to_us = next((m for m in all_metrics if normalize_exp_name(m['experiment']) == 'US_to_US'), None)
        us_to_vn = next((m for m in all_metrics if normalize_exp_name(m['experiment']) == 'US_to_VN'), None)
        
        # Transfer success analysis
        if vn_to_us and us_to_us:
            degradation = (vn_to_us['mae'] / us_to_us['mae'] - 1) * 100
            print(f"\n1. Vietnam → US Transfer:")
            print(f"   Within-US MAE: {us_to_us['mae']:.1f} A")
            print(f"   Vietnam→US MAE: {vn_to_us['mae']:.1f} A")
            print(f"   Degradation: +{degradation:.1f}%")
            if vn_to_us['mae'] > 400:
                print(f"   ⚠️  Transfer FAILED (MAE > 400 A)")
            else:
                print(f"   ✅ Transfer reasonably successful")
        
        if us_to_vn and vn_to_vn:
            degradation = (us_to_vn['mae'] / vn_to_vn['mae'] - 1) * 100
            print(f"\n2. US → Vietnam Transfer:")
            print(f"   Within-VN MAE: {vn_to_vn['mae']:.1f} A")
            print(f"   US→VN MAE: {us_to_vn['mae']:.1f} A")
            print(f"   Degradation: +{degradation:.1f}%")
            if us_to_vn['mae'] < 300:
                print(f"   ✅ Transfer successful (MAE < 300 A)")
            else:
                print(f"   ⚠️  Transfer shows room for improvement")
        
        print("\n" + "="*70)
        print("✅ CROSS-REGION VALIDATION COMPLETE")
        print("="*70)
        print(f"\nResults saved to: {output_dir}")
        print("Files:")
        print(f"  - results_table.md: Markdown table for paper")
        print(f"  - metrics.json: All metrics in JSON format")
        print(f"  - figure_A_cross_region_scatter.png: Scatter plots")
        print(f"  - figure_B_error_distributions.png: Error distributions")
        print(f"  - figure_C_per_line_performance.png: Per-line analysis")
        print(f"  - model_*.pt: Trained model checkpoints")


if __name__ == "__main__":
    main()
