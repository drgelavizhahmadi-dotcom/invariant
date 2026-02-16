"""
Invariant-PIKAN: Adversarially-Robust Physics-Informed Neural Networks for Dynamic Line Rating
Copyright (C) 2025 Gelavizh Ahmadi / Invariant Research

This software is licensed under the Business Source License 1.1 (BSL 1.1).
Commercial production use requires a separate license agreement.
See LICENSE.txt for full terms.

HWF-PIKAN for plasma physics (Heravifard et al., Sharif University, 2025).
"""

#!/usr/bin/env python3
"""
Train InvariantPIKAN with Learnable Per-Line Physics Parameters

Extends the production training script to include:
- LinePhysicsParams module for per-line physics
- Physics loss with line-specific resistance, emissivity, absorptivity
- Hierarchical Bayesian regularization

Usage:
    python -m scripts.train_per_line_physics \
        --data-path data/processed/unified_dlr_training.h5 \
        --line-id-col line_id \
        --epochs 100 \
        --reg-weight 0.01
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torch.utils.tensorboard import SummaryWriter

# Project imports
sys.path.append(str(Path(__file__).parent.parent))

from core.data import VietnamDataset, InputNormalizer
from models.invariant_pikan_v2 import create_invariant_pikan_v2
from models.line_physics import LinePhysicsParams, LinePhysicsConfig, create_line_physics_for_dataset
from core.physics_per_line import heat_balance_residual_per_line


def parse_args():
    parser = argparse.ArgumentParser(description='Train InvariantPIKAN with per-line physics')
    
    # Data args
    parser.add_argument('--data-path', type=str, required=True,
                       help='Path to unified training data (HDF5 or CSV)')
    parser.add_argument('--line-id-col', type=str, default='line_id',
                       help='Column name for line identifiers')
    parser.add_argument('--region-col', type=str, default='region',
                       help='Column name for region (US/VN)')
    
    # Model args
    parser.add_argument('--base-model', type=str, default=None,
                       help='Optional: initialize from base model checkpoint')
    
    # Training args
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--lr-line-physics', type=float, default=1e-3,
                       help='Learning rate for line physics params (typically higher)')
    parser.add_argument('--device', type=str, default='auto')
    
    # Loss weights
    parser.add_argument('--lambda-physics', type=float, default=0.05)
    parser.add_argument('--reg-weight', type=float, default=0.01,
                       help='Regularization weight for line physics deviations')
    
    # Output
    parser.add_argument('--output-dir', type=str, default=None)
    parser.add_argument('--save-every', type=int, default=10)
    
    # Line physics config
    parser.add_argument('--init-resistance-mean', type=float, default=0.0,
                       help='Initial log(resistance_factor) mean')
    parser.add_argument('--init-emissivity-mean', type=float, default=0.8)
    parser.add_argument('--init-absorptivity-mean', type=float, default=0.8)
    
    return parser.parse_args()


def load_data_with_line_ids(data_path: str, line_id_col: str, region_col: str):
    """Load data and extract line ID mapping."""
    print(f"\n📊 Loading data from {data_path}...")
    
    if data_path.endswith('.h5') or data_path.endswith('.hdf5'):
        df = pd.read_hdf(data_path, key='data')
    else:
        df = pd.read_csv(data_path)
    
    print(f"  Total samples: {len(df)}")
    
    # Create line ID mapping
    if line_id_col in df.columns:
        # Convert to string to handle mixed types
        df[line_id_col] = df[line_id_col].astype(str)
        unique_lines = df[line_id_col].unique()
        print(f"  Unique lines: {len(unique_lines)}")
        
        # Create mapping
        line_to_idx = {line: idx for idx, line in enumerate(sorted(unique_lines))}
        df['line_idx'] = df[line_id_col].map(line_to_idx)
    else:
        print(f"  Warning: {line_id_col} not found. Using single line.")
        line_to_idx = {'default': 0}
        df['line_idx'] = 0
    
    # Show region distribution
    if region_col in df.columns:
        print(f"\n  Region distribution:")
        print(df[region_col].value_counts())
    
    return df, line_to_idx


def create_dataloaders(df: pd.DataFrame, line_to_idx: dict, batch_size: int, test_frac: float = 0.1):
    """Create train/test dataloaders from dataframe."""
    
    # Rename columns to match VietnamDataset expectations
    col_mapping = {
        'temperature': 'T_ambient',
        'actual': 'Ampacity',
    }
    df = df.rename(columns=col_mapping)
    
    # Ensure required columns exist
    if 'wind_direction' not in df.columns and 'WinDir' in df.columns:
        df['wind_direction'] = df['WinDir']
    
    # Save temp CSV for VietnamDataset
    temp_path = Path('temp_per_line_training.csv')
    df.to_csv(temp_path, index=False)
    
    # Create dataset
    dataset = VietnamDataset(str(temp_path))
    
    # Add line_idx to dataset
    dataset.line_indices = df['line_idx'].values
    
    # Train/test split
    n_total = len(dataset)
    n_test = int(n_total * test_frac)
    n_train = n_total - n_test
    
    # Random split
    indices = np.random.permutation(n_total)
    train_idx = indices[:n_train]
    test_idx = indices[n_train:]
    
    train_dataset = Subset(dataset, train_idx)
    test_dataset = Subset(dataset, test_idx)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    
    print(f"\n  Train samples: {n_train}")
    print(f"  Test samples: {n_test}")
    
    # Cleanup temp file
    temp_path.unlink(missing_ok=True)
    
    return train_loader, test_loader, dataset


def train_epoch(model, line_physics, train_loader, optimizer, scheduler, 
                lambda_physics, reg_weight, device, epoch):
    """Train for one epoch."""
    model.train()
    line_physics.train()
    
    total_loss = 0.0
    total_data_loss = 0.0
    total_physics_loss = 0.0
    total_reg_loss = 0.0
    n_batches = 0
    
    base_physics_params = {
        'diameter': 0.02814,
        'R_ref': 7.283e-5,
        'alpha_R': 0.00403
    }
    
    for batch_idx, (x, y) in enumerate(train_loader):
        x = x.to(device)
        y = y.to(device)
        
        # Get line indices for this batch
        if hasattr(train_loader.dataset, 'line_indices'):
            # For Subset, access underlying dataset
            if isinstance(train_loader.dataset, Subset):
                indices = train_loader.dataset.indices
                line_ids = torch.tensor(
                    train_loader.dataset.dataset.line_indices[indices[batch_idx * train_loader.batch_size:(batch_idx + 1) * train_loader.batch_size]],
                    dtype=torch.long, device=device
                )
            else:
                start_idx = batch_idx * train_loader.batch_size
                end_idx = min(start_idx + train_loader.batch_size, len(train_loader.dataset))
                line_ids = torch.tensor(
                    train_loader.dataset.line_indices[start_idx:end_idx],
                    dtype=torch.long, device=device
                )
        else:
            line_ids = torch.zeros(len(x), dtype=torch.long, device=device)
        
        # Ensure line_ids matches batch size
        if len(line_ids) != len(x):
            line_ids = torch.zeros(len(x), dtype=torch.long, device=device)
        
        # Forward pass
        weather = x[:, :4]
        weather_dict = {
            'T_amb': weather[:, 0],
            'wind_speed': weather[:, 1],
            'solar': weather[:, 3]
        }
        
        predictions = model(weather, weather_dict)
        targets = {'temperature': y[:, 0], 'ampacity': y[:, 1]}
        
        # Get per-line physics parameters
        line_params = line_physics(line_ids)
        
        # Data loss
        temp_loss = nn.functional.mse_loss(predictions['temperature'], targets['temperature'])
        amp_loss = nn.functional.mse_loss(predictions['ampacity'], targets['ampacity'])
        data_loss = temp_loss + amp_loss
        
        # Physics loss with per-line parameters
        current = predictions['ampacity'].squeeze()
        T_conductor = predictions['temperature'].squeeze()
        T_ambient = weather[:, 0]
        wind_speed = weather[:, 1]
        solar_irradiance = weather[:, 3]
        
        residual = heat_balance_residual_per_line(
            current=current,
            T_conductor=T_conductor,
            T_ambient=T_ambient,
            wind_speed=wind_speed,
            solar_irradiance=solar_irradiance,
            line_params=line_params,
            base_physics_params=base_physics_params,
            wind_angle=None
        )
        physics_loss = torch.mean(residual ** 2)
        
        # Regularization loss for line physics
        reg_loss = line_physics.regularization_loss()
        
        # Total loss
        loss = data_loss + lambda_physics * physics_loss + reg_weight * reg_loss
        
        # Backward
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            list(model.parameters()) + list(line_physics.parameters()),
            max_norm=1.0
        )
        optimizer.step()
        
        total_loss += loss.item()
        total_data_loss += data_loss.item()
        total_physics_loss += physics_loss.item()
        total_reg_loss += reg_loss.item()
        n_batches += 1
        
        if batch_idx % 50 == 0:
            print(f"  Batch {batch_idx}: loss={loss.item():.4f}, "
                  f"data={data_loss.item():.4f}, physics={physics_loss.item():.4f}, "
                  f"reg={reg_loss.item():.4f}")
    
    if scheduler:
        scheduler.step()
    
    return {
        'loss': total_loss / n_batches,
        'data_loss': total_data_loss / n_batches,
        'physics_loss': total_physics_loss / n_batches,
        'reg_loss': total_reg_loss / n_batches
    }


@torch.no_grad()
def evaluate(model, line_physics, test_loader, lambda_physics, reg_weight, device):
    """Evaluate on test set."""
    model.eval()
    line_physics.eval()
    
    total_loss = 0.0
    all_preds = []
    all_targets = []
    n_batches = 0
    
    base_physics_params = {
        'diameter': 0.02814,
        'R_ref': 7.283e-5,
        'alpha_R': 0.00403
    }
    
    for x, y in test_loader:
        x = x.to(device)
        y = y.to(device)
        
        # Use default line (0) for evaluation
        line_ids = torch.zeros(len(x), dtype=torch.long, device=device)
        
        # Forward
        weather = x[:, :4]
        weather_dict = {
            'T_amb': weather[:, 0],
            'wind_speed': weather[:, 1],
            'solar': weather[:, 3]
        }
        
        predictions = model(weather, weather_dict)
        
        # Collect for metrics
        all_preds.extend(predictions['ampacity'].cpu().numpy())
        all_targets.extend(y[:, 1].cpu().numpy())
        n_batches += 1
    
    # Compute MAE
    all_preds = np.array(all_preds).flatten()
    all_targets = np.array(all_targets).flatten()
    mae = np.mean(np.abs(all_preds - all_targets))
    bias = np.mean(all_preds - all_targets)
    
    return {'mae': mae, 'bias': bias}


def main():
    args = parse_args()
    
    # Device
    if args.device == 'auto':
        device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    print(f"\n🔧 Device: {device}")
    
    # Load data
    df, line_to_idx = load_data_with_line_ids(
        args.data_path, args.line_id_col, args.region_col
    )
    
    # Create dataloaders
    train_loader, test_loader, dataset = create_dataloaders(
        df, line_to_idx, args.batch_size
    )
    
    # Create model
    print("\n🏗️  Creating InvariantPIKAN v2 model...")
    model = create_invariant_pikan_v2(config={})
    
    if args.base_model:
        print(f"Loading base model from {args.base_model}")
        ckpt = torch.load(args.base_model, map_location=device, weights_only=False)
        model.load_state_dict(ckpt['model_state_dict'])
    
    model = model.to(device)
    
    # Create line physics module
    print(f"\n📐 Creating LinePhysicsParams for {len(line_to_idx)} lines...")
    line_config = LinePhysicsConfig(
        num_lines=len(line_to_idx),
        reg_weight=args.reg_weight,
        init_resistance_factor_mean=args.init_resistance_mean,
        init_emissivity_mean=args.init_emissivity_mean,
        init_absorptivity_mean=args.init_absorptivity_mean
    )
    line_physics = LinePhysicsParams(line_config)
    line_physics = line_physics.to(device)
    
    # Optimizer with different learning rates
    optimizer = torch.optim.AdamW([
        {'params': model.parameters(), 'lr': args.lr},
        {'params': line_physics.parameters(), 'lr': args.lr_line_physics}
    ])
    
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    
    # Output directory
    if args.output_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.output_dir = f"runs/per_line_physics_{timestamp}"
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Save config
    with open(os.path.join(args.output_dir, 'config.json'), 'w') as f:
        json.dump(vars(args), f, indent=2, default=str)
    
    # Save line mapping
    with open(os.path.join(args.output_dir, 'line_mapping.json'), 'w') as f:
        json.dump(line_to_idx, f, indent=2)
    
    # TensorBoard
    writer = SummaryWriter(args.output_dir)
    
    print(f"\n🚀 Starting training for {args.epochs} epochs...")
    print(f"Output: {args.output_dir}")
    
    best_mae = float('inf')
    
    for epoch in range(args.epochs):
        epoch_start = time.time()
        
        # Train
        train_metrics = train_epoch(
            model, line_physics, train_loader, optimizer, scheduler,
            args.lambda_physics, args.reg_weight, device, epoch
        )
        
        # Evaluate
        eval_metrics = evaluate(
            model, line_physics, test_loader,
            args.lambda_physics, args.reg_weight, device
        )
        
        epoch_time = time.time() - epoch_start
        
        # Log
        print(f"\n📊 Epoch {epoch+1}/{args.epochs} ({epoch_time:.1f}s):")
        print(f"  Train loss: {train_metrics['loss']:.4f}")
        print(f"  Test MAE: {eval_metrics['mae']:.2f} A, Bias: {eval_metrics['bias']:.2f} A")
        
        writer.add_scalar('Loss/train', train_metrics['loss'], epoch)
        writer.add_scalar('Loss/data', train_metrics['data_loss'], epoch)
        writer.add_scalar('Loss/physics', train_metrics['physics_loss'], epoch)
        writer.add_scalar('Loss/regularization', train_metrics['reg_loss'], epoch)
        writer.add_scalar('Metrics/test_mae', eval_metrics['mae'], epoch)
        writer.add_scalar('Metrics/test_bias', eval_metrics['bias'], epoch)
        
        # Save best
        if eval_metrics['mae'] < best_mae:
            best_mae = eval_metrics['mae']
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'line_physics_state_dict': line_physics.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'mae': best_mae,
                'config': vars(args)
            }, os.path.join(args.output_dir, 'best_model.pt'))
            print(f"  -> Saved best model (MAE: {best_mae:.2f})")
        
        # Periodic save
        if (epoch + 1) % args.save_every == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'line_physics_state_dict': line_physics.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'config': vars(args)
            }, os.path.join(args.output_dir, f'checkpoint_epoch{epoch+1}.pt'))
    
    # Final save
    torch.save({
        'epoch': args.epochs,
        'model_state_dict': model.state_dict(),
        'line_physics_state_dict': line_physics.state_dict(),
        'config': vars(args)
    }, os.path.join(args.output_dir, 'final_model.pt'))
    
    writer.close()
    
    # Print final line physics summary
    print("\n" + "="*70)
    line_physics.summary()
    
    print(f"\n✅ Training complete! Best MAE: {best_mae:.2f} A")
    print(f"Results saved to: {args.output_dir}")


if __name__ == '__main__':
    main()
