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
Fine-tune a pre-trained model specifically on US DLR data.
Uses the unified dataset but filters for US region samples.
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
from core.physics import IEEE738HeatBalance

# Import loss from production training
try:
    from scripts.train_invariant_pikan_production import PhysicsInformedLoss
except Exception:
    from core.model import PhysicsInformedLoss


def main():
    parser = argparse.ArgumentParser(description="Fine-tune on US DLR data")
    parser.add_argument("--base-model", type=str, required=True,
                       help="Path to base model checkpoint")
    parser.add_argument("--data-path", type=str, 
                       default="data/processed/unified_dlr_training.h5",
                       help="Path to unified training data (HDF5)")
    parser.add_argument("--epochs", type=int, default=50,
                       help="Number of fine-tuning epochs")
    parser.add_argument("--lr", type=float, default=5e-5,
                       help="Learning rate for fine-tuning (small for stability)")
    parser.add_argument("--batch-size", type=int, default=32,
                       help="Batch size")
    parser.add_argument("--device", type=str, default="auto",
                       help="Device: 'cpu', 'mps', 'cuda', or 'auto'")
    parser.add_argument("--output-dir", type=str, default=None,
                       help="Output directory (default: auto-generated)")
    parser.add_argument("--lambda-physics", type=float, default=0.0,
                       help="Physics loss weight (0 to focus on data fit)")
    parser.add_argument("--save-every", type=int, default=10,
                       help="Save checkpoint every N epochs")

    args = parser.parse_args()

    # Setup device
    if args.device == "auto":
        device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    print(f"Using device: {device}")

    # Load base model
    print(f"Loading base model from {args.base_model}")
    checkpoint = torch.load(args.base_model, map_location=device, weights_only=False)
    
    model_config = checkpoint.get('config', {}).get('model', None) if isinstance(checkpoint, dict) and 'config' in checkpoint else {}
    if not model_config:
        model_config = {}
    
    print(f"Using model config: {model_config}")
    
    model = create_invariant_pikan_v2(config=model_config)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.train()
    
    # Count parameters
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model has {n_params:,} trainable parameters")

    # Load unified dataset
    print(f"Loading unified dataset from {args.data_path}...")
    
    # Create a temporary CSV from HDF5 for VietnamDataset compatibility
    df = pd.read_hdf(args.data_path, key='data')
    print(f"Loaded {len(df)} total samples")
    
    # Filter for US region
    if 'region' in df.columns:
        us_df = df[df['region'] == 'US'].copy()
        print(f"US samples: {len(us_df)}")
    else:
        print("Warning: No 'region' column found. Using all data.")
        us_df = df
    
    if len(us_df) == 0:
        print("No US samples found! Exiting.")
        sys.exit(1)
    
    # Rename columns to match VietnamDataset expectations
    column_mapping = {
        'temperature': 'T_ambient',
        'actual': 'Ampacity',
        'wind_speed': 'wind_speed',  # same
        'solar_irradiance': 'solar_irradiance',  # same
    }
    us_df = us_df.rename(columns=column_mapping)
    
    # Ensure required columns exist
    if 'wind_direction' not in us_df.columns and 'WinDir' in us_df.columns:
        us_df['wind_direction'] = us_df['WinDir']
    
    # Save temporary CSV for VietnamDataset
    temp_dir = Path("temp_finetune_us")
    temp_dir.mkdir(exist_ok=True)
    temp_csv = temp_dir / "us_training_data.csv"
    us_df.to_csv(temp_csv, index=False)
    
    # Load as VietnamDataset
    dataset = VietnamDataset(str(temp_csv))
    print(f"Dataset loaded: {len(dataset)} samples")
    
    # Create data loader
    train_loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,  # Avoid multiprocessing issues
        pin_memory=False,
    )

    # Setup output directory
    if args.output_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.output_dir = f"runs/finetune_us_{timestamp}"

    os.makedirs(args.output_dir, exist_ok=True)
    print(f"Output directory: {args.output_dir}")
    
    # Save config
    with open(os.path.join(args.output_dir, 'config.json'), 'w') as f:
        json.dump(vars(args), f, indent=2, default=str)

    # Setup TensorBoard
    writer = SummaryWriter(args.output_dir)

    # Setup loss and optimizer
    physics_engine = IEEE738HeatBalance()
    loss_fn = PhysicsInformedLoss(
        physics_engine=physics_engine, 
        lambda_physics=args.lambda_physics
    )
    
    # Fine-tuning with small LR
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # Training loop
    print(f"\nStarting US fine-tuning for {args.epochs} epochs...")
    print(f"LR: {args.lr}, Physics lambda: {args.lambda_physics}")
    best_loss = float('inf')

    for epoch in range(args.epochs):
        epoch_start = time.time()
        total_loss = 0.0
        n_batches = 0

        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            
            # Split inputs
            weather = batch_x[:, :4]
            weather_dict = {
                'T_amb': weather[:, 0],
                'wind_speed': weather[:, 1], 
                'solar': weather[:, 3]
            }

            # Forward pass
            predictions = model(weather, weather_dict)
            targets = {'temperature': batch_y[:, 0], 'ampacity': batch_y[:, 1]}

            # Loss
            weather_for_loss = torch.cat([weather[:, :3], batch_x[:, 4:5]], dim=-1)
            loss, loss_components = loss_fn(
                predictions, targets, weather_for_loss, return_components=True
            )

            # Backward
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # Gradient clipping
            optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        avg_loss = total_loss / n_batches
        epoch_time = time.time() - epoch_start
        
        # Update learning rate
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']

        print(f"Epoch {epoch+1}/{args.epochs} - Loss: {avg_loss:.4f} - LR: {current_lr:.2e} - Time: {epoch_time:.2f}s")

        # Log to TensorBoard
        writer.add_scalar('Loss/train', avg_loss, epoch)
        writer.add_scalar('LR', current_lr, epoch)
        
        for comp_name, comp_value in loss_components.items():
            if isinstance(comp_value, (int, float)):
                writer.add_scalar(f'Loss/{comp_name}', comp_value, epoch)

        # Save best model
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': best_loss,
                'config': {'model': model_config},
            }, os.path.join(args.output_dir, 'best_model.pt'))
            print(f"  -> Saved best model (loss: {best_loss:.4f})")

        # Periodic save
        if (epoch + 1) % args.save_every == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
                'config': {'model': model_config},
            }, os.path.join(args.output_dir, f'checkpoint_epoch{epoch+1}.pt'))

    # Save final model
    torch.save({
        'epoch': args.epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': avg_loss,
        'config': {'model': model_config},
    }, os.path.join(args.output_dir, 'final_model.pt'))

    writer.close()
    
    # Cleanup temp file
    temp_csv.unlink(missing_ok=True)
    
    print(f"\n✅ US fine-tuning complete!")
    print(f"Best loss: {best_loss:.4f}")
    print(f"Models saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
