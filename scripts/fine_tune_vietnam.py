"""
Invariant-PIKAN: Adversarially-Robust Physics-Informed Neural Networks for Dynamic Line Rating
Copyright (C) 2025 Gelavizh Ahmadi / Invariant Research

This software is licensed under the Business Source License 1.1 (BSL 1.1).
Commercial production use requires a separate license agreement.
See LICENSE.txt for full terms.

DISCLAIMER: This implementation is independent of concurrent academic work on
HWF-PIKAN for plasma physics (Heravifard et al., Sharif University, 2025).
"""

#!/usr/bin/env python3
"""
Fine-tune the DLR model on Vietnam transmission line data

This script loads a pre-trained model and fine-tunes it on real Vietnam
transmission line data to improve generalization to real-world conditions.

Author: Dr. Gelavizh Ahmadi
Copyright (c) 2026 Invariant Energy GmbH
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from pathlib import Path
import sys
import json
from datetime import datetime

# Import project modules
sys.path.append('.')

from core.data import create_dataloaders, VietnamDataset
from core.model import PhysicsDLR
from core.physics import IEEE738HeatBalance
from core.train import quick_train


def fine_tune_on_vietnam(
    model_path: str = 'models/best_model.pt',
    vietnam_csv: str = 'data/mendeley/vietnam_220kv.csv',
    n_epochs: int = 50,
    batch_size: int = 128,
    learning_rate: float = 1e-4,
    synthetic_ratio: float = 0.5,  # More real data for fine-tuning
    device: str = 'cpu',
    save_path: str = 'models/vietnam_finetuned.pt',
):
    """
    Fine-tune model on Vietnam data
    
    Args:
        model_path: Path to pre-trained model
        vietnam_csv: Path to Vietnam dataset
        n_epochs: Number of fine-tuning epochs
        batch_size: Training batch size
        learning_rate: Learning rate for fine-tuning
        synthetic_ratio: Ratio of synthetic to real data
        device: Device to use
        save_path: Where to save fine-tuned model
    """
    print("Vietnam Fine-tuning")
    print("=" * 50)
    
    # Load pre-trained model
    model = PhysicsDLR()
    checkpoint = torch.load(model_path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    print(f"Loaded pre-trained model from {model_path}")
    
    # Create dataloaders with Vietnam data
    train_loader, val_loader, train_dataset, val_dataset = create_dataloaders(
        n_train=10000,  # Smaller dataset for fine-tuning
        n_val=2000,
        batch_size=batch_size,
        vietnam_data_path=vietnam_csv,
        synthetic_ratio=synthetic_ratio,
        seed=42,
    )
    
    print(f"Training with {len(train_loader)} batches, validation with {len(val_loader)} batches")
    print(f"Synthetic ratio: {synthetic_ratio}")
    
    # Fine-tuning setup
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=10
    )
    
    # Training parameters
    physics_weight = 1.0  # Lower physics weight for fine-tuning on real data
    log_var_data = torch.tensor(0.0, requires_grad=True, device=device)
    log_var_phys = torch.tensor(0.0, requires_grad=True, device=device)
    
    best_val_loss = float('inf')
    patience = 15
    patience_counter = 0
    
    # Training loop
    for epoch in range(n_epochs):
        model.train()
        train_losses = []
        
        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            
            optimizer.zero_grad()
            
            # Forward pass
            pred_temp, pred_amp = model(batch_x)
            pred_temp = pred_temp.squeeze(-1)
            pred_amp = pred_amp.squeeze(-1)
            
            # Data loss
            data_loss = nn.MSELoss()(pred_temp, batch_y[:, 0]) + nn.MSELoss()(pred_amp, batch_y[:, 1])
            
            # Physics loss (denormalize inputs first)
            raw_inputs = train_dataset.normalizer.inverse_transform(batch_x.cpu().numpy())
            raw_inputs = torch.tensor(raw_inputs, dtype=torch.float32, device=device)
            
            physics = IEEE738HeatBalance()
            physics.to(device)  # Move physics to device
            with torch.no_grad():
                true_temp = physics.steady_state_temperature(
                    current=raw_inputs[:, 4],  # current
                    T_ambient=raw_inputs[:, 0],  # T_ambient
                    wind_speed=raw_inputs[:, 1],  # wind_speed
                    solar_irradiance=raw_inputs[:, 3],  # solar
                    wind_angle=raw_inputs[:, 2],  # wind_angle
                )
                true_amp = physics.ampacity(
                    T_max=75.0,
                    T_ambient=raw_inputs[:, 0],
                    wind_speed=raw_inputs[:, 1],
                    solar_irradiance=raw_inputs[:, 3],
                    wind_angle=raw_inputs[:, 2],
                )
            
            phys_loss_temp = nn.MSELoss()(pred_temp, true_temp)
            phys_loss_amp = nn.MSELoss()(pred_amp, true_amp)
            physics_loss = phys_loss_temp + phys_loss_amp
            
            # Combined loss with uncertainty weighting
            total_loss = (
                torch.exp(-log_var_data) * data_loss + 
                torch.exp(-log_var_phys) * physics_loss * physics_weight +
                log_var_data + log_var_phys
            )
            
            total_loss.backward()
            optimizer.step()
            
            train_losses.append(total_loss.item())
        
        # Validation
        model.eval()
        val_losses = []
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x = batch_x.to(device)
                batch_y = batch_y.to(device)
                
                pred_temp, pred_amp = model(batch_x)
                val_loss = nn.MSELoss()(pred_temp, batch_y[:, 0]) + nn.MSELoss()(pred_amp, batch_y[:, 1])
                val_losses.append(val_loss.item())
        
        avg_train_loss = np.mean(train_losses)
        avg_val_loss = np.mean(val_losses)
        
        print(f"Epoch {epoch+1:3d}/{n_epochs} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")
        
        # Learning rate scheduling
        scheduler.step(avg_val_loss)
        
        # Early stopping
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            
            # Save best model
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': best_val_loss,
                'fine_tuned_on': 'vietnam',
            }, save_path)
            print(f"  Saved best model to {save_path}")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break
    
    print(f"\nFine-tuning complete! Best validation loss: {best_val_loss:.4f}")
    print(f"Model saved to {save_path}")


def main():
    """Main fine-tuning function"""
    # Check for pre-trained model
    model_paths = [
        'models/best_model.pt',
        'models/tuned_model.pt',
        'models/governed_model.pt',
        'models/conservative_model.pt',
    ]
    
    model_path = None
    for path in model_paths:
        if Path(path).exists():
            model_path = path
            break
    
    if not model_path:
        print("No pre-trained model found. Please train a base model first.")
        return
    
    # Check for Vietnam data
    vietnam_csv = 'data/mendeley/vietnam_220kv.csv'
    if not Path(vietnam_csv).exists():
        print(f"Vietnam dataset not found at {vietnam_csv}")
        return
    
    # Set device
    device = 'mps' if torch.backends.mps.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Fine-tune
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_path = f'models/vietnam_finetuned_{timestamp}.pt'
    
    fine_tune_on_vietnam(
        model_path=model_path,
        vietnam_csv=vietnam_csv,
        n_epochs=50,
        batch_size=128,
        learning_rate=1e-4,
        synthetic_ratio=0.3,  # 30% synthetic, 70% Vietnam
        device=device,
        save_path=save_path,
    )
    
    # Validate the fine-tuned model
    print("\nValidating fine-tuned model...")
    from scripts.validate_vietnam import validate_on_vietnam, load_model
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    
    vietnam_dataset = VietnamDataset(vietnam_csv)
    model = load_model(save_path, device)
    
    metrics, pred_temp, true_temp, pred_amp, true_amp = validate_on_vietnam(
        model, vietnam_dataset, device
    )
    
    print("\nFine-tuned Model Validation Metrics:")
    print("-" * 40)
    print(f"Temperature MAE:  {metrics['temp_mae']:.2f} °C")
    print(f"Temperature RMSE: {metrics['temp_rmse']:.2f} °C")
    print(f"Temperature R²:   {metrics['temp_r2']:.3f}")
    print(f"Ampacity MAE:  {metrics['amp_mae']:.2f} A")
    print(f"Ampacity RMSE: {metrics['amp_rmse']:.2f} A")
    print(f"Ampacity R²:   {metrics['amp_r2']:.3f}")


if __name__ == "__main__":
    main()
