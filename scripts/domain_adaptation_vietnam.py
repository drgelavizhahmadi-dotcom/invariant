"""
Invariant-PIKAN: Adversarially-Robust Physics-Informed Neural Networks for Dynamic Line Rating
Copyright (C) 2025 Gelavizh Ahmadi / Invariant Research

This software is licensed under the Business Source License 1.1 (BSL 1.1).
Commercial production use requires a separate license agreement.
See LICENSE.txt for full terms.

"""

#!/usr/bin/env python3
"""
Domain Adaptation: Fine-tune best model on Vietnam data

This script properly fine-tunes the best performing model (best_model.pt)
on Vietnam transmission line data to adapt to the real-world domain.

Key improvements over previous fine-tuning:
- Proper checkpoint loading with normalizer preservation
- Domain-specific data handling
- Better loss weighting for real data
- Validation during training

Author: Dr. Gelavizh Ahmadi
Copyright (c) 2026 Invariant Energy GmbH
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd
from pathlib import Path
import sys
import json
import time
from datetime import datetime
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Import project modules
sys.path.append('.')

from core.model import PhysicsDLR
from core.physics import IEEE738HeatBalance
from core.data import InputNormalizer


class VietnamDataset(torch.utils.data.Dataset):
    """Vietnam dataset for fine-tuning"""

    def __init__(self, csv_path: str = "data/mendeley/vietnam_220kv.csv", normalizer: InputNormalizer = None):
        df = pd.read_csv(csv_path)

        # Extract features (matching the expected input format)
        self.T_ambient = df['temp'].values.astype(np.float32)
        self.wind_speed = df['Wind1'].values.astype(np.float32)
        self.wind_angle = df['WinDir'].values.astype(np.float32)
        self.solar_irradiance = df['GHI'].values.astype(np.float32)
        self.ampacity = df['Ampacity'].values.astype(np.float32)

        # For temperature, use ambient + offset as proxy (since actual conductor temp not measured)
        self.conductor_temp = self.T_ambient + 10.0

        self.n_samples = len(df)
        self.normalizer = normalizer

        print(f"Loaded Vietnam dataset: {self.n_samples:,} samples for fine-tuning")

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx: int):
        # Use fixed current/resistance for DLR calculation
        current = 1000.0  # A (typical for this line)
        resistance = 0.08  # ohm/m

        # Input features
        x = np.array([
            self.T_ambient[idx],
            self.wind_speed[idx],
            self.wind_angle[idx],
            self.solar_irradiance[idx],
            current,
            resistance,
        ])

        # Target: conductor temperature proxy + measured ampacity
        y = np.array([
            self.conductor_temp[idx],
            self.ampacity[idx],
        ])

        # Normalize inputs if normalizer provided
        if self.normalizer:
            x = self.normalizer.transform(x.reshape(1, -1)).flatten()

        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)


def fine_tune_best_model(
    base_model_path: str = 'models/best_model.pt',
    vietnam_csv: str = 'data/mendeley/vietnam_220kv.csv',
    n_epochs: int = 100,
    batch_size: int = 64,
    learning_rate: int = 5e-5,  # Lower LR for fine-tuning
    val_split: float = 0.2,
    device: str = 'cpu',
    save_path: str = None,
):
    """
    Fine-tune the best model on Vietnam data with proper domain adaptation

    Args:
        base_model_path: Path to the best base model
        vietnam_csv: Path to Vietnam dataset
        n_epochs: Number of fine-tuning epochs
        batch_size: Training batch size
        learning_rate: Learning rate for fine-tuning
        val_split: Fraction of data for validation
        device: Device to use
        save_path: Where to save fine-tuned model
    """
    print("🔄 Domain Adaptation: Fine-tuning Best Model on Vietnam Data")
    print("=" * 70)

    # Set device
    device = torch.device(device)
    print(f"Using device: {device}")

    # Load base model and normalizer
    print(f"Loading base model from {base_model_path}...")
    checkpoint = torch.load(base_model_path, map_location=device, weights_only=False)

    model = PhysicsDLR()
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.train()

    # Load normalizer from checkpoint
    normalizer = InputNormalizer()
    if 'normalizer' in checkpoint:
        normalizer.mean = np.array(checkpoint['normalizer']['mean'])
        normalizer.std = np.array(checkpoint['normalizer']['std'])
        print("✅ Loaded normalizer from checkpoint")
    else:
        print("⚠️  No normalizer in checkpoint, using defaults")

    # Load Vietnam dataset
    print(f"Loading Vietnam dataset from {vietnam_csv}...")
    vietnam_dataset = VietnamDataset(vietnam_csv, normalizer)

    # Split into train/val
    n_val = int(len(vietnam_dataset) * val_split)
    n_train = len(vietnam_dataset) - n_val

    train_dataset, val_dataset = torch.utils.data.random_split(
        vietnam_dataset, [n_train, n_val],
        generator=torch.Generator().manual_seed(42)
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    print(f"Training samples: {n_train}, Validation samples: {n_val}")

    # Fine-tuning setup
    # Freeze most layers, only fine-tune last layers
    for name, param in model.named_parameters():
        if 'encoder' in name and ('6' in name or '7' in name):  # Fine-tune only last encoder layers
            param.requires_grad = True
        elif 'temp_head' in name or 'amp_head' in name:  # Always fine-tune heads
            param.requires_grad = True
        else:
            param.requires_grad = False

    # Count trainable parameters
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Trainable parameters: {trainable_params:,} / {total_params:,} ({100*trainable_params/total_params:.1f}%)")

    optimizer = optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=learning_rate,
        weight_decay=1e-4
    )

    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs, eta_min=learning_rate/10)

    # Loss function - focus more on data fidelity for domain adaptation
    temp_criterion = nn.MSELoss()
    amp_criterion = nn.MSELoss()

    # Training loop
    best_val_loss = float('inf')
    patience = 20
    patience_counter = 0

    history = {
        'train_loss': [],
        'val_loss': [],
        'temp_mae': [],
        'amp_mae': [],
        'lr': []
    }

    print(f"\n🚀 Starting fine-tuning for {n_epochs} epochs...")
    print("-" * 70)

    for epoch in range(n_epochs):
        epoch_start = time.time()

        # Training
        model.train()
        train_losses = []

        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)

            optimizer.zero_grad()

            # Forward pass
            pred_temp, pred_amp = model(batch_x)

            # Compute losses
            temp_loss = temp_criterion(pred_temp.squeeze(), batch_y[:, 0])
            amp_loss = amp_criterion(pred_amp.squeeze(), batch_y[:, 1])

            # Weighted loss (focus more on ampacity for Vietnam data)
            total_loss = temp_loss + 2.0 * amp_loss

            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_losses.append(total_loss.item())

        # Validation
        model.eval()
        val_losses = []
        temp_preds, temp_trues = [], []
        amp_preds, amp_trues = [], []

        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)

                pred_temp, pred_amp = model(batch_x)

                # Validation loss
                val_temp_loss = temp_criterion(pred_temp.squeeze(), batch_y[:, 0])
                val_amp_loss = amp_criterion(pred_amp.squeeze(), batch_y[:, 1])
                val_loss = val_temp_loss + 2.0 * val_amp_loss
                val_losses.append(val_loss.item())

                # Collect predictions for metrics
                temp_preds.extend(pred_temp.detach().cpu().numpy().flatten().tolist())
                temp_trues.extend(batch_y[:, 0].cpu().numpy().tolist())
                amp_preds.extend(pred_amp.detach().cpu().numpy().flatten().tolist())
                amp_trues.extend(batch_y[:, 1].cpu().numpy().tolist())

        # Calculate metrics
        avg_train_loss = np.mean(train_losses)
        avg_val_loss = np.mean(val_losses)

        temp_mae = mean_absolute_error(temp_trues, temp_preds)
        amp_mae = mean_absolute_error(amp_trues, amp_preds)

        current_lr = optimizer.param_groups[0]['lr']

        # Record history
        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(avg_val_loss)
        history['temp_mae'].append(temp_mae)
        history['amp_mae'].append(amp_mae)
        history['lr'].append(current_lr)

        # Logging
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"Epoch {epoch+1:3d}/{n_epochs} │ "
                  f"Train: {avg_train_loss:.4f} │ "
                  f"Val: {avg_val_loss:.4f} │ "
                  f"T_MAE: {temp_mae:.2f}°C │ "
                  f"I_MAE: {amp_mae:.0f}A │ "
                  f"LR: {current_lr:.1e}")

        # Learning rate scheduling
        scheduler.step()

        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0

            if save_path:
                checkpoint = {
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_loss': best_val_loss,
                    'fine_tuned_on': 'vietnam',
                    'normalizer': {
                        'mean': normalizer.mean.tolist(),
                        'std': normalizer.std.tolist(),
                    },
                    'base_model': base_model_path,
                    'training_history': history,
                }
                torch.save(checkpoint, save_path)
                print(f"💾 Saved best model to {save_path}")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"⏹️  Early stopping at epoch {epoch+1}")
                break

    print("-" * 70)
    print(f"✅ Fine-tuning complete! Best validation loss: {best_val_loss:.4f}")

    # Final validation metrics
    print("📈 Final Validation Metrics:")
    print(f"   Temperature MAE: {temp_mae:.2f}°C")
    print(f"   Ampacity MAE: {amp_mae:.0f}A")

    return model, history


def main():
    """Main domain adaptation function"""
    import time

    # Check for base model
    base_model_path = 'models/best_model.pt'
    if not Path(base_model_path).exists():
        print(f"❌ Base model not found: {base_model_path}")
        return

    # Check for Vietnam data
    vietnam_csv = 'data/mendeley/vietnam_220kv.csv'
    if not Path(vietnam_csv).exists():
        print(f"❌ Vietnam dataset not found: {vietnam_csv}")
        return

    # Set device
    device = 'mps' if torch.backends.mps.is_available() else 'cpu'

    # Create timestamped save path
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_path = f'models/vietnam_domain_adapted_{timestamp}.pt'

    # Fine-tune
    model, history = fine_tune_best_model(
        base_model_path=base_model_path,
        vietnam_csv=vietnam_csv,
        n_epochs=100,
        batch_size=64,
        learning_rate=5e-5,
        val_split=0.2,
        device=device,
        save_path=save_path,
    )

    print(f"\n🎯 Domain adaptation complete!")
    print(f"   Fine-tuned model saved to: {save_path}")

    # Quick validation on full Vietnam dataset
    print("\n🔍 Quick validation on full Vietnam dataset...")
    from scripts.validate_vietnam import validate_on_vietnam, load_model, VietnamDataset as ValidateVietnamDataset

    vietnam_dataset = ValidateVietnamDataset(vietnam_csv)
    model, _ = load_model(save_path, device)

    metrics, pred_temp, true_temp, pred_amp, true_amp = validate_on_vietnam(
        model, vietnam_dataset, device, use_consistency_loss=False
    )

    print("\n🎯 Domain Adapted Model - Full Vietnam Validation:")
    print("-" * 55)
    print(f"Temperature MAE:  {metrics['temp_mae']:.2f} °C")
    print(f"Temperature RMSE: {metrics['temp_rmse']:.2f} °C")
    print(f"Temperature R²:   {metrics['temp_r2']:.3f}")
    print(f"Ampacity MAE:  {metrics['amp_mae']:.2f} A")
    print(f"Ampacity RMSE: {metrics['amp_rmse']:.2f} A")
    print(f"Ampacity R²:   {metrics['amp_r2']:.3f}")

    # Save final results
    results_path = f"results/domain_adaptation_{timestamp}_results.json"
    final_results = {
        'timestamp': timestamp,
        'base_model': base_model_path,
        'vietnam_validation': metrics,
        'training_history': history,
    }

    with open(results_path, 'w') as f:
        json.dump(final_results, f, indent=2, default=str)

    print(f"📊 Results saved to {results_path}")


if __name__ == "__main__":
    main()
