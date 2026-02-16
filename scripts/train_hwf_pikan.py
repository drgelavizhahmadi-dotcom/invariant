"""
Train HWF-PIKAN model for Dynamic Line Rating
Advanced multi-resolution physics-informed neural network
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
import time
import json
from pathlib import Path
import argparse

# Add project root to path
import sys
sys.path.append('.')

from models.hwf_pikan import create_hwf_pikan_model
from core.data import VietnamDataset
from core.physics import physics_loss_fn


def train_hwf_pikan_model(
    vietnam_csv='data/mendeley/vietnam_220kv.csv',
    batch_size=32,
    epochs=200,
    device='cpu',
    save_path='models/hwf_pikan_dlr.pt'
):
    """
    Two-stage training protocol for HWF-PIKAN:
    Stage 1: Adam for initial convergence (first 100 epochs)
    Stage 2: L-BFGS for fine-tuning (remaining epochs)
    """

    print("🚀 Training HWF-PIKAN DLR Model (Two-Stage Protocol)")
    print("=" * 60)

    device = torch.device(device)
    print(f"Using device: {device}")

    # Load Vietnam dataset
    print(f"Loading training data from {vietnam_csv}...")
    vietnam_dataset = VietnamDataset(vietnam_csv)
    dataloader = DataLoader(vietnam_dataset, batch_size=batch_size, shuffle=True)

    print(f"Dataset: {len(vietnam_dataset)} samples")
    print(f"Temperature range: {vietnam_dataset.T_ambient.min():.1f} - {vietnam_dataset.T_ambient.max():.1f} °C")
    print(f"Wind speed range: {vietnam_dataset.wind_speed.min():.1f} - {vietnam_dataset.wind_speed.max():.1f} m/s")
    print(f"Ampacity range: {vietnam_dataset.ampacity.min():.0f} - {vietnam_dataset.ampacity.max():.0f} A")

    # Create model
    print("Creating HWF-PIKAN model...")
    model = create_hwf_pikan_model()
    model = model.to(device)

    # Optimizer and scheduler
    lr = 1e-3  # Adam learning rate for first stage
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    # Training history
    history = {
        'epoch': [],
        'train_loss': [],
        'data_loss': [],
        'physics_loss': [],
        'learning_rate': [],
        'physics_residual': [],
        'loss_weights': [],
        'stage': []
    }

    print(f"\n🏃 Starting two-stage training for {epochs} epochs...")

    start_time = time.time()
    best_mae = float('inf')

    # Stage 1: Adam optimization (first 100 epochs or half the epochs)
    adam_epochs = min(100, epochs // 2)
    print(f"\n📈 Stage 1: Adam Optimization ({adam_epochs} epochs)")

    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=adam_epochs)

    for epoch in range(adam_epochs):
        model.train()
        epoch_losses = []
        epoch_data_losses = []
        epoch_physics_losses = []
        epoch_residuals = []
        epoch_weights = []

        for batch in dataloader:
            # Move to device - batch is (x, y) tuple
            x, targets = batch
            x = x.to(device)
            targets = targets.to(device)

            # Create weather dict for physics
            weather_dict = {
                'T_amb': x[:, 0],      # Ambient temperature
                'wind_speed': x[:, 1], # Wind speed
                'solar': x[:, 3]       # Solar irradiance (index 3, not 2)
            }

            # Forward pass
            outputs = model(x, weather_dict)

            # Data loss (MSE on temperature and ampacity)
            T_pred = outputs['temperature']
            I_pred = outputs['ampacity']

            T_target = targets[:, 0]  # Conductor temperature
            I_target = targets[:, 1]  # Ampacity

            data_loss = (
                nn.functional.mse_loss(T_pred, T_target) +
                nn.functional.mse_loss(I_pred, I_target)
            )

            # Physics loss (heat balance constraint)
            physics_loss = outputs['physics_residual']

            # Adaptive loss balancing
            losses = {
                'data_temp': nn.functional.mse_loss(T_pred, T_target),
                'data_amp': nn.functional.mse_loss(I_pred, I_target),
                'physics': physics_loss
            }

            total_loss, loss_weights = model.loss_balancer(losses)

            # Backward pass
            optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            # Record losses
            epoch_losses.append(total_loss.item())
            epoch_data_losses.append(data_loss.item())
            epoch_physics_losses.append(physics_loss.item())
            epoch_residuals.append(outputs['physics_residual'].item())

        # Average losses for epoch
        avg_loss = np.mean(epoch_losses)
        avg_data_loss = np.mean(epoch_data_losses)
        avg_physics_loss = np.mean(epoch_physics_losses)
        avg_residual = np.mean(epoch_residuals)
        current_lr = optimizer.param_groups[0]['lr']

        # Update history
        history['epoch'].append(epoch + 1)
        history['train_loss'].append(avg_loss)
        history['data_loss'].append(avg_data_loss)
        history['physics_loss'].append(avg_physics_loss)
        history['learning_rate'].append(current_lr)
        history['physics_residual'].append(avg_residual)
        history['stage'].append('Adam')

        # Learning rate scheduling
        scheduler.step()

        # Print progress
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1:3d} | Loss: {avg_loss:.4f} | Data: {avg_data_loss:.4f} | "
                  f"Physics: {avg_physics_loss:.4f} | LR: {current_lr:.6f}")

    # Stage 2: L-BFGS fine-tuning (disabled due to numerical instability)
    lbfgs_epochs = epochs - adam_epochs
    if lbfgs_epochs > 0:
        print(f"\n🔬 Stage 2: L-BFGS Fine-tuning (skipped - using Adam result)")
        print("Note: L-BFGS optimization caused numerical instability. Using Adam-trained model.")

    training_time = time.time() - start_time
    print(f"⏱️  Training completed in {training_time:.1f} seconds")

    training_time = time.time() - start_time
    print(f"⏱️  Training completed in {training_time:.1f} seconds")
    # Save trained model
    print(f"\n💾 Saving HWF-PIKAN model to {save_path}...")
    torch.save({
        'model_state_dict': model.state_dict(),
        'config': {
            'input_dim': 4,
            'hidden_dim': 32,
            'fourier_bands': 16,
            'wavelet_scales': 8,
            'batch_size': batch_size,
            'epochs': epochs,
            'learning_rate': lr
        },
        'history': history,
        'training_time': training_time
    }, save_path)

    print("✅ HWF-PIKAN training completed!")

    return model, history


def evaluate_hwf_pikan_model(
    model_path='models/hwf_pikan_dlr.pt',
    vietnam_csv='data/mendeley/vietnam_220kv.csv',
    n_samples=None,
    device='cpu'
):
    """
    Evaluate trained HWF-PIKAN model
    """

    print("🔍 Evaluating HWF-PIKAN Model")
    print("=" * 40)

    device = torch.device(device)

    # Load model
    print(f"Loading model from {model_path}...")
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)

    model = create_hwf_pikan_model()
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()

    # Load test data
    print(f"Loading test data from {vietnam_csv}...")
    vietnam_dataset = VietnamDataset(vietnam_csv)

    if n_samples is None:
        n_samples = len(vietnam_dataset)
    else:
        n_samples = min(n_samples, len(vietnam_dataset))

    print(f"Evaluating on {n_samples} samples...")

    # Collect predictions
    predictions = []
    targets = []
    conditions = []

    start_time = time.time()

    with torch.no_grad():
        for i in range(n_samples):
            if (i + 1) % 100 == 0:
                print(f"   Processed {i+1}/{n_samples} samples...")

            sample = vietnam_dataset[i]
            x, target = sample  # Unpack tuple
            x = x.unsqueeze(0).to(device)

            # Create weather dict
            weather_dict = {
                'T_amb': x[:, 0],
                'wind_speed': x[:, 1],
                'solar': x[:, 3]  # Solar is at index 3
            }

            # Forward pass
            outputs = model(x, weather_dict)

            predictions.append({
                'temperature': outputs['temperature'].item(),
                'ampacity': outputs['ampacity'].item(),
                'physics_residual': outputs['physics_residual'].item()
            })

            targets.append(target.tolist())

            conditions.append({
                'T_ambient': x[0, 0].item(),
                'wind_speed': x[0, 1].item(),
                'wind_angle': x[0, 2].item(),  # Wind angle at index 2
                'solar_irradiance': x[0, 3].item(),  # Solar at index 3
                'current': x[0, 4].item()  # Current at index 4
            })

    eval_time = time.time() - start_time
    print(f"⏱️  Evaluation completed in {eval_time:.1f} seconds")
    # Convert to arrays
    pred_temps = np.array([p['temperature'] for p in predictions])
    pred_amps = np.array([p['ampacity'] for p in predictions])
    true_temps = np.array([t[0] for t in targets])
    true_amps = np.array([t[1] for t in targets])
    physics_residuals = np.array([p['physics_residual'] for p in predictions])
    conditions_df = pd.DataFrame(conditions)

    # Calculate metrics
    temp_errors = pred_temps - true_temps
    amp_errors = pred_amps - true_amps

    metrics = {
        'n_samples': n_samples,
        'evaluation_time': eval_time,
        'temperature': {
            'mae': np.mean(np.abs(temp_errors)),
            'rmse': np.sqrt(np.mean(temp_errors**2)),
            'mean_error': np.mean(temp_errors)
        },
        'ampacity': {
            'mae': np.mean(np.abs(amp_errors)),
            'rmse': np.sqrt(np.mean(amp_errors**2)),
            'mean_error': np.mean(amp_errors)
        },
        'physics_consistency': {
            'mean_residual': np.mean(physics_residuals),
            'max_residual': np.max(physics_residuals),
            'residual_std': np.std(physics_residuals)
        },
        'environmental_conditions': {
            'T_ambient_range': [conditions_df['T_ambient'].min(), conditions_df['T_ambient'].max()],
            'wind_speed_range': [conditions_df['wind_speed'].min(), conditions_df['wind_speed'].max()],
            'solar_range': [conditions_df['solar_irradiance'].min(), conditions_df['solar_irradiance'].max()]
        }
    }

    # Print results
    print("\n📊 HWF-PIKAN Evaluation Results:")
    print(f"🌡️  Temperature - MAE: {metrics['temperature']['mae']:.2f}°C, RMSE: {metrics['temperature']['rmse']:.2f}°C")
    print(f"⚡ Ampacity - MAE: {metrics['ampacity']['mae']:.1f}A, RMSE: {metrics['ampacity']['rmse']:.1f}A")
    print(f"🔬 Physics Residual - Mean: {metrics['physics_consistency']['mean_residual']:.4f}")

    return metrics, predictions, targets, conditions_df


def main():
    """Main training function"""
    parser = argparse.ArgumentParser(description='Train HWF-PIKAN DLR Model')
    parser.add_argument('--vietnam-csv', type=str, default='data/mendeley/vietnam_220kv.csv',
                       help='Path to Vietnam dataset')
    parser.add_argument('--batch-size', type=int, default=32, help='Training batch size')
    parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--device', type=str, default='cpu', help='Device for training')
    parser.add_argument('--save-path', type=str, default='models/hwf_pikan_dlr.pt',
                       help='Path to save trained model')
    parser.add_argument('--n-samples', type=int, default=None,
                       help='Number of samples to evaluate (None = all)')
    parser.add_argument('--eval-only', action='store_true',
                       help='Only evaluate existing model')

    args = parser.parse_args()

    if args.eval_only:
        # Only evaluate
        metrics, predictions, targets, conditions = evaluate_hwf_pikan_model(
            model_path=args.save_path,
            vietnam_csv=args.vietnam_csv,
            n_samples=args.n_samples,
            device=args.device
        )
    else:
        # Train and evaluate
        model, history = train_hwf_pikan_model(
            vietnam_csv=args.vietnam_csv,
            batch_size=args.batch_size,
            epochs=args.epochs,
            device=args.device,
            save_path=args.save_path
        )

        # Evaluate trained model
        metrics, predictions, targets, conditions = evaluate_hwf_pikan_model(
            model_path=args.save_path,
            vietnam_csv=args.vietnam_csv,
            n_samples=args.n_samples,
            device=args.device
        )

    # Save results
    timestamp = pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')
    results_file = f"reports/hwf_pikan_results_{timestamp}.json"

    final_results = {
        'experiment': 'hwf_pikan_dlr',
        'timestamp': timestamp,
        'evaluation': metrics,
        'config': vars(args)
    }

    with open(results_file, 'w') as f:
        json.dump(final_results, f, indent=2, default=str)

    print(f"💾 Detailed results saved to {results_file}")
    print("\n✅ HWF-PIKAN experiment completed!")


if __name__ == "__main__":
    main()