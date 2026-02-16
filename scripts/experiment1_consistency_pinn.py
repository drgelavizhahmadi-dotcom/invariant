#!/usr/bin/env python3
"""
Experiment 1: Multi-Head PINN with Physics Consistency Constraints

This script implements the improved PINN architecture that enforces physics
consistency between temperature and ampacity predictions.

Key improvements:
- Physics consistency loss: I²R = q_conv + q_rad - q_solar at predicted T
- Multi-head architecture with shared encoder
- Adaptive loss weighting with consistency term

Author: Dr. Gelavizh Ahmadi
Copyright (c) 2026 Invariant Energy GmbH
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from pathlib import Path
import json
from datetime import datetime
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from typing import Dict

from core.model import PhysicsDLR, PhysicsInformedLoss
from core.physics import IEEE738HeatBalance
from core.data import create_dataloaders
from core.train import TrainConfig, train, get_device


def evaluate_model_on_vietnam(
    model: PhysicsDLR,
    vietnam_data_path: str = "data/vietnam_220kv_processed.csv",
    save_results: bool = True,
) -> Dict[str, float]:
    """
    Evaluate model performance on Vietnam transmission line data

    Args:
        model: Trained PhysicsDLR model
        vietnam_data_path: Path to Vietnam dataset
        save_results: Whether to save detailed results

    Returns:
        Dictionary of evaluation metrics
    """
    device = next(model.parameters()).device

    # Load Vietnam data
    if not Path(vietnam_data_path).exists():
        print(f"❌ Vietnam data not found at {vietnam_data_path}")
        return {}

    vietnam_df = pd.read_csv(vietnam_data_path)
    print(f"📊 Loaded Vietnam dataset: {len(vietnam_df)} samples")

    # Prepare inputs (using same feature engineering as training)
    physics = IEEE738HeatBalance().to(device)

    # Vietnam data features
    features = ['T_ambient', 'wind_speed', 'wind_angle', 'solar_irradiance', 'current', 'resistance']
    targets = ['conductor_temp', 'ampacity']

    # Normalize inputs (using training normalizer if available)
    # For now, use simple min-max scaling based on expected ranges
    x_data = vietnam_df[features].values.astype(np.float32)

    # Simple normalization (can be improved with proper normalizer)
    x_min = np.array([0, 0, 0, 0, 0, 0.05])  # Approximate mins
    x_max = np.array([50, 20, 360, 1000, 2000, 0.15])  # Approximate maxes
    x_norm = (x_data - x_min) / (x_max - x_min + 1e-8)

    y_true = vietnam_df[targets].values.astype(np.float32)

    # Convert to tensors
    x_tensor = torch.from_numpy(x_norm).to(device)
    y_tensor = torch.from_numpy(y_true).to(device)

    model.eval()
    with torch.no_grad():
        # Use physics consistency forward pass
        temp_pred, rating_pred, _ = model.forward_with_physics_consistency(
            x_tensor, physics, consistency_weight=0.1
        )

        temp_pred = temp_pred.cpu().numpy().squeeze()
        rating_pred = rating_pred.cpu().numpy().squeeze()

    # Calculate metrics
    temp_true = y_true[:, 0]
    rating_true = y_true[:, 1]

    metrics = {
        'temp_r2': r2_score(temp_true, temp_pred),
        'temp_mae': mean_absolute_error(temp_true, temp_pred),
        'temp_rmse': np.sqrt(mean_squared_error(temp_true, temp_pred)),
        'rating_r2': r2_score(rating_true, rating_pred),
        'rating_mae': mean_absolute_error(rating_true, rating_pred),
        'rating_rmse': np.sqrt(mean_squared_error(rating_true, rating_pred)),
    }

    print("\n📈 Vietnam Validation Results:")
    print(f"   Temperature R²: {metrics['temp_r2']:.4f}")
    print(f"   Temperature MAE: {metrics['temp_mae']:.1f}°C")
    print(f"   Temperature RMSE: {metrics['temp_rmse']:.1f}°C")
    print(f"   Ampacity R²: {metrics['rating_r2']:.4f}")
    print(f"   Ampacity MAE: {metrics['rating_mae']:.1f}A")
    print(f"   Ampacity RMSE: {metrics['rating_rmse']:.1f}A")
    if save_results:
        results_df = pd.DataFrame({
            'temp_true': temp_true,
            'temp_pred': temp_pred,
            'rating_true': rating_true,
            'rating_pred': rating_pred,
            'T_ambient': x_data[:, 0],
            'wind_speed': x_data[:, 1],
            'wind_angle': x_data[:, 2],
            'solar_irradiance': x_data[:, 3],
            'current': x_data[:, 4],
        })

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_path = f"reports/experiment1_vietnam_results_{timestamp}.csv"
        results_df.to_csv(results_path, index=False)
        print(f"💾 Detailed results saved to {results_path}")

        # Save metrics
        metrics_path = f"reports/experiment1_vietnam_metrics_{timestamp}.json"
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=2)
        print(f"💾 Metrics saved to {metrics_path}")

    return metrics


def run_experiment1():
    """
    Run Experiment 1: Multi-Head PINN with Physics Consistency Constraints
    """
    print("🔬 Experiment 1: Multi-Head PINN with Physics Consistency")
    print("=" * 60)

    # Configuration for improved model
    config = TrainConfig(
        n_train=20000,  # Increased training data
        n_val=4000,
        batch_size=256,
        n_epochs=50,  # Reduced epochs for testing
        learning_rate=1e-3,
        physics_weight=0.1,  # Increased physics weight
        temp_weight=1.0,
        rating_weight=0.5,
        consistency_weight=0.1,  # New consistency weight
        real_data_path=None,  # No real data for now
        synthetic_ratio=1.0,  # All synthetic
        save_path="models/experiment1_consistency_model.pt",
    )

    print("⚙️  Configuration:")
    print(f"   Training samples: {config.n_train:,}")
    print(f"   Validation samples: {config.n_val:,}")
    print(f"   Epochs: {config.n_epochs}")
    print(f"   Physics weight: {config.physics_weight}")
    print(f"   Consistency weight: {config.consistency_weight}")
    print(f"   Real data ratio: {1 - config.synthetic_ratio:.1%}")

    # Train the model
    print("\n🚀 Starting training...")
    model, history = train(config)

    # Evaluate on Vietnam data
    print("\n📊 Evaluating on Vietnam dataset...")
    vietnam_metrics = evaluate_model_on_vietnam(model)

    # Save training history
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    history_path = f"reports/experiment1_training_history_{timestamp}.json"
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)
    print(f"💾 Training history saved to {history_path}")

    # Plot training curves
    plt.figure(figsize=(12, 8))

    # Extract loss values from history
    train_losses = [epoch['total_loss'] for epoch in history['train']]
    val_losses = [epoch['total_loss'] for epoch in history['val']]
    
    plt.subplot(2, 3, 1)
    plt.plot(train_losses, label='Train')
    plt.plot(val_losses, label='Val')
    plt.title('Total Loss')
    plt.legend()

    plt.subplot(2, 3, 2)
    train_temp_loss = [epoch['temp_loss'] for epoch in history['train']]
    val_temp_loss = [epoch['temp_loss'] for epoch in history['val']]
    plt.plot(train_temp_loss, label='Train')
    plt.plot(val_temp_loss, label='Val')
    plt.title('Temperature Loss')
    plt.legend()

    plt.subplot(2, 3, 3)
    train_rating_loss = [epoch['rating_loss'] for epoch in history['train']]
    val_rating_loss = [epoch['rating_loss'] for epoch in history['val']]
    plt.plot(train_rating_loss, label='Train')
    plt.plot(val_rating_loss, label='Val')
    plt.title('Rating Loss')
    plt.legend()

    plt.subplot(2, 3, 4)
    train_physics_loss = [epoch['physics_loss'] for epoch in history['train']]
    val_physics_loss = [epoch['physics_loss'] for epoch in history['val']]
    plt.plot(train_physics_loss, label='Train')
    plt.plot(val_physics_loss, label='Val')
    plt.title('Physics Loss')
    plt.legend()

    plt.subplot(2, 3, 5)
    train_consistency_loss = [epoch.get('consistency_loss', 0) for epoch in history['train']]
    val_consistency_loss = [epoch.get('consistency_loss', 0) for epoch in history['val']]
    plt.plot(train_consistency_loss, label='Train')
    plt.plot(val_consistency_loss, label='Val')
    plt.title('Consistency Loss')
    plt.legend()

    plt.subplot(2, 3, 6)
    plt.plot(history['lr'])
    plt.title('Learning Rate')
    plt.yscale('log')

    plt.tight_layout()
    plot_path = f"reports/experiment1_training_curves_{timestamp}.png"
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"💾 Training curves saved to {plot_path}")

    # Summary
    print("\n🎯 Experiment 1 Summary:")
    temp_r2 = vietnam_metrics.get('temp_r2', 'N/A')
    rating_r2 = vietnam_metrics.get('rating_r2', 'N/A')
    temp_mae = vietnam_metrics.get('temp_mae', 'N/A')
    rating_mae = vietnam_metrics.get('rating_mae', 'N/A')
    
    if isinstance(temp_r2, (int, float)) and not np.isnan(temp_r2):
        print(f"   Temperature R²: {temp_r2:.4f}")
    else:
        print(f"   Temperature R²: {temp_r2}")
        
    if isinstance(rating_r2, (int, float)) and not np.isnan(rating_r2):
        print(f"   Ampacity R²: {rating_r2:.4f}")
    else:
        print(f"   Ampacity R²: {rating_r2}")
        
    if isinstance(temp_mae, (int, float)) and not np.isnan(temp_mae):
        print(f"   Temperature MAE: {temp_mae:.1f}°C")
    else:
        print(f"   Temperature MAE: {temp_mae}")
        
    if isinstance(rating_mae, (int, float)) and not np.isnan(rating_mae):
        print(f"   Ampacity MAE: {rating_mae:.1f}A")
    else:
        print(f"   Ampacity MAE: {rating_mae}")

    return model, history, vietnam_metrics


if __name__ == "__main__":
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)

    # Run the experiment
    model, history, metrics = run_experiment1()

    print("\n✅ Experiment 1 completed successfully!")