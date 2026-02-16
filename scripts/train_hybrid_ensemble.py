"""
Invariant-PIKAN: Adversarially-Robust Physics-Informed Neural Networks for Dynamic Line Rating
Copyright (C) 2025 Gelavizh Ahmadi / Invariant Research

This software is licensed under the Business Source License 1.1 (BSL 1.1).
Commercial production use requires a separate license agreement.
See LICENSE.txt for full terms.

"""

#!/usr/bin/env python3
"""
Train Hybrid Ensemble Model

This script trains the HybridEnsemble to learn optimal blending weights
between neural network and calibrated physics predictions.

The ensemble learns when to trust each approach based on the data,
potentially improving overall performance while maintaining interpretability.

Author: Dr. Gelavizh Ahmadi
Copyright (c) 2026 Invariant Energy GmbH
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import json
from typing import Dict, List, Tuple
import time
import sys

# Add project root to path
sys.path.append('.')

from core.model import PhysicsDLR, HybridEnsemble
from core.data import SyntheticDLRDataset, DataConfig
from scripts.validate_vietnam import VietnamDataset, load_model
from calibration_results.vietnam_params import VIETNAM_LINE_PARAMS, CALIBRATION_METRICS


def create_training_data(
    vietnam_csv: str = 'data/mendeley/vietnam_220kv.csv',
    synthetic_samples: int = 10000,
    seed: int = 42
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Create training data combining Vietnam and synthetic data

    Args:
        vietnam_csv: Path to Vietnam dataset
        synthetic_samples: Number of synthetic samples to generate
        seed: Random seed

    Returns:
        X_train, y_train tensors
    """
    print("Creating training data...")

    # Load Vietnam data
    vietnam_dataset = VietnamDataset(vietnam_csv)
    vietnam_data = []
    vietnam_targets = []

    for i in range(len(vietnam_dataset)):
        x, y = vietnam_dataset[i]
        vietnam_data.append(x)
        vietnam_targets.append(y)

    X_vietnam = torch.tensor(np.array(vietnam_data), dtype=torch.float32)
    y_vietnam = torch.tensor(np.array(vietnam_targets), dtype=torch.float32)

    print(f"Vietnam data: {len(X_vietnam)} samples")

    # Generate synthetic data with similar distribution to Vietnam
    config = DataConfig()

    # Adjust config to match Vietnam conditions (hotter, windier)
    config.T_ambient_min = 25.0
    config.T_ambient_max = 45.0
    config.wind_speed_min = 0.5
    config.wind_speed_max = 8.0
    config.solar_min = 200
    config.solar_max = 1000

    synthetic_dataset = SyntheticDLRDataset(
        n_samples=synthetic_samples,
        config=config,
        seed=seed,
        add_noise=True
    )

    X_synthetic = synthetic_dataset.X
    y_synthetic = torch.cat([
        synthetic_dataset.T_conductor.unsqueeze(1),
        synthetic_dataset.I_rating.unsqueeze(1)
    ], dim=1)

    print(f"Synthetic data: {len(X_synthetic)} samples")

    # Combine datasets
    X_train = torch.cat([X_vietnam, X_synthetic], dim=0)
    y_train = torch.cat([y_vietnam, y_synthetic], dim=0)

    print(f"Combined training data: {len(X_train)} samples")

    return X_train, y_train


def train_hybrid_ensemble(
    model_path: str = 'models/best_model.pt',
    vietnam_csv: str = 'data/mendeley/vietnam_220kv.csv',
    synthetic_samples: int = 10000,
    batch_size: int = 64,
    epochs: int = 50,
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-4,
    device: str = 'cpu',
    save_path: str = 'models/hybrid_ensemble.pt'
) -> Dict[str, any]:
    """
    Train the hybrid ensemble model

    Args:
        model_path: Path to trained neural model
        vietnam_csv: Path to Vietnam dataset
        synthetic_samples: Number of synthetic samples
        batch_size: Training batch size
        epochs: Number of training epochs
        learning_rate: Learning rate for optimizer
        weight_decay: Weight decay for regularization
        device: Device to train on
        save_path: Path to save trained ensemble

    Returns:
        Training history and final metrics
    """
    print("🔄 Training Hybrid Ensemble")
    print("=" * 60)

    device = torch.device(device)
    print(f"Using device: {device}")

    # Load trained neural model
    print(f"Loading neural model from {model_path}...")
    neural_model, normalizer = load_model(model_path, device)
    neural_model.eval()  # Freeze neural weights

    # Load calibrated parameters
    print("Loading calibrated physics parameters...")
    calibrated_params = VIETNAM_LINE_PARAMS

    # Create hybrid ensemble
    print("Creating hybrid ensemble...")
    ensemble = HybridEnsemble(neural_model, calibrated_params, learnable_weights=True)
    ensemble = ensemble.to(device)
    ensemble.train()

    # Create training data
    X_train, y_train = create_training_data(vietnam_csv, synthetic_samples)

    # Create data loader
    train_dataset = TensorDataset(X_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # Optimizer (only optimize blending weights)
    optimizer = optim.Adam([
        {'params': ensemble.physics_weight_logit, 'lr': learning_rate},
        {'params': ensemble.neural_weight_logit, 'lr': learning_rate}
    ], weight_decay=weight_decay)

    # Loss function (MSE on ampacity prediction)
    criterion = nn.MSELoss()

    # Training history
    history = {
        'epoch': [],
        'train_loss': [],
        'physics_weight': [],
        'neural_weight': []
    }

    print("\n🚀 Starting training...")
    start_time = time.time()

    for epoch in range(epochs):
        epoch_loss = 0.0
        n_batches = 0

        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            optimizer.zero_grad()

            # Forward pass
            predictions = ensemble(batch_x)
            pred_amp = predictions['ampacity']
            target_amp = batch_y[:, 1]  # Ampacity is second column

            # Compute loss
            loss = criterion(pred_amp, target_amp)

            # Backward pass
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

        # Record metrics
        avg_loss = epoch_loss / n_batches
        phys_weight, neur_weight = ensemble.get_blending_weights()

        history['epoch'].append(epoch + 1)
        history['train_loss'].append(avg_loss)
        history['physics_weight'].append(phys_weight)
        history['neural_weight'].append(neur_weight)

        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1:3d} | Loss: {avg_loss:.4f} | "
                  f"Physics: {phys_weight:.3f} | Neural: {neur_weight:.3f}")

    training_time = time.time() - start_time
    print(f"⏱️  Training completed in {training_time:.1f} seconds")
    # Save trained ensemble
    print(f"\n💾 Saving ensemble to {save_path}...")
    torch.save({
        'ensemble_state_dict': ensemble.state_dict(),
        'neural_model_path': model_path,
        'calibrated_params': calibrated_params,
        'training_config': {
            'synthetic_samples': synthetic_samples,
            'batch_size': batch_size,
            'epochs': epochs,
            'learning_rate': learning_rate,
            'weight_decay': weight_decay
        },
        'history': history,
        'final_weights': ensemble.get_blending_weights(),
        'final_loss': history['train_loss'][-1]
    }, save_path)

    # Final evaluation
    print("\n📊 Final Results:")
    final_phys_weight, final_neur_weight = ensemble.get_blending_weights()
    print(f"   Physics Weight: {final_phys_weight:.3f}")
    print(f"   Neural Weight: {final_neur_weight:.3f}")
    print(f"   Final Loss: {history['train_loss'][-1]:.4f}")

    return {
        'model': ensemble,
        'history': history,
        'final_weights': (final_phys_weight, final_neur_weight),
        'final_loss': history['train_loss'][-1],
        'training_time': training_time
    }


def evaluate_ensemble(
    ensemble_path: str = 'models/hybrid_ensemble.pt',
    vietnam_csv: str = 'data/mendeley/vietnam_220kv.csv',
    device: str = 'cpu'
) -> Dict[str, float]:
    """
    Evaluate trained ensemble on Vietnam data

    Args:
        ensemble_path: Path to saved ensemble
        vietnam_csv: Path to Vietnam dataset
        device: Device for evaluation

    Returns:
        Evaluation metrics
    """
    print("🔍 Evaluating Hybrid Ensemble")
    print("=" * 40)

    device = torch.device(device)

    # Load ensemble
    print(f"Loading ensemble from {ensemble_path}...")
    checkpoint = torch.load(ensemble_path, map_location=device)

    # Recreate ensemble
    neural_model, _ = load_model(checkpoint['neural_model_path'], device)
    neural_model.eval()

    ensemble = HybridEnsemble(neural_model, checkpoint['calibrated_params'])
    ensemble.load_state_dict(checkpoint['ensemble_state_dict'])
    ensemble = ensemble.to(device)
    ensemble.eval()

    # Load Vietnam data
    vietnam_dataset = VietnamDataset(vietnam_csv)
    vietnam_data = []
    vietnam_targets = []

    for i in range(len(vietnam_dataset)):
        x, y = vietnam_dataset[i]
        vietnam_data.append(x)
        vietnam_targets.append(y)

    X_test = torch.tensor(np.array(vietnam_data), dtype=torch.float32).to(device)
    y_test = torch.tensor(np.array(vietnam_targets), dtype=torch.float32).to(device)

    print(f"Evaluating on {len(X_test)} Vietnam samples...")

    # Evaluate
    with torch.no_grad():
        predictions = ensemble(X_test, return_components=True)

        pred_temp = predictions['temperature'].cpu().numpy()
        pred_amp = predictions['ampacity'].cpu().numpy()
        pred_amp_physics = predictions['ampacity_physics'].cpu().numpy()
        pred_amp_neural = predictions['ampacity_neural'].cpu().numpy()

        true_temp = y_test[:, 0].cpu().numpy()
        true_amp = y_test[:, 1].cpu().numpy()

    # Calculate metrics
    temp_mae = np.mean(np.abs(pred_temp - true_temp))
    temp_rmse = np.sqrt(np.mean((pred_temp - true_temp)**2))

    amp_mae = np.mean(np.abs(pred_amp - true_amp))
    amp_rmse = np.sqrt(np.mean((pred_amp - true_amp)**2))

    # Component metrics
    physics_mae = np.mean(np.abs(pred_amp_physics - true_amp))
    neural_mae = np.mean(np.abs(pred_amp_neural - true_amp))

    phys_weight, neur_weight = ensemble.get_blending_weights()

    metrics = {
        'temperature_mae': temp_mae,
        'temperature_rmse': temp_rmse,
        'ampacity_mae': amp_mae,
        'ampacity_rmse': amp_rmse,
        'physics_ampacity_mae': physics_mae,
        'neural_ampacity_mae': neural_mae,
        'physics_weight': phys_weight,
        'neural_weight': neur_weight,
        'ensemble_improvement': (neural_mae - amp_mae) / neural_mae * 100
    }

    print("\n📊 Evaluation Results:")
    print(f"   Temperature MAE: {temp_mae:.2f}°C")
    print(f"   Temperature RMSE: {temp_rmse:.2f}°C")
    print(f"   Ampacity MAE: {amp_mae:.0f}A")
    print(f"   Ampacity RMSE: {amp_rmse:.0f}A")
    print(f"   Physics Ampacity MAE: {physics_mae:.0f}A")
    print(f"   Neural Ampacity MAE: {neural_mae:.0f}A")
    print(f"   Physics Weight: {phys_weight:.3f}")
    print(f"   Neural Weight: {neur_weight:.3f}")
    print(f"   Ensemble Improvement: {metrics['ensemble_improvement']:+.1f}%")

    return metrics


def plot_training_history(history: Dict, save_path: str = 'reports/hybrid_training.png'):
    """Plot training history"""
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    # Loss
    axes[0, 0].plot(history['epoch'], history['train_loss'], 'b-', linewidth=2)
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Training Loss (MSE)')
    axes[0, 0].set_title('Training Loss')
    axes[0, 0].grid(True, alpha=0.3)

    # Weights
    axes[0, 1].plot(history['epoch'], history['physics_weight'], 'r-', label='Physics', linewidth=2)
    axes[0, 1].plot(history['epoch'], history['neural_weight'], 'b-', label='Neural', linewidth=2)
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Weight')
    axes[0, 1].set_title('Blending Weights')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # Weight ratio
    weight_ratio = np.array(history['physics_weight']) / np.array(history['neural_weight'])
    axes[1, 0].plot(history['epoch'], weight_ratio, 'g-', linewidth=2)
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Physics/Neural Weight Ratio')
    axes[1, 0].set_title('Weight Ratio Evolution')
    axes[1, 0].grid(True, alpha=0.3)

    # Final weights bar chart
    final_phys = history['physics_weight'][-1]
    final_neur = history['neural_weight'][-1]
    axes[1, 1].bar(['Physics', 'Neural'], [final_phys, final_neur],
                   color=['red', 'blue'], alpha=0.7)
    axes[1, 1].set_ylabel('Final Weight')
    axes[1, 1].set_title('Final Blending Weights')
    axes[1, 1].grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"📈 Training plot saved to {save_path}")


def main():
    """Main training function"""
    import argparse

    parser = argparse.ArgumentParser(description='Train Hybrid Ensemble')
    parser.add_argument('--model-path', type=str, default='models/best_model.pt',
                       help='Path to trained neural model')
    parser.add_argument('--vietnam-csv', type=str, default='data/mendeley/vietnam_220kv.csv',
                       help='Path to Vietnam dataset')
    parser.add_argument('--synthetic-samples', type=int, default=10000,
                       help='Number of synthetic training samples')
    parser.add_argument('--batch-size', type=int, default=64,
                       help='Training batch size')
    parser.add_argument('--epochs', type=int, default=50,
                       help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=1e-3,
                       help='Learning rate')
    parser.add_argument('--weight-decay', type=float, default=1e-4,
                       help='Weight decay')
    parser.add_argument('--device', type=str, default='cpu',
                       help='Device to train on')
    parser.add_argument('--save-path', type=str, default='models/hybrid_ensemble.pt',
                       help='Path to save trained ensemble')
    parser.add_argument('--evaluate-only', action='store_true',
                       help='Only evaluate existing model')

    args = parser.parse_args()

    if args.evaluate_only:
        # Only evaluate
        metrics = evaluate_ensemble(args.save_path, args.vietnam_csv, args.device)

        # Save metrics
        with open('reports/hybrid_evaluation.json', 'w') as f:
            json.dump(metrics, f, indent=2)

    else:
        # Train and evaluate
        results = train_hybrid_ensemble(
            model_path=args.model_path,
            vietnam_csv=args.vietnam_csv,
            synthetic_samples=args.synthetic_samples,
            batch_size=args.batch_size,
            epochs=args.epochs,
            learning_rate=args.lr,
            weight_decay=args.weight_decay,
            device=args.device,
            save_path=args.save_path
        )

        # Plot training history
        plot_training_history(results['history'])

        # Evaluate
        metrics = evaluate_ensemble(args.save_path, args.vietnam_csv, args.device)

        # Save comprehensive results
        final_results = {
            'training': results,
            'evaluation': metrics,
            'config': vars(args)
        }

        with open('reports/hybrid_ensemble_results.json', 'w') as f:
            json.dump(final_results, f, indent=2, default=str)

        print("\n✅ Hybrid ensemble training and evaluation completed!")
        print("Results saved to reports/hybrid_ensemble_results.json")


if __name__ == "__main__":
    main()
