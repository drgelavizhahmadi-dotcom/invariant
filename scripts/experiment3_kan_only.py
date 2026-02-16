"""
Invariant-PIKAN: Adversarially-Robust Physics-Informed Neural Networks for Dynamic Line Rating
Copyright (C) 2025 Gelavizh Ahmadi / Invariant Research

This software is licensed under the Business Source License 1.1 (BSL 1.1).
Commercial production use requires a separate license agreement.
See LICENSE.txt for full terms.

"""

#!/usr/bin/env python3
"""
Experiment 3: KAN Residual Only

This experiment tests KAN (Kolmogorov-Arnold Network) residual correction:
- Start with neural ampacity prediction (415A baseline)
- Add KAN to predict residual correction
- Train only KAN, neural network remains frozen
- No physics blending

Expected: ~390-400A improvement over neural baseline

Author: Dr. Gelavizh Ahmadi
Copyright (c) 2026 Invariant Energy GmbH
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import json
from typing import Dict, List, Tuple
import time
import math
import sys

# Add project root to path
sys.path.append('.')

from core.model import PhysicsDLR
from scripts.validate_vietnam import VietnamDataset, load_model


class KANLayer(nn.Module):
    """
    Kolmogorov-Arnold Network Layer using B-splines

    Based on the KAN paper: "KAN: Kolmogorov-Arnold Networks"
    Uses B-spline basis functions instead of traditional activations.
    """

    def __init__(self, input_dim: int, output_dim: int, num_grids: int = 8, spline_order: int = 3):
        """
        Initialize KAN layer

        Args:
            input_dim: Input dimension
            output_dim: Output dimension
            num_grids: Number of grid points for B-splines
            spline_order: Order of B-splines (3 = cubic)
        """
        super().__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_grids = num_grids
        self.spline_order = spline_order

        # Grid points for basis functions (learnable)
        grid_tensor = torch.linspace(-1, 1, num_grids)
        self.register_buffer('grid', grid_tensor.unsqueeze(0).unsqueeze(0).repeat(input_dim, output_dim, 1))

        # Basis coefficients (learnable)
        self.coeffs = nn.Parameter(torch.randn(input_dim, output_dim, num_grids))

        # Scaling factors
        self.scale = nn.Parameter(torch.ones(input_dim, output_dim))
        self.bias = nn.Parameter(torch.zeros(input_dim, output_dim))

    def b_spline_basis(self, x: torch.Tensor, grid: torch.Tensor, k: int) -> torch.Tensor:
        """
        Simplified B-spline basis using polynomial interpolation
        For now, use a simple piecewise linear basis for stability
        """
        x = x.unsqueeze(-1).unsqueeze(-1)  # [batch, input_dim, 1, 1]
        grid = grid.unsqueeze(0)  # [1, input_dim, output_dim, grid_size]

        # Simple piecewise linear basis
        # Find which grid interval x falls into
        grid_expanded = grid.expand(x.shape[0], -1, -1, -1)  # [batch, input_dim, output_dim, grid_size]

        # Compute distances to grid points
        dist = torch.abs(x - grid_expanded)  # [batch, input_dim, output_dim, grid_size]

        # Use inverse distance weighting as a simple basis
        # Avoid division by zero
        weights = 1.0 / (dist + 1e-6)
        # Normalize
        weights = weights / weights.sum(dim=-1, keepdim=True)

        return weights

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through KAN layer

        Args:
            x: Input tensor [batch, input_dim]

        Returns:
            Output tensor [batch, output_dim]
        """
        batch_size = x.shape[0]

        # Normalize input to [-1, 1] range
        x_norm = 2 * (x - x.min(dim=0, keepdim=True)[0]) / \
                 (x.max(dim=0, keepdim=True)[0] - x.min(dim=0, keepdim=True)[0] + 1e-6) - 1

        # Compute B-spline basis
        basis = self.b_spline_basis(x_norm, self.grid, self.spline_order)
        # basis shape: [batch, input_dim, output_dim, num_basis]

        # Apply coefficients
        # coeffs shape: [input_dim, output_dim, num_basis]
        coeffs_expanded = self.coeffs.unsqueeze(0)  # [1, input_dim, output_dim, num_basis]

        # Element-wise multiplication and sum
        weighted_basis = basis * coeffs_expanded  # [batch, input_dim, output_dim, num_basis]
        output = weighted_basis.sum(dim=-1)  # [batch, input_dim, output_dim]

        # Apply scaling and bias
        scale_expanded = self.scale.unsqueeze(0)  # [1, input_dim, output_dim]
        bias_expanded = self.bias.unsqueeze(0)  # [1, input_dim, output_dim]

        output = output * scale_expanded + bias_expanded

        # Sum over input dimensions
        output = output.sum(dim=1)  # [batch, output_dim]

        return output


class KANResidual(nn.Module):
    """
    KAN-based residual correction for neural ampacity predictions

    This model:
    1. Takes neural network predictions as input
    2. Uses KAN to predict residual corrections
    3. Adds residual to neural prediction
    4. Only KAN is trained, neural network is frozen
    """

    def __init__(self, neural_model: nn.Module, kan_hidden_dim: int = 32, num_grids: int = 8):
        """
        Initialize KAN residual model

        Args:
            neural_model: Frozen neural network
            kan_hidden_dim: Hidden dimension for KAN layers
            num_grids: Number of grid points for B-splines
        """
        super().__init__()

        self.neural_model = neural_model

        # KAN layers for residual prediction
        input_dim = 6  # Same as neural model input
        self.kan_layers = nn.Sequential(
            KANLayer(input_dim, kan_hidden_dim, num_grids=num_grids),
            nn.ReLU(),
            KANLayer(kan_hidden_dim, kan_hidden_dim // 2, num_grids=num_grids),
            nn.ReLU(),
            KANLayer(kan_hidden_dim // 2, 1, num_grids=num_grids)  # Output residual
        )

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass: neural prediction + KAN residual

        Args:
            x: Input tensor [batch, input_dim]

        Returns:
            Dictionary with predictions
        """
        # Get neural predictions (frozen)
        with torch.no_grad():
            neural_temp, neural_amp = self.neural_model(x)

        # KAN residual prediction
        residual = self.kan_layers(x).squeeze(-1)

        # Add residual to neural ampacity
        corrected_amp = neural_amp.squeeze() + residual

        return {
            'temperature': neural_temp.squeeze(),
            'ampacity': corrected_amp,
            'ampacity_neural': neural_amp.squeeze(),
            'residual': residual
        }


def train_kan_residual(
    neural_model_path: str = 'models/best_model.pt',
    vietnam_csv: str = 'data/mendeley/vietnam_220kv.csv',
    kan_hidden_dim: int = 32,
    num_grids: int = 8,
    batch_size: int = 32,
    epochs: int = 100,
    learning_rate: float = 1e-3,
    device: str = 'cpu',
    save_path: str = 'models/kan_residual.pt'
) -> Dict[str, any]:
    """
    Train KAN residual correction model

    Args:
        neural_model_path: Path to trained neural model
        vietnam_csv: Path to Vietnam dataset
        kan_hidden_dim: Hidden dimension for KAN
        num_grids: Number of B-spline grids
        batch_size: Training batch size
        epochs: Number of training epochs
        learning_rate: Learning rate
        device: Training device
        save_path: Path to save trained model

    Returns:
        Training history and final metrics
    """
    print("🚀 Training KAN Residual Model")
    print("=" * 50)

    device = torch.device(device)
    print(f"Using device: {device}")

    # Load neural model (frozen)
    print(f"Loading neural model from {neural_model_path}...")
    neural_model, normalizer = load_model(neural_model_path, device)
    neural_model.eval()  # Freeze neural weights

    # Create KAN residual model
    print("Creating KAN residual model...")
    kan_model = KANResidual(neural_model, kan_hidden_dim, num_grids)
    kan_model = kan_model.to(device)
    kan_model.train()

    # Load training data
    print(f"Loading training data from {vietnam_csv}...")
    vietnam_dataset = VietnamDataset(vietnam_csv)

    # Create data loader
    train_data = []
    train_targets = []

    for i in range(len(vietnam_dataset)):
        x, y = vietnam_dataset[i]
        train_data.append(x)
        train_targets.append(y)

    X_train = torch.tensor(np.array(train_data), dtype=torch.float32)
    y_train = torch.tensor(np.array(train_targets), dtype=torch.float32)

    train_dataset = torch.utils.data.TensorDataset(X_train, y_train)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    print(f"Training on {len(X_train)} samples")

    # Optimizer (only KAN parameters)
    optimizer = optim.Adam(kan_model.kan_layers.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)

    # Loss function
    criterion = nn.MSELoss()

    # Training history
    history = {
        'epoch': [],
        'train_loss': [],
        'learning_rate': []
    }

    print("\n🏃 Starting training...")
    start_time = time.time()

    for epoch in range(epochs):
        epoch_loss = 0.0
        n_batches = 0

        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            optimizer.zero_grad()

            # Forward pass
            predictions = kan_model(batch_x)
            pred_amp = predictions['ampacity']
            target_amp = batch_y[:, 1]  # Ampacity target

            # Compute loss
            loss = criterion(pred_amp, target_amp)

            # Backward pass
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

        # Record metrics
        avg_loss = epoch_loss / n_batches
        current_lr = optimizer.param_groups[0]['lr']

        history['epoch'].append(epoch + 1)
        history['train_loss'].append(avg_loss)
        history['learning_rate'].append(current_lr)

        # Learning rate scheduling
        scheduler.step(avg_loss)

        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1:3d} | Loss: {avg_loss:.4f} | "
                  f"LR: {current_lr:.6f}")

    training_time = time.time() - start_time
    print(f"⏱️  Training completed in {training_time:.1f} seconds")
    # Save trained model
    print(f"\n💾 Saving KAN residual model to {save_path}...")
    torch.save({
        'kan_model_state_dict': kan_model.kan_layers.state_dict(),
        'neural_model_path': neural_model_path,
        'config': {
            'kan_hidden_dim': kan_hidden_dim,
            'num_grids': num_grids,
            'batch_size': batch_size,
            'epochs': epochs,
            'learning_rate': learning_rate
        },
        'history': history,
        'final_loss': history['train_loss'][-1]
    }, save_path)

    return {
        'model': kan_model,
        'history': history,
        'final_loss': history['train_loss'][-1],
        'training_time': training_time
    }


def evaluate_kan_residual(
    model_path: str = 'models/kan_residual.pt',
    vietnam_csv: str = 'data/mendeley/vietnam_220kv.csv',
    device: str = 'cpu',
    n_samples: int = None
) -> Dict[str, float]:
    """
    Evaluate trained KAN residual model

    Args:
        model_path: Path to saved KAN model
        vietnam_csv: Path to Vietnam dataset
        device: Device for evaluation
        n_samples: Number of samples to evaluate

    Returns:
        Evaluation metrics
    """
    print("🔍 Evaluating KAN Residual Model")
    print("=" * 40)

    device = torch.device(device)

    # Load checkpoint
    print(f"Loading KAN model from {model_path}...")
    checkpoint = torch.load(model_path, map_location=device)

    # Recreate models
    neural_model, _ = load_model(checkpoint['neural_model_path'], device)
    neural_model.eval()

    kan_model = KANResidual(neural_model,
                           checkpoint['config']['kan_hidden_dim'],
                           checkpoint['config']['num_grids'])
    kan_model.kan_layers.load_state_dict(checkpoint['kan_model_state_dict'])
    kan_model = kan_model.to(device)
    kan_model.eval()

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

            x, y = vietnam_dataset[i]
            x_batch = torch.tensor(x, device=device).unsqueeze(0)

            pred = kan_model(x_batch)

            predictions.append({
                'temperature': pred['temperature'].item(),
                'ampacity': pred['ampacity'].item(),
                'ampacity_neural': pred['ampacity_neural'].item(),
                'residual': pred['residual'].item()
            })

            targets.append(y)

            conditions.append({
                'T_ambient': x[0],
                'wind_speed': x[1],
                'wind_angle': x[2],
                'solar_irradiance': x[3],
                'current': x[4]
            })

    eval_time = time.time() - start_time
    print(f"⏱️  Evaluation completed in {eval_time:.1f} seconds")
    # Convert to arrays
    pred_temps = np.array([p['temperature'] for p in predictions])
    pred_amps = np.array([p['ampacity'] for p in predictions])
    pred_amps_neural = np.array([p['ampacity_neural'] for p in predictions])
    residuals = np.array([p['residual'] for p in predictions])
    true_temps = np.array([t[0] for t in targets])
    true_amps = np.array([t[1] for t in targets])
    conditions_df = pd.DataFrame(conditions)

    # Calculate metrics
    temp_errors = pred_temps - true_temps
    amp_errors = pred_amps - true_amps
    neural_errors = pred_amps_neural - true_amps

    metrics = {
        'n_samples': n_samples,
        'temperature': {
            'mae': np.mean(np.abs(temp_errors)),
            'rmse': np.sqrt(np.mean(temp_errors**2))
        },
        'ampacity': {
            'mae': np.mean(np.abs(amp_errors)),
            'rmse': np.sqrt(np.mean(amp_errors**2)),
            'mean_error': np.mean(amp_errors),
            'std_error': np.std(amp_errors)
        },
        'neural_baseline': {
            'mae': np.mean(np.abs(neural_errors)),
            'rmse': np.sqrt(np.mean(neural_errors**2))
        },
        'kan_residual': {
            'mean': np.mean(residuals),
            'std': np.std(residuals),
            'min': np.min(residuals),
            'max': np.max(residuals),
            'mae': np.mean(np.abs(residuals))
        },
        'improvement': {
            'mae_reduction': np.mean(np.abs(neural_errors)) - np.mean(np.abs(amp_errors)),
            'percentage': ((np.mean(np.abs(neural_errors)) - np.mean(np.abs(amp_errors))) /
                          np.mean(np.abs(neural_errors)) * 100)
        },
        'conditions': {
            'T_ambient_range': [conditions_df['T_ambient'].min(), conditions_df['T_ambient'].max()],
            'wind_speed_range': [conditions_df['wind_speed'].min(), conditions_df['wind_speed'].max()],
            'solar_range': [conditions_df['solar_irradiance'].min(), conditions_df['solar_irradiance'].max()]
        },
        'evaluation_time': eval_time
    }

    return metrics, predictions, targets, conditions_df


def plot_kan_results(
    metrics: Dict,
    predictions: List,
    targets: List,
    conditions: pd.DataFrame,
    history: Dict = None,
    save_path: str = 'reports/experiment3_kan_residual.png'
):
    """Create comprehensive results plot"""
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))

    pred_temps = np.array([p['temperature'] for p in predictions])
    pred_amps = np.array([p['ampacity'] for p in predictions])
    pred_amps_neural = np.array([p['ampacity_neural'] for p in predictions])
    residuals = np.array([p['residual'] for p in predictions])
    true_temps = np.array([t[0] for t in targets])
    true_amps = np.array([t[1] for t in targets])

    temp_errors = pred_temps - true_temps
    amp_errors = pred_amps - true_amps
    neural_errors = pred_amps_neural - true_amps

    # 1. Temperature scatter
    axes[0, 0].scatter(true_temps, pred_temps, alpha=0.6, s=10, color='blue')
    axes[0, 0].plot([true_temps.min(), true_temps.max()], [true_temps.min(), true_temps.max()],
                   'r--', linewidth=2, label='Perfect prediction')
    axes[0, 0].set_xlabel('True Temperature (°C)')
    axes[0, 0].set_ylabel('Predicted Temperature (°C)')
    axes[0, 0].set_title('Temperature: True vs Predicted')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # 2. Ampacity comparison
    axes[0, 1].scatter(true_amps, pred_amps_neural, alpha=0.6, s=10, color='red', label='Neural Only')
    axes[0, 1].scatter(true_amps, pred_amps, alpha=0.6, s=10, color='green', label='KAN Corrected')
    axes[0, 1].plot([true_amps.min(), true_amps.max()], [true_amps.min(), true_amps.max()],
                   'k--', linewidth=2, label='Perfect')
    axes[0, 1].set_xlabel('True Ampacity (A)')
    axes[0, 1].set_ylabel('Predicted Ampacity (A)')
    axes[0, 1].set_title('Ampacity: Neural vs KAN Corrected')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # 3. Residual distribution
    axes[0, 2].hist(residuals, bins=50, edgecolor='black', alpha=0.7, color='purple')
    axes[0, 2].axvline(x=0, color='red', linestyle='--', linewidth=2)
    axes[0, 2].set_xlabel('KAN Residual (A)')
    axes[0, 2].set_ylabel('Frequency')
    axes[0, 2].set_title('KAN Residual Distribution')
    axes[0, 2].grid(True, alpha=0.3)

    # 4. Performance comparison
    methods = ['Neural Only', 'KAN Corrected']
    maes = [metrics['neural_baseline']['mae'], metrics['ampacity']['mae']]
    colors = ['red', 'green']

    bars = axes[1, 0].bar(methods, maes, color=colors, alpha=0.7)
    axes[1, 0].set_ylabel('MAE (A)')
    axes[1, 0].set_title('Performance Comparison')
    axes[1, 0].grid(True, alpha=0.3, axis='y')

    # Add value labels
    for bar, mae in zip(bars, maes):
        height = bar.get_height()
        axes[1, 0].text(bar.get_x() + bar.get_width()/2., height + 2,
                       f'{mae:.0f}', ha='center', va='bottom')

    # 5. Training history (if available)
    if history is not None:
        axes[1, 1].plot(history['epoch'], history['train_loss'], 'b-', linewidth=2)
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Training Loss (MSE)')
        axes[1, 1].set_title('KAN Training Loss')
        axes[1, 1].grid(True, alpha=0.3)
        axes[1, 1].set_yscale('log')
    else:
        axes[1, 1].text(0.5, 0.5, 'No training\nhistory\navailable',
                       ha='center', va='center', transform=axes[1, 1].transAxes)
        axes[1, 1].set_title('Training History')

    # 6. Summary statistics
    axes[1, 2].axis('off')
    summary_text = f"""Experiment 3: KAN Residual Only
================================

Samples: {metrics['n_samples']:,}

Performance:
• Neural Only MAE: {metrics['neural_baseline']['mae']:.0f}A
• KAN Corrected MAE: {metrics['ampacity']['mae']:.0f}A
• Improvement: {metrics['improvement']['mae_reduction']:.0f}A ({metrics['improvement']['percentage']:.1f}%)

KAN Residual Stats:
• Mean: {metrics['kan_residual']['mean']:.1f}A
• Std: {metrics['kan_residual']['std']:.1f}A
• Range: [{metrics['kan_residual']['min']:.1f}, {metrics['kan_residual']['max']:.1f}]A

Expected: ~390-400A
Actual: {metrics['ampacity']['mae']:.0f}A

Method: Neural ampacity + KAN residual
Neural network frozen, only KAN trained
"""

    axes[1, 2].text(0.05, 0.95, summary_text, transform=axes[1, 2].transAxes,
                    fontsize=10, verticalalignment='top', fontfamily='monospace',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"📈 Results plot saved to {save_path}")


def print_experiment_results(metrics: Dict):
    """Print detailed experiment results"""
    print("\n" + "="*60)
    print("🔬 EXPERIMENT 3: KAN RESIDUAL ONLY")
    print("="*60)

    print(f"Samples evaluated: {metrics['n_samples']:,}")

    print("\n🌡️  Temperature Performance:")
    print(f"   MAE: {metrics['temperature']['mae']:.2f}°C")
    print(f"   RMSE: {metrics['temperature']['rmse']:.2f}°C")

    print("\n⚡ Ampacity Performance:")
    print(f"   Neural Only MAE: {metrics['neural_baseline']['mae']:.0f}A")
    print(f"   KAN Corrected MAE: {metrics['ampacity']['mae']:.0f}A")
    print(f"   Improvement: {metrics['improvement']['mae_reduction']:.0f}A ({metrics['improvement']['percentage']:.1f}%)")

    print("\n🧠 KAN Residual Statistics:")
    print(f"   Mean: {metrics['kan_residual']['mean']:.1f}A")
    print(f"   Std: {metrics['kan_residual']['std']:.1f}A")
    print(f"   Range: [{metrics['kan_residual']['min']:.1f}, {metrics['kan_residual']['max']:.1f}]A")
    print(f"   MAE: {metrics['kan_residual']['mae']:.1f}A")

    print("\n📍 Environmental Conditions:")
    print(f"   Ambient Temperature: {metrics['conditions']['T_ambient_range'][0]:.1f} - {metrics['conditions']['T_ambient_range'][1]:.1f}°C")
    print(f"   Wind Speed: {metrics['conditions']['wind_speed_range'][0]:.1f} - {metrics['conditions']['wind_speed_range'][1]:.1f}m/s")
    print(f"   Solar Irradiance: {metrics['conditions']['solar_range'][0]:.0f} - {metrics['conditions']['solar_range'][1]:.0f}W/m²")

    print("\n🎯 Expected vs Actual:")
    expected_mae = 395  # midpoint of 390-400A range
    actual_mae = metrics['ampacity']['mae']
    diff = actual_mae - expected_mae
    print(f"   Expected MAE: {expected_mae:.0f}A")
    print(f"   Actual MAE: {actual_mae:.0f}A")
    print(f"   Difference: {diff:+.0f}A")

    if abs(diff) < 15:
        print("   ✅ Result matches expectation!")
    elif diff < 0:
        print("   ✅ Better than expected!")
    else:
        print("   ⚠️  Higher than expected.")

    print("\n📋 Method Summary:")
    print("   • Start with neural ampacity (415A baseline)")
    print("   • Add KAN to predict residual corrections")
    print("   • Train only KAN, neural network frozen")
    print("   • No physics blending")

    print("="*60)


def main():
    """Main experiment function"""
    import argparse

    parser = argparse.ArgumentParser(description='Experiment 3: KAN Residual Only')
    parser.add_argument('--model-path', type=str, default='models/best_model.pt',
                       help='Path to trained neural model')
    parser.add_argument('--vietnam-csv', type=str, default='data/mendeley/vietnam_220kv.csv',
                       help='Path to Vietnam dataset')
    parser.add_argument('--kan-hidden-dim', type=int, default=32,
                       help='Hidden dimension for KAN layers')
    parser.add_argument('--num-grids', type=int, default=8,
                       help='Number of B-spline grids')
    parser.add_argument('--batch-size', type=int, default=32,
                       help='Training batch size')
    parser.add_argument('--epochs', type=int, default=100,
                       help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=1e-3,
                       help='Learning rate')
    parser.add_argument('--device', type=str, default='cpu',
                       help='Device for training/evaluation')
    parser.add_argument('--n-samples', type=int, default=None,
                       help='Number of samples to evaluate (None = all)')
    parser.add_argument('--train-only', action='store_true',
                       help='Only train the model')
    parser.add_argument('--eval-only', action='store_true',
                       help='Only evaluate existing model')
    parser.add_argument('--kan-model-path', type=str, default='models/kan_residual.pt',
                       help='Path to save/load KAN model')

    args = parser.parse_args()

    if args.train_only:
        # Only train
        results = train_kan_residual(
            neural_model_path=args.model_path,
            vietnam_csv=args.vietnam_csv,
            kan_hidden_dim=args.kan_hidden_dim,
            num_grids=args.num_grids,
            batch_size=args.batch_size,
            epochs=args.epochs,
            learning_rate=args.lr,
            device=args.device,
            save_path=args.kan_model_path
        )
        print("✅ KAN training completed!")

    elif args.eval_only:
        # Only evaluate
        metrics, predictions, targets, conditions = evaluate_kan_residual(
            model_path=args.kan_model_path,
            vietnam_csv=args.vietnam_csv,
            device=args.device,
            n_samples=args.n_samples
        )
        print_experiment_results(metrics)

        # Load training history if available
        try:
            checkpoint = torch.load(args.kan_model_path)
            history = checkpoint.get('history')
        except:
            history = None

        plot_kan_results(metrics, predictions, targets, conditions, history)

    else:
        # Train and evaluate
        print("🚀 Training KAN residual model...")
        train_results = train_kan_residual(
            neural_model_path=args.model_path,
            vietnam_csv=args.vietnam_csv,
            kan_hidden_dim=args.kan_hidden_dim,
            num_grids=args.num_grids,
            batch_size=args.batch_size,
            epochs=args.epochs,
            learning_rate=args.lr,
            device=args.device,
            save_path=args.kan_model_path
        )

        print("\n🔍 Evaluating trained model...")
        metrics, predictions, targets, conditions = evaluate_kan_residual(
            model_path=args.kan_model_path,
            vietnam_csv=args.vietnam_csv,
            device=args.device,
            n_samples=args.n_samples
        )

        print_experiment_results(metrics)
        plot_kan_results(metrics, predictions, targets, conditions, train_results['history'])

        # Save comprehensive results
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        results_file = f"reports/experiment3_kan_residual_{timestamp}.json"

        final_results = {
            'experiment': 'kan_residual_only',
            'timestamp': timestamp,
            'training': train_results,
            'evaluation': metrics,
            'config': vars(args)
        }

        with open(results_file, 'w') as f:
            json.dump(final_results, f, indent=2, default=str)

        print(f"💾 Detailed results saved to {results_file}")
        print("\n✅ Experiment 3 completed!")


if __name__ == "__main__":
    main()
