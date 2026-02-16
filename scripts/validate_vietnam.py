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
Validate model performance on Vietnam 220kV transmission line dataset

This script loads a trained model and evaluates it on real transmission line data
from Vietnam to assess generalization to real-world conditions.

Supports physics consistency loss evaluation.

Author: Dr. Gelavizh Ahmadi
Copyright (c) 2026 Invariant Energy GmbH
"""

import torch
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from typing import Dict, Tuple
import argparse
import sys
import os

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.model import PhysicsDLR
from core.physics import IEEE738HeatBalance


class VietnamDataset:
    """
    Dataset for Vietnam 220kV transmission line data

    Loads real transmission line data from Vietnam for validation.
    Since conductor temperature is not directly measured, we use ambient
    temperature as a proxy and focus on ampacity prediction accuracy.
    """

    def __init__(self, csv_path: str = "data/mendeley/vietnam_220kv.csv"):
        """
        Load Vietnam dataset from CSV

        Args:
            csv_path: Path to Vietnam CSV data
        """
        self.csv_path = csv_path
        self._load_data()

    def _load_data(self):
        """Load and preprocess the data"""
        df = pd.read_csv(self.csv_path)

        # Extract features
        self.T_ambient = df['temp'].values.astype(np.float32)  # Ambient temperature (°C)
        self.wind_speed = df['Wind1'].values.astype(np.float32)  # Wind speed (m/s)
        self.wind_angle = df['WinDir'].values.astype(np.float32)  # Wind direction (degrees)
        self.solar_irradiance = df['GHI'].values.astype(np.float32)  # Solar irradiance (W/m²)
        self.ampacity = df['Ampacity'].values.astype(np.float32)  # Target ampacity (A)

        # For conductor temperature, we'll use ambient + some offset as proxy
        # since actual conductor temperature isn't measured
        self.conductor_temp = self.T_ambient + 10.0  # Rough estimate

        self.n_samples = len(df)

        print(f"Loaded Vietnam dataset: {self.n_samples:,} samples")
        print(f"Temperature range: {self.T_ambient.min():.1f} - {self.T_ambient.max():.1f} °C")
        print(f"Wind speed range: {self.wind_speed.min():.1f} - {self.wind_speed.max():.1f} m/s")
        print(f"Ampacity range: {self.ampacity.min():.0f} - {self.ampacity.max():.0f} A")

    def get_statistics(self) -> Dict[str, Dict[str, float]]:
        """Get dataset statistics"""
        return {
            'T_ambient': {
                'min': self.T_ambient.min(),
                'max': self.T_ambient.max(),
                'mean': self.T_ambient.mean(),
            },
            'wind_speed': {
                'min': self.wind_speed.min(),
                'max': self.wind_speed.max(),
                'mean': self.wind_speed.mean(),
            },
            'wind_angle': {
                'min': self.wind_angle.min(),
                'max': self.wind_angle.max(),
                'mean': self.wind_angle.mean(),
            },
            'solar_irradiance': {
                'min': self.solar_irradiance.min(),
                'max': self.solar_irradiance.max(),
                'mean': self.solar_irradiance.mean(),
            },
            'ampacity': {
                'min': self.ampacity.min(),
                'max': self.ampacity.max(),
                'mean': self.ampacity.mean(),
            },
        }

    def __len__(self) -> int:
        return self.n_samples

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get input and target for validation

        Returns:
            x: Input features [T_ambient, wind_speed, wind_angle, solar, current, resistance]
            y: Targets [conductor_temp, ampacity]
        """
        # Use typical current for DLR calculation (around 1000A for this line)
        current = 1000.0  # A
        resistance = 0.08  # ohm/m (typical for ACSR conductor)

        x = np.array([
            self.T_ambient[idx],
            self.wind_speed[idx],
            self.wind_angle[idx],
            self.solar_irradiance[idx],
            current,
            resistance,
        ])

        y = np.array([
            self.conductor_temp[idx],  # Proxy conductor temperature
            self.ampacity[idx],        # True ampacity
        ])

        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)


def load_model(model_path: str, device: str = 'cpu') -> Tuple[PhysicsDLR, 'InputNormalizer']:
    """Load trained model from checkpoint"""
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)

    # Create model with same config as training
    model = PhysicsDLR()
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()

    # Load normalizer from checkpoint
    from core.data import InputNormalizer
    normalizer = InputNormalizer()
    if 'normalizer' in checkpoint:
        normalizer.mean = np.array(checkpoint['normalizer']['mean'])
        normalizer.std = np.array(checkpoint['normalizer']['std'])
        print("Loaded normalizer from checkpoint")
    else:
        print("Warning: No normalizer found in checkpoint, using defaults")

    return model, normalizer


def validate_on_vietnam(
    model: PhysicsDLR,
    normalizer: 'InputNormalizer',
    vietnam_dataset: VietnamDataset,
    device: str = 'cpu',
    batch_size: int = 256,
    use_consistency_loss: bool = False,
) -> Tuple[Dict[str, float], np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Validate model on Vietnam dataset

    Returns:
        metrics: Dictionary of validation metrics
        pred_temp: Predicted conductor temperatures
        true_temp: True conductor temperatures (proxy)
        pred_amp: Predicted ampacities
        true_amp: True ampacities
    """
    physics = IEEE738HeatBalance().to(device)

    all_pred_temp = []
    all_true_temp = []
    all_pred_amp = []
    all_true_amp = []

    # Process in batches
    for i in range(0, len(vietnam_dataset), batch_size):
        batch_end = min(i + batch_size, len(vietnam_dataset))
        batch_x = []
        batch_y = []

        for j in range(i, batch_end):
            x, y = vietnam_dataset[j]
            batch_x.append(x)
            batch_y.append(y)

        x_batch = torch.stack(batch_x).to(device)
        y_batch = torch.stack(batch_y).to(device)

        # Normalize inputs using the training normalizer
        x_batch_normalized = torch.tensor(
            normalizer.transform(x_batch.cpu().numpy()),
            dtype=torch.float32,
            device=device
        )

        with torch.no_grad():
            if use_consistency_loss:
                # Use physics consistency forward pass
                pred_temp, pred_amp, _ = model.forward_with_physics_consistency(
                    x_batch_normalized, physics, consistency_weight=0.1
                )
            else:
                # Standard forward pass
                pred_temp, pred_amp = model(x_batch_normalized)

        # Store predictions
        all_pred_temp.extend(pred_temp.squeeze().cpu().numpy())
        all_true_temp.extend(y_batch[:, 0].cpu().numpy())
        all_pred_amp.extend(pred_amp.squeeze().cpu().numpy())
        all_true_amp.extend(y_batch[:, 1].cpu().numpy())

    # Convert to numpy arrays
    pred_temp = np.array(all_pred_temp)
    true_temp = np.array(all_true_temp)
    pred_amp = np.array(all_pred_amp)
    true_amp = np.array(all_true_amp)

    # Calculate metrics
    metrics = {
        'temp_mae': mean_absolute_error(true_temp, pred_temp),
        'temp_rmse': np.sqrt(mean_squared_error(true_temp, pred_temp)),
        'temp_r2': r2_score(true_temp, pred_temp),
        'temp_mean_error': np.mean(pred_temp - true_temp),
        'amp_mae': mean_absolute_error(true_amp, pred_amp),
        'amp_rmse': np.sqrt(mean_squared_error(true_amp, pred_amp)),
        'amp_r2': r2_score(true_amp, pred_amp),
        'amp_mean_error': np.mean(pred_amp - true_amp),
    }

    return metrics, pred_temp, true_temp, pred_amp, true_amp


def plot_validation_results(pred_temp, true_temp, pred_amp, true_amp, output_dir: str = "results"):
    """Plot validation results"""
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))

    # Temperature scatter plot
    ax1.scatter(true_temp, pred_temp, alpha=0.6, s=1)
    ax1.plot([true_temp.min(), true_temp.max()], [true_temp.min(), true_temp.max()], 'r--', linewidth=2)
    ax1.set_xlabel('True Conductor Temperature (°C)')
    ax1.set_ylabel('Predicted Conductor Temperature (°C)')
    ax1.set_title('Temperature Prediction')
    ax1.grid(True, alpha=0.3)

    # Temperature error histogram
    temp_errors = pred_temp - true_temp
    ax2.hist(temp_errors, bins=50, alpha=0.7, edgecolor='black')
    ax2.axvline(x=0, color='red', linestyle='--', linewidth=2)
    ax2.set_xlabel('Temperature Error (°C)')
    ax2.set_ylabel('Frequency')
    ax2.set_title('Temperature Prediction Errors')
    ax2.grid(True, alpha=0.3)

    # Ampacity scatter plot
    ax3.scatter(true_amp, pred_amp, alpha=0.6, s=1, color='orange')
    ax3.plot([true_amp.min(), true_amp.max()], [true_amp.min(), true_amp.max()], 'r--', linewidth=2)
    ax3.set_xlabel('True Ampacity (A)')
    ax3.set_ylabel('Predicted Ampacity (A)')
    ax3.set_title('Ampacity Prediction')
    ax3.grid(True, alpha=0.3)

    # Ampacity error histogram
    amp_errors = pred_amp - true_amp
    ax4.hist(amp_errors, bins=50, alpha=0.7, edgecolor='black', color='orange')
    ax4.axvline(x=0, color='red', linestyle='--', linewidth=2)
    ax4.set_xlabel('Ampacity Error (A)')
    ax4.set_ylabel('Frequency')
    ax4.set_title('Ampacity Prediction Errors')
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"{output_dir}/vietnam_validation_plots.png", dpi=150, bbox_inches='tight')
    plt.close()


def main(model_path: str, use_consistency_loss: bool = False, output_dir: str = "results"):
    """Main validation function"""
    print("🔬 Vietnam Transmission Line Validation")
    print("=" * 50)

    # Set device
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load model
    print(f"Loading model from {model_path}")
    model, normalizer = load_model(model_path, device)

    # Load Vietnam dataset
    print("Loading Vietnam dataset...")
    vietnam_dataset = VietnamDataset()

    # Print dataset statistics
    print("\nVietnam Dataset Statistics:")
    stats = vietnam_dataset.get_statistics()
    for key, values in stats.items():
        print(f"  {key}: min={values['min']:.1f}, max={values['max']:.1f}, mean={values['mean']:.1f}")

    # Validate
    print(f"\nRunning validation{' with consistency loss' if use_consistency_loss else ''}...")
    metrics, pred_temp, true_temp, pred_amp, true_amp = validate_on_vietnam(
        model, normalizer, vietnam_dataset, device, use_consistency_loss=use_consistency_loss
    )

    # Print metrics
    print("\nValidation Metrics:")
    print("-" * 40)
    print(f"Temperature MAE:  {metrics['temp_mae']:.2f} °C")
    print(f"Temperature RMSE: {metrics['temp_rmse']:.2f} °C")
    print(f"Temperature R²:   {metrics['temp_r2']:.3f}")
    print(f"Temperature Mean Error: {metrics['temp_mean_error']:.2f} °C")
    print()
    print(f"Ampacity MAE:  {metrics['amp_mae']:.2f} A")
    print(f"Ampacity RMSE: {metrics['amp_rmse']:.2f} A")
    print(f"Ampacity R²:   {metrics['amp_r2']:.3f}")
    print(f"Ampacity Mean Error: {metrics['amp_mean_error']:.2f} A")

    # Plot results
    plot_validation_results(pred_temp, true_temp, pred_amp, true_amp, output_dir)

    # Save detailed results
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    results_df = pd.DataFrame({
        'true_temp': true_temp,
        'pred_temp': pred_temp,
        'temp_error': pred_temp - true_temp,
        'true_amp': true_amp,
        'pred_amp': pred_amp,
        'amp_error': pred_amp - true_amp,
    })

    results_path = f"{output_dir}/vietnam_validation_results.csv"
    results_df.to_csv(results_path, index=False)
    print(f"\n💾 Detailed results saved to {results_path}")

    # Save metrics
    metrics_path = f"{output_dir}/vietnam_validation_metrics.json"
    import json
    # Convert numpy types to Python types for JSON serialization
    serializable_metrics = {k: float(v) for k, v in metrics.items()}
    with open(metrics_path, 'w') as f:
        json.dump(serializable_metrics, f, indent=2)
    print(f"💾 Metrics saved to {metrics_path}")

    # Summary
    amp_performance = "excellent" if metrics['amp_r2'] > 0.8 else "good" if metrics['amp_r2'] > 0.6 else "moderate" if metrics['amp_r2'] > 0.4 else "poor"
    temp_performance = "excellent" if abs(metrics['temp_mae']) < 5 else "good" if abs(metrics['temp_mae']) < 10 else "moderate" if abs(metrics['temp_mae']) < 15 else "poor"

    print("\n🎯 Validation Summary:")
    print(f"   Ampacity performance: {amp_performance} (R² = {metrics['amp_r2']:.3f})")
    print(f"   Temperature performance: {temp_performance} (MAE = {metrics['temp_mae']:.1f}°C)")
    print(f"   Consistency loss used: {use_consistency_loss}")

    print("\n✅ Validation Complete!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Validate model on Vietnam 220kV transmission line dataset")
    parser.add_argument("--model-path", type=str, required=True, help="Path to the trained model checkpoint")
    parser.add_argument("--consistency-loss", action="store_true", help="Use physics consistency loss during validation")
    parser.add_argument("--output-dir", type=str, default="results", help="Directory to save results")

    args = parser.parse_args()
    main(
        model_path=args.model_path,
        use_consistency_loss=args.consistency_loss,
        output_dir=args.output_dir
    )
