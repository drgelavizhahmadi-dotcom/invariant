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
Experiment 1: Physics-Constrained Ampacity Alone

This experiment tests using neural temperature predictions (1.74°C MAE)
with purely physics-based ampacity derivation using calibrated IEEE 738 equations.

No neural ampacity head, no blending - just physics + accurate temperature.

Expected: ~390-400A MAE improvement over direct neural ampacity (415A MAE)

Author: Dr. Gelavizh Ahmadi
Copyright (c) 2026 Invariant Energy GmbH
"""

import torch
import torch.nn as nn
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

from core.model import PhysicsDLR
from core.physics import IEEE738HeatBalance
from scripts.validate_vietnam import VietnamDataset, load_model
from calibration_results.vietnam_params import VIETNAM_LINE_PARAMS, CALIBRATION_METRICS


class PhysicsConstrainedAmpacity(nn.Module):
    """
    Uses neural temperature + calibrated physics for ampacity derivation

    This model:
    1. Uses neural network for accurate temperature prediction
    2. Derives ampacity using calibrated IEEE 738 physics equations
    3. No neural ampacity head - purely physics-constrained
    """

    def __init__(self, neural_model: nn.Module, calibrated_params: Dict):
        """
        Initialize physics-constrained model

        Args:
            neural_model: Trained neural network (temperature predictor)
            calibrated_params: Calibrated physics parameters
        """
        super().__init__()

        self.neural_model = neural_model
        self.calibrated_params = calibrated_params

        # Create physics engine with calibrated parameters
        self.physics = IEEE738HeatBalance(
            conductor_diameter=calibrated_params['diameter'],
            conductor_emissivity=calibrated_params['emissivity'],
            conductor_absorptivity=calibrated_params['absorptivity'],
            resistance_per_meter_25C=calibrated_params['resistance_ac'],
            temp_coeff_resistance=calibrated_params['temp_coefficient'],
            max_conductor_temp=100.0
        )

    def derive_ampacity_from_temperature(
        self,
        T_conductor: torch.Tensor,
        T_ambient: torch.Tensor,
        wind_speed: torch.Tensor,
        solar_irradiance: torch.Tensor,
        wind_angle: torch.Tensor = None,
        T_limit: float = 75.0
    ) -> torch.Tensor:
        """
        Derive ampacity from temperature using calibrated physics

        For each temperature prediction, find the current that would cause
        exactly that temperature under the given conditions.

        Args:
            T_conductor: Predicted conductor temperatures [batch]
            T_ambient: Ambient temperatures [batch]
            wind_speed: Wind speeds [batch]
            solar_irradiance: Solar irradiance [batch]
            wind_angle: Wind angles [batch] (optional)
            T_limit: Not used in this method (kept for compatibility)

        Returns:
            Derived ampacities [batch]
        """
        from scipy.optimize import minimize_scalar
        import numpy as np

        batch_size = T_conductor.shape[0]
        ampacities = []

        # Process each sample in batch
        for i in range(batch_size):
            T_target = T_conductor[i].item()
            T_amb = T_ambient[i].item()
            v_wind = wind_speed[i].item()
            solar = solar_irradiance[i].item()

            def objective(current):
                """Find current where conductor temperature equals T_target"""
                try:
                    # Use standalone physics function for efficiency
                    from core.physics import ieee738_temperature
                    T_pred = ieee738_temperature(
                        current, T_amb, v_wind, solar,
                        diameter=self.physics.D.item(),
                        emissivity=self.physics.epsilon.item(),
                        absorptivity=self.physics.alpha_s.item(),
                        R_20=self.physics.R_ref.item(),
                        alpha=self.physics.alpha_R.item()
                    )
                    return abs(T_pred - T_target)
                except:
                    return 1000.0  # Large penalty for invalid conditions

            # Optimize current to reach target temperature
            try:
                result = minimize_scalar(objective, bounds=(100, 3000), method='bounded')
                if result.success:
                    ampacities.append(result.x)
                else:
                    ampacities.append(1500.0)  # Fallback
            except:
                ampacities.append(1500.0)  # Fallback

        return torch.tensor(ampacities, device=T_conductor.device, dtype=T_conductor.dtype)

    def forward(
        self,
        x: torch.Tensor,
        T_limit: float = 75.0
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass: neural temperature + physics ampacity

        Args:
            x: Input tensor [batch, input_dim]
            T_limit: Temperature limit for ampacity calculation

        Returns:
            Dictionary with temperature and derived ampacity
        """
        # Extract weather conditions
        T_ambient = x[:, 0]
        wind_speed = x[:, 1]
        wind_angle = x[:, 2]
        solar_irradiance = x[:, 3]

        # Neural temperature prediction
        with torch.no_grad():  # Don't backprop through neural model
            temp_pred, _ = self.neural_model(x)  # Ignore neural ampacity

        # Derive ampacity from temperature using physics
        amp_pred = self.derive_ampacity_from_temperature(
            temp_pred.squeeze(),
            T_ambient,
            wind_speed,
            solar_irradiance,
            wind_angle,
            T_limit
        )

        return {
            'temperature': temp_pred.squeeze(),
            'ampacity': amp_pred,
            'method': 'physics_constrained'
        }


def evaluate_physics_constrained(
    model_path: str = 'models/best_model.pt',
    vietnam_csv: str = 'data/mendeley/vietnam_220kv.csv',
    device: str = 'cpu',
    n_samples: int = None
) -> Dict[str, float]:
    """
    Evaluate physics-constrained ampacity model

    Args:
        model_path: Path to trained neural model
        vietnam_csv: Path to Vietnam dataset
        device: Device for evaluation
        n_samples: Number of samples to evaluate (None = all)

    Returns:
        Evaluation metrics
    """
    print("🔬 Experiment 1: Physics-Constrained Ampacity Alone")
    print("=" * 60)

    device = torch.device(device)
    print(f"Using device: {device}")

    # Load neural model
    print(f"Loading neural model from {model_path}...")
    neural_model, normalizer = load_model(model_path, device)
    neural_model.eval()

    # Create physics-constrained model
    print("Creating physics-constrained model...")
    physics_model = PhysicsConstrainedAmpacity(neural_model, VIETNAM_LINE_PARAMS)
    physics_model = physics_model.to(device)
    physics_model.eval()

    # Load Vietnam data
    print(f"Loading Vietnam dataset from {vietnam_csv}...")
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

            # Get data sample
            x, y = vietnam_dataset[i]
            x_batch = torch.tensor(x, device=device).unsqueeze(0)

            # Forward pass
            pred = physics_model(x_batch)

            predictions.append({
                'temperature': pred['temperature'].item(),
                'ampacity': pred['ampacity'].item()
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
    # Convert to arrays for analysis
    pred_temps = np.array([p['temperature'] for p in predictions])
    pred_amps = np.array([p['ampacity'] for p in predictions])
    true_temps = np.array([t[0] for t in targets])
    true_amps = np.array([t[1] for t in targets])
    conditions_df = pd.DataFrame(conditions)

    # Calculate metrics
    temp_errors = pred_temps - true_temps
    amp_errors = pred_amps - true_amps

    metrics = {
        'n_samples': n_samples,
        'temperature': {
            'mae': np.mean(np.abs(temp_errors)),
            'rmse': np.sqrt(np.mean(temp_errors**2)),
            'mean_error': np.mean(temp_errors),
            'std_error': np.std(temp_errors)
        },
        'ampacity': {
            'mae': np.mean(np.abs(amp_errors)),
            'rmse': np.sqrt(np.mean(amp_errors**2)),
            'mean_error': np.mean(amp_errors),
            'std_error': np.std(amp_errors)
        },
        'conditions': {
            'T_ambient_range': [conditions_df['T_ambient'].min(), conditions_df['T_ambient'].max()],
            'wind_speed_range': [conditions_df['wind_speed'].min(), conditions_df['wind_speed'].max()],
            'solar_range': [conditions_df['solar_irradiance'].min(), conditions_df['solar_irradiance'].max()]
        },
        'calibration_info': {
            'rms_residual': CALIBRATION_METRICS['rms_residual'],
            'resistance_factor': VIETNAM_LINE_PARAMS['resistance_factor'],
            'emissivity': VIETNAM_LINE_PARAMS['emissivity'],
            'absorptivity': VIETNAM_LINE_PARAMS['absorptivity']
        },
        'evaluation_time': eval_time
    }

    return metrics, predictions, targets, conditions_df


def plot_physics_constrained_results(
    metrics: Dict,
    predictions: List,
    targets: List,
    conditions: pd.DataFrame,
    save_path: str = 'reports/experiment1_physics_constrained.png'
):
    """Create comprehensive results plot"""
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))

    pred_temps = np.array([p['temperature'] for p in predictions])
    pred_amps = np.array([p['ampacity'] for p in predictions])
    true_temps = np.array([t[0] for t in targets])
    true_amps = np.array([t[1] for t in targets])

    temp_errors = pred_temps - true_temps
    amp_errors = pred_amps - true_amps

    # 1. Temperature scatter
    axes[0, 0].scatter(true_temps, pred_temps, alpha=0.6, s=10, color='blue')
    axes[0, 0].plot([true_temps.min(), true_temps.max()], [true_temps.min(), true_temps.max()],
                   'r--', linewidth=2, label='Perfect prediction')
    axes[0, 0].set_xlabel('True Temperature (°C)')
    axes[0, 0].set_ylabel('Predicted Temperature (°C)')
    axes[0, 0].set_title('Temperature: True vs Predicted')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # 2. Ampacity scatter
    axes[0, 1].scatter(true_amps, pred_amps, alpha=0.6, s=10, color='green')
    axes[0, 1].plot([true_amps.min(), true_amps.max()], [true_amps.min(), true_amps.max()],
                   'r--', linewidth=2, label='Perfect prediction')
    axes[0, 1].set_xlabel('True Ampacity (A)')
    axes[0, 1].set_ylabel('Physics-Constrained Ampacity (A)')
    axes[0, 1].set_title('Ampacity: True vs Physics-Constrained')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # 3. Temperature error distribution
    axes[0, 2].hist(temp_errors, bins=50, edgecolor='black', alpha=0.7, color='lightblue')
    axes[0, 2].axvline(x=0, color='red', linestyle='--', linewidth=2)
    axes[0, 2].set_xlabel('Temperature Error (°C)')
    axes[0, 2].set_ylabel('Frequency')
    axes[0, 2].set_title('Temperature Prediction Errors')
    axes[0, 2].grid(True, alpha=0.3)

    # 4. Ampacity error distribution
    axes[1, 0].hist(amp_errors, bins=50, edgecolor='black', alpha=0.7, color='lightgreen')
    axes[1, 0].axvline(x=0, color='red', linestyle='--', linewidth=2)
    axes[1, 0].set_xlabel('Ampacity Error (A)')
    axes[1, 0].set_ylabel('Frequency')
    axes[1, 0].set_title('Ampacity Prediction Errors')
    axes[1, 0].grid(True, alpha=0.3)

    # 5. Error vs wind speed
    axes[1, 1].scatter(conditions['wind_speed'], amp_errors, alpha=0.6, s=10, color='orange')
    axes[1, 1].axhline(y=0, color='red', linestyle='--', linewidth=2)
    axes[1, 1].set_xlabel('Wind Speed (m/s)')
    axes[1, 1].set_ylabel('Ampacity Error (A)')
    axes[1, 1].set_title('Ampacity Error vs Wind Speed')
    axes[1, 1].grid(True, alpha=0.3)

    # 6. Summary statistics
    axes[1, 2].axis('off')
    summary_text = f"""Experiment 1: Physics-Constrained Ampacity
==========================================

Samples: {metrics['n_samples']:,}

Temperature Performance:
• MAE: {metrics['temperature']['mae']:.2f}°C
• RMSE: {metrics['temperature']['rmse']:.2f}°C

Ampacity Performance:
• MAE: {metrics['ampacity']['mae']:.0f}A
• RMSE: {metrics['ampacity']['rmse']:.0f}A

Calibration Info:
• RMS Residual: {metrics['calibration_info']['rms_residual']:.3f}
• Resistance Factor: {metrics['calibration_info']['resistance_factor']:.3f}
• Emissivity: {metrics['calibration_info']['emissivity']:.3f}
• Absorptivity: {metrics['calibration_info']['absorptivity']:.3f}

Expected: ~390-400A MAE
Actual: {metrics['ampacity']['mae']:.0f}A

Method: Neural temperature (1.74°C MAE) + Calibrated physics
"""

    axes[1, 2].text(0.05, 0.95, summary_text, transform=axes[1, 2].transAxes,
                    fontsize=10, verticalalignment='top', fontfamily='monospace',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"📈 Results plot saved to {save_path}")


def print_experiment_results(metrics: Dict):
    """Print detailed experiment results"""
    print("\n" + "="*70)
    print("🔬 EXPERIMENT 1: PHYSICS-CONSTRAINED AMPACITY ALONE")
    print("="*70)

    print(f"Samples evaluated: {metrics['n_samples']:,}")

    print("\n🌡️  Temperature Performance (Neural Network):")
    print(f"   MAE: {metrics['temperature']['mae']:.2f}°C")
    print(f"   RMSE: {metrics['temperature']['rmse']:.2f}°C")
    print(f"   Mean Error: {metrics['temperature']['mean_error']:.2f}°C")
    print(f"   Std Error: {metrics['temperature']['std_error']:.2f}°C")

    print("\n⚡ Ampacity Performance (Physics-Constrained):")
    print(f"   MAE: {metrics['ampacity']['mae']:.0f}A")
    print(f"   RMSE: {metrics['ampacity']['rmse']:.0f}A")
    print(f"   Mean Error: {metrics['ampacity']['mean_error']:.0f}A")
    print(f"   Std Error: {metrics['ampacity']['std_error']:.0f}A")

    print("\n🔧 Calibration Parameters:")
    print(f"   RMS Residual: {metrics['calibration_info']['rms_residual']:.3f}")
    print(f"   Resistance Factor: {metrics['calibration_info']['resistance_factor']:.3f}")
    print(f"   Emissivity: {metrics['calibration_info']['emissivity']:.3f}")
    print(f"   Absorptivity: {metrics['calibration_info']['absorptivity']:.3f}")

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
    if abs(diff) < 20:
        print("   ✅ Result matches expectation!")
    elif diff < 0:
        print("   ✅ Better than expected!")
    else:
        print("   ⚠️  Higher than expected - investigate.")

    print("\n📋 Method Summary:")
    print("   • Temperature: Neural network prediction (1.74°C MAE)")
    print("   • Ampacity: Derived from temperature using calibrated IEEE 738")
    print("   • No neural ampacity head, no blending, purely physics-constrained")

    print("="*70)


def main():
    """Main experiment function"""
    import argparse

    parser = argparse.ArgumentParser(description='Experiment 1: Physics-Constrained Ampacity')
    parser.add_argument('--model-path', type=str, default='models/best_model.pt',
                       help='Path to trained neural model')
    parser.add_argument('--vietnam-csv', type=str, default='data/mendeley/vietnam_220kv.csv',
                       help='Path to Vietnam dataset')
    parser.add_argument('--device', type=str, default='cpu',
                       help='Device for evaluation')
    parser.add_argument('--n-samples', type=int, default=None,
                       help='Number of samples to evaluate (None = all)')
    parser.add_argument('--save-results', action='store_true', default=True,
                       help='Save results to file')

    args = parser.parse_args()

    # Run evaluation
    metrics, predictions, targets, conditions = evaluate_physics_constrained(
        model_path=args.model_path,
        vietnam_csv=args.vietnam_csv,
        device=args.device,
        n_samples=args.n_samples
    )

    # Print results
    print_experiment_results(metrics)

    # Create plots
    plot_physics_constrained_results(metrics, predictions, targets, conditions)

    # Save results
    if args.save_results:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        results_file = f"reports/experiment1_physics_constrained_{timestamp}.json"

        results_data = {
            'experiment': 'physics_constrained_ampacity',
            'timestamp': timestamp,
            'metrics': metrics,
            'config': vars(args)
        }

        with open(results_file, 'w') as f:
            json.dump(results_data, f, indent=2, default=str)

        print(f"💾 Detailed results saved to {results_file}")

    print("\n✅ Experiment 1 completed!")


if __name__ == "__main__":
    main()
