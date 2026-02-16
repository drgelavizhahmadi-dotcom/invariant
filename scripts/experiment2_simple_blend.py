"""
Invariant-PIKAN: Adversarially-Robust Physics-Informed Neural Networks for Dynamic Line Rating
Copyright (C) 2025 Gelavizh Ahmadi / Invariant Research

This software is licensed under the Business Source License 1.1 (BSL 1.1).
Commercial production use requires a separate license agreement.
See LICENSE.txt for full terms.

"""

#!/usr/bin/env python3
"""
Experiment 2: Simple Blend (No KAN)

This experiment tests simple fixed-weight blending:
- 60% physics-constrained ampacity (from neural temperature)
- 40% neural ampacity prediction
- Fixed weights (not learned)
- No KAN residual network

Expected: ~385A MAE (similar to hybrid ensemble)

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


class SimpleBlend(nn.Module):
    """
    Simple fixed-weight blending of physics and neural predictions

    This model combines:
    - Physics-constrained ampacity (from neural temperature)
    - Neural ampacity prediction
    - Fixed blending weights: 60% physics, 40% neural
    - No learning, no KAN residual
    """

    def __init__(self, neural_model: nn.Module, calibrated_params: Dict, physics_weight: float = 0.6):
        """
        Initialize simple blend model

        Args:
            neural_model: Trained neural network
            calibrated_params: Calibrated physics parameters
            physics_weight: Fixed weight for physics component (0.6 = 60%)
        """
        super().__init__()

        self.neural_model = neural_model
        self.calibrated_params = calibrated_params
        self.physics_weight = physics_weight
        self.neural_weight = 1.0 - physics_weight

        # Create physics engine
        self.physics = IEEE738HeatBalance(
            conductor_diameter=calibrated_params['diameter'],
            conductor_emissivity=calibrated_params['emissivity'],
            conductor_absorptivity=calibrated_params['absorptivity'],
            resistance_per_meter_25C=calibrated_params['resistance_ac'],
            temp_coeff_resistance=calibrated_params['temp_coefficient'],
            max_conductor_temp=100.0
        )

    def derive_physics_ampacity(
        self,
        T_conductor: torch.Tensor,
        T_ambient: torch.Tensor,
        wind_speed: torch.Tensor,
        solar_irradiance: torch.Tensor,
        wind_angle: torch.Tensor = None
    ) -> torch.Tensor:
        """
        Derive ampacity from temperature using calibrated physics
        """
        from scipy.optimize import minimize_scalar
        import numpy as np

        batch_size = T_conductor.shape[0]
        ampacities = []

        for i in range(batch_size):
            T_target = T_conductor[i].item()
            T_amb = T_ambient[i].item()
            v_wind = wind_speed[i].item()
            solar = solar_irradiance[i].item()

            def objective(current):
                try:
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
                    return 1000.0

            try:
                result = minimize_scalar(objective, bounds=(100, 3000), method='bounded')
                if result.success:
                    ampacities.append(result.x)
                else:
                    ampacities.append(1500.0)
            except:
                ampacities.append(1500.0)

        return torch.tensor(ampacities, device=T_conductor.device, dtype=T_conductor.dtype)

    def forward(
        self,
        x: torch.Tensor,
        return_components: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass with simple blending

        Args:
            x: Input tensor [batch, input_dim]
            return_components: Whether to return individual predictions

        Returns:
            Dictionary with blended predictions
        """
        # Extract weather conditions
        T_ambient = x[:, 0]
        wind_speed = x[:, 1]
        wind_angle = x[:, 2]
        solar_irradiance = x[:, 3]

        # Neural predictions
        with torch.no_grad():
            neural_temp, neural_amp = self.neural_model(x)

        # Physics-constrained ampacity
        physics_amp = self.derive_physics_ampacity(
            neural_temp.squeeze(),
            T_ambient,
            wind_speed,
            solar_irradiance,
            wind_angle
        )

        # Simple fixed-weight blending
        blended_amp = (self.physics_weight * physics_amp +
                      self.neural_weight * neural_amp.squeeze())

        result = {
            'temperature': neural_temp.squeeze(),
            'ampacity': blended_amp,
            'physics_weight': torch.tensor(self.physics_weight),
            'neural_weight': torch.tensor(self.neural_weight)
        }

        if return_components:
            result.update({
                'ampacity_physics': physics_amp,
                'ampacity_neural': neural_amp.squeeze()
            })

        return result


def evaluate_simple_blend(
    model_path: str = 'models/best_model.pt',
    vietnam_csv: str = 'data/mendeley/vietnam_220kv.csv',
    physics_weight: float = 0.6,
    device: str = 'cpu',
    n_samples: int = None
) -> Dict[str, float]:
    """
    Evaluate simple blend model

    Args:
        model_path: Path to trained neural model
        vietnam_csv: Path to Vietnam dataset
        physics_weight: Fixed weight for physics component
        device: Device for evaluation
        n_samples: Number of samples to evaluate

    Returns:
        Evaluation metrics
    """
    print("🔬 Experiment 2: Simple Blend (No KAN)")
    print("=" * 50)
    print(f"Physics weight: {physics_weight:.1f} ({physics_weight*100:.0f}%)")
    print(f"Neural weight: {1-physics_weight:.1f} ({(1-physics_weight)*100:.0f}%)")

    device = torch.device(device)

    # Load neural model
    print(f"Loading neural model from {model_path}...")
    neural_model, normalizer = load_model(model_path, device)
    neural_model.eval()

    # Create simple blend model
    print("Creating simple blend model...")
    blend_model = SimpleBlend(neural_model, VIETNAM_LINE_PARAMS, physics_weight)
    blend_model = blend_model.to(device)
    blend_model.eval()

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

            x, y = vietnam_dataset[i]
            x_batch = torch.tensor(x, device=device).unsqueeze(0)

            pred = blend_model(x_batch, return_components=True)

            predictions.append({
                'temperature': pred['temperature'].item(),
                'ampacity': pred['ampacity'].item(),
                'ampacity_physics': pred['ampacity_physics'].item(),
                'ampacity_neural': pred['ampacity_neural'].item()
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
    pred_amps_physics = np.array([p['ampacity_physics'] for p in predictions])
    pred_amps_neural = np.array([p['ampacity_neural'] for p in predictions])
    true_temps = np.array([t[0] for t in targets])
    true_amps = np.array([t[1] for t in targets])
    conditions_df = pd.DataFrame(conditions)

    # Calculate metrics
    temp_errors = pred_temps - true_temps
    amp_errors = pred_amps - true_amps
    physics_errors = pred_amps_physics - true_amps
    neural_errors = pred_amps_neural - true_amps

    metrics = {
        'n_samples': n_samples,
        'blend_weights': {
            'physics': physics_weight,
            'neural': 1 - physics_weight
        },
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
        'component_performance': {
            'physics_only': {
                'mae': np.mean(np.abs(physics_errors)),
                'rmse': np.sqrt(np.mean(physics_errors**2))
            },
            'neural_only': {
                'mae': np.mean(np.abs(neural_errors)),
                'rmse': np.sqrt(np.mean(neural_errors**2))
            }
        },
        'conditions': {
            'T_ambient_range': [conditions_df['T_ambient'].min(), conditions_df['T_ambient'].max()],
            'wind_speed_range': [conditions_df['wind_speed'].min(), conditions_df['wind_speed'].max()],
            'solar_range': [conditions_df['solar_irradiance'].min(), conditions_df['solar_irradiance'].max()]
        },
        'calibration_info': {
            'rms_residual': CALIBRATION_METRICS['rms_residual'],
            'resistance_factor': VIETNAM_LINE_PARAMS['resistance_factor']
        },
        'evaluation_time': eval_time
    }

    return metrics, predictions, targets, conditions_df


def plot_simple_blend_results(
    metrics: Dict,
    predictions: List,
    targets: List,
    conditions: pd.DataFrame,
    save_path: str = 'reports/experiment2_simple_blend.png'
):
    """Create comprehensive results plot"""
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))

    pred_temps = np.array([p['temperature'] for p in predictions])
    pred_amps = np.array([p['ampacity'] for p in predictions])
    pred_amps_physics = np.array([p['ampacity_physics'] for p in predictions])
    pred_amps_neural = np.array([p['ampacity_neural'] for p in predictions])
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
    axes[0, 1].set_ylabel('Blended Ampacity (A)')
    axes[0, 1].set_title('Ampacity: True vs Simple Blend')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # 3. Component comparison
    physics_mae = metrics['component_performance']['physics_only']['mae']
    neural_mae = metrics['component_performance']['neural_only']['mae']
    blend_mae = metrics['ampacity']['mae']

    components = ['Physics Only', 'Neural Only', 'Simple Blend']
    maes = [physics_mae, neural_mae, blend_mae]
    colors = ['red', 'blue', 'green']

    bars = axes[0, 2].bar(components, maes, color=colors, alpha=0.7)
    axes[0, 2].set_ylabel('MAE (A)')
    axes[0, 2].set_title('Component Performance Comparison')
    axes[0, 2].grid(True, alpha=0.3, axis='y')

    # Add value labels on bars
    for bar, mae in zip(bars, maes):
        height = bar.get_height()
        axes[0, 2].text(bar.get_x() + bar.get_width()/2., height + 5,
                       f'{mae:.0f}', ha='center', va='bottom')

    # 4. Ampacity error distribution
    axes[1, 0].hist(amp_errors, bins=50, edgecolor='black', alpha=0.7, color='lightgreen')
    axes[1, 0].axvline(x=0, color='red', linestyle='--', linewidth=2)
    axes[1, 0].set_xlabel('Ampacity Error (A)')
    axes[1, 0].set_ylabel('Frequency')
    axes[1, 0].set_title('Blend Ampacity Errors')
    axes[1, 0].grid(True, alpha=0.3)

    # 5. Error vs wind speed
    axes[1, 1].scatter(conditions['wind_speed'], amp_errors, alpha=0.6, s=10, color='orange')
    axes[1, 1].axhline(y=0, color='red', linestyle='--', linewidth=2)
    axes[1, 1].set_xlabel('Wind Speed (m/s)')
    axes[1, 1].set_ylabel('Ampacity Error (A)')
    axes[1, 1].set_title('Blend Error vs Wind Speed')
    axes[1, 1].grid(True, alpha=0.3)

    # 6. Summary statistics
    axes[1, 2].axis('off')
    summary_text = f"""Experiment 2: Simple Blend (No KAN)
================================

Samples: {metrics['n_samples']:,}

Blend Weights:
• Physics: {metrics['blend_weights']['physics']:.1f} ({metrics['blend_weights']['physics']*100:.0f}%)
• Neural: {metrics['blend_weights']['neural']:.1f} ({metrics['blend_weights']['neural']*100:.0f}%)

Performance:
• Temperature MAE: {metrics['temperature']['mae']:.2f}°C
• Blend Ampacity MAE: {metrics['ampacity']['mae']:.0f}A
• Physics Only MAE: {metrics['component_performance']['physics_only']['mae']:.0f}A
• Neural Only MAE: {metrics['component_performance']['neural_only']['mae']:.0f}A

Expected: ~385A
Actual: {metrics['ampacity']['mae']:.0f}A

Method: Fixed-weight blending
No learning, no KAN residual
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
    print("🔬 EXPERIMENT 2: SIMPLE BLEND (NO KAN)")
    print("="*60)

    print(f"Samples evaluated: {metrics['n_samples']:,}")

    print("\n⚖️  Blend Weights:")
    print(f"   Physics: {metrics['blend_weights']['physics']:.1f} ({metrics['blend_weights']['physics']*100:.0f}%)")
    print(f"   Neural: {metrics['blend_weights']['neural']:.1f} ({metrics['blend_weights']['neural']*100:.0f}%)")

    print("\n🌡️  Temperature Performance:")
    print(f"   MAE: {metrics['temperature']['mae']:.2f}°C")
    print(f"   RMSE: {metrics['temperature']['rmse']:.2f}°C")

    print("\n⚡ Ampacity Performance:")
    print(f"   Blend MAE: {metrics['ampacity']['mae']:.0f}A")
    print(f"   Blend RMSE: {metrics['ampacity']['rmse']:.0f}A")
    print(f"   Physics Only MAE: {metrics['component_performance']['physics_only']['mae']:.0f}A")
    print(f"   Neural Only MAE: {metrics['component_performance']['neural_only']['mae']:.0f}A")

    print("\n🔧 Calibration Info:")
    print(f"   RMS Residual: {metrics['calibration_info']['rms_residual']:.3f}")
    print(f"   Resistance Factor: {metrics['calibration_info']['resistance_factor']:.3f}")

    print("\n📍 Environmental Conditions:")
    print(f"   Ambient Temperature: {metrics['conditions']['T_ambient_range'][0]:.1f} - {metrics['conditions']['T_ambient_range'][1]:.1f}°C")
    print(f"   Wind Speed: {metrics['conditions']['wind_speed_range'][0]:.1f} - {metrics['conditions']['wind_speed_range'][1]:.1f}m/s")
    print(f"   Solar Irradiance: {metrics['conditions']['solar_range'][0]:.0f} - {metrics['conditions']['solar_range'][1]:.0f}W/m²")

    print("\n🎯 Expected vs Actual:")
    expected_mae = 385  # expected value
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

    # Component comparison
    physics_mae = metrics['component_performance']['physics_only']['mae']
    neural_mae = metrics['component_performance']['neural_only']['mae']
    blend_mae = metrics['ampacity']['mae']

    print("\n📊 Component Analysis:")
    print(f"   Physics Only: {physics_mae:.0f}A")
    print(f"   Neural Only: {neural_mae:.0f}A")
    print(f"   Simple Blend: {blend_mae:.0f}A")

    physics_improvement = (neural_mae - physics_mae) / neural_mae * 100
    blend_improvement = (neural_mae - blend_mae) / neural_mae * 100

    print(f"   Physics improvement over neural: {physics_improvement:+.1f}%")
    print(f"   Blend improvement over neural: {blend_improvement:+.1f}%")

    print("\n📋 Method Summary:")
    print("   • Fixed weights: 60% physics, 40% neural")
    print("   • No learning of blend weights")
    print("   • No KAN residual network")
    print("   • Simple linear combination")

    print("="*60)


def main():
    """Main experiment function"""
    import argparse

    parser = argparse.ArgumentParser(description='Experiment 2: Simple Blend')
    parser.add_argument('--model-path', type=str, default='models/best_model.pt',
                       help='Path to trained neural model')
    parser.add_argument('--vietnam-csv', type=str, default='data/mendeley/vietnam_220kv.csv',
                       help='Path to Vietnam dataset')
    parser.add_argument('--physics-weight', type=float, default=0.6,
                       help='Fixed weight for physics component (0.0-1.0)')
    parser.add_argument('--device', type=str, default='cpu',
                       help='Device for evaluation')
    parser.add_argument('--n-samples', type=int, default=None,
                       help='Number of samples to evaluate (None = all)')
    parser.add_argument('--save-results', action='store_true', default=True,
                       help='Save results to file')

    args = parser.parse_args()

    # Run evaluation
    metrics, predictions, targets, conditions = evaluate_simple_blend(
        model_path=args.model_path,
        vietnam_csv=args.vietnam_csv,
        physics_weight=args.physics_weight,
        device=args.device,
        n_samples=args.n_samples
    )

    # Print results
    print_experiment_results(metrics)

    # Create plots
    plot_simple_blend_results(metrics, predictions, targets, conditions)

    # Save results
    if args.save_results:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        results_file = f"reports/experiment2_simple_blend_{timestamp}.json"

        results_data = {
            'experiment': 'simple_blend',
            'timestamp': timestamp,
            'metrics': metrics,
            'config': vars(args)
        }

        with open(results_file, 'w') as f:
            json.dump(results_data, f, indent=2, default=str)

        print(f"💾 Detailed results saved to {results_file}")

    print("\n✅ Experiment 2 completed!")


if __name__ == "__main__":
    main()
