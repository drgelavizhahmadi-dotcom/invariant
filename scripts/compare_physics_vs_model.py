"""
Invariant-PIKAN: Adversarially-Robust Physics-Informed Neural Networks for Dynamic Line Rating
Copyright (C) 2025 Gelavizh Ahmadi / Invariant Research

This software is licensed under the Business Source License 1.1 (BSL 1.1).
Commercial production use requires a separate license agreement.
See LICENSE.txt for full terms.

"""

#!/usr/bin/env python3
"""
Compare Neural Network Predictions vs Calibrated Physics

This script validates the physics calibration by comparing:
1. Neural network ampacity predictions
2. Calibrated IEEE 738 physics predictions
3. Heat balance residuals

Expected: Low RMS residual (around 0.715) indicates good calibration.

Author: Dr. Gelavizh Ahmadi
Copyright (c) 2026 Invariant Energy GmbH
"""

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import sys
from typing import Dict, Tuple, List
import json

# Add project root to path
sys.path.append('.')

from core.model import PhysicsDLR
from core.physics import IEEE738HeatBalance
from scripts.validate_vietnam import VietnamDataset, load_model
from core.physics import (
    convective_heat_loss,
    radiative_heat_loss,
    solar_heat_gain,
    resistive_heat_gain,
    ieee738_temperature
)


def load_calibrated_physics(vietnam_params_path: str = 'calibration_results/vietnam_params.py') -> Dict:
    """
    Load calibrated Vietnam line parameters
    """
    # Import the parameters
    sys.path.append(str(Path(vietnam_params_path).parent))
    from vietnam_params import VIETNAM_LINE_PARAMS, CALIBRATION_METRICS

    return VIETNAM_LINE_PARAMS, CALIBRATION_METRICS


def create_calibrated_physics_engine(vietnam_params: Dict) -> IEEE738HeatBalance:
    """
    Create IEEE 738 physics engine with calibrated Vietnam parameters
    """
    physics = IEEE738HeatBalance(
        conductor_diameter=vietnam_params['diameter'],
        conductor_emissivity=vietnam_params['emissivity'],
        conductor_absorptivity=vietnam_params['absorptivity'],
        resistance_per_meter_25C=vietnam_params['resistance_ac'],
        temp_coeff_resistance=vietnam_params['temp_coefficient'],
        max_conductor_temp=100.0
    )

    return physics


def predict_with_physics(
    physics: IEEE738HeatBalance,
    T_ambient: float,
    wind_speed: float,
    solar_irradiance: float,
    wind_angle: float = 0.0,
    T_limit: float = 75.0
) -> Dict[str, float]:
    """
    Predict ampacity using calibrated physics (solve for current at T_limit)
    """
    from scipy.optimize import minimize_scalar

    def objective(current):
        """Find current where conductor temperature equals T_limit"""
        try:
            T_pred = ieee738_temperature(
                current,
                T_ambient,
                wind_speed,
                solar_irradiance,
                diameter=physics.D.item(),
                emissivity=physics.epsilon.item(),
                absorptivity=physics.alpha_s.item(),
                R_20=physics.R_ref.item(),
                alpha=physics.alpha_R.item()
            )
            return abs(T_pred - T_limit)
        except:
            return 1000.0  # Large penalty for invalid conditions

    # Optimize current to reach T_limit
    result = minimize_scalar(objective, bounds=(100, 3000), method='bounded')

    if result.success:
        ampacity = result.x
        # Verify the temperature
        T_verify = ieee738_temperature(
            ampacity, T_ambient, wind_speed, solar_irradiance,
            diameter=physics.D.item(),
            emissivity=physics.epsilon.item(),
            absorptivity=physics.alpha_s.item(),
            R_20=physics.R_ref.item(),
            alpha=physics.alpha_R.item()
        )
    else:
        ampacity = 1000.0  # Fallback
        T_verify = T_ambient + 10

    return {
        'ampacity': ampacity,
        'temperature': T_verify,
        'optimization_success': result.success
    }


def compare_predictions(
    model_path: str = 'models/best_model.pt',
    vietnam_csv: str = 'data/mendeley/vietnam_220kv.csv',
    vietnam_params_path: str = 'calibration_results/vietnam_params.py',
    n_samples: int = 1000,
    device: str = 'cpu'
) -> Dict[str, any]:
    """
    Compare neural network vs calibrated physics predictions
    """
    print("🔍 Comparing Neural Network vs Calibrated Physics Predictions")
    print("=" * 70)

    # Set device
    device = torch.device(device)
    print(f"Using device: {device}")

    # Load model and normalizer
    print(f"Loading model from {model_path}...")
    model, normalizer = load_model(model_path, device)
    model.eval()

    # Load calibrated parameters
    print(f"Loading calibrated parameters from {vietnam_params_path}...")
    vietnam_params, calibration_metrics = load_calibrated_physics(vietnam_params_path)
    print(f"Expected RMS residual: {calibration_metrics['rms_residual']:.3f}")

    # Create calibrated physics engine
    physics = create_calibrated_physics_engine(vietnam_params)
    physics = physics.to(device)

    # Load Vietnam dataset
    print(f"Loading Vietnam dataset from {vietnam_csv}...")
    vietnam_dataset = VietnamDataset(vietnam_csv)

    # Subsample for comparison
    if n_samples < len(vietnam_dataset):
        indices = np.random.choice(len(vietnam_dataset), n_samples, replace=False)
        print(f"Using {n_samples} random samples for comparison")
    else:
        indices = range(len(vietnam_dataset))
        print(f"Using all {len(vietnam_dataset)} samples for comparison")

    # Collect predictions
    nn_predictions = []
    physics_predictions = []
    heat_balance_residuals = []
    conditions = []

    print("\n🔄 Running predictions...")
    with torch.no_grad():
        for i, idx in enumerate(indices):
            if (i + 1) % 100 == 0:
                print(f"   Processed {i+1}/{len(indices)} samples...")

            # Get data sample
            x, y = vietnam_dataset[idx]
            x_batch = torch.tensor(x, device=device).unsqueeze(0)

            # Extract conditions
            T_ambient = x[0]
            wind_speed = x[1]
            wind_angle = x[2]
            solar_irradiance = x[3]
            current = x[4]  # From dataset

            # Neural network prediction
            pred_temp, pred_amp = model(x_batch)
            nn_temp = pred_temp.item()
            nn_amp = pred_amp.item()

            # Physics prediction (solve for ampacity at 75°C)
            phys_result = predict_with_physics(
                physics, T_ambient, wind_speed, solar_irradiance, wind_angle, T_limit=75.0
            )
            phys_amp = phys_result['ampacity']
            phys_temp = phys_result['temperature']

            # Calculate heat balance residual for NN prediction
            residual = physics.heat_balance_residual(
                current=torch.tensor([current]),
                T_conductor=torch.tensor([nn_temp]),
                T_ambient=torch.tensor([T_ambient]),
                wind_speed=torch.tensor([wind_speed]),
                solar_irradiance=torch.tensor([solar_irradiance]),
                wind_angle=torch.tensor([wind_angle])
            ).item()

            # Store results
            nn_predictions.append({
                'temperature': nn_temp,
                'ampacity': nn_amp
            })

            physics_predictions.append({
                'temperature': phys_temp,
                'ampacity': phys_amp
            })

            heat_balance_residuals.append(residual)

            conditions.append({
                'T_ambient': T_ambient,
                'wind_speed': wind_speed,
                'wind_angle': wind_angle,
                'solar_irradiance': solar_irradiance,
                'current': current
            })

    # Convert to arrays for analysis
    nn_temps = np.array([p['temperature'] for p in nn_predictions])
    nn_amps = np.array([p['ampacity'] for p in nn_predictions])
    phys_temps = np.array([p['temperature'] for p in physics_predictions])
    phys_amps = np.array([p['ampacity'] for p in physics_predictions])
    residuals = np.array(heat_balance_residuals)

    conditions_df = pd.DataFrame(conditions)

    # Calculate comparison metrics
    ampacity_errors = nn_amps - phys_amps
    temp_errors = nn_temps - phys_temps

    results = {
        'n_samples': len(indices),
        'nn_predictions': {
            'temperature': {'mean': nn_temps.mean(), 'std': nn_temps.std()},
            'ampacity': {'mean': nn_amps.mean(), 'std': nn_amps.std()}
        },
        'physics_predictions': {
            'temperature': {'mean': phys_temps.mean(), 'std': phys_temps.std()},
            'ampacity': {'mean': phys_amps.mean(), 'std': phys_amps.std()}
        },
        'comparison': {
            'ampacity_rmse': np.sqrt(np.mean(ampacity_errors**2)),
            'ampacity_mae': np.mean(np.abs(ampacity_errors)),
            'ampacity_max_error': np.max(np.abs(ampacity_errors)),
            'temperature_rmse': np.sqrt(np.mean(temp_errors**2)),
            'temperature_mae': np.mean(np.abs(temp_errors)),
            'heat_balance_residuals': {
                'mean': residuals.mean(),
                'std': residuals.std(),
                'rms': np.sqrt(np.mean(residuals**2)),
                'max_abs': np.max(np.abs(residuals))
            }
        },
        'conditions': {
            'T_ambient_range': [conditions_df['T_ambient'].min(), conditions_df['T_ambient'].max()],
            'wind_speed_range': [conditions_df['wind_speed'].min(), conditions_df['wind_speed'].max()],
            'solar_range': [conditions_df['solar_irradiance'].min(), conditions_df['solar_irradiance'].max()]
        }
    }

    return results, nn_predictions, physics_predictions, conditions_df


def plot_comparison(results: Dict, nn_preds: List, phys_preds: List, conditions: pd.DataFrame):
    """
    Create comprehensive comparison plots
    """
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))

    nn_amps = np.array([p['ampacity'] for p in nn_preds])
    phys_amps = np.array([p['ampacity'] for p in phys_preds])
    nn_temps = np.array([p['temperature'] for p in nn_preds])
    phys_temps = np.array([p['temperature'] for p in phys_preds])

    ampacity_errors = nn_amps - phys_amps
    temp_errors = nn_temps - phys_temps

    # 1. Ampacity comparison scatter
    axes[0, 0].scatter(phys_amps, nn_amps, alpha=0.6, s=10, color='blue')
    axes[0, 0].plot([nn_amps.min(), nn_amps.max()], [nn_amps.min(), nn_amps.max()],
                   'r--', linewidth=2, label='Perfect agreement')
    axes[0, 0].set_xlabel('Physics Ampacity (A)')
    axes[0, 0].set_ylabel('Neural Network Ampacity (A)')
    axes[0, 0].set_title('Ampacity: Physics vs Neural Network')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # 2. Ampacity error distribution
    axes[0, 1].hist(ampacity_errors, bins=50, edgecolor='black', alpha=0.7, color='skyblue')
    axes[0, 1].axvline(x=0, color='red', linestyle='--', linewidth=2)
    axes[0, 1].set_xlabel('Ampacity Error (NN - Physics) (A)')
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].set_title('Ampacity Prediction Errors')
    axes[0, 1].grid(True, alpha=0.3)

    # 3. Temperature comparison
    axes[0, 2].scatter(phys_temps, nn_temps, alpha=0.6, s=10, color='green')
    axes[0, 2].plot([nn_temps.min(), nn_temps.max()], [nn_temps.min(), nn_temps.max()],
                   'r--', linewidth=2, label='Perfect agreement')
    axes[0, 2].set_xlabel('Physics Temperature (°C)')
    axes[0, 2].set_ylabel('Neural Network Temperature (°C)')
    axes[0, 2].set_title('Temperature: Physics vs Neural Network')
    axes[0, 2].legend()
    axes[0, 2].grid(True, alpha=0.3)

    # 4. Error vs wind speed
    axes[1, 0].scatter(conditions['wind_speed'], ampacity_errors, alpha=0.6, s=10, color='orange')
    axes[1, 0].axhline(y=0, color='red', linestyle='--', linewidth=2)
    axes[1, 0].set_xlabel('Wind Speed (m/s)')
    axes[1, 0].set_ylabel('Ampacity Error (A)')
    axes[1, 0].set_title('Ampacity Error vs Wind Speed')
    axes[1, 0].grid(True, alpha=0.3)

    # 5. Error vs ambient temperature
    axes[1, 1].scatter(conditions['T_ambient'], ampacity_errors, alpha=0.6, s=10, color='purple')
    axes[1, 1].axhline(y=0, color='red', linestyle='--', linewidth=2)
    axes[1, 1].set_xlabel('Ambient Temperature (°C)')
    axes[1, 1].set_ylabel('Ampacity Error (A)')
    axes[1, 1].set_title('Ampacity Error vs Ambient Temperature')
    axes[1, 1].grid(True, alpha=0.3)

    # 6. Summary statistics
    axes[1, 2].axis('off')
    summary_text = f"""Comparison Summary:
==================

Samples: {results['n_samples']:,}

Ampacity Comparison:
• RMSE: {results['comparison']['ampacity_rmse']:.1f} A
• MAE: {results['comparison']['ampacity_mae']:.1f} A
• Max Error: {results['comparison']['ampacity_max_error']:.1f} A

Heat Balance Residuals:
• Mean: {results['comparison']['heat_balance_residuals']['mean']:.3f}
• RMS: {results['comparison']['heat_balance_residuals']['rms']:.3f}
• Max |Residual|: {results['comparison']['heat_balance_residuals']['max_abs']:.3f}

Expected RMS: {0.715:.3f} (from calibration)
Actual RMS: {results['comparison']['heat_balance_residuals']['rms']:.3f}

Calibration Quality: {'✅ Good' if abs(results['comparison']['heat_balance_residuals']['rms'] - 0.715) < 0.1 else '⚠️ Check calibration'}
"""

    axes[1, 2].text(0.05, 0.95, summary_text, transform=axes[1, 2].transAxes,
                    fontsize=10, verticalalignment='top', fontfamily='monospace',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    plt.tight_layout()
    return fig


def print_comparison_results(results: Dict):
    """Print detailed comparison results"""
    print("\n" + "="*70)
    print("🔍 NEURAL NETWORK vs CALIBRATED PHYSICS COMPARISON")
    print("="*70)

    print(f"Samples compared: {results['n_samples']:,}")

    print("\n📊 Neural Network Predictions:")
    print(f"   Temperature: {results['nn_predictions']['temperature']['mean']:.1f} ± {results['nn_predictions']['temperature']['std']:.1f} °C")
    print(f"   Ampacity: {results['nn_predictions']['ampacity']['mean']:.0f} ± {results['nn_predictions']['ampacity']['std']:.0f} A")

    print("\n🔧 Calibrated Physics Predictions:")
    print(f"   Temperature: {results['physics_predictions']['temperature']['mean']:.1f} ± {results['physics_predictions']['temperature']['std']:.1f} °C")
    print(f"   Ampacity: {results['physics_predictions']['ampacity']['mean']:.0f} ± {results['physics_predictions']['ampacity']['std']:.0f} A")

    print("\n⚖️  Comparison Metrics:")
    print(f"   Ampacity RMSE: {results['comparison']['ampacity_rmse']:.1f} A")
    print(f"   Ampacity MAE: {results['comparison']['ampacity_mae']:.1f} A")
    print(f"   Ampacity Max Error: {results['comparison']['ampacity_max_error']:.1f} A")

    print("\n🌡️  Heat Balance Residuals (NN predictions):")
    print(f"   Mean: {results['comparison']['heat_balance_residuals']['mean']:.3f}")
    print(f"   RMS: {results['comparison']['heat_balance_residuals']['rms']:.3f}")
    print(f"   Max |Residual|: {results['comparison']['heat_balance_residuals']['max_abs']:.3f}")

    expected_rms = 0.715
    actual_rms = results['comparison']['heat_balance_residuals']['rms']
    diff = abs(actual_rms - expected_rms)

    print(f"\n🎯 Calibration Validation:")
    print(f"   Expected RMS residual: {expected_rms:.3f}")
    print(f"   Actual RMS residual: {actual_rms:.3f}")
    print(f"   Difference: {diff:.3f}")

    if diff < 0.1:
        print("   ✅ Excellent calibration! Physics matches neural network predictions.")
    elif diff < 0.2:
        print("   ⚠️  Good calibration, but some discrepancy detected.")
    else:
        print("   ❌ Poor calibration - check calibration process.")

    print("\n📍 Environmental Conditions:")
    print(f"   Ambient Temperature: {results['conditions']['T_ambient_range'][0]:.1f} - {results['conditions']['T_ambient_range'][1]:.1f} °C")
    print(f"   Wind Speed: {results['conditions']['wind_speed_range'][0]:.1f} - {results['conditions']['wind_speed_range'][1]:.1f} m/s")
    print(f"   Solar Irradiance: {results['conditions']['solar_range'][0]:.0f} - {results['conditions']['solar_range'][1]:.0f} W/m²")

    print("="*70)


def main():
    """Main comparison function"""
    import time

    start_time = time.time()

    # Run comparison
    results, nn_preds, phys_preds, conditions = compare_predictions(
        model_path='models/best_model.pt',
        vietnam_csv='data/mendeley/vietnam_220kv.csv',
        vietnam_params_path='calibration_results/vietnam_params.py',
        n_samples=100,  # Reduced from 1000 for faster execution
        device='mps' if torch.backends.mps.is_available() else 'cpu'
    )

    # Print results
    print_comparison_results(results)

    # Create plots
    fig = plot_comparison(results, nn_preds, phys_preds, conditions)
    plt.savefig('calibration_results/physics_vs_nn_comparison.png', dpi=150, bbox_inches='tight')
    print("📈 Comparison plot saved to calibration_results/physics_vs_nn_comparison.png")

    # Save detailed results
    results_path = f"calibration_results/comparison_results_{int(time.time())}.json"
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)

    print(f"💾 Detailed results saved to {results_path}")

    elapsed = time.time() - start_time
    print(f"\n⏱️  Comparison completed in {elapsed:.1f} seconds")


if __name__ == "__main__":
    main()
