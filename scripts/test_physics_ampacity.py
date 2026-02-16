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
Test Physics-Constrained Ampacity Derivation

This script tests whether deriving ampacity from accurate temperature predictions
using IEEE 738 physics equations provides better ampacity accuracy than direct prediction.

Approach:
1. Load domain-adapted model with excellent temperature prediction (1.74°C MAE)
2. Use model to predict conductor temperatures
3. Use IEEE 738 physics to derive ampacity from predicted temperatures
4. Compare derived ampacity to actual measured ampacity

Author: Dr. Gelavizh Ahmadi
Copyright (c) 2026 Invariant Energy GmbH
"""

import torch
import numpy as np
import pandas as pd
from pathlib import Path
import sys
import json
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from typing import Dict, Tuple

# Import project modules
sys.path.append('.')

from core.model import PhysicsDLR
from core.physics import IEEE738HeatBalance
from scripts.validate_vietnam import VietnamDataset, load_model


def derive_ampacity_from_temperature(
    T_conductor: torch.Tensor,
    T_ambient: torch.Tensor,
    wind_speed: torch.Tensor,
    wind_angle: torch.Tensor,
    solar_irradiance: torch.Tensor,
    physics_engine: IEEE738HeatBalance,
    tolerance: float = 0.1,
    max_iterations: int = 50,
) -> torch.Tensor:
    """
    Derive ampacity from predicted conductor temperature using physics

    Uses binary search to find current that produces the predicted temperature

    Args:
        T_conductor: Predicted conductor temperature [batch]
        T_ambient: Ambient temperature [batch]
        wind_speed: Wind speed [batch]
        wind_angle: Wind angle [batch]
        solar_irradiance: Solar irradiance [batch]
        physics_engine: IEEE 738 physics engine
        tolerance: Temperature tolerance for convergence (°C)
        max_iterations: Maximum binary search iterations

    Returns:
        Derived ampacity (current) [batch]
    """
    # Initialize current bounds
    current_min = torch.full_like(T_conductor, 100.0)   # Minimum current (A)
    current_max = torch.full_like(T_conductor, 3000.0)  # Maximum current (A)

    # Target temperature (predicted conductor temp)
    T_target = T_conductor.clone()

    for iteration in range(max_iterations):
        # Current guess (midpoint)
        current_guess = (current_min + current_max) / 2

        # Calculate heat balance residual for this current
        # residual = q_c + q_r - q_s - I²R
        # We want residual = 0 for steady state
        residual = physics_engine.heat_balance_residual(
            current=current_guess,
            T_conductor=T_target,
            T_ambient=T_ambient,
            wind_speed=wind_speed,
            solar_irradiance=solar_irradiance,
            wind_angle=wind_angle,
        )

        # If residual > 0, temperature would be too low (current too high)
        # If residual < 0, temperature would be too high (current too low)
        current_max = torch.where(residual > 0, current_guess, current_max)
        current_min = torch.where(residual < 0, current_guess, current_min)

        # Check convergence (residual close to zero)
        if torch.all(torch.abs(residual) < tolerance):
            break

    # Final ampacity is the current that gives zero residual
    derived_ampacity = (current_min + current_max) / 2

    return derived_ampacity


def test_physics_constrained_ampacity(
    model_path: str = 'models/vietnam_domain_adapted_20260213_212616.pt',
    vietnam_csv: str = 'data/mendeley/vietnam_220kv.csv',
    device: str = 'cpu',
    batch_size: int = 256,
) -> Dict[str, float]:
    """
    Test physics-constrained ampacity derivation

    Args:
        model_path: Path to model with good temperature prediction
        vietnam_csv: Path to Vietnam dataset
        device: Device to use
        batch_size: Batch size for processing

    Returns:
        Dictionary of comparison metrics
    """
    print("🔬 Testing Physics-Constrained Ampacity Derivation")
    print("=" * 70)

    # Set device
    device = torch.device(device)
    print(f"Using device: {device}")

    # Load model and normalizer
    print(f"Loading model from {model_path}...")
    model, normalizer = load_model(model_path, device)
    model.eval()

    # Load Vietnam dataset
    print(f"Loading Vietnam dataset from {vietnam_csv}...")
    vietnam_dataset = VietnamDataset(vietnam_csv)

    # Initialize physics engine
    physics_engine = IEEE738HeatBalance(
        conductor_diameter=0.02814,  # Drake ACSR typical
        conductor_emissivity=0.8,
        conductor_absorptivity=0.8,
        resistance_per_meter_25C=7.283e-5,
        temp_coeff_resistance=0.00403,
        max_conductor_temp=100.0
    )

    print("\n📊 Dataset Statistics:")
    print(f"   Samples: {len(vietnam_dataset):,}")
    print(f"   Temperature range: {vietnam_dataset.T_ambient.min():.1f} - {vietnam_dataset.T_ambient.max():.1f} °C")
    print(f"   Wind speed range: {vietnam_dataset.wind_speed.min():.1f} - {vietnam_dataset.wind_speed.max():.1f} m/s")
    print(f"   Ampacity range: {vietnam_dataset.ampacity.min():.0f} - {vietnam_dataset.ampacity.max():.0f} A")

    # Process in batches
    print("\n🔄 Processing predictions...")
    all_pred_temp_direct = []
    all_true_temp = []
    all_pred_amp_direct = []
    all_pred_amp_physics = []
    all_true_amp = []

    n_batches = (len(vietnam_dataset) + batch_size - 1) // batch_size

    with torch.no_grad():
        for batch_idx in range(n_batches):
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, len(vietnam_dataset))

            # Prepare batch data
            batch_x = []
            batch_y = []

            for j in range(start_idx, end_idx):
                x, y = vietnam_dataset[j]
                batch_x.append(x)
                batch_y.append(y)

            x_batch = torch.stack(batch_x).to(device)
            y_batch = torch.stack(batch_y).to(device)

            # Normalize inputs
            x_batch_normalized = torch.tensor(
                normalizer.transform(x_batch.cpu().numpy()),
                dtype=torch.float32,
                device=device
            )

            # Direct model predictions
            pred_temp_direct, pred_amp_direct = model(x_batch_normalized)

            # Denormalize inputs for physics calculations
            raw_inputs = x_batch.cpu().numpy()
            T_ambient = torch.tensor(raw_inputs[:, 0], device=device)
            wind_speed = torch.tensor(raw_inputs[:, 1], device=device)
            wind_angle = torch.tensor(raw_inputs[:, 2], device=device)
            solar_irradiance = torch.tensor(raw_inputs[:, 3], device=device)
            current_fixed = torch.tensor(raw_inputs[:, 4], device=device)  # Fixed current from dataset
            resistance = torch.tensor(raw_inputs[:, 5], device=device)

            # Derive ampacity from predicted temperature using physics
            pred_temp_denorm = pred_temp_direct.squeeze()
            derived_ampacity = derive_ampacity_from_temperature(
                T_conductor=pred_temp_denorm,
                T_ambient=T_ambient,
                wind_speed=wind_speed,
                wind_angle=wind_angle,
                solar_irradiance=solar_irradiance,
                physics_engine=physics_engine,
            )

            # Collect results
            all_pred_temp_direct.extend(pred_temp_direct.squeeze().cpu().numpy())
            all_true_temp.extend(y_batch[:, 0].cpu().numpy())
            all_pred_amp_direct.extend(pred_amp_direct.squeeze().cpu().numpy())
            all_pred_amp_physics.extend(derived_ampacity.cpu().numpy())
            all_true_amp.extend(y_batch[:, 1].cpu().numpy())

            if (batch_idx + 1) % 10 == 0:
                print(f"   Processed {end_idx}/{len(vietnam_dataset)} samples...")

    # Convert to numpy arrays
    pred_temp_direct = np.array(all_pred_temp_direct)
    true_temp = np.array(all_true_temp)
    pred_amp_direct = np.array(all_pred_amp_direct)
    pred_amp_physics = np.array(all_pred_amp_physics)
    true_amp = np.array(all_true_amp)

    # Debug: Print some statistics
    print("\n🔍 Debug Statistics:")
    print(f"   Predicted temp range: {pred_temp_direct.min():.1f} - {pred_temp_direct.max():.1f} °C")
    print(f"   True temp range: {true_temp.min():.1f} - {true_temp.max():.1f} °C")
    print(f"   Direct amp range: {pred_amp_direct.min():.0f} - {pred_amp_direct.max():.0f} A")
    print(f"   Physics amp range: {pred_amp_physics.min():.0f} - {pred_amp_physics.max():.0f} A")
    print(f"   True amp range: {true_amp.min():.0f} - {true_amp.max():.0f} A")

    # Calculate metrics
    print("\n📈 Calculating Metrics...")
    metrics = {}

    # Temperature metrics (direct prediction)
    metrics['temp_mae_direct'] = mean_absolute_error(true_temp, pred_temp_direct)
    metrics['temp_rmse_direct'] = np.sqrt(mean_squared_error(true_temp, pred_temp_direct))
    metrics['temp_r2_direct'] = r2_score(true_temp, pred_temp_direct)

    # Ampacity metrics (direct prediction)
    metrics['amp_mae_direct'] = mean_absolute_error(true_amp, pred_amp_direct)
    metrics['amp_rmse_direct'] = np.sqrt(mean_squared_error(true_amp, pred_amp_direct))
    metrics['amp_r2_direct'] = r2_score(true_amp, pred_amp_direct)

    # Ampacity metrics (physics-derived)
    metrics['amp_mae_physics'] = mean_absolute_error(true_amp, pred_amp_physics)
    metrics['amp_rmse_physics'] = np.sqrt(mean_squared_error(true_amp, pred_amp_physics))
    metrics['amp_r2_physics'] = r2_score(true_amp, pred_amp_physics)

    # Improvement ratios
    metrics['amp_mae_improvement'] = metrics['amp_mae_direct'] / metrics['amp_mae_physics']
    metrics['amp_rmse_improvement'] = metrics['amp_rmse_direct'] / metrics['amp_rmse_physics']
    metrics['amp_r2_improvement'] = metrics['amp_r2_physics'] / metrics['amp_r2_direct'] if metrics['amp_r2_direct'] != 0 else float('inf')

    return metrics, pred_temp_direct, true_temp, pred_amp_direct, pred_amp_physics, true_amp


def print_comparison_table(metrics: Dict[str, float]):
    """Print formatted comparison table"""
    print("\n" + "="*70)
    print("🎯 PHYSICS-CONSTRAINED AMPACITY RESULTS")
    print("="*70)

    print(f"{'Metric':<15} {'MAE':<10} {'RMSE':<10} {'R²':<10}")
    print("-" * 70)

    # Temperature (direct)
    print(f"{'Temperature':<15} "
          f"{metrics['temp_mae_direct']:<10.2f} "
          f"{metrics['temp_rmse_direct']:<10.2f} "
          f"{metrics['temp_r2_direct']:<10.3f}")

    print("-" * 70)

    # Ampacity (direct)
    print(f"{'Ampacity Direct':<15} "
          f"{metrics['amp_mae_direct']:<10.0f} "
          f"{metrics['amp_rmse_direct']:<10.0f} "
          f"{metrics['amp_r2_direct']:<10.3f}")

    # Ampacity (physics-derived)
    print(f"{'Ampacity Physics':<15} "
          f"{metrics['amp_mae_physics']:<10.0f} "
          f"{metrics['amp_rmse_physics']:<10.0f} "
          f"{metrics['amp_r2_physics']:<10.3f}")

    print("-" * 70)

    # Improvements
    print(f"{'Improvement (x)':<15} "
          f"{metrics['amp_mae_improvement']:<10.2f} "
          f"{metrics['amp_rmse_improvement']:<10.2f} "
          f"{metrics['amp_r2_improvement']:<10.2f}")

    print("="*70)


def main():
    """Main test function"""
    import time

    start_time = time.time()

    # Test physics-constrained ampacity
    metrics, pred_temp, true_temp, pred_amp_direct, pred_amp_physics, true_amp = test_physics_constrained_ampacity(
        model_path='models/vietnam_domain_adapted_20260213_212616.pt',
        vietnam_csv='data/mendeley/vietnam_220kv.csv',
        device='mps' if torch.backends.mps.is_available() else 'cpu',
        batch_size=256,
    )

    # Print results
    print_comparison_table(metrics)

    # Summary
    improvement = metrics['amp_mae_improvement']
    if improvement > 1.1:
        print(f"\n✅ SUCCESS: Physics-constrained ampacity improves MAE by {improvement:.1f}x!")
        print("   Deriving ampacity from temperature predictions is better than direct prediction.")
    elif improvement > 0.9:
        print(f"\n⚪ NEUTRAL: Physics-constrained ampacity similar to direct prediction ({improvement:.2f}x)")
    else:
        print(f"\n❌ WORSE: Physics-constrained ampacity worse than direct prediction ({improvement:.2f}x)")

    # Save detailed results
    results = {
        'timestamp': pd.Timestamp.now().isoformat(),
        'model': 'vietnam_domain_adapted_20260213_212616.pt',
        'metrics': metrics,
        'summary': {
            'temperature_mae': metrics['temp_mae_direct'],
            'ampacity_mae_direct': metrics['amp_mae_direct'],
            'ampacity_mae_physics': metrics['amp_mae_physics'],
            'improvement_ratio': improvement,
        }
    }

    results_path = f"results/physics_constrained_ampacity_{int(time.time())}.json"
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)

    print(f"\n💾 Detailed results saved to {results_path}")

    elapsed = time.time() - start_time
    print(f"\n⏱️  Test completed in {elapsed:.1f} seconds")

if __name__ == "__main__":
    main()
