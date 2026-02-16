#!/usr/bin/env python3
"""
Simple Physics vs Model Comparison
"""

import torch
import numpy as np
import sys
sys.path.append('.')

from core.model import PhysicsDLR
from core.physics import IEEE738HeatBalance
from scripts.validate_vietnam import VietnamDataset, load_model
from calibration_results.vietnam_params import VIETNAM_LINE_PARAMS, CALIBRATION_METRICS

def simple_comparison():
    print("🔍 Physics vs Model Comparison")
    print("=" * 50)

    # Load model
    print("Loading model...")
    model, normalizer = load_model('models/best_model.pt', 'cpu')
    model.eval()

    # Load dataset
    print("Loading Vietnam dataset...")
    dataset = VietnamDataset('data/mendeley/vietnam_220kv.csv')
    print(f"Dataset size: {len(dataset)}")

    # Create physics with calibrated params
    print("Creating calibrated physics engine...")
    physics = IEEE738HeatBalance(
        conductor_diameter=VIETNAM_LINE_PARAMS['diameter'],
        conductor_emissivity=VIETNAM_LINE_PARAMS['emissivity'],
        conductor_absorptivity=VIETNAM_LINE_PARAMS['absorptivity'],
        resistance_per_meter_25C=VIETNAM_LINE_PARAMS['resistance_ac'],
        temp_coeff_resistance=VIETNAM_LINE_PARAMS['temp_coefficient'],
        max_conductor_temp=100.0
    )

    print(f"Expected RMS residual: {CALIBRATION_METRICS['rms_residual']:.3f}")

    # Test on 10 random samples
    n_samples = 10
    indices = np.random.choice(len(dataset), n_samples, replace=False)

    print(f"\nTesting on {n_samples} random samples:")
    print("Sample | NN Temp | NN Amp | Residual")
    print("-" * 40)

    residuals = []
    nn_amps = []
    nn_temps = []

    with torch.no_grad():
        for i, idx in enumerate(indices):
            x, y = dataset[idx]
            x_batch = torch.tensor(x).unsqueeze(0)

            # NN prediction
            pred_temp, pred_amp = model(x_batch)
            nn_temp = pred_temp.item()
            nn_amp = pred_amp.item()

            # Calculate heat balance residual
            T_amb, wind_speed, wind_angle, solar, current = x
            residual = physics.heat_balance_residual(
                current=torch.tensor([current]),
                T_conductor=torch.tensor([nn_temp]),
                T_ambient=torch.tensor([T_amb]),
                wind_speed=torch.tensor([wind_speed]),
                solar_irradiance=torch.tensor([solar]),
                wind_angle=torch.tensor([wind_angle])
            ).item()

            residuals.append(residual)
            nn_amps.append(nn_amp)
            nn_temps.append(nn_temp)

            print("3d")

    residuals = np.array(residuals)
    nn_amps = np.array(nn_amps)
    nn_temps = np.array(nn_temps)

    print("
📊 Summary:")
    print(".3f")
    print(".3f")
    print(".1f")
    print(".1f")
    print(".3f")
    print(".3f")

    expected_rms = CALIBRATION_METRICS['rms_residual']
    actual_rms = np.sqrt(np.mean(residuals**2))
    diff = abs(actual_rms - expected_rms)

    print("
🎯 Calibration Validation:")
    print(".3f")
    print(".3f")
    print(".3f")

    if diff < 0.05:
        print("   ✅ Excellent! Physics matches neural network predictions.")
    elif diff < 0.1:
        print("   ⚠️  Good calibration, minor discrepancy.")
    else:
        print("   ❌ Significant discrepancy - check calibration.")

    print("\n✅ Comparison completed!")

if __name__ == "__main__":
    simple_comparison()