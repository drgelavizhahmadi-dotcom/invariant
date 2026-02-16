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
Quick Comparison Test: Neural Network vs Calibrated Physics
"""

import torch
import numpy as np
import sys
sys.path.append('.')

from core.model import PhysicsDLR
from core.physics import IEEE738HeatBalance
from scripts.validate_vietnam import VietnamDataset, load_model
from calibration_results.vietnam_params import VIETNAM_LINE_PARAMS, CALIBRATION_METRICS

def quick_test():
    print("🔍 Quick Physics vs Model Comparison Test")
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

    # Test on first 5 samples
    print("\nTesting on first 5 samples:")
    print("Sample | NN Temp | NN Amp | Physics Amp | Residual")
    print("-" * 55)

    residuals = []

    with torch.no_grad():
        for i in range(min(5, len(dataset))):
            x, y = dataset[i]
            x_batch = torch.tensor(x).unsqueeze(0)

            # NN prediction
            pred_temp, pred_amp = model(x_batch)
            nn_temp = pred_temp.item()
            nn_amp = pred_amp.item()

            # Physics prediction (simple approximation)
            T_amb, wind_speed, wind_angle, solar, current = x

            # Calculate heat balance residual
            residual = physics.heat_balance_residual(
                current=torch.tensor([current]),
                T_conductor=torch.tensor([nn_temp]),
                T_ambient=torch.tensor([T_amb]),
                wind_speed=torch.tensor([wind_speed]),
                solar_irradiance=torch.tensor([solar]),
                wind_angle=torch.tensor([wind_angle])
            ).item()

            residuals.append(residual)

            print("5d")

    rms_residual = np.sqrt(np.mean(np.array(residuals)**2))
    print(".3f")
    print(".3f")
    print("✅ Test completed successfully!")

if __name__ == "__main__":
    quick_test()
