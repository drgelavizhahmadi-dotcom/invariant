#!/usr/bin/env python3
"""
Evaluate Hybrid Ensemble
"""

import torch
import numpy as np
import sys

# Add project root to path
sys.path.append('.')

from core.model import HybridEnsemble
from scripts.validate_vietnam import VietnamDataset, load_model
from calibration_results.vietnam_params import VIETNAM_LINE_PARAMS


def evaluate_hybrid():
    print("🔍 Evaluating Hybrid Ensemble")
    print("=" * 30)

    # Load components
    neural_model, _ = load_model('models/best_model.pt', 'cpu')
    neural_model.eval()

    # Load ensemble
    checkpoint = torch.load('models/hybrid_ensemble_simple.pt')
    ensemble = HybridEnsemble(neural_model, VIETNAM_LINE_PARAMS)
    ensemble.load_state_dict(checkpoint['ensemble_state_dict'])
    ensemble.eval()

    phys_weight, neur_weight = ensemble.get_blending_weights()
    print(f"Blending weights - Physics: {phys_weight:.3f}, Neural: {neur_weight:.3f}")

    # Load test data
    vietnam_dataset = VietnamDataset('data/mendeley/vietnam_220kv.csv')
    print(f"Evaluating on {len(vietnam_dataset)} samples")

    # Evaluate
    predictions = []
    targets = []

    with torch.no_grad():
        for i in range(len(vietnam_dataset)):
            x, y = vietnam_dataset[i]
            x_batch = torch.tensor(x).unsqueeze(0)

            pred = ensemble(x_batch, return_components=True)

            predictions.append({
                'temp': pred['temperature'].item(),
                'amp': pred['ampacity'].item(),
                'amp_physics': pred['ampacity_physics'].item(),
                'amp_neural': pred['ampacity_neural'].item()
            })
            targets.append(y)

    # Convert to arrays
    pred_temps = np.array([p['temp'] for p in predictions])
    pred_amps = np.array([p['amp'] for p in predictions])
    pred_amps_physics = np.array([p['amp_physics'] for p in predictions])
    pred_amps_neural = np.array([p['amp_neural'] for p in predictions])

    true_temps = np.array([t[0] for t in targets])
    true_amps = np.array([t[1] for t in targets])

    # Calculate metrics
    temp_mae = np.mean(np.abs(pred_temps - true_temps))
    amp_mae = np.mean(np.abs(pred_amps - true_amps))
    physics_mae = np.mean(np.abs(pred_amps_physics - true_amps))
    neural_mae = np.mean(np.abs(pred_amps_neural - true_amps))

    print("
📊 Results:")
    print(".2f")
    print(".0f")
    print(".0f")
    print(".0f")

    improvement = (neural_mae - amp_mae) / neural_mae * 100
    print("+.1f")

    print("\n✅ Evaluation completed!")


if __name__ == "__main__":
    evaluate_hybrid()