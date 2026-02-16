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
Simple Hybrid Ensemble Training
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import sys

# Add project root to path
sys.path.append('.')

from core.model import PhysicsDLR, HybridEnsemble
from core.data import SyntheticDLRDataset, DataConfig
from scripts.validate_vietnam import VietnamDataset, load_model
from calibration_results.vietnam_params import VIETNAM_LINE_PARAMS


def simple_train():
    print("🔄 Simple Hybrid Ensemble Training")
    print("=" * 40)

    # Load neural model
    print("Loading neural model...")
    neural_model, _ = load_model('models/best_model.pt', 'cpu')
    neural_model.eval()

    # Create ensemble
    ensemble = HybridEnsemble(neural_model, VIETNAM_LINE_PARAMS)
    ensemble.train()

    # Create simple training data (just Vietnam data)
    print("Loading Vietnam data...")
    vietnam_dataset = VietnamDataset('data/mendeley/vietnam_220kv.csv')
    X_train = []
    y_train = []

    for i in range(len(vietnam_dataset)):
        x, y = vietnam_dataset[i]
        X_train.append(x)
        y_train.append(y)

    X_train = torch.tensor(np.array(X_train), dtype=torch.float32)
    y_train = torch.tensor(np.array(y_train), dtype=torch.float32)

    print(f"Training on {len(X_train)} samples")

    # Simple training loop
    optimizer = optim.Adam([ensemble.physics_weight_logit, ensemble.neural_weight_logit], lr=1e-2)
    criterion = nn.MSELoss()

    print("Training for 10 epochs...")
    for epoch in range(10):
        optimizer.zero_grad()

        # Forward pass
        pred = ensemble(X_train)
        loss = criterion(pred['ampacity'], y_train[:, 1])  # Ampacity target

        # Backward pass
        loss.backward()
        optimizer.step()

        phys_weight, neur_weight = ensemble.get_blending_weights()
        print("2d"
    # Save model
    print("\n💾 Saving ensemble...")
    save_path = 'models/hybrid_ensemble_simple.pt'
    torch.save({
        'ensemble_state_dict': ensemble.state_dict(),
        'final_weights': ensemble.get_blending_weights()
    }, save_path)
    print(f"Saved to {save_path}")
    print(f"Final weights - Physics: {phys_weight:.3f}, Neural: {neur_weight:.3f}")


if __name__ == "__main__":
    simple_train()
