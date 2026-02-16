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
import torch
import pandas as pd
from pathlib import Path
import sys
sys.path.append('/Users/gelavizhahmadi/Projects/invariant')

from models.invariant_pikan_v2 import InvariantPIKANV2
from core.data import VietnamDataset
from core.physics import IEEE738HeatBalance
from torch.utils.data import DataLoader
import torch.nn as nn

def test_weight_update():
    # Load model
    model = InvariantPIKANV2()
    device = torch.device('mps')  # Use MPS like training
    model.to(device)

    # Get initial weights
    initial_weights = {name: param.clone() for name, param in model.named_parameters()}

    # Load small dataset
    data_path = 'runs/invariant_pikan_production_20260215_165605/temp_unified_data.csv'
    dataset = VietnamDataset(data_path)
    dataloader = DataLoader(dataset, batch_size=8, shuffle=True)

    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    # Physics engine
    physics = IEEE738HeatBalance()

    # Use PhysicsInformedLoss like in training
    from scripts.train_invariant_pikan_production import PhysicsInformedLoss
    loss_fn = PhysicsInformedLoss(physics)

    # Get one batch
    batch = next(iter(dataloader))
    x, y = batch
    weather = x[:, :4].to(device)
    weather_dict = {'T_amb': weather[:, 0], 'wind_speed': weather[:, 1], 'solar': weather[:, 3]}

    # Multiple steps
    for step in range(5):
        # Forward
        preds = model(weather, weather_dict)
        targets = {'temperature': y[:, 0].to(device), 'ampacity': y[:, 1].to(device)}
        weather_input = torch.cat([weather[:, :3], x[:, 4:5].to(device)], dim=-1)
        loss, _ = loss_fn(preds, targets, weather_input, return_components=True)
        
        print(f"Step {step}: Loss = {loss.item():.4f}")
        
        # Backward and step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

if __name__ == "__main__":
    test_weight_update()
