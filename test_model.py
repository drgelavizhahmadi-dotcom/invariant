"""
Invariant-PIKAN: Adversarially-Robust Physics-Informed Neural Networks for Dynamic Line Rating
Copyright (C) 2025 Gelavizh Ahmadi / Invariant Research

This software is licensed under the Business Source License 1.1 (BSL 1.1).
Commercial production use requires a separate license agreement.
See LICENSE.txt for full terms.

"""

import torch
import numpy as np
from core.model import PhysicsDLR
from core.data import InputNormalizer

# Load model and normalizer
checkpoint = torch.load('models/experiment1_consistency_model.pt', map_location='cpu', weights_only=False)
model = PhysicsDLR()
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

normalizer = InputNormalizer()
normalizer.mean = np.array(checkpoint['normalizer']['mean'])
normalizer.std = np.array(checkpoint['normalizer']['std'])

print("Normalizer stats:")
print(f"  Temp mean: {normalizer.mean[0]:.1f}°C, std: {normalizer.std[0]:.1f}")
print(f"  Wind mean: {normalizer.mean[1]:.1f}m/s, std: {normalizer.std[1]:.1f}")
print(f"  Current mean: {normalizer.mean[4]:.0f}A, std: {normalizer.std[4]:.0f}")
print()

# Test on training-like conditions
test_input = np.array([[15.0, 2.0, 45.0, 500.0, 700.0, 7.283e-5]])
test_input_norm = normalizer.transform(test_input)
test_tensor = torch.tensor(test_input_norm, dtype=torch.float32)

with torch.no_grad():
    temp_pred, amp_pred = model(test_tensor)

print('Training conditions (15°C, 2m/s, 500W/m², 700A):')
print(f'  Temp: {temp_pred.item():.1f}°C, Amp: {amp_pred.item():.0f}A')

# Test on Vietnam-like conditions
test_input_viet = np.array([[28.0, 14.0, 150.0, 200.0, 1000.0, 7.283e-5]])
test_input_viet_norm = normalizer.transform(test_input_viet)
test_tensor_viet = torch.tensor(test_input_viet_norm, dtype=torch.float32)

with torch.no_grad():
    temp_pred_viet, amp_pred_viet = model(test_tensor_viet)

print('Vietnam conditions (28°C, 14m/s, 200W/m², 1000A):')
print(f'  Temp: {temp_pred_viet.item():.1f}°C, Amp: {amp_pred_viet.item():.0f}A')
