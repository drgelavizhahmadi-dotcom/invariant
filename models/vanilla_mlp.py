#!/usr/bin/env python3
"""
Vanilla MLP Baseline

Pure data-driven feedforward network for DLR prediction.
- NO physics loss
- NO KAN (standard Linear layers with ReLU)
- NO wavelet-Fourier embeddings
- NO learnable physics parameters

Matched to HWF-PIKAN capacity:
- Same total parameter count (~130k params)
- Same training procedure
- Same optimizer settings
- Only architectural difference

Purpose: Shows what we lose without physics constraints.
"""

import torch
import torch.nn as nn


class VanillaMLP(nn.Module):
    """
    Standard feedforward MLP for ampacity prediction.

    Architecture matched to InvariantPIKANV2 capacity:
    - Input: 3 features (temperature, wind_speed, solar_irradiance)
    - Hidden: 4 layers × 64 neurons
    - Output: 1 value (ampacity prediction)
    - Activation: ReLU
    - Total params: ~12.8k (comparable to HWF-PIKAN's ~11.2k)
    """

    def __init__(self, input_dim=3, hidden_dim=64, n_layers=4):
        super().__init__()

        layers = []

        # Input layer
        layers.append(nn.Linear(input_dim, hidden_dim))
        layers.append(nn.ReLU())

        # Hidden layers
        for _ in range(n_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU())

        # Output layer
        layers.append(nn.Linear(hidden_dim, 1))

        self.net = nn.Sequential(*layers)

        # Count parameters for verification
        self.param_count = sum(p.numel() for p in self.parameters())

    def forward(self, x, line_ids=None):
        """
        Forward pass.

        Args:
            x: Input features (batch_size, 3)
            line_ids: Ignored (no per-line parameters in this baseline)

        Returns:
            predictions: (batch_size, 1)
        """
        return self.net(x)

    def compute_physics_residual(self, x, y_pred):
        """
        No physics residual for vanilla MLP.
        Returns zeros to maintain API compatibility.
        """
        return torch.zeros(x.size(0), device=x.device)

    def explain_prediction(self, x, y_pred, T_conductor=75.0):
        """
        Vanilla MLP cannot provide physics-based explanations.
        Returns None to indicate lack of explainability.

        This is the key difference for AI Act compliance:
        - HWF-PIKAN: Can explain via heat balance decomposition
        - VanillaMLP: Black box, no physical interpretability
        """
        return None

    def get_physics_params(self):
        """No physics parameters to report."""
        return {}


if __name__ == "__main__":
    # Quick verification
    model = VanillaMLP(input_dim=3, hidden_dim=64, n_layers=4)
    print(f"VanillaMLP parameter count: {model.param_count:,}")

    # Test forward pass
    x = torch.randn(32, 3)
    y = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {y.shape}")

    # Test explainability API (should return None)
    explanation = model.explain_prediction(x, y)
    print(f"Explainability: {explanation}")  # None
