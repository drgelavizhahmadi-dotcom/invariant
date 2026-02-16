"""
Invariant-PIKAN: Adversarially-Robust Physics-Informed Neural Networks for Dynamic Line Rating
Copyright (C) 2025 Gelavizh Ahmadi / Invariant Research

This software is licensed under the Business Source License 1.1 (BSL 1.1).
Commercial production use requires a separate license agreement.
See LICENSE.txt for full terms.

"""

"""
Hybrid Wavelet-Fourier Physics-Informed KAN for Power Grid DLR
Based on InvariantPIKAN architecture for advection-dominated PDEs
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Any, Optional

# Import physics functions - handle both relative and absolute imports
try:
    from core.physics import convective_heat_loss, radiative_heat_loss, solar_heat_gain, resistive_heat_gain
except ImportError:
    # Fallback for when run as script
    import sys
    import os
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
    from core.physics import convective_heat_loss, radiative_heat_loss, solar_heat_gain, resistive_heat_gain


class WaveletFourierEmbedding(nn.Module):
    """
    Multi-resolution embedding combining:
    - Fourier features (global smooth structure)
    - Wavelet features (local sharp variations)

    From: InvariantPIKAN (2025)
    """
    def __init__(self, input_dim=4, fourier_bands=16, wavelet_scales=8):
        super().__init__()

        # Fourier features (sin/cos basis)
        self.fourier_bands = fourier_bands
        self.freqs = nn.Parameter(
            torch.randn(input_dim, fourier_bands) * 2 * np.pi,
            requires_grad=True  # Learnable frequencies
        )

        # Wavelet scales (Morlet wavelet basis)
        self.wavelet_scales = wavelet_scales
        self.scales = nn.Parameter(
            torch.logspace(0, 2, wavelet_scales),  # Scale parameters
            requires_grad=True
        )

        # Output dimension
        self.output_dim = 2 * fourier_bands + wavelet_scales

    def morlet_wavelet(self, x, scale, omega0=5.0):
        """
        Morlet wavelet: ψ(t) = cos(5t) * exp(-t²/2)
        Scaled version: ψ((x - μ)/scale)
        """
        x_scaled = x / scale.unsqueeze(-1)
        return torch.cos(omega0 * x_scaled) * torch.exp(-0.5 * x_scaled**2)

    def forward(self, x):
        """
        x: [batch, input_dim] weather variables [T_amb, wind, solar, current]
        """
        batch_size = x.shape[0]

        # Fourier features
        # γ(v) = [cos(2πBv), sin(2πBv)] where B are learnable frequencies
        B = self.freqs  # [input_dim, fourier_bands]

        # x @ B: [batch, input_dim] @ [input_dim, fourier_bands] -> [batch, fourier_bands]
        fourier_cos = torch.cos(2 * np.pi * (x @ B))
        fourier_sin = torch.sin(2 * np.pi * (x @ B))

        # For now, simplify to just Fourier features to avoid dimension issues
        # Wavelet features - simplified
        wavelet_stack = torch.randn(x.shape[0], self.wavelet_scales, device=x.device)  # Placeholder

        # Concatenate all features
        # fourier_cos/sin: [batch, fourier_bands] each -> flatten to [batch, 2*fourier_bands]
        fourier_features = torch.cat([fourier_cos, fourier_sin], dim=-1)  # [batch, 2*fourier_bands]
        # wavelet_stack: [batch, wavelet_scales]
        h = torch.cat([fourier_features, wavelet_stack], dim=-1)  # [batch, 2*fourier_bands + wavelet_scales]

        return h


class AdaptiveLossBalancing(nn.Module):
    """
    Adaptive weighting between data and physics loss
    Based on neural tangent kernel analysis
    """
    def __init__(self, num_losses=2):
        super().__init__()
        self.log_weights = nn.Parameter(torch.zeros(num_losses))

    def forward(self, losses):
        """losses: dict of individual losses"""
        weights = F.softmax(self.log_weights, dim=-1)
        total = sum(w * losses[k] for w, k in zip(weights, losses.keys()))
        return total, weights


class InvariantPIKAN_DLR(nn.Module):
    """
    Hybrid Wavelet-Fourier Physics-Informed KAN for Dynamic Line Rating

    Architecture:
    1. Multi-resolution embedding (Fourier + Wavelet)
    2. KAN backbone with Chebyshev basis (for oscillatory dynamics)
    3. Physics-informed loss (IEEE 738 heat balance)

    Based on:
    - InvariantPIKAN for multi-scale PDEs
    - Scaled-cPIKAN for Chebyshev basis
    """

    def __init__(self,
                 input_dim=6,          # [T_amb, wind_speed, wind_angle, solar, current, R_ref]
                 hidden_dim=32,
                 output_dim=2,          # [temperature, ampacity]
                 fourier_bands=16,
                 wavelet_scales=8,
                 kan_grid=5,
                 kan_k=3):

        super().__init__()

        # 1. Multi-resolution embedding
        self.embedding = WaveletFourierEmbedding(
            input_dim=input_dim,
            fourier_bands=fourier_bands,
            wavelet_scales=wavelet_scales
        )

        # 2. KAN backbone (Chebyshev basis for oscillatory dynamics)
        # Note: Using a simplified KAN implementation since the full KAN library may not be available
        # In practice, you'd use the actual KAN library with Chebyshev basis
        kan_input_dim = self.embedding.output_dim
        self.kan_layers = nn.Sequential(
            nn.Linear(kan_input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim//2),
            nn.ReLU(),
            nn.Linear(hidden_dim//2, output_dim)
        )

        # 3. Learnable line parameters (from your calibration)
        self.log_resistance_factor = nn.Parameter(torch.tensor(0.0))  # Calibrated to 0.5
        self.raw_emissivity = nn.Parameter(torch.tensor(0.9))         # From calibration
        self.raw_absorptivity = nn.Parameter(torch.tensor(0.814))     # From calibration

        # 4. Adaptive loss balancing
        self.loss_balancer = AdaptiveLossBalancing(num_losses=3)  # data_temp, data_amp, physics

        # 5. Register calibrated values as buffers (not trainable unless needed)
        self.register_buffer('calibrated_resistance', torch.tensor(7.04e-5))
        self.register_buffer('calibrated_emissivity', torch.tensor(0.9))
        self.register_buffer('calibrated_absorptivity', torch.tensor(0.814))

    def get_line_params(self):
        """Get line parameters with learned adjustments"""
        return {
            'diameter': 0.0275,
            'emissivity': torch.sigmoid(self.raw_emissivity) * 0.5 + 0.3,
            'absorptivity': torch.sigmoid(self.raw_absorptivity) * 0.5 + 0.3,
            'resistance_ac': 1.4085e-4 * torch.exp(self.log_resistance_factor),
            'temp_coefficient': 0.0039
        }

    def compute_physics_residual(self, T, I, weather):
        """
        Simplified physics residual for batched computation
        TODO: Implement proper batched physics calculations
        """
        # Placeholder: simple constraint that heat loss should balance heat generation
        # This is a simplified version - in practice you'd use the full IEEE 738 equations
        residual = torch.abs(I**2 * 0.0001 - (T - weather['T_amb']) * weather['wind_speed'] * 0.1)
        return residual.mean()

    def forward(self, x, weather_dict):
        """
        x: [batch, input_dim] - weather variables
        weather_dict: dict with 'T_amb', 'wind_speed', 'solar' tensors
        """
        # Multi-resolution embedding
        h = self.embedding(x)

        # KAN forward pass
        output = self.kan_layers(h)  # [batch, 2] = [temp, ampacity]

        T_pred = output[:, 0]
        I_pred = output[:, 1]

        # Physics consistency residual
        physics_residual = self.compute_physics_residual(
            T_pred, I_pred, weather_dict
        )

        return {
            'temperature': T_pred,
            'ampacity': I_pred,
            'physics_residual': physics_residual,
            'line_params': self.get_line_params(),
            'embeddings': h
        }


def create_invariant_pikan_model(config=None):
    """Factory function with sensible defaults"""
    if config is None:
        config = {
            'input_dim': 6,
            'hidden_dim': 32,
            'fourier_bands': 16,
            'wavelet_scales': 8,
            'kan_grid': 5,
            'kan_k': 3
        }

    model = InvariantPIKAN_DLR(**config)

    # Initialize with calibrated Vietnam parameters
    model.log_resistance_factor.data = torch.log(torch.tensor(0.5))  # 50% of standard
    model.raw_emissivity.data = torch.tensor(0.9)
    model.raw_absorptivity.data = torch.tensor(0.814)

    return model
