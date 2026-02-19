#!/usr/bin/env python3
"""
Vanilla PINN Baseline

MLP + IEEE 738 physics loss + learnable global physics parameters.
- NO KAN (standard Linear layers with ReLU)
- NO wavelet-Fourier embeddings
- YES physics loss (IEEE 738 heat balance residual)
- YES learnable physics parameters (emissivity, absorptivity, resistance)
- YES explain_prediction (heat balance decomposition)

Same capacity as VanillaMLP and HWF-PIKAN (~12k params).
Only difference from VanillaMLP: physics constraint in training loss
and ability to produce audit-ready explanations.

Purpose: Isolates the contribution of physics constraints
from the KAN/wavelet/Fourier architectural choices.
"""

import torch
import torch.nn as nn
import math


class VanillaPINN(nn.Module):
    """
    MLP + Physics-Informed loss for ampacity prediction.

    Architecture matched to VanillaMLP and InvariantPIKANV2 capacity:
    - Input: 3 features (temperature, wind_speed, solar_irradiance)
    - Hidden: 4 layers x 64 neurons (ReLU)
    - Output: 1 value (ampacity prediction)
    - Physics: IEEE 738 heat balance residual in loss
    - Learnable: emissivity, absorptivity, resistance_factor
    - Total params: ~12.8k + 3 physics params
    """

    def __init__(self, input_dim=3, hidden_dim=64, n_layers=4):
        super().__init__()

        layers = []
        layers.append(nn.Linear(input_dim, hidden_dim))
        layers.append(nn.ReLU())
        for _ in range(n_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(hidden_dim, 1))

        self.net = nn.Sequential(*layers)

        # Learnable physics parameters (same parameterization as HWF-PIKAN)
        self.physics_params = nn.ParameterDict({
            'log_resistance': nn.Parameter(torch.tensor(0.0)),
            'raw_emissivity': nn.Parameter(torch.tensor(0.8)),
            'raw_absorptivity': nn.Parameter(torch.tensor(0.8)),
        })

        self.param_count = sum(p.numel() for p in self.parameters())

    def get_physics_params(self):
        """Get physics parameters in physical space."""
        return {
            'resistance_factor': torch.exp(self.physics_params['log_resistance']),
            'emissivity': torch.sigmoid(self.physics_params['raw_emissivity']),
            'absorptivity': torch.sigmoid(self.physics_params['raw_absorptivity']),
        }

    def forward(self, x, line_ids=None):
        """
        Forward pass. Returns (batch_size, 1) tensor.
        Same API as VanillaMLP for drop-in comparison.
        """
        return self.net(x)

    def compute_physics_residual(self, x, y_pred):
        """
        IEEE 738 heat balance residual per sample.

        Uses the same simplified analytical formulation as
        InvariantPIKANV2.compute_physics_residual for fair comparison.

        Args:
            x: (batch, 3) normalized input features
            y_pred: (batch,) predicted ratio (dimensionless)

        Returns:
            residual: (batch,) per-sample physics violation
        """
        params = self.get_physics_params()

        # Approximate physics residual on normalized inputs
        # This matches InvariantPIKANV2's ieee738_analytical behavior
        T_amb = x[:, 0]
        V_wind = x[:, 1]
        Q_solar = x[:, 2]

        D = 0.02814
        R_ref = 7.283e-5
        alpha_R = 0.00403

        wind_factor = torch.clamp(V_wind, min=0.1).abs() ** 0.5
        q_conv = 2.5 * wind_factor * (D * math.pi)
        q_solar = Q_solar.abs() * params['absorptivity'] * (D * math.pi)
        q_resist = y_pred ** 2 * R_ref * params['resistance_factor']

        residual = torch.abs(q_conv - q_solar - q_resist)
        return residual

    def explain_prediction(self, x, I_pred, T_conductor=75.0):
        """
        IEEE 738 heat balance decomposition — identical formulation
        to InvariantPIKANV2.explain_prediction.

        Uses THIS model's learned physics parameters.
        This is the AI Act audit trail: a regulator sees the same
        physical decomposition regardless of whether the backbone
        is an MLP or a KAN.

        Args:
            x: (batch, 3) weather features [temperature, wind_speed, solar_irradiance]
            I_pred: (batch,) predicted ampacity (A)
            T_conductor: conductor temperature limit (C), default 75.0

        Returns:
            dict with per-sample explanation (same schema as HWF-PIKAN)
        """
        params = self.get_physics_params()

        T_amb = x[:, 0]
        wind_speed = x[:, 1]
        solar_irradiance = x[:, 2]

        D = 0.02814
        R_ref = 7.283e-5
        alpha_R = 0.00403
        sigma = 5.67e-8
        pi = torch.tensor(math.pi)

        emissivity = params['emissivity']
        absorptivity = params['absorptivity']
        resistance_factor = params['resistance_factor']

        T_c = torch.full_like(T_amb, T_conductor)

        # Convective cooling (IEEE 738)
        rho_air = 1.1
        mu_air = 1.96e-5
        k_air = 0.0277
        V_w = torch.clamp(wind_speed, min=1e-4)
        Re = rho_air * V_w * D / mu_air
        Pr = 0.71
        Nu_low = 0.3 + (
            0.62 * torch.pow(Re, 0.5) * (Pr ** (1.0 / 3.0))
        ) / ((1.0 + (0.4 / Pr) ** (2.0 / 3.0)) ** 0.25)
        Nu_morgan = 0.583 * torch.pow(Re, 0.471)
        Nu = torch.where(Re < 1000.0, Nu_low, Nu_morgan)
        q_convective = pi * k_air * Nu * (T_c - T_amb)

        # Radiative cooling
        T_c_K = T_c + 273.15
        T_a_K = T_amb + 273.15
        q_radiative = pi * D * emissivity * sigma * (
            torch.pow(T_c_K, 4) - torch.pow(T_a_K, 4)
        )

        # Solar heating
        q_solar = absorptivity * solar_irradiance * D

        # Joule heating
        R_T = R_ref * resistance_factor * (1 + alpha_R * (T_conductor - 25.0))
        q_joule = I_pred ** 2 * R_T

        # Binding constraint
        cooling_total = q_convective + q_radiative
        binding = torch.where(
            q_convective < q_radiative,
            torch.zeros_like(q_convective),
            torch.ones_like(q_radiative),
        )
        convective_fraction = q_convective / (cooling_total + 1e-8)
        heating_total = q_solar + q_joule
        physics_residual = torch.abs(cooling_total - heating_total)

        return {
            'q_convective': q_convective,
            'q_radiative': q_radiative,
            'q_solar': q_solar,
            'q_joule': q_joule,
            'binding_constraint': binding,
            'physics_residual': physics_residual,
            'convective_fraction': convective_fraction,
            'cooling_total': cooling_total,
            'heating_total': heating_total,
        }


if __name__ == "__main__":
    model = VanillaPINN(input_dim=3, hidden_dim=64, n_layers=4)
    print(f"VanillaPINN parameter count: {model.param_count:,}")
    print(f"Physics params: {model.get_physics_params()}")

    x = torch.randn(32, 3)
    y = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {y.shape}")

    residual = model.compute_physics_residual(x, y.squeeze())
    print(f"Physics residual shape: {residual.shape}")

    explanation = model.explain_prediction(x, y.squeeze())
    print(f"Explainability: {list(explanation.keys())}")
