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
Per-Line Physics Functions for InvariantPIKAN

Extension to core.physics that accepts per-line parameters for:
- resistance_factor
- emissivity  
- absorptivity

These functions are used with LinePhysicsParams module for learnable
line-specific physics.
"""

import torch
import torch.nn as nn
import math
from typing import Dict, Optional


def resistance_per_line(
    T_conductor: torch.Tensor,
    R_ref: float,
    resistance_factor: torch.Tensor,
    alpha_R: float,
    T_ref: float = 25.0
) -> torch.Tensor:
    """
    Temperature-dependent AC resistance with per-line factor.
    
    R(T) = R_ref * resistance_factor * [1 + α_R * (T - T_ref)]
    
    Args:
        T_conductor: Conductor temperature (°C) [batch]
        R_ref: Base resistance at T_ref (Ω/m)
        resistance_factor: Per-line resistance multiplier [batch]
        alpha_R: Temperature coefficient of resistance (1/°C)
        T_ref: Reference temperature (°C)
        
    Returns:
        Resistance per meter (Ω/m) [batch]
    """
    return R_ref * resistance_factor * (1 + alpha_R * (T_conductor - T_ref))


def joule_heating_per_line(
    current: torch.Tensor,
    T_conductor: torch.Tensor,
    R_ref: float,
    resistance_factor: torch.Tensor,
    alpha_R: float,
    T_ref: float = 25.0
) -> torch.Tensor:
    """
    Joule (I²R) heating with per-line resistance.
    
    Args:
        current: Line current (A) [batch]
        T_conductor: Conductor temperature (°C) [batch]
        R_ref: Base resistance at T_ref (Ω/m)
        resistance_factor: Per-line resistance multiplier [batch]
        alpha_R: Temperature coefficient
        T_ref: Reference temperature
        
    Returns:
        Heat gain rate (W/m) [batch]
    """
    R = resistance_per_line(T_conductor, R_ref, resistance_factor, alpha_R, T_ref)
    return current ** 2 * R


def solar_heat_gain_per_line(
    solar_irradiance: torch.Tensor,
    diameter: float,
    absorptivity: torch.Tensor
) -> torch.Tensor:
    """
    Solar heat absorption with per-line absorptivity.
    
    q_s = α_s * Q_s * D
    
    Args:
        solar_irradiance: Total solar irradiance (W/m²) [batch]
        diameter: Conductor diameter (m)
        absorptivity: Per-line absorptivity [batch]
        
    Returns:
        Heat gain rate (W/m) [batch]
    """
    return absorptivity * solar_irradiance * diameter


def radiative_heat_loss_per_line(
    T_conductor: torch.Tensor,
    T_ambient: torch.Tensor,
    diameter: float,
    emissivity: torch.Tensor,
    sigma: float = 5.67e-8
) -> torch.Tensor:
    """
    Radiative heat loss with per-line emissivity.
    
    q_r = π * D * ε * σ * (T_c⁴ - T_a⁴)
    
    Args:
        T_conductor: Conductor temperature (°C) [batch]
        T_ambient: Ambient temperature (°C) [batch]
        diameter: Conductor diameter (m)
        emissivity: Per-line emissivity [batch]
        sigma: Stefan-Boltzmann constant
        
    Returns:
        Heat loss rate (W/m) [batch]
    """
    T_c_K = T_conductor + 273.15
    T_a_K = T_ambient + 273.15
    
    pi = torch.tensor(math.pi, device=T_conductor.device)
    q_r = pi * diameter * emissivity * sigma * (
        torch.pow(T_c_K, 4) - torch.pow(T_a_K, 4)
    )
    
    return q_r


def heat_balance_residual_per_line(
    current: torch.Tensor,
    T_conductor: torch.Tensor,
    T_ambient: torch.Tensor,
    wind_speed: torch.Tensor,
    solar_irradiance: torch.Tensor,
    line_params: Dict[str, torch.Tensor],
    base_physics_params: Dict[str, float],
    wind_angle: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Heat balance residual with per-line physics parameters.
    
    Residual = (q_c + q_r) - (q_s + I²R)
    
    Args:
        current: Line current (A) [batch]
        T_conductor: Conductor temperature (°C) [batch]
        T_ambient: Ambient temperature (°C) [batch]
        wind_speed: Wind speed (m/s) [batch]
        solar_irradiance: Solar irradiance (W/m²) [batch]
        line_params: Dictionary with 'resistance_factor', 'emissivity', 'absorptivity'
        base_physics_params: Dictionary with 'diameter', 'R_ref', 'alpha_R'
        wind_angle: Wind angle (degrees, optional) [batch]
        
    Returns:
        Heat balance residual (W/m) [batch]
    """
    # Extract line parameters
    resistance_factor = line_params['resistance_factor']
    emissivity = line_params['emissivity']
    absorptivity = line_params['absorptivity']
    
    # Extract base parameters
    diameter = base_physics_params['diameter']
    R_ref = base_physics_params['R_ref']
    alpha_R = base_physics_params['alpha_R']
    
    # Heat gains with per-line parameters
    q_joule = joule_heating_per_line(
        current, T_conductor, R_ref, resistance_factor, alpha_R
    )
    q_solar = solar_heat_gain_per_line(
        solar_irradiance, diameter, absorptivity
    )
    
    # Heat losses
    # Convective heat loss (doesn't depend on emissivity/absorptivity)
    import sys
    from pathlib import Path
    sys.path.append(str(Path(__file__).parent.parent))
    from core.physics import IEEE738HeatBalance
    physics = IEEE738HeatBalance(
        conductor_diameter=diameter,
        resistance_per_meter_25C=R_ref,
        temp_coeff_resistance=alpha_R
    ).to(T_conductor.device)
    
    q_conv = physics.convective_heat_loss(
        T_conductor, T_ambient, wind_speed, wind_angle
    )
    
    # Radiative heat loss with per-line emissivity
    q_rad = radiative_heat_loss_per_line(
        T_conductor, T_ambient, diameter, emissivity
    )
    
    # Residual
    residual = (q_conv + q_rad) - (q_solar + q_joule)
    
    return residual


def physics_loss_per_line(
    predictions: Dict[str, torch.Tensor],
    targets: Dict[str, torch.Tensor],
    weather: torch.Tensor,
    line_params: Dict[str, torch.Tensor],
    base_physics_params: Dict[str, float],
    physics_weight: float = 0.1
) -> tuple:
    """
    Compute physics-informed loss with per-line parameters.
    
    Args:
        predictions: Dict with 'temperature', 'ampacity' from model
        targets: Dict with 'temperature', 'ampacity' ground truth
        weather: Weather tensor [batch, 4] with [T_amb, wind_speed, wind_angle, solar]
        line_params: Per-line physics parameters
        base_physics_params: Base physics constants
        physics_weight: Weight for physics loss component
        
    Returns:
        Tuple of (total_loss, loss_components_dict)
    """
    # Data loss
    temp_loss = nn.functional.mse_loss(predictions['temperature'], targets['temperature'])
    amp_loss = nn.functional.mse_loss(predictions['ampacity'], targets['ampacity'])
    
    # Physics loss
    T_ambient = weather[:, 0]
    wind_speed = weather[:, 1]
    wind_angle = weather[:, 2] if weather.shape[1] > 2 else None
    solar_irradiance = weather[:, 3]
    
    # Use predicted ampacity as current for physics check
    current = predictions['ampacity'].squeeze()
    T_conductor = predictions['temperature'].squeeze()
    
    residual = heat_balance_residual_per_line(
        current=current,
        T_conductor=T_conductor,
        T_ambient=T_ambient,
        wind_speed=wind_speed,
        solar_irradiance=solar_irradiance,
        line_params=line_params,
        base_physics_params=base_physics_params,
        wind_angle=wind_angle
    )
    
    physics_loss = torch.mean(residual ** 2)
    
    # Total loss
    total_loss = temp_loss + amp_loss + physics_weight * physics_loss
    
    loss_components = {
        'temp_loss': temp_loss.item(),
        'amp_loss': amp_loss.item(),
        'physics_loss': physics_loss.item(),
        'physics_residual_mean': torch.mean(torch.abs(residual)).item()
    }
    
    return total_loss, loss_components


# Quick test
if __name__ == "__main__":
    print("Testing per-line physics functions...")
    
    # Create test data
    batch_size = 5
    device = torch.device('cpu')
    
    current = torch.full((batch_size,), 800.0, device=device)
    T_conductor = torch.full((batch_size,), 60.0, device=device)
    T_ambient = torch.full((batch_size,), 25.0, device=device)
    wind_speed = torch.full((batch_size,), 5.0, device=device)
    solar_irradiance = torch.full((batch_size,), 800.0, device=device)
    
    # Per-line parameters (varying across batch)
    line_params = {
        'resistance_factor': torch.tensor([0.8, 1.0, 1.2, 0.9, 1.1], device=device),
        'emissivity': torch.tensor([0.7, 0.8, 0.9, 0.75, 0.85], device=device),
        'absorptivity': torch.tensor([0.7, 0.8, 0.9, 0.75, 0.85], device=device)
    }
    
    base_params = {
        'diameter': 0.02814,
        'R_ref': 7.283e-5,
        'alpha_R': 0.00403
    }
    
    # Compute residual
    residual = heat_balance_residual_per_line(
        current, T_conductor, T_ambient, wind_speed, solar_irradiance,
        line_params, base_params
    )
    
    print(f"\nResiduals for {batch_size} lines with different parameters:")
    print(f"  {residual.cpu().numpy()}")
    print(f"  Mean: {torch.mean(residual).item():.2f} W/m")
    print(f"  Std:  {torch.std(residual).item():.2f} W/m")
    
    print("\n✅ Per-line physics test passed!")
