"""
Invariant-PIKAN: Adversarially-Robust Physics-Informed Neural Networks for Dynamic Line Rating
Copyright (C) 2025 Gelavizh Ahmadi / Invariant Research

This software is licensed under the Business Source License 1.1 (BSL 1.1).
Commercial production use requires a separate license agreement.
See LICENSE.txt for full terms.

HWF-PIKAN for plasma physics (Heravifard et al., Sharif University, 2025).
"""

"""
IEEE 738 Heat Balance Equations
Differentiable implementation for physics-informed neural network loss

Reference: IEEE Std 738-2012 - Standard for Calculating the Current-Temperature
Relationship of Bare Overhead Conductors

Author: Dr. Gelavizh Ahmadi
Copyright (c) 2026 Invariant Energy GmbH
"""

import torch
import torch.nn as nn
import math
from typing import Optional, Tuple
from dataclasses import dataclass


@dataclass
class ConductorProperties:
    """Standard conductor properties for common ACSR types"""
    
    # Drake ACSR (typical 230kV transmission)
    DRAKE = {
        "diameter": 0.02814,  # m
        "resistance_25C": 7.283e-5,  # Ohm/m
        "emissivity": 0.8,
        "absorptivity": 0.8,
        "temp_coeff": 0.00403,  # 1/°C
    }
    
    # Cardinal ACSR (typical 345kV transmission)
    CARDINAL = {
        "diameter": 0.03048,  # m
        "resistance_25C": 5.738e-5,  # Ohm/m
        "emissivity": 0.8,
        "absorptivity": 0.8,
        "temp_coeff": 0.00403,  # 1/°C
    }


class IEEE738HeatBalance(nn.Module):
    """
    IEEE 738-2012 Steady-State Heat Balance
    
    The fundamental equation:
        q_c + q_r = q_s + I²R
    
    Where:
        q_c = convective heat loss rate (W/m)
        q_r = radiative heat loss rate (W/m)
        q_s = solar heat gain rate (W/m)
        I²R = Joule heating rate (W/m)
    
    All methods are differentiable for use in neural network training.
    """
    
    def __init__(
        self,
        conductor_diameter: float = 0.02814,  # m (Drake ACSR default)
        conductor_emissivity: float = 0.8,
        conductor_absorptivity: float = 0.8,
        resistance_per_meter_25C: float = 7.283e-5,  # Ohm/m at 25°C
        temp_coeff_resistance: float = 0.00403,  # 1/°C for aluminum
        max_conductor_temp: float = 100.0,  # °C safety limit
    ):
        super().__init__()
        
        # Register as buffers (move to device with model)
        self.register_buffer('D', torch.tensor(conductor_diameter))
        self.register_buffer('epsilon', torch.tensor(conductor_emissivity))
        self.register_buffer('alpha_s', torch.tensor(conductor_absorptivity))
        self.register_buffer('R_ref', torch.tensor(resistance_per_meter_25C))
        self.register_buffer('alpha_R', torch.tensor(temp_coeff_resistance))
        self.register_buffer('T_ref', torch.tensor(25.0))
        self.register_buffer('T_max', torch.tensor(max_conductor_temp))
        
        # Physical constants
        self.register_buffer('sigma', torch.tensor(5.67e-8))  # Stefan-Boltzmann W/(m²·K⁴)
        self.register_buffer('pi', torch.tensor(math.pi))
        
        # Air properties at ~40°C (representative)
        self.register_buffer('rho_air', torch.tensor(1.1))  # kg/m³
        self.register_buffer('mu_air', torch.tensor(1.96e-5))  # Pa·s dynamic viscosity
        self.register_buffer('k_air', torch.tensor(0.0277))  # W/(m·K) thermal conductivity
        # Typical Prandtl number for air (used in low-Re Nu correlation)
        self.register_buffer('Pr', torch.tensor(0.71))
        
    def resistance(self, T_conductor: torch.Tensor) -> torch.Tensor:
        """
        Temperature-dependent AC resistance per unit length (Ohm/m)
        
        R(T) = R_ref * [1 + α_R * (T - T_ref)]
        
        Args:
            T_conductor: Conductor temperature (°C)
            
        Returns:
            Resistance per meter (Ohm/m)
        """
        return self.R_ref * (1 + self.alpha_R * (T_conductor - self.T_ref))
    
    def joule_heating(
        self, 
        current: torch.Tensor, 
        T_conductor: torch.Tensor
    ) -> torch.Tensor:
        """
        Joule (I²R) heating per unit length (W/m)
        
        q_j = I² * R(T)
        
        Args:
            current: Line current (A)
            T_conductor: Conductor temperature (°C)
            
        Returns:
            Heat gain rate (W/m)
        """
        R = self.resistance(T_conductor)
        return current ** 2 * R
    
    def solar_heat_gain(
        self, 
        solar_irradiance: torch.Tensor,
        elevation_correction: float = 1.0,
    ) -> torch.Tensor:
        """
        Solar heat absorption per unit length (W/m)
        
        q_s = α_s * Q_s * D
        
        Args:
            solar_irradiance: Total solar irradiance (W/m²)
            elevation_correction: Altitude correction factor
            
        Returns:
            Heat gain rate (W/m)
        """
        return self.alpha_s * solar_irradiance * self.D * elevation_correction
    
    def reynolds_number(self, wind_speed: torch.Tensor) -> torch.Tensor:
        """
        Reynolds number for flow around conductor

        Re = ρ * V * D / μ

        Use a very small clamp for numerical stability so low-Re regimes
        (<1000) are preserved instead of being artificially bumped.
        """
        V_w = torch.clamp(wind_speed, min=1e-4)  # preserve low-Re behaviour
        return self.rho_air * V_w * self.D / self.mu_air
    
    def wind_direction_factor(
        self, 
        wind_angle: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Wind direction factor K_angle
        
        Perpendicular wind (90°) = 1.0 (maximum cooling)
        Parallel wind (0°) = ~0.4 (minimum cooling)
        
        IEEE 738 correlation:
        K_angle = 1.194 - cos(φ) + 0.194*cos(2φ) + 0.368*sin(2φ)
        
        Simplified version used here.
        """
        if wind_angle is None:
            return torch.tensor(1.0, device=self.D.device)
        
        angle_rad = wind_angle * self.pi / 180
        K_angle = 1.194 - torch.cos(angle_rad) + 0.194 * torch.cos(2 * angle_rad)
        return torch.clamp(K_angle, min=0.388, max=1.0)
    
    def convective_heat_loss_forced(
        self,
        T_conductor: torch.Tensor,
        T_ambient: torch.Tensor,
        wind_speed: torch.Tensor,
        wind_angle: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forced convection heat loss per unit length (W/m).

        Implements a piecewise Nusselt number correlation that matches IEEE
        guidance at low Reynolds numbers (Re < 1000) and retains the Morgan
        correlation in its validated range. This replaces the previous
        artificial wind-speed clamp so the forced-convective Nu behaves
        correctly across low‑wind regimes.

        Regions implemented:
        - Re < 1000 : low‑Re correlation (IEEE-like / Churchill form)
        - Re >= 1000: Morgan correlation (existing implementation)

        q_c = π * k_air * Nu * (T_c - T_a) * K_angle
        """
        # Use the raw wind speed (small clamp only for numerical stability)
        V_w = torch.clamp(wind_speed, min=1e-4)
        Re = self.reynolds_number(V_w)

        # Low-Re correlation (common IEEE/empirical form used in validations)
        # Nu_low = 0.3 + (0.62 * Re^0.5 * Pr^(1/3)) / (1 + (0.4/Pr)^(2/3))^0.25
        Pr = self.Pr
        Nu_low = 0.3 + (
            0.62 * torch.pow(Re, 0.5) * torch.pow(Pr, 1.0 / 3.0)
        ) / torch.pow((1.0 + torch.pow(0.4 / Pr, 2.0 / 3.0)), 0.25)

        # Morgan correlation (original implementation) for higher Re
        Nu_morgan = 0.583 * torch.pow(Re, 0.471)

        # Select Nu based on Reynolds number
        Nu = torch.where(Re < 1000.0, Nu_low, Nu_morgan)

        # Wind direction factor
        K_angle = self.wind_direction_factor(wind_angle)

        # Heat loss
        delta_T = T_conductor - T_ambient
        q_c = self.pi * self.k_air * Nu * delta_T * K_angle

        return q_c
    
    def convective_heat_loss_natural(
        self,
        T_conductor: torch.Tensor,
        T_ambient: torch.Tensor,
    ) -> torch.Tensor:
        """
        Natural convection heat loss per unit length (W/m)
        
        For low wind conditions, natural convection dominates.
        Using simplified correlation for horizontal cylinder.
        
        q_cn = 3.645 * ρ^0.5 * D^0.75 * (T_c - T_a)^1.25
        """
        delta_T = torch.clamp(T_conductor - T_ambient, min=0.1)
        
        q_cn = 3.645 * torch.sqrt(self.rho_air) * torch.pow(self.D, 0.75) * torch.pow(delta_T, 1.25)
        
        return q_cn
    
    def convective_heat_loss(
        self,
        T_conductor: torch.Tensor,
        T_ambient: torch.Tensor,
        wind_speed: torch.Tensor,
        wind_angle: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Combined convective heat loss (W/m).

        Uses the larger of forced or natural convection (consistent with IEEE
        practice). The forced-convection Nu now uses a low‑Re correlation so
        no artificial wind-speed clamp is required.
        """
        q_forced = self.convective_heat_loss_forced(
            T_conductor, T_ambient, wind_speed, wind_angle
        )
        q_natural = self.convective_heat_loss_natural(T_conductor, T_ambient)

        return torch.maximum(q_forced, q_natural)
    
    def radiative_heat_loss(
        self,
        T_conductor: torch.Tensor,
        T_ambient: torch.Tensor,
    ) -> torch.Tensor:
        """
        Radiative heat loss per unit length (W/m)
        
        Stefan-Boltzmann law:
        q_r = π * D * ε * σ * (T_c⁴ - T_a⁴)
        
        Note: Temperatures must be in Kelvin for Stefan-Boltzmann
        
        Args:
            T_conductor: Conductor temperature (°C)
            T_ambient: Ambient air temperature (°C)
            
        Returns:
            Heat loss rate (W/m)
        """
        # Convert to Kelvin
        T_c_K = T_conductor + 273.15
        T_a_K = T_ambient + 273.15
        
        q_r = self.pi * self.D * self.epsilon * self.sigma * (
            torch.pow(T_c_K, 4) - torch.pow(T_a_K, 4)
        )
        
        return q_r
    
    def heat_balance_residual(
        self,
        current: torch.Tensor,
        T_conductor: torch.Tensor,
        T_ambient: torch.Tensor,
        wind_speed: torch.Tensor,
        solar_irradiance: torch.Tensor,
        wind_angle: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Heat balance residual (should approach 0 at steady state)
        
        Residual = (q_c + q_r) - (q_s + I²R)
        
        Positive residual: cooling > heating → temperature should drop
        Negative residual: heating > cooling → temperature should rise
        
        At steady state: residual ≈ 0
        
        This is the core physics constraint used in the PINN loss function.
        
        Args:
            current: Line current (A)
            T_conductor: Conductor temperature (°C)
            T_ambient: Ambient temperature (°C)
            wind_speed: Wind speed (m/s)
            solar_irradiance: Solar irradiance (W/m²)
            wind_angle: Wind angle to conductor (degrees, optional)
            
        Returns:
            Heat balance residual (W/m) - should be near zero
        """
        # Heat gains
        q_joule = self.joule_heating(current, T_conductor)
        q_solar = self.solar_heat_gain(solar_irradiance)
        
        # Heat losses
        q_conv = self.convective_heat_loss(T_conductor, T_ambient, wind_speed, wind_angle)
        q_rad = self.radiative_heat_loss(T_conductor, T_ambient)
        
        # Residual
        residual = (q_conv + q_rad) - (q_solar + q_joule)
        
        return residual
    
    def steady_state_temperature(
        self,
        current: torch.Tensor,
        T_ambient: torch.Tensor,
        wind_speed: torch.Tensor,
        solar_irradiance: torch.Tensor,
        wind_angle: Optional[torch.Tensor] = None,
        max_iter: int = 50,
        tol: float = 0.1,
    ) -> torch.Tensor:
        """
        Solve for steady-state conductor temperature using Newton-Raphson
        
        This is used for generating ground-truth training data.
        Note: This method is NOT differentiable (uses iterative solve).
        
        Args:
            current: Line current (A)
            T_ambient: Ambient temperature (°C)
            wind_speed: Wind speed (m/s)
            solar_irradiance: Solar irradiance (W/m²)
            wind_angle: Wind angle (degrees, optional)
            max_iter: Maximum iterations
            tol: Convergence tolerance (°C)
            
        Returns:
            Steady-state conductor temperature (°C)
        """
        # Initial guess: ambient + 20°C
        T_c = T_ambient.clone() + 20.0
        
        for _ in range(max_iter):
            residual = self.heat_balance_residual(
                current, T_c, T_ambient, wind_speed, solar_irradiance, wind_angle
            )
            
            # Numerical derivative dR/dT
            dT = 0.1
            residual_plus = self.heat_balance_residual(
                current, T_c + dT, T_ambient, wind_speed, solar_irradiance, wind_angle
            )
            dR_dT = (residual_plus - residual) / dT
            
            # Newton-Raphson step
            T_c = T_c - residual / (dR_dT + 1e-8)
            
            # Clamp to reasonable range
            T_c = torch.clamp(T_c, min=T_ambient, max=self.T_max + 50)
            
            if torch.abs(residual).max() < tol:
                break
        
        return T_c
    
    def ampacity(
        self,
        T_max: float,
        T_ambient: torch.Tensor,
        wind_speed: torch.Tensor,
        solar_irradiance: torch.Tensor,
        wind_angle: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Calculate maximum allowable current (ampacity) for given conditions
        
        This is the Dynamic Line Rating (DLR) calculation.
        
        Solves: I = sqrt[(q_c + q_r - q_s) / R(T_max)]
        
        Args:
            T_max: Maximum allowable conductor temperature (°C)
            T_ambient: Ambient temperature (°C)
            wind_speed: Wind speed (m/s)
            solar_irradiance: Solar irradiance (W/m²)
            wind_angle: Wind angle (degrees, optional)
            
        Returns:
            Maximum allowable current (A)
        """
        # Handle both tensor and scalar T_max
        if isinstance(T_max, torch.Tensor):
            T_max_val = T_max.item()
        else:
            T_max_val = T_max
            
        T_c = torch.full_like(T_ambient, T_max_val)
        
        # Heat loss at max temperature
        q_conv = self.convective_heat_loss(T_c, T_ambient, wind_speed, wind_angle)
        q_rad = self.radiative_heat_loss(T_c, T_ambient)
        
        # Heat gain from solar
        q_solar = self.solar_heat_gain(solar_irradiance)
        
        # Resistance at max temperature
        R = self.resistance(T_c)
        
        # Available heat dissipation capacity
        q_available = q_conv + q_rad - q_solar
        q_available = torch.clamp(q_available, min=1.0)  # Prevent negative/zero
        
        # Maximum current: I = sqrt(q_available / R)
        I_max = torch.sqrt(q_available / R)
        
        return I_max
    
    def static_rating(
        self,
        T_max: float = 75.0,
        T_ambient: float = 35.0,
        wind_speed: float = 0.61,  # 2 ft/s per IEEE 738
        solar_irradiance: float = 1000.0,
    ) -> float:
        """
        Calculate conservative static rating (traditional method)
        
        Uses worst-case assumptions:
        - High ambient temperature (35°C default)
        - Low wind speed (0.61 m/s = 2 ft/s)
        - Full solar load (1000 W/m²)
        
        Returns:
            Static ampacity rating (A)
        """
        T_amb = torch.tensor([T_ambient])
        V_w = torch.tensor([wind_speed])
        Q_s = torch.tensor([solar_irradiance])
        
        return self.ampacity(T_max, T_amb, V_w, Q_s).item()


def physics_loss_fn(
    physics: IEEE738HeatBalance,
    pred_temperature: torch.Tensor,
    current: torch.Tensor,
    T_ambient: torch.Tensor,
    wind_speed: torch.Tensor,
    solar_irradiance: torch.Tensor,
    wind_angle: Optional[torch.Tensor] = None,
    reduction: str = 'mean',
) -> torch.Tensor:
    """
    Physics-informed loss function
    
    Penalizes violations of the IEEE 738 heat balance equation.
    The model learns to predict temperatures that satisfy physics.
    
    Args:
        physics: IEEE738HeatBalance instance
        pred_temperature: Model's temperature prediction (°C)
        current: Line current (A)
        T_ambient: Ambient temperature (°C)
        wind_speed: Wind speed (m/s)
        solar_irradiance: Solar irradiance (W/m²)
        wind_angle: Wind angle (degrees, optional)
        reduction: 'mean', 'sum', or 'none'
        
    Returns:
        Physics loss (MSE of heat balance residual)
    """
    residual = physics.heat_balance_residual(
        current=current,
        T_conductor=pred_temperature.squeeze(),
        T_ambient=T_ambient,
        wind_speed=wind_speed,
        solar_irradiance=solar_irradiance,
        wind_angle=wind_angle,
    )
    
    # Squared residual (heat balance violation)
    loss = residual ** 2
    
    if reduction == 'mean':
        return loss.mean()
    elif reduction == 'sum':
        return loss.sum()
    else:
        return loss


# Convenience function for quick testing
def test_physics():
    """Quick test of physics module"""
    physics = IEEE738HeatBalance()
    
    # Test conditions
    T_amb = torch.tensor([25.0])
    wind = torch.tensor([5.0])
    solar = torch.tensor([800.0])
    current = torch.tensor([800.0])
    
    # Calculate steady-state temperature
    T_c = physics.steady_state_temperature(current, T_amb, wind, solar)
    print(f"Steady-state temperature: {T_c.item():.1f}°C")
    
    # Calculate residual (should be near zero)
    residual = physics.heat_balance_residual(current, T_c, T_amb, wind, solar)
    print(f"Heat balance residual: {residual.item():.2f} W/m")
    
    # Calculate dynamic rating
    rating = physics.ampacity(75.0, T_amb, wind, solar)
    print(f"Dynamic rating (75°C limit): {rating.item():.0f} A")
    
    # Compare to static rating
    static = physics.static_rating()
    print(f"Static rating: {static:.0f} A")
    print(f"Capacity gain: {((rating.item() - static) / static * 100):.1f}%")


# Standalone functions for calibration (numpy-based for scipy optimization)
import numpy as np

def convective_heat_loss(T, T_amb, v_wind, diameter):
    """
    Convective heat loss (W/m)
    Simplified IEEE 738 forced convection approximation

    Args:
        T: Conductor temperature (°C)
        T_amb: Ambient temperature (°C)
        v_wind: Wind speed (m/s)
        diameter: Conductor diameter (m)

    Returns:
        Convective heat loss (W/m)
    """
    # Simplified correlation - use actual physics for production
    T_film = (T + T_amb) / 2 + 273.15  # Kelvin

    # Air properties (approximate)
    rho_air = 1.1  # kg/m³
    mu_air = 1.96e-5  # Pa·s
    k_air = 0.0277  # W/(m·K)
    Pr = 0.71

    # Reynolds number
    Re = rho_air * v_wind * diameter / mu_air
    Re = max(Re, 1e-4)  # Prevent division by zero

    # Nusselt number (simplified Morgan correlation)
    Nu = 0.583 * (Re ** 0.471)

    # Heat transfer coefficient
    h = Nu * k_air / diameter

    # Convective heat loss
    q_conv = np.pi * diameter * h * (T - T_amb)

    return q_conv


def radiative_heat_loss(T, T_amb, diameter, emissivity):
    """
    Radiative heat loss (W/m)
    Stefan-Boltzmann law

    Args:
        T: Conductor temperature (°C)
        T_amb: Ambient temperature (°C)
        diameter: Conductor diameter (m)
        emissivity: Surface emissivity (dimensionless)

    Returns:
        Radiative heat loss (W/m)
    """
    sigma = 5.670e-8  # Stefan-Boltzmann constant
    T_k = T + 273.15
    T_amb_k = T_amb + 273.15
    return sigma * emissivity * np.pi * diameter * (T_k**4 - T_amb_k**4)


def solar_heat_gain(solar_irradiance, diameter, absorptivity):
    """
    Solar heat gain (W/m)

    Args:
        solar_irradiance: Solar irradiance (W/m²)
        diameter: Conductor diameter (m)
        absorptivity: Surface absorptivity (dimensionless)

    Returns:
        Solar heat gain (W/m)
    """
    return absorptivity * solar_irradiance * np.pi * diameter


def resistive_heat_gain(I, T, R_20, alpha):
    """
    Resistive heating (W/m)
    I²R with temperature-dependent resistance

    Args:
        I: Current (A)
        T: Conductor temperature (°C)
        R_20: Resistance at 20°C (Ω/m)
        alpha: Temperature coefficient of resistance (1/°C)

    Returns:
        Resistive heat gain (W/m)
    """
    R = R_20 * (1 + alpha * (T - 20))
    return I**2 * R


def ieee738_temperature(I, T_amb, v_wind, solar, diameter=0.028, emissivity=0.8, absorptivity=0.8, R_20=7.28e-5, alpha=0.004):
    """
    Solve for steady-state conductor temperature using IEEE 738 heat balance

    Args:
        I: Current (A)
        T_amb: Ambient temperature (°C)
        v_wind: Wind speed (m/s)
        solar: Solar irradiance (W/m²)
        diameter: Conductor diameter (m)
        emissivity: Surface emissivity
        absorptivity: Surface absorptivity
        R_20: Resistance at 20°C (Ω/m)
        alpha: Temperature coefficient (1/°C)

    Returns:
        Steady-state conductor temperature (°C)
    """
    from scipy.optimize import fsolve

    def heat_balance(T):
        q_conv = convective_heat_loss(T, T_amb, v_wind, diameter)
        q_rad = radiative_heat_loss(T, T_amb, diameter, emissivity)
        q_solar = solar_heat_gain(solar, diameter, absorptivity)
        q_resist = resistive_heat_gain(I, T, R_20, alpha)

        return q_conv + q_rad - q_solar - q_resist

    # Initial guess: ambient + current-dependent rise
    T_guess = T_amb + I * 0.05

    # Solve for T where heat balance = 0
    T_solution = fsolve(heat_balance, T_guess)[0]

    return T_solution


def ieee738_ampacity(T_max, T_amb, v_wind, solar, diameter=0.028, emissivity=0.8, absorptivity=0.8, R_20=7.28e-5, alpha=0.004):
    """
    Calculate maximum allowable current (ampacity) for given conditions

    Solves: I = sqrt[(q_c + q_r - q_s) / R(T_max)]

    Args:
        T_max: Maximum allowable conductor temperature (°C)
        T_amb: Ambient temperature (°C)
        v_wind: Wind speed (m/s)
        solar: Solar irradiance (W/m²)
        diameter: Conductor diameter (m)
        emissivity: Emissivity coefficient
        absorptivity: Absorptivity coefficient
        R_20: Resistance at 20°C (Ohm/m)
        alpha: Temperature coefficient (1/°C)

    Returns:
        Maximum allowable current (A)
    """
    from scipy.optimize import fsolve

    def current_balance(I):
        # Heat losses at max temperature
        q_conv = convective_heat_loss(T_max, T_amb, v_wind, diameter)
        q_rad = radiative_heat_loss(T_max, T_amb, diameter, emissivity)
        q_solar = solar_heat_gain(solar, diameter, absorptivity)

        # Resistive heat generation
        R_T = R_20 * (1 + alpha * (T_max - 20))  # Resistance at T_max
        q_resist = I**2 * R_T

        return q_conv + q_rad - q_solar - q_resist

    # Initial guess based on typical ampacity
    I_guess = 1000  # A

    # Solve for I where heat balance = 0
    I_solution = fsolve(current_balance, I_guess)[0]

    return I_solution


def ieee738_analytical(T_amb, V_wind, Q_solar, I_load, T_max=None, **kwargs):
    """
    Analytical solution for IEEE 738 heat balance
    
    This is a differentiable approximation that can be used in neural networks.
    For exact solutions, use the IEEE738HeatBalance class.
    
    Args:
        T_amb: Ambient temperature (°C) - tensor
        V_wind: Wind speed (m/s) - tensor  
        Q_solar: Solar irradiance (W/m²) - tensor
        I_load: Current (A) - tensor
        T_max: Optional max temperature constraint
        
    Returns:
        T_conductor: Conductor temperature (°C) - tensor
        I_max: Maximum allowable current (A) - tensor (if T_max provided)
    """
    # Default conductor parameters (can be overridden via kwargs)
    diameter = kwargs.get('diameter', 0.02814)  # Drake ACSR
    emissivity = kwargs.get('emissivity', 0.8)
    absorptivity = kwargs.get('absorptivity', 0.8)
    R_20 = kwargs.get('R_20', 7.283e-5)  # Ohm/m at 20°C
    alpha = kwargs.get('alpha', 0.00403)  # 1/°C
    
    # Simplified analytical approximation
    # T_conductor ≈ T_amb + k * I² / (1 + V_wind^0.5)
    # where k is a calibrated heat transfer coefficient
    
    # Calibrated coefficients (from typical IEEE 738 results)
    k_resistive = 0.15  # °C / (A²/m) - resistive heating coefficient
    k_convective = 2.5  # Wind cooling coefficient
    
    # Resistive heating term
    q_resist = I_load**2 * R_20 * (1 + alpha * (25 - 20))  # Approximate at 25°C
    
    # Convective cooling (simplified)
    wind_factor = torch.clamp(V_wind, min=0.1)**0.5
    q_conv = k_convective * wind_factor * (diameter * torch.pi)
    
    # Solar gain
    q_solar = Q_solar * absorptivity * (diameter * torch.pi)
    
    # Radiative loss (approximate)
    T_approx = T_amb + 20  # Initial guess
    q_rad = emissivity * 5.67e-8 * (T_approx + 273)**4 * (diameter * torch.pi)
    
    # Heat balance: q_resist + q_solar = q_conv + q_rad
    # Solve for T_conductor
    net_heat = q_resist + q_solar - q_conv - q_rad * 0.1  # Simplified
    
    # Temperature rise proportional to net heat
    T_rise = net_heat * k_resistive
    T_conductor = T_amb + T_rise
    
    # Clamp to reasonable range
    T_conductor = torch.clamp(T_conductor, min=T_amb, max=T_amb + 100)
    
    if T_max is not None:
        # Calculate maximum current for given T_max
        # I_max² = (q_conv + q_rad - q_solar) / R(T_max)
        R_Tmax = R_20 * (1 + alpha * (T_max - 20))
        q_cool = k_convective * wind_factor * (diameter * torch.pi) + q_rad
        I_max = torch.sqrt(torch.clamp(q_cool - q_solar, min=0) / R_Tmax)
        return T_conductor, I_max
    
    return T_conductor, None


if __name__ == "__main__":
    test_physics()
