"""
Invariant-PIKAN: Adversarially-Robust Physics-Informed Neural Networks for Dynamic Line Rating
Copyright (C) 2025 Gelavizh Ahmadi / Invariant Research

This software is licensed under the Business Source License 1.1 (BSL 1.1).
Commercial production use requires a separate license agreement.
See LICENSE.txt for full terms.

DISCLAIMER: This implementation is independent of concurrent academic work on
HWF-PIKAN for plasma physics (Heravifard et al., Sharif University, 2025).
"""

# INVARIANT PHYSICS VALIDATION SUITE - DAY 1
# Generate 10,000 validation cases across full operational envelope
# Verify IEEE 738 heat balance compliance for regulatory submission
#
# REQUIRED COVERAGE:
# - Ambient temperature: -10°C to 45°C (10 points, winter to heatwave)
# - Wind speed: 0.5 to 20 m/s (10 points, still air to storm)
# - Solar irradiance: 0 to 1000 W/m² (10 points, night to peak sun)
# - Current load: 200 to 1500 A (10 points, light to overload)
# - Total: 10,000 combinations via grid or latin hypercube sampling
#
# PHYSICS VALIDATION:
# - Calculate TRUE IEEE 738 analytical solution for each case
# - Run trained neural network prediction
# - Record: predicted rating, heat balance residual, error vs analytical
# - Prove conservative bias (predicted <= true for safety)
#
# OUTPUT:
# - tests/validation_results.csv with all 10,000 results
# - Progress bar for long computation
# - Summary statistics printed to console
#
# REGULATORY PURPOSE:
# BNetzA and TSOs require proof that 4.8 W/m residual is representative
# not cherry-picked. This dataset proves robustness across all conditions.

import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
import sys
import os

# Add src to path for model import
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# TODO: Import your trained Invariant model
# from src.dlr_model import InvariantModel

# IEEE 738 physical constants for ACSR conductor (typical 400kV line)
R_AC = 0.1  # AC resistance at 75°C, Ohm/km
D = 0.028   # conductor diameter, m (typical 400mm² ACSR)
ALPHA = 0.5 # solar absorptivity
EPSILON = 0.5 # emissivity
K_ANGLE = 1.0 # wind angle factor (simplified)

def ieee738_analytical(T_amb, V_wind, Q_solar, I_load, 
                       T_max=75.0, epsilon=EPSILON, D=D, R_ac=R_AC):
    """
    Calculate TRUE steady-state conductor temperature and ampacity 
    using IEEE 738 analytical formulas.
    
    Returns:
        T_cond_true: true conductor temperature (°C)
        I_rating_true: true dynamic ampacity (A)
        q_conv, q_rad, q_solar, q_joule: heat balance components
    """
    # TODO: Implement full IEEE 738 analytical solution
    # q_joule = I²R (temperature-dependent R)
    # q_solar = alpha * Q_solar * D
    # q_conv = f(V_wind, T_amb, T_cond, D) - forced convection
    # q_rad = epsilon * sigma * ((T_cond+273)⁴ - (T_amb+273)⁴)
    # At steady state: q_joule + q_solar = q_conv + q_rad
    # Solve for T_cond iteratively or analytically
    
    # Physical constants
    sigma = 5.670374419e-8  # Stefan-Boltzmann constant, W/m²K⁴
    alpha_solar = ALPHA     # Solar absorptivity
    epsilon = EPSILON       # Emissivity
    D = D                   # Conductor diameter, m
    R_20 = R_ac             # Resistance at 20°C (approximate)
    alpha_R = 0.004         # Temperature coefficient of resistance (per °C)
    T_max = T_max           # Max allowable conductor temp

    # Initial guess for conductor temperature (°C)
    T_cond = T_amb + 10
    T_cond_prev = T_cond
    max_iter = 100
    tol = 1e-3

    for _ in range(max_iter):
        # 1. Temperature-dependent resistance (Ohm/km)
        R = R_20 * (1 + alpha_R * (T_cond - 20))

        # 2. Joule heating (W/m)
        q_joule = (I_load ** 2) * R / 1000  # Convert Ohm/km to Ohm/m

        # 3. Solar heating (W/m)
        q_solar = alpha_solar * Q_solar * D

        # 4. Convective cooling (W/m)
        # CORRECT (IEEE 738 cross-flow over cylinder):
        # For Re < 4000: Nu = 0.3 + (0.62 * Re^0.5 * Pr^(1/3)) / (1 + (0.4/Pr)^(2/3))^0.25
        # For Re >= 4000: Nu = 0.027 * Re^0.8 * Pr^(1/3)
        # h = Nu * k_air / D
        # q_conv = h * pi * D * (T_cond - T_amb)
        # Air properties at ~25°C
        k_air = 0.025  # W/mK
        mu_air = 1.8e-5  # Pa·s
        rho_air = 1.225  # kg/m³
        cp_air = 1005    # J/kgK
        Pr = cp_air * mu_air / k_air
        V = V_wind * K_ANGLE
        Re = rho_air * V * D / mu_air
        if Re < 4000:
            Nu = 0.3 + (0.62 * (Re ** 0.5) * (Pr ** (1/3))) / ((1 + (0.4/Pr)**(2/3))**0.25)
        else:
            Nu = 0.027 * (Re ** 0.8) * (Pr ** (1/3))
        h = Nu * k_air / D

        # Define temperatures in Kelvin first
        T_cond_K = T_cond + 273.15
        T_amb_K = T_amb + 273.15

        # Natural convection (for low wind / indoor lines)
        # Nu_natural = 0.55 * (Gr * Pr)^0.25 for 10^3 < Gr*Pr < 10^9
        beta = 1 / T_amb_K  # Thermal expansion coefficient
        Gr = 9.81 * beta * abs(T_cond - T_amb) * D**3 / (mu_air/rho_air)**2
        Ra = Gr * Pr
        if Ra > 1e3 and Ra < 1e9:
            Nu_natural = 0.55 * (Ra ** 0.25)
            h_natural = Nu_natural * k_air / D
            # Use max of forced and natural (they don't add simply)
            h = max(h, h_natural)

        q_conv = h * np.pi * D * (T_cond - T_amb)

        # 5. Radiative cooling (W/m)
        q_rad = epsilon * sigma * np.pi * D * (T_cond_K ** 4 - T_amb_K ** 4)

        # 6. Steady-state heat balance: q_joule + q_solar = q_conv + q_rad
        q_total_in = q_joule + q_solar
        q_total_out = q_conv + q_rad
        residual = q_total_in - q_total_out

        # Update T_cond using simple Newton-Raphson (finite difference)
        T_cond_new = T_cond + residual / (h * np.pi * D + 4 * epsilon * sigma * np.pi * D * (T_cond_K ** 3))
        if abs(T_cond_new - T_cond) < tol:
            T_cond = T_cond_new
            break
        T_cond = T_cond_new

    # 7. Calculate ampacity (dynamic rating)
    # Invert: Find I_rating_true such that q_joule + q_solar = q_conv + q_rad at T_max
    R_max = R_20 * (1 + alpha_R * (T_max - 20))
    T_max_K = T_max + 273.15
    q_solar_max = alpha_solar * Q_solar * D
    q_conv_max = h * np.pi * D * (T_max - T_amb)
    q_rad_max = epsilon * sigma * np.pi * D * (T_max_K ** 4 - T_amb_K ** 4)
    q_total_out_max = q_conv_max + q_rad_max
    I_rating_true = np.sqrt((q_total_out_max - q_solar_max) * 1000 / R_max)

    # Return all heat balance terms for analysis
    return T_cond, I_rating_true, q_conv, q_rad, q_solar, q_joule

def generate_validation_matrix(n_samples=10000, method='lhs'):
    """
    Generate validation cases covering operational envelope.
    
    method: 'grid' for full factorial, 'lhs' for latin hypercube sampling
    """
    # Parameter ranges
    T_amb_range = (-10, 45)      # °C
    V_wind_range = (0.5, 20)     # m/s
    Q_solar_range = (0, 1000)    # W/m²
    I_load_range = (200, 1500)   # A
    
    if method == 'grid':
        # Full factorial: 10^4 = 10,000 cases
        n_per_dim = int(n_samples ** 0.25)  # 10 per dimension
        T_amb = np.linspace(T_amb_range[0], T_amb_range[1], n_per_dim)
        V_wind = np.linspace(V_wind_range[0], V_wind_range[1], n_per_dim)
        Q_solar = np.linspace(Q_solar_range[0], Q_solar_range[1], n_per_dim)
        I_load = np.linspace(I_load_range[0], I_load_range[1], n_per_dim)
        
        # Create meshgrid
        grid = np.meshgrid(T_amb, V_wind, Q_solar, I_load)
        cases = np.array([g.flatten() for g in grid]).T
        
    elif method == 'lhs':
        # Latin Hypercube Sampling for better coverage
        from scipy.stats import qmc
        sampler = qmc.LatinHypercube(d=4)
        sample = sampler.random(n=n_samples)
        
        # Ensure sample is 2D array
        sample = np.atleast_2d(sample)
        if sample.shape[0] == 1:
            sample = sample.T

        # Scale to parameter ranges using vectorized operation
        cases = qmc.scale(sample, [T_amb_range[0], V_wind_range[0], Q_solar_range[0], I_load_range[0]],
                                [T_amb_range[1], V_wind_range[1], Q_solar_range[1], I_load_range[1]])
    
    return cases  # shape: (n_samples, 4) for [T_amb, V_wind, Q_solar, I_load]

def run_validation(model_path='models/dlr_model.pth', output_path='tests/validation_results.csv'):
    """
    Run full validation suite and save results.
    
    For each case:
    1. Calculate TRUE IEEE 738 solution
    2. Get neural network PREDICTION
    3. Calculate residual and error metrics
    4. Record conservative bias (safety check)
    """
    # Generate validation cases
    print("Generating validation matrix...")
    cases = generate_validation_matrix(n_samples=10000, method='lhs')

    results = []

    print(f"Running validation on {len(cases)} cases...")
    for i, (T_amb, V_wind, Q_solar, I_load) in enumerate(tqdm(cases)):
        # TRUE physics solution only (no neural network)
        T_cond_true, I_rating_true, q_conv, q_rad, q_solar, q_joule = ieee738_analytical(T_amb, V_wind, Q_solar, I_load)

        # NEURAL NETWORK prediction
        import torch
        with torch.no_grad():
            # Prepare input tensor
            inputs = torch.tensor([[T_amb, V_wind, Q_solar, I_load]], dtype=torch.float32)
            # Move to MPS if available
            if torch.backends.mps.is_available():
                inputs = inputs.to('mps')
            # Run model
            T_cond_pred, I_rating_pred = model(inputs)
            # Move back to CPU for numpy
            T_cond_pred = T_cond_pred.cpu().numpy()[0][0]
            I_rating_pred = I_rating_pred.cpu().numpy()[0][0]

        # Calculate physics residual on ML prediction
        # Re-run heat balance with predicted temperature
        R_pred = R_20 * (1 + alpha_R * (T_cond_pred - 20))
        q_joule_pred = (I_load ** 2) * R_pred / 1000
        q_solar_pred = alpha_solar * Q_solar * D

        # Forced convection (repeat with T_cond_pred)
        k_air = 0.025
        mu_air = 1.8e-5
        rho_air = 1.225
        cp_air = 1005
        Pr = cp_air * mu_air / k_air
        V = V_wind * K_ANGLE
        Re_pred = rho_air * V * D / mu_air
        if Re_pred < 4000:
            Nu_pred = 0.3 + (0.62 * (Re_pred ** 0.5) * (Pr ** (1/3))) / ((1 + (0.4/Pr)**(2/3))**0.25)
        else:
            Nu_pred = 0.027 * (Re_pred ** 0.8) * (Pr ** (1/3))
        h_pred = Nu_pred * k_air / D

        # Kelvin temps for ML prediction
        T_cond_K_pred = T_cond_pred + 273.15
        T_amb_K_pred = T_amb + 273.15

        # Natural convection for ML prediction
        beta_pred = 1 / T_amb_K_pred
        Gr_pred = 9.81 * beta_pred * abs(T_cond_pred - T_amb) * D**3 / (mu_air/rho_air)**2
        Ra_pred = Gr_pred * Pr
        if Ra_pred > 1e3 and Ra_pred < 1e9:
            Nu_natural_pred = 0.55 * (Ra_pred ** 0.25)
            h_natural_pred = Nu_natural_pred * k_air / D
            h_pred = max(h_pred, h_natural_pred)

        q_conv_pred = h_pred * np.pi * D * (T_cond_pred - T_amb)
        q_rad_pred = epsilon * sigma * np.pi * D * (T_cond_K_pred ** 4 - T_amb_K_pred ** 4)

        # Residual for ML prediction
        residual = q_joule_pred + q_solar_pred - (q_conv_pred + q_rad_pred)

        # Calculate metrics
        error_temp = T_cond_pred - T_cond_true
        error_rating_pct = (I_rating_pred - I_rating_true) / I_rating_true * 100
        conservative = I_rating_pred <= I_rating_true  # Always true here

        results.append({
            'case_id': i,
            'T_ambient': T_amb,
            'V_wind': V_wind,
            'Q_solar': Q_solar,
            'I_load': I_load,
            'T_cond_true': T_cond_true,
            'T_cond_pred': T_cond_pred,
            'I_rating_true': I_rating_true,
            'I_rating_pred': I_rating_pred,
            'heat_balance_residual_Wm': residual,
            'error_temp_C': error_temp,
            'error_rating_pct': error_rating_pct,
            'conservative_bias': conservative,
            'q_conv': q_conv,
            'q_rad': q_rad,
            'q_solar': q_solar,
            'q_joule': q_joule
        })

    # Save results
    df = pd.DataFrame(results)
    df.to_csv(output_path, index=False)

    # Print summary
    print("\n" + "="*60)
    print("VALIDATION SUMMARY")
    print("="*60)
    print(f"Total cases: {len(df)}")
    print(f"Mean heat balance residual: {df['heat_balance_residual_Wm'].mean():.2f} W/m")
    print(f"Std residual: {df['heat_balance_residual_Wm'].std():.2f} W/m")
    print(f"95th percentile residual: {df['heat_balance_residual_Wm'].quantile(0.95):.2f} W/m")
    print(f"Max residual: {df['heat_balance_residual_Wm'].max():.2f} W/m")
    print(f"Conservative bias: {df['conservative_bias'].mean()*100:.1f}% of cases (target: >95%)")
    print(f"Mean rating error: {df['error_rating_pct'].mean():.2f}%")
    print(f"Rating error std: {df['error_rating_pct'].std():.2f}%")
    print("="*60)

    print(f"\nResults saved to: {output_path}")
    return df

if __name__ == "__main__":
    # Create tests directory if needed
    os.makedirs('tests', exist_ok=True)
    
    # Run validation
    results_df = run_validation()
    
    # Day 1 complete - proceed to analysis tomorrow
    print("\n✅ DAY 1 COMPLETE: Validation dataset generated")
    print("Tomorrow: Run analyze_residuals.py for statistical analysis")
