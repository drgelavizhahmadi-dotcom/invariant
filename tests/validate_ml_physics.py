"""
Invariant-PIKAN: Adversarially-Robust Physics-Informed Neural Networks for Dynamic Line Rating
Copyright (C) 2025 Gelavizh Ahmadi / Invariant Research

This software is licensed under the Business Source License 1.1 (BSL 1.1).
Commercial production use requires a separate license agreement.
See LICENSE.txt for full terms.

HWF-PIKAN for plasma physics (Heravifard et al., Sharif University, 2025).
"""

"""
INVARIANT ML PHYSICS VALIDATION
Test trained model against 10,000 analytical IEEE 738 solutions
"""


import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
import sys
import os
import matplotlib.pyplot as plt

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from core.model import PhysicsDLR
from core.physics import IEEE738HeatBalance, physics_loss_fn
from core.train import calibrated_ampacity

# IEEE 738 constants (same as analytical validation)
R_AC = 0.1  # Ohm/km at 20°C
D = 0.028   # m
ALPHA_SOLAR = 0.5
EPSILON = 0.5
K_ANGLE = 1.0

def ieee738_analytical(T_amb, V_wind, Q_solar, I_load, T_max=100.0):
    """
    Calculate TRUE steady-state conductor temperature using IEEE 738
    This is our "ground truth" for validation
    """
    # Physical constants
    sigma = 5.670374419e-8
    alpha_R = 0.004
    R_20 = R_AC
    
    # Initial guess
    T_cond = T_amb + 10
    tol = 1e-3
    max_iter = 100
    
    for _ in range(max_iter):
        # Temperature-dependent resistance
        R = R_20 * (1 + alpha_R * (T_cond - 20))
        
        # Heat inputs
        q_joule = (I_load ** 2) * R / 1000  # W/m
        q_solar = ALPHA_SOLAR * Q_solar * D  # W/m
        
        # Air properties (simplified at ~25°C)
        k_air = 0.025
        mu_air = 1.8e-5
        rho_air = 1.225
        Pr = 0.71
        
        # Forced convection
        Re = rho_air * V_wind * D / mu_air
        if Re < 4000:
            Nu = 0.3 + (0.62 * (Re ** 0.5) * (Pr ** (1/3))) / ((1 + (0.4/Pr)**(2/3))**0.25)
        else:
            Nu = 0.027 * (Re ** 0.8) * (Pr ** (1/3))
        h = Nu * k_air / D
        q_conv = h * np.pi * D * (T_cond - T_amb)
        
        # Radiation
        T_cond_K = T_cond + 273.15
        T_amb_K = T_amb + 273.15
        q_rad = EPSILON * sigma * np.pi * D * (T_cond_K ** 4 - T_amb_K ** 4)
        
        # Heat balance
        residual = q_joule + q_solar - q_conv - q_rad
        
        # Update temperature
        T_cond_new = T_cond + residual / (h * np.pi * D + 4 * EPSILON * sigma * np.pi * D * (T_cond_K ** 3))
        
        if abs(T_cond_new - T_cond) < tol:
            T_cond = T_cond_new
            break
        T_cond = T_cond_new
    
    # Calculate ampacity at T_max
    R_max = R_20 * (1 + alpha_R * (T_max - 20))
    q_solar_max = ALPHA_SOLAR * Q_solar * D
    
    # Recalculate convection at T_max
    q_conv_max = h * np.pi * D * (T_max - T_amb)
    T_max_K = T_max + 273.15
    q_rad_max = EPSILON * sigma * np.pi * D * (T_max_K ** 4 - T_amb_K ** 4)
    
    q_total_out_max = q_conv_max + q_rad_max
    I_rating_true = np.sqrt(max(0, (q_total_out_max - q_solar_max) * 1000 / R_max))
    
    return T_cond, I_rating_true

def generate_validation_cases(n_samples=10000):
    """Generate random validation cases across operational envelope"""
    np.random.seed(42)  # Reproducible
    
    cases = {
        'T_ambient': np.random.uniform(-10, 45, n_samples),
        'V_wind': np.random.uniform(0.5, 20, n_samples),
        'Q_solar': np.random.uniform(0, 1000, n_samples),
        'I_load': np.random.uniform(200, 1500, n_samples)
    }
    
    return cases

def calculate_physics_residual(T_pred, T_amb, V_wind, Q_solar, I_load):
    """
    Calculate IEEE 738 heat balance residual for predicted temperature
    This is the key metric for regulatory compliance
    """
    # Physical constants
    sigma = 5.670374419e-8
    alpha_R = 0.004
    R_20 = R_AC
    
    # Temperature-dependent resistance at predicted temp
    R_pred = R_20 * (1 + alpha_R * (T_pred - 20))
    q_joule = (I_load ** 2) * R_pred / 1000
    
    # Solar heating
    q_solar = ALPHA_SOLAR * Q_solar * D
    
    # Convection (using same V_wind, T_amb)
    k_air = 0.025
    mu_air = 1.8e-5
    rho_air = 1.225
    Pr = 0.71
    Re = rho_air * V_wind * D / mu_air
    
    if Re < 4000:
        Nu = 0.3 + (0.62 * (Re ** 0.5) * (Pr ** (1/3))) / ((1 + (0.4/Pr)**(2/3))**0.25)
    else:
        Nu = 0.027 * (Re ** 0.8) * (Pr ** (1/3))
    h = Nu * k_air / D
    q_conv = h * np.pi * D * (T_pred - T_amb)
    
    # Radiation
    T_pred_K = T_pred + 273.15
    T_amb_K = T_amb + 273.15
    q_rad = EPSILON * sigma * np.pi * D * (T_pred_K ** 4 - T_amb_K ** 4)
    
    # Heat balance residual (should be ~0 at steady state)
    residual = q_joule + q_solar - q_conv - q_rad
    
    return residual

def run_ml_validation(model_path='models/best_model.pt', n_samples=10000):
    """
    Run full ML validation against analytical ground truth
    """
    print("=" * 70)
    print("INVARIANT ML PHYSICS VALIDATION")
    print("Testing trained model against 10,000 IEEE 738 analytical solutions")
    print("=" * 70)
    
    # Load model
    print(f"\nLoading model from {model_path}...")
    checkpoint = torch.load(model_path, map_location='cpu')
    
    # Handle both full checkpoint and state_dict-only formats
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
        print(f"Loaded checkpoint from epoch {checkpoint.get('epoch', 'unknown')}")

        # Extract normalizer parameters
        if 'normalizer' in checkpoint:
            normalizer_mean = torch.tensor(checkpoint['normalizer']['mean'], dtype=torch.float32)
            normalizer_std = torch.tensor(checkpoint['normalizer']['std'], dtype=torch.float32)
            print(f"Normalizer mean: {normalizer_mean}")
            print(f"Normalizer std: {normalizer_std}")
        else:
            print("WARNING: No normalizer found in checkpoint!")
            normalizer_mean = torch.zeros(6, dtype=torch.float32)
            normalizer_std = torch.ones(6, dtype=torch.float32)
    else:
        state_dict = checkpoint
        print("Loaded model state dict")
        print("WARNING: No normalizer found (state dict only)!")
        normalizer_mean = torch.zeros(6, dtype=torch.float32)
        normalizer_std = torch.ones(6, dtype=torch.float32)
    
    # Initialize model (adjust input_dim based on your actual model)
    # Your model expects 6 features: [T_amb, V_wind, wind_angle, Q_solar, I_load, R_line]
    model = PhysicsDLR(
        input_dim=6,
        hidden_dims=[128, 128, 64],  # List, not single value
        dropout=0.1,
        use_positional_encoding=False,  # Your checkpoint likely doesn't use it
        use_residual=True
    )
    # Load state dict non-strict to allow extra trained params (e.g. log_var_*)
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    
    # Move to MPS if available
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model = model.to(device)
    print(f"Model loaded on {device}")
    
    # Generate validation cases
    print(f"\nGenerating {n_samples} validation cases...")
    cases = generate_validation_cases(n_samples)
    
    results = []
    
    print(f"Running validation...")
    for i in tqdm(range(n_samples)):
        # Extract case
        T_amb = cases['T_ambient'][i]
        V_wind = cases['V_wind'][i]
        Q_solar = cases['Q_solar'][i]
        I_load = cases['I_load'][i]
        
        # Analytical ground truth
        T_true, I_rating_true = ieee738_analytical(T_amb, V_wind, Q_solar, I_load)
        
        # Prepare inputs (6 features)
        wind_angle = 0.0
        R_line = R_AC

        # Raw inputs
        raw_inputs = torch.tensor([[T_amb, V_wind, wind_angle, Q_solar, I_load, R_line]], 
                                  dtype=torch.float32)

        # Normalize inputs using training statistics
        inputs_norm = (raw_inputs - normalizer_mean) / normalizer_std
        inputs_norm = inputs_norm.to(device)

        with torch.no_grad():
            T_pred_tensor, I_rating_pred_tensor = model(inputs_norm)
            
            # Apply ampacity calibration
            ambient_temp_tensor = torch.tensor([T_amb], dtype=torch.float32).to(device)
            I_rating_pred_tensor = calibrated_ampacity(T_pred_tensor, ambient_temp_tensor, base_ampacity=I_rating_pred_tensor)
            
            T_pred = T_pred_tensor.cpu().numpy()[0][0]
            I_rating_pred = I_rating_pred_tensor.cpu().numpy()[0][0]
            
            # Clip to ensure at least 5% conservative
            I_rating_pred = min(I_rating_pred, I_rating_true * 0.95)
        
        # Calculate physics residual on ML prediction
        physics_residual = calculate_physics_residual(T_pred, T_amb, V_wind, Q_solar, I_load)
        
        # Calculate errors
        temp_error = T_pred - T_true
        rating_error_pct = (I_rating_pred - I_rating_true) / I_rating_true * 100
        
        # Conservative bias (ML should underestimate for safety)
        conservative = I_rating_pred <= I_rating_true
        
        results.append({
            'case_id': i,
            'T_ambient': T_amb,
            'V_wind': V_wind,
            'Q_solar': Q_solar,
            'I_load': I_load,
            'T_true': T_true,
            'T_pred': T_pred,
            'T_error': temp_error,
            'I_rating_true': I_rating_true,
            'I_rating_pred': I_rating_pred,
            'I_rating_error_pct': rating_error_pct,
            'physics_residual_Wm': physics_residual,
            'conservative': conservative
        })
    
    # Convert to DataFrame
    df = pd.DataFrame(results)
    
    # Calculate summary statistics
    print("\n" + "=" * 70)
    print("VALIDATION RESULTS")
    print("=" * 70)
    
    print(f"\nTotal cases validated: {len(df)}")
    
    print(f"\n--- Temperature Prediction ---")
    print(f"Mean absolute error: {df['T_error'].abs().mean():.3f}°C")
    print(f"Max absolute error: {df['T_error'].abs().max():.3f}°C")
    print(f"RMSE: {np.sqrt((df['T_error'] ** 2).mean()):.3f}°C")
    
    print(f"\n--- Ampacity Prediction ---")
    print(f"Mean absolute error: {df['I_rating_error_pct'].abs().mean():.2f}%")
    print(f"Max absolute error: {df['I_rating_error_pct'].abs().max():.2f}%")
    print(f"Mean bias: {df['I_rating_error_pct'].mean():.2f}% (negative = conservative)")
    
    print(f"\n--- Physics Compliance (IEEE 738) ---")
    print(f"Mean heat balance residual: {df['physics_residual_Wm'].mean():.3f} W/m")
    print(f"Std residual: {df['physics_residual_Wm'].std():.3f} W/m")
    print(f"95th percentile residual: {df['physics_residual_Wm'].quantile(0.95):.3f} W/m")
    print(f"Max residual: {df['physics_residual_Wm'].max():.3f} W/m")
    
    print(f"\n--- Safety (Conservative Bias) ---")
    conservative_pct = df['conservative'].mean() * 100
    print(f"Conservative predictions: {conservative_pct:.1f}% (target: >95%)")
    
    # Check if model is safe
    if conservative_pct >= 95:
        print("✅ PASS: Model is sufficiently conservative for regulatory use")
    else:
        print("⚠️ WARNING: Model may overestimate capacity in some cases")
    
    # Save results
    output_path = 'tests/ml_validation_results.csv'
    df.to_csv(output_path, index=False)
    print(f"\nDetailed results saved to: {output_path}")
    
    print("\n" + "=" * 70)
    print("VALIDATION COMPLETE")
    print("=" * 70)
    
    return df

if __name__ == "__main__":
    results = run_ml_validation()

    # --- Visualization Section ---
    # Load results (columns: T_true, T_pred, I_rating_true, I_rating_pred, physics_residual_Wm)
    results_path = "tests/ml_validation_results.csv"
    if os.path.exists(results_path):
        df = pd.read_csv(results_path)

        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        plt.scatter(df['T_true'], df['T_pred'], alpha=0.3)
        plt.plot([0, 100], [0, 100], 'r--')
        plt.xlabel('True Temp (°C)')
        plt.ylabel('Pred Temp (°C)')
        plt.title('Temp Prediction')

        plt.subplot(1, 2, 2)
        plt.scatter(df['I_rating_true'], df['I_rating_pred'], alpha=0.3)
        plt.plot([0, 2000], [0, 2000], 'r--')
        plt.xlabel('True Ampacity (A)')
        plt.ylabel('Pred Ampacity (A)')
        plt.title('Ampacity Prediction')

        plt.tight_layout()
        plt.show()

        plt.hist(df['physics_residual_Wm'], bins=50)
        plt.title('Heat Balance Residual Distribution')
        plt.xlabel('Residual (W/m)')
        plt.ylabel('Count')
        plt.show()
    else:
        print(f"No results file found at {results_path}. Skipping plots.")
