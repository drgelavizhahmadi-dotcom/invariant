# scripts/validate_mendeley_corrected.py
import pandas as pd
import numpy as np
import sys
sys.path.append('.')

# Load your existing physics functions
from core.physics import ieee738_temperature, ieee738_ampacity

# Load Mendeley data
df = pd.read_csv('data/mendeley/vietnam_220kv.csv', parse_dates=['datetime'])

# Column mapping (verify these match your CSV)
df = df.rename(columns={
    'temp': 'T_amb',           # Ambient temperature (°C)
    'Wind1': 'wind_speed',     # Wind speed (m/s)
    'WinDir': 'wind_angle',    # Wind direction (degrees)
    'GHI': 'solar',            # Global horizontal irradiance (W/m²)
    'Ampacity': 'dlr'          # Dynamic line rating (A) — this is our "current"
})

# Conductor parameters (Trang-Thap Cham 220kV line)
conductor_params = {
    'diameter': 0.0275,        # m (approx for ACSR 400mm²)
    'emissivity': 0.7,
    'absorptivity': 0.8,
    'R_20': 1.4085e-4, # Ω/m at 20°C
    'alpha': 0.0039 # per °C
}

# CRITICAL: The dataset gives DLR (safe current), not actual current
# For validation, we compute: "What conductor temp results from running DLR amps?"
# Then compare to typical max (75°C) — but we need measured temp for real validation

# Check if dataset has measured conductor temperature
print("Available columns:", df.columns.tolist())

# If 'conductor_temp' or similar exists:
if 'conductor_temp' in df.columns or 'T_cond' in df.columns:
    # Direct temperature validation
    T_measured = df.get('conductor_temp', df.get('T_cond'))

    predictions = []
    for idx, row in df.iterrows():
        T_pred = ieee738_temperature(
            I=row['dlr'],  # Use DLR as the current
            T_amb=row['T_amb'],
            v_wind=row['wind_speed'],
            solar=row['solar'],
            **conductor_params
        )
        predictions.append(T_pred)

    df['T_pred'] = predictions
    df['error'] = df['T_pred'] - T_measured
    mae_temp = df['error'].abs().mean()

    print(f"\nTemperature MAE: {mae_temp:.2f}°C")
    print(f"RMSE: {np.sqrt((df['error']**2).mean()):.2f}°C")

else:
    # No measured temp — validate ampacity calculation instead
    # Compute ampacity from weather, compare to published DLR

    predictions = []
    for idx, row in df.iterrows():
        dlr_pred = ieee738_ampacity(
            T_max=75,  # Standard max conductor temp
            T_amb=row['T_amb'],
            v_wind=row['wind_speed'],
            solar=row['solar'],
            **conductor_params
        )
        predictions.append(dlr_pred)

    df['dlr_pred'] = predictions
    df['error'] = df['dlr'] - df['dlr_pred']
    mae_amp = df['error'].abs().mean()

    print(f"\nAmpacity MAE: {mae_amp:.2f} A")
    print(f"Relative to mean DLR ({df['dlr'].mean():.0f}A): {100*mae_amp/df['dlr'].mean():.1f}%")

# Common output
print(f"\nDataset: {len(df)} records")
print(f"Date range: {df['datetime'].min()} to {df['datetime'].max()}")
print(f"Weather range: {df['T_amb'].min():.1f}°C to {df['T_amb'].max():.1f}°C")

# Save results
df.to_csv('mendeley_validation_results.csv', index=False)

# Plot
import matplotlib.pyplot as plt
fig, axes = plt.subplots(1, 2, figsize=(12,5))

if 'T_measured' in locals():
    axes[0].scatter(T_measured, df['T_pred'], alpha=0.1, s=1)
    axes[0].plot([0,150],[0,150],'r--')
    axes[0].set_xlabel('Measured Temp (°C)')
    axes[0].set_ylabel('Predicted Temp (°C)')
    axes[0].set_title(f'Temperature: MAE = {mae_temp:.1f}°C')
else:
    axes[0].scatter(df['dlr'], df['dlr_pred'], alpha=0.1, s=1)
    axes[0].plot([0,2000],[0,2000],'r--')
    axes[0].set_xlabel('Published DLR (A)')
    axes[0].set_ylabel('Physics DLR (A)')
    axes[0].set_title(f'Ampacity: MAE = {mae_amp:.1f}A')

axes[1].hist(df['error'], bins=100)
axes[1].set_xlabel('Error')
axes[1].set_ylabel('Frequency')

plt.tight_layout()
plt.savefig('mendeley_validation.png', dpi=150)
print("\nSaved: mendeley_validation.png")