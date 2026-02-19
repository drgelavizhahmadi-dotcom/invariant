#!/usr/bin/env python3
"""
Cross-Region Validation v3 - Ratio-Based Training with Physics-Informed Model

CRITICAL: Uses InvariantPIKANV2 (HWF-PIKAN) with CORRECT physics parameters
- emissivity = 0.7 (from DynaLiRD Table 2)
- absorptivity = 0.9 (from DynaLiRD Table 2)
- SLR = region-specific (850 for VN, 1000 for US)
"""

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from pathlib import Path
import json
import sys

sys.path.append(str(Path(__file__).parent.parent))

from models.invariant_pikan_v2 import InvariantPIKANV2
from scripts.evaluation_helpers import evaluate_ratio_model
from core.physics import IEEE738HeatBalance

# ============================================================================
# CONFIGURATION
# ============================================================================

UNIFIED_DATA_PATH = "data/processed/unified_dlr_training_NORMALIZED.h5"
TARGET_COL = 'target_ratio'  # DLR/SLR ratio (dimensionless)
FEATURE_COLS = ['temperature', 'wind_speed', 'solar_irradiance']

# Physics parameters from DynaLiRD Table 2
PHYSICS_PARAMS = {
    'emissivity': 0.7,      # From DynaLiRD Table 2
    'absorptivity': 0.9,    # From DynaLiRD Table 2
}

# SLR values (documented)
SLR_VALUES = {
    'VN': 850.0,  # DynaLiRD Table 2
    'US': 1000.0,  # NREL documentation
}

# Training config
BATCH_SIZE = 256
EPOCHS = 100
LEARNING_RATE = 1e-3
PHYSICS_WEIGHT = 0.1  # Weight for physics loss
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

RESULTS_DIR = Path("cross_region_results/NORMALIZED_v3")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 80)
print("CROSS-REGION VALIDATION v3 - PHYSICS-INFORMED RATIO TRAINING")
print("=" * 80)
print(f"\n🔬 Physics Parameters (DynaLiRD Table 2):")
print(f"   Emissivity:    {PHYSICS_PARAMS['emissivity']}")
print(f"   Absorptivity:  {PHYSICS_PARAMS['absorptivity']}")
print(f"\n📊 SLR Values:")
print(f"   VN_SLR: {SLR_VALUES['VN']:.0f} A")
print(f"   US_SLR: {SLR_VALUES['US']:.0f} A")
print(f"\n⚙️  Model: InvariantPIKANV2 (Hybrid Wavelet-Fourier PIKAN)")
print(f"   Dataset: {UNIFIED_DATA_PATH}")
print(f"   Device: {DEVICE}")

# ============================================================================
# LOAD DATA
# ============================================================================

print("\n" + "=" * 80)
print("LOADING DATA")
print("=" * 80)

df = pd.read_hdf(UNIFIED_DATA_PATH, key='data')
vn_df = df[df['region'] == 'VN'].copy()
us_df = df[df['region'] == 'US'].copy()

print(f"\n✅ Loaded:")
print(f"   VN: {len(vn_df):,} samples")
print(f"   US: {len(us_df):,} samples")

# ============================================================================
# PREPARE DATA
# ============================================================================

def prepare_data(df, feature_cols, target_col):
    """Prepare features and targets"""
    X = df[feature_cols].values.astype(np.float32)
    y = df[target_col].values.astype(np.float32)

    # Normalize features
    X_mean = X.mean(axis=0)
    X_std = X.std(axis=0)
    X_normalized = (X - X_mean) / (X_std + 1e-8)

    return X_normalized, y, X_mean, X_std

X_vn, y_vn, vn_mean, vn_std = prepare_data(vn_df, FEATURE_COLS, TARGET_COL)
X_us, y_us, us_mean, us_std = prepare_data(us_df, FEATURE_COLS, TARGET_COL)

print(f"\n✅ Data prepared:")
print(f"   VN: X={X_vn.shape}, y={y_vn.shape}")
print(f"   US: X={X_us.shape}, y={y_us.shape}")

# ============================================================================
# CREATE MODEL WITH CORRECT PHYSICS PARAMETERS
# ============================================================================

def create_model_with_physics(slr, emissivity=0.7, absorptivity=0.9):
    """
    Create InvariantPIKANV2 with correct physics parameters

    CRITICAL: Initialize with DynaLiRD Table 2 values, not defaults
    """
    model = InvariantPIKANV2(
        input_dim=3,  # temp, wind, solar
        hidden_dim=64,
        output_dim=1,  # predict ratio directly
        fourier_bands=16,
        wavelet_scales=8,
    ).to(DEVICE)

    # CRITICAL: Set physics parameters from DynaLiRD Table 2
    # Use inverse sigmoid to get raw values
    with torch.no_grad():
        # emissivity: sigmoid(raw) = target → raw = logit(target)
        model.physics_params['raw_emissivity'].fill_(
            torch.logit(torch.tensor(emissivity))
        )
        model.physics_params['raw_absorptivity'].fill_(
            torch.logit(torch.tensor(absorptivity))
        )

    # Store SLR for physics calculations
    model.slr = slr

    print(f"\n🔬 Model created with physics params:")
    print(f"   SLR: {slr:.0f} A")
    params = model.get_physics_params()
    print(f"   Emissivity: {params['emissivity'].item():.3f}")
    print(f"   Absorptivity: {params['absorptivity'].item():.3f}")

    return model

# ============================================================================
# TRAINING FUNCTION
# ============================================================================

def train_model(X_train, y_train, X_val, y_val, slr, region_name):
    """Train InvariantPIKANV2 on ratio targets"""

    model = create_model_with_physics(
        slr=slr,
        emissivity=PHYSICS_PARAMS['emissivity'],
        absorptivity=PHYSICS_PARAMS['absorptivity'],
    )

    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # Convert to tensors
    X_train_t = torch.FloatTensor(X_train).to(DEVICE)
    y_train_t = torch.FloatTensor(y_train).to(DEVICE)
    X_val_t = torch.FloatTensor(X_val).to(DEVICE)
    y_val_t = torch.FloatTensor(y_val).to(DEVICE)

    best_val_loss = float('inf')

    for epoch in range(EPOCHS):
        model.train()
        optimizer.zero_grad()

        # Forward pass - model predicts ratio directly
        # Note: InvariantPIKANV2 expects weather_dict, but we'll simplify
        # by just predicting from features
        h = model.embedding(X_train_t)
        predictions = model.kan(h)
        y_pred = predictions[:, 0]  # First output is ratio

        # MSE loss on ratios
        mse_loss = torch.mean((y_pred - y_train_t) ** 2)

        # Total loss (no physics loss for now - model is trained on ratios)
        loss = mse_loss

        # Backward
        loss.backward()
        optimizer.step()

        # Validation
        if (epoch + 1) % 10 == 0:
            model.eval()
            with torch.no_grad():
                h_val = model.embedding(X_val_t)
                pred_val = model.kan(h_val)
                y_val_pred = pred_val[:, 0]
                val_loss = torch.mean((y_val_pred - y_val_t) ** 2)

            print(f"  Epoch {epoch+1:3d}/{EPOCHS}: "
                  f"train_loss={mse_loss.item():.4f}, "
                  f"val_loss={val_loss.item():.4f}")

            if val_loss < best_val_loss:
                best_val_loss = val_loss

    return model

# ============================================================================
# EVALUATION FUNCTION
# ============================================================================

def evaluate_model(model, X_test, y_test_ratio, region, experiment_name):
    """Evaluate and convert to amps"""
    model.eval()
    with torch.no_grad():
        X_test_t = torch.FloatTensor(X_test).to(DEVICE)
        h = model.embedding(X_test_t)
        pred = model.kan(h)
        y_pred_ratio = pred[:, 0].cpu().numpy()

    metrics = evaluate_ratio_model(
        predictions=y_pred_ratio,
        actuals=y_test_ratio,
        region=region,
    )

    print(f"\n{'='*60}")
    print(f"{experiment_name}")
    print(f"{'='*60}")
    print(f"📊 Metrics in AMPS:")
    print(f"   MAE:  {metrics['mae_amps']:7.1f} A")
    print(f"   RMSE: {metrics['rmse_amps']:7.1f} A")
    print(f"   Bias: {metrics['bias_amps']:+7.1f} A")
    print(f"📐 Metrics in RATIO:")
    print(f"   MAE:  {metrics['mae_ratio']:.3f}")

    return metrics

# ============================================================================
# RUN 4 EXPERIMENTS
# ============================================================================

print("\n" + "=" * 80)
print("EXPERIMENT 1: VN_only")
print("=" * 80)

n_vn_train = int(0.8 * len(X_vn))
X_vn_train, X_vn_test = X_vn[:n_vn_train], X_vn[n_vn_train:]
y_vn_train, y_vn_test = y_vn[:n_vn_train], y_vn[n_vn_train:]

print(f"\n🔧 Training VN_only model (SLR={SLR_VALUES['VN']:.0f}A)...")
vn_only_model = train_model(X_vn_train, y_vn_train, X_vn_test, y_vn_test,
                             SLR_VALUES['VN'], 'VN')

vn_only_metrics = evaluate_model(vn_only_model, X_vn_test, y_vn_test,
                                  'VN', 'VN_only')

# ============================================================================

print("\n" + "=" * 80)
print("EXPERIMENT 2: US_only")
print("=" * 80)

n_us_train = int(0.8 * len(X_us))
X_us_train, X_us_test = X_us[:n_us_train], X_us[n_us_train:]
y_us_train, y_us_test = y_us[:n_us_train], y_us[n_us_train:]

print(f"\n🔧 Training US_only model (SLR={SLR_VALUES['US']:.0f}A)...")
us_only_model = train_model(X_us_train, y_us_train, X_us_test, y_us_test,
                             SLR_VALUES['US'], 'US')

us_only_metrics = evaluate_model(us_only_model, X_us_test, y_us_test,
                                  'US', 'US_only')

# ============================================================================

print("\n" + "=" * 80)
print("EXPERIMENT 3: VN→US")
print("=" * 80)

vn_to_us_metrics = evaluate_model(vn_only_model, X_us_test, y_us_test,
                                   'US', 'VN→US Transfer')

# ============================================================================

print("\n" + "=" * 80)
print("EXPERIMENT 4: US→VN")
print("=" * 80)

us_to_vn_metrics = evaluate_model(us_only_model, X_vn_test, y_vn_test,
                                   'VN', 'US→VN Transfer')

# ============================================================================
# SUMMARY
# ============================================================================

print("\n" + "=" * 80)
print("SUMMARY - CROSS-REGION TRANSFER RESULTS")
print("=" * 80)

results = pd.DataFrame([
    {
        'Experiment': 'VN_only',
        'Train': 'VN',
        'Test': 'VN',
        'MAE (A)': vn_only_metrics['mae_amps'],
        'RMSE (A)': vn_only_metrics['rmse_amps'],
        'Bias (A)': vn_only_metrics['bias_amps'],
        'MAE (ratio)': vn_only_metrics['mae_ratio'],
        'Degradation (%)': 0.0,
    },
    {
        'Experiment': 'US_only',
        'Train': 'US',
        'Test': 'US',
        'MAE (A)': us_only_metrics['mae_amps'],
        'RMSE (A)': us_only_metrics['rmse_amps'],
        'Bias (A)': us_only_metrics['bias_amps'],
        'MAE (ratio)': us_only_metrics['mae_ratio'],
        'Degradation (%)': 0.0,
    },
    {
        'Experiment': 'VN→US',
        'Train': 'VN',
        'Test': 'US',
        'MAE (A)': vn_to_us_metrics['mae_amps'],
        'RMSE (A)': vn_to_us_metrics['rmse_amps'],
        'Bias (A)': vn_to_us_metrics['bias_amps'],
        'MAE (ratio)': vn_to_us_metrics['mae_ratio'],
        'Degradation (%)': ((vn_to_us_metrics['mae_amps'] - us_only_metrics['mae_amps']) /
                            us_only_metrics['mae_amps'] * 100),
    },
    {
        'Experiment': 'US→VN',
        'Train': 'US',
        'Test': 'VN',
        'MAE (A)': us_to_vn_metrics['mae_amps'],
        'RMSE (A)': us_to_vn_metrics['rmse_amps'],
        'Bias (A)': us_to_vn_metrics['bias_amps'],
        'MAE (ratio)': us_to_vn_metrics['mae_ratio'],
        'Degradation (%)': ((us_to_vn_metrics['mae_amps'] - vn_only_metrics['mae_amps']) /
                            vn_only_metrics['mae_amps'] * 100),
    },
])

print("\n📊 RESULTS (in AMPS for paper):")
print(results.to_string(index=False, float_format=lambda x: f'{x:.1f}'))

# Save
results.to_csv(RESULTS_DIR / 'metrics_amps.csv', index=False)
results.to_json(RESULTS_DIR / 'metrics_amps.json', orient='records', indent=2)

print(f"\n✅ Results saved to {RESULTS_DIR}/")

# Key metrics
print("\n" + "=" * 80)
print("KEY METRICS")
print("=" * 80)

vn_base = results.loc[results['Experiment'] == 'VN_only', 'MAE (A)'].values[0]
us_base = results.loc[results['Experiment'] == 'US_only', 'MAE (A)'].values[0]
vn_to_us_mae = results.loc[results['Experiment'] == 'VN→US', 'MAE (A)'].values[0]
us_to_vn_mae = results.loc[results['Experiment'] == 'US→VN', 'MAE (A)'].values[0]

print(f"\n📋 Baseline:")
print(f"   VN_only: {vn_base:.1f} A")
print(f"   US_only: {us_base:.1f} A")

print(f"\n🔄 Cross-Region:")
print(f"   VN→US: {vn_to_us_mae:.1f} A ({((vn_to_us_mae-us_base)/us_base*100):+.1f}% vs baseline)")
print(f"   US→VN: {us_to_vn_mae:.1f} A ({((us_to_vn_mae-vn_base)/vn_base*100):+.1f}% vs baseline)")

print("\n" + "=" * 80)
