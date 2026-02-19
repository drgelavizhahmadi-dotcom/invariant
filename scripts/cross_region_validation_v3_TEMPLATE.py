#!/usr/bin/env python3
"""
Cross-Region Validation v3 - TEMPLATE with Ratio-Based Training

CRITICAL CHANGES FROM v2:
1. ✅ Train on 'target_ratio' (dimensionless DLR/SLR ratio)
2. ✅ Predict ratio (dimensionless)
3. ✅ Convert predictions to amps for reporting
4. ✅ Report MAE in amps (interpretable) and ratio (for completeness)

This template shows how to modify cross_region_validation_v2.py to:
- Use normalized dataset (target_ratio column)
- Predict dimensionless ratios
- Convert back to amps for evaluation
- Report metrics in both units
"""

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from pathlib import Path
import json
import sys

sys.path.append(str(Path(__file__).parent.parent))

from core.model import SimpleDLRModel
from scripts.evaluation_helpers import evaluate_ratio_model, compare_cross_region_transfer

# ============================================================================
# CONFIGURATION - UPDATED FOR RATIO-BASED TRAINING
# ============================================================================

# CHANGE 1: Use normalized dataset
UNIFIED_DATA_PATH = "data/processed/unified_dlr_training_NORMALIZED.h5"

# CHANGE 2: Target column is now 'target_ratio' (not 'actual' or 'Ampacity')
TARGET_COL = 'target_ratio'  # DLR/SLR ratio (dimensionless)

# Feature columns (unchanged)
FEATURE_COLS = ['temperature', 'wind_speed', 'solar_irradiance']

# Training config (unchanged)
BATCH_SIZE = 256
EPOCHS = 100
LEARNING_RATE = 1e-3
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Output directory
RESULTS_DIR = Path("cross_region_results/NORMALIZED_v3")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 80)
print("CROSS-REGION VALIDATION v3 - RATIO-BASED TRAINING")
print("=" * 80)
print(f"\n📊 Configuration:")
print(f"   Dataset:      {UNIFIED_DATA_PATH}")
print(f"   Target:       {TARGET_COL} (DLR/SLR ratio)")
print(f"   Features:     {FEATURE_COLS}")
print(f"   Device:       {DEVICE}")
print(f"   Results dir:  {RESULTS_DIR}")

# ============================================================================
# LOAD NORMALIZED DATA
# ============================================================================

print("\n" + "=" * 80)
print("LOADING NORMALIZED DATA")
print("=" * 80)

df = pd.read_hdf(UNIFIED_DATA_PATH, key='data')

# Split by region
vn_df = df[df['region'] == 'VN'].copy()
us_df = df[df['region'] == 'US'].copy()

print(f"\n✅ Loaded normalized dataset:")
print(f"   VN samples: {len(vn_df):,}")
print(f"   US samples: {len(us_df):,}")
print(f"   Total:      {len(df):,}")

# Verify target column
print(f"\n📊 Target statistics (ratio units):")
print(f"   VN {TARGET_COL}: {vn_df[TARGET_COL].mean():.3f} ± {vn_df[TARGET_COL].std():.3f}")
print(f"   US {TARGET_COL}: {us_df[TARGET_COL].mean():.3f} ± {us_df[TARGET_COL].std():.3f}")

# ============================================================================
# PREPARE DATASETS
# ============================================================================

def prepare_data(df, feature_cols, target_col):
    """Prepare features and targets for training/evaluation"""
    X = df[feature_cols].values.astype(np.float32)
    y = df[target_col].values.astype(np.float32)

    # Normalize features (using dataset statistics)
    X_mean = X.mean(axis=0)
    X_std = X.std(axis=0)
    X_normalized = (X - X_mean) / (X_std + 1e-8)

    return X_normalized, y, X_mean, X_std


# Prepare VN data
X_vn, y_vn, vn_mean, vn_std = prepare_data(vn_df, FEATURE_COLS, TARGET_COL)

# Prepare US data
X_us, y_us, us_mean, us_std = prepare_data(us_df, FEATURE_COLS, TARGET_COL)

print(f"\n✅ Data prepared:")
print(f"   VN: X shape {X_vn.shape}, y shape {y_vn.shape}")
print(f"   US: X shape {X_us.shape}, y shape {y_us.shape}")

# ============================================================================
# TRAINING FUNCTION (UNCHANGED - WORKS WITH ANY TARGET)
# ============================================================================

def train_model(X_train, y_train, X_val, y_val, epochs=EPOCHS):
    """Train SimpleDLRModel on ratio targets"""

    # Create model
    model = SimpleDLRModel(input_dim=X_train.shape[1], hidden_dim=128).to(DEVICE)

    # Loss and optimizer
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # Convert to tensors
    X_train_t = torch.FloatTensor(X_train).to(DEVICE)
    y_train_t = torch.FloatTensor(y_train).to(DEVICE)
    X_val_t = torch.FloatTensor(X_val).to(DEVICE)
    y_val_t = torch.FloatTensor(y_val).to(DEVICE)

    # Training loop
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()

        # Forward pass
        y_pred = model(X_train_t).squeeze()
        loss = criterion(y_pred, y_train_t)

        # Backward pass
        loss.backward()
        optimizer.step()

        # Validation
        if (epoch + 1) % 10 == 0:
            model.eval()
            with torch.no_grad():
                y_val_pred = model(X_val_t).squeeze()
                val_loss = criterion(y_val_pred, y_val_t)

            print(f"  Epoch {epoch+1:3d}/{epochs}: "
                  f"train_loss={loss.item():.4f}, val_loss={val_loss.item():.4f}")

    return model

# ============================================================================
# EVALUATION FUNCTION - UPDATED FOR RATIO → AMP CONVERSION
# ============================================================================

def evaluate_model(model, X_test, y_test_ratio, region, experiment_name):
    """
    Evaluate model and convert predictions to amps for reporting

    Args:
        model: Trained model (predicts ratio)
        X_test: Test features
        y_test_ratio: True ratios (dimensionless)
        region: 'VN' or 'US' (determines SLR for conversion)
        experiment_name: Name for display

    Returns:
        Dictionary with metrics in both ratio and amp units
    """
    model.eval()
    with torch.no_grad():
        X_test_t = torch.FloatTensor(X_test).to(DEVICE)
        y_pred_ratio = model(X_test_t).squeeze().cpu().numpy()

    # Use evaluation_helpers to compute metrics in both units
    metrics = evaluate_ratio_model(
        predictions=y_pred_ratio,
        actuals=y_test_ratio,
        region=region,
        return_predictions=True,
    )

    # Print results
    print(f"\n{'='*60}")
    print(f"{experiment_name}")
    print(f"{'='*60}")
    print(f"\n📊 Metrics in AMPS (for paper):")
    print(f"   MAE:  {metrics['mae_amps']:7.1f} A")
    print(f"   RMSE: {metrics['rmse_amps']:7.1f} A")
    print(f"   Bias: {metrics['bias_amps']:+7.1f} A")

    print(f"\n📐 Metrics in RATIO units:")
    print(f"   MAE:  {metrics['mae_ratio']:.3f}")
    print(f"   RMSE: {metrics['rmse_ratio']:.3f}")

    return metrics

# ============================================================================
# RUN 4 EXPERIMENTS
# ============================================================================

print("\n" + "=" * 80)
print("EXPERIMENT 1: VN_only (Train on VN, Test on VN)")
print("=" * 80)

# Split VN data (80/20)
n_vn_train = int(0.8 * len(X_vn))
X_vn_train, X_vn_test = X_vn[:n_vn_train], X_vn[n_vn_train:]
y_vn_train, y_vn_test = y_vn[:n_vn_train], y_vn[n_vn_train:]

# Train
print("\n🔧 Training VN_only model...")
vn_only_model = train_model(X_vn_train, y_vn_train, X_vn_test, y_vn_test)

# Evaluate (predictions are ratios, converted to amps for reporting)
vn_only_metrics = evaluate_model(vn_only_model, X_vn_test, y_vn_test,
                                  region='VN', experiment_name='VN_only')

# ============================================================================

print("\n" + "=" * 80)
print("EXPERIMENT 2: US_only (Train on US, Test on US)")
print("=" * 80)

# Split US data (80/20)
n_us_train = int(0.8 * len(X_us))
X_us_train, X_us_test = X_us[:n_us_train], X_us[n_us_train:]
y_us_train, y_us_test = y_us[:n_us_train], y_us[n_us_train:]

# Train
print("\n🔧 Training US_only model...")
us_only_model = train_model(X_us_train, y_us_train, X_us_test, y_us_test)

# Evaluate (predictions are ratios, converted to amps for reporting)
us_only_metrics = evaluate_model(us_only_model, X_us_test, y_us_test,
                                  region='US', experiment_name='US_only')

# ============================================================================

print("\n" + "=" * 80)
print("EXPERIMENT 3: VN→US (Train on VN, Test on US)")
print("=" * 80)

# Use VN-trained model, test on US data
# CRITICAL: Must convert US features using US normalization stats, not VN!
X_us_test_normalized = (X_us_test * (vn_std + 1e-8)) + vn_mean  # Denormalize
X_us_test_renorm = (X_us_test_normalized - us_mean) / (us_std + 1e-8)  # Re-normalize with US stats

# Evaluate on US test set
vn_to_us_metrics = evaluate_model(vn_only_model, X_us_test_renorm, y_us_test,
                                   region='US', experiment_name='VN→US Transfer')

# ============================================================================

print("\n" + "=" * 80)
print("EXPERIMENT 4: US→VN (Train on US, Test on VN)")
print("=" * 80)

# Use US-trained model, test on VN data
# CRITICAL: Must convert VN features using VN normalization stats, not US!
X_vn_test_normalized = (X_vn_test * (us_std + 1e-8)) + us_mean  # Denormalize
X_vn_test_renorm = (X_vn_test_normalized - vn_mean) / (vn_std + 1e-8)  # Re-normalize with VN stats

# Evaluate on VN test set
us_to_vn_metrics = evaluate_model(us_only_model, X_vn_test_renorm, y_vn_test,
                                   region='VN', experiment_name='US→VN Transfer')

# ============================================================================
# SUMMARY TABLE (IN AMPS FOR PAPER)
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
    },
    {
        'Experiment': 'US_only',
        'Train': 'US',
        'Test': 'US',
        'MAE (A)': us_only_metrics['mae_amps'],
        'RMSE (A)': us_only_metrics['rmse_amps'],
        'Bias (A)': us_only_metrics['bias_amps'],
        'MAE (ratio)': us_only_metrics['mae_ratio'],
    },
    {
        'Experiment': 'VN→US',
        'Train': 'VN',
        'Test': 'US',
        'MAE (A)': vn_to_us_metrics['mae_amps'],
        'RMSE (A)': vn_to_us_metrics['rmse_amps'],
        'Bias (A)': vn_to_us_metrics['bias_amps'],
        'MAE (ratio)': vn_to_us_metrics['mae_ratio'],
    },
    {
        'Experiment': 'US→VN',
        'Train': 'US',
        'Test': 'VN',
        'MAE (A)': us_to_vn_metrics['mae_amps'],
        'RMSE (A)': us_to_vn_metrics['rmse_amps'],
        'Bias (A)': us_to_vn_metrics['bias_amps'],
        'MAE (ratio)': us_to_vn_metrics['mae_ratio'],
    },
])

# Calculate degradation
results['Degradation (%)'] = 0.0
results.loc[results['Experiment'] == 'VN→US', 'Degradation (%)'] = (
    (results.loc[results['Experiment'] == 'VN→US', 'MAE (A)'].values[0] -
     results.loc[results['Experiment'] == 'US_only', 'MAE (A)'].values[0]) /
    results.loc[results['Experiment'] == 'US_only', 'MAE (A)'].values[0] * 100
)
results.loc[results['Experiment'] == 'US→VN', 'Degradation (%)'] = (
    (results.loc[results['Experiment'] == 'US→VN', 'MAE (A)'].values[0] -
     results.loc[results['Experiment'] == 'VN_only', 'MAE (A)'].values[0]) /
    results.loc[results['Experiment'] == 'VN_only', 'MAE (A)'].values[0] * 100
)

print("\n📊 RESULTS TABLE (for paper - in AMPS):")
print(results.to_string(index=False, float_format=lambda x: f'{x:.1f}'))

# Save results
results.to_csv(RESULTS_DIR / 'metrics_amps.csv', index=False)
results.to_json(RESULTS_DIR / 'metrics_amps.json', orient='records', indent=2)

print(f"\n✅ Results saved to {RESULTS_DIR}/")

# ============================================================================
# KEY INSIGHTS
# ============================================================================

print("\n" + "=" * 80)
print("KEY INSIGHTS")
print("=" * 80)

vn_baseline = results.loc[results['Experiment'] == 'VN_only', 'MAE (A)'].values[0]
us_baseline = results.loc[results['Experiment'] == 'US_only', 'MAE (A)'].values[0]
vn_to_us = results.loc[results['Experiment'] == 'VN→US', 'MAE (A)'].values[0]
us_to_vn = results.loc[results['Experiment'] == 'US→VN', 'MAE (A)'].values[0]

vn_to_us_deg = (vn_to_us - us_baseline) / us_baseline * 100
us_to_vn_deg = (us_to_vn - vn_baseline) / vn_baseline * 100

print(f"\n📋 Baseline Performance:")
print(f"   VN_only: {vn_baseline:.1f} A MAE")
print(f"   US_only: {us_baseline:.1f} A MAE")

print(f"\n🔄 Cross-Region Transfer:")
print(f"   VN→US: {vn_to_us:.1f} A MAE ({vn_to_us_deg:+.1f}% degradation)")
print(f"   US→VN: {us_to_vn:.1f} A MAE ({us_to_vn_deg:+.1f}% degradation)")

if vn_to_us_deg < 100 and us_to_vn_deg < 100:
    print(f"\n✅ Transfer learning WORKS with normalized targets!")
    print(f"   Degradation is reasonable (<100%)")
elif vn_to_us_deg < 300:
    print(f"\n🟡 Transfer learning PARTIALLY works")
    print(f"   Some degradation but better than before normalization")
else:
    print(f"\n❌ Transfer learning still fails significantly")
    print(f"   Need further investigation or domain adaptation")

print("\n" + "=" * 80)
print("TEMPLATE COMPLETE")
print("=" * 80)
print("\n💡 To use this template:")
print("   1. Copy to cross_region_validation_v3.py")
print("   2. Run: python scripts/cross_region_validation_v3.py")
print("   3. Compare results with v2 (unnormalized targets)")
print("   4. Report MAE in AMPS for paper, ratio for completeness")
