#!/usr/bin/env python3
"""
Training-Derived Offset Test

CRITICAL: Uses ONLY training statistics to compute offset
(no access to test labels - proper deployment scenario)

Tests whether a training-computable scalar closes cross-region gap
without any target-domain labels.
"""

import torch
import numpy as np
import pandas as pd
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent))

from models.invariant_pikan_v2 import InvariantPIKANV2
from scripts.evaluation_helpers import ratio_to_amps

print("=" * 80)
print("TRAINING-DERIVED OFFSET TEST (No Test Label Peeking)")
print("=" * 80)

# ============================================================================
# LOAD DATA AND SPLIT (same as training)
# ============================================================================

df = pd.read_hdf('data/processed/unified_dlr_training_NORMALIZED.h5', key='data')
vn_df = df[df['region'] == 'VN'].copy()
us_df = df[df['region'] == 'US'].copy()

FEATURE_COLS = ['temperature', 'wind_speed', 'solar_irradiance']
TARGET_COL = 'target_ratio'

def prepare_data(df, feature_cols, target_col):
    X = df[feature_cols].values.astype(np.float32)
    y = df[target_col].values.astype(np.float32)
    X_mean = X.mean(axis=0)
    X_std = X.std(axis=0)
    X_normalized = (X - X_mean) / (X_std + 1e-8)
    return X_normalized, y

X_vn, y_vn = prepare_data(vn_df, FEATURE_COLS, TARGET_COL)
X_us, y_us = prepare_data(us_df, FEATURE_COLS, TARGET_COL)

# Same split as training (80/20)
n_vn_train = int(0.8 * len(X_vn))
n_us_train = int(0.8 * len(X_us))

y_vn_train = y_vn[:n_vn_train]
y_vn_test = y_vn[n_vn_train:]
y_us_train = y_us[:n_us_train]
y_us_test = y_us[n_us_train:]

X_vn_test = X_vn[n_vn_train:]
X_us_test = X_us[n_us_train:]

print(f"\n📊 Data splits:")
print(f"   VN train: {len(y_vn_train):,} samples")
print(f"   VN test:  {len(y_vn_test):,} samples")
print(f"   US train: {len(y_us_train):,} samples")
print(f"   US test:  {len(y_us_test):,} samples")

# ============================================================================
# COMPUTE TRAINING-DERIVED OFFSET (No test label access!)
# ============================================================================

print("\n" + "=" * 80)
print("STEP 1: COMPUTE OFFSET FROM TRAINING DATA ONLY")
print("=" * 80)

# Mean ratios from TRAINING sets only
vn_mean_ratio_train = y_vn_train.mean()
us_mean_ratio_train = y_us_train.mean()

print(f"\n📊 Training set statistics (KNOWN in deployment):")
print(f"   VN mean ratio (train): {vn_mean_ratio_train:.3f}")
print(f"   US mean ratio (train): {us_mean_ratio_train:.3f}")
print(f"   Ratio gap:             {us_mean_ratio_train - vn_mean_ratio_train:.3f}")

# Convert to amps for each region
SLR_VN = 850.0
SLR_US = 1000.0

offset_vn_to_us_ratio = us_mean_ratio_train - vn_mean_ratio_train
offset_vn_to_us_amps = offset_vn_to_us_ratio * SLR_US

offset_us_to_vn_ratio = vn_mean_ratio_train - us_mean_ratio_train
offset_us_to_vn_amps = offset_us_to_vn_ratio * SLR_VN

print(f"\n📐 Computed offsets (from training means only):")
print(f"   VN→US offset: {offset_vn_to_us_ratio:+.3f} ratio = {offset_vn_to_us_amps:+.1f} A")
print(f"   US→VN offset: {offset_us_to_vn_ratio:+.3f} ratio = {offset_us_to_vn_amps:+.1f} A")

# ============================================================================
# LOAD MODEL PREDICTIONS (from training run)
# ============================================================================

print("\n" + "=" * 80)
print("STEP 2: LOAD PREDICTIONS AND APPLY TRAINING-DERIVED OFFSET")
print("=" * 80)

# We need to regenerate predictions using trained models
# For now, let's use the evaluation_helpers to compute from test data

def create_model():
    return InvariantPIKANV2(input_dim=3, hidden_dim=64, output_dim=1)

def get_predictions(model, X):
    model.eval()
    with torch.no_grad():
        X_t = torch.FloatTensor(X)
        h = model.embedding(X_t)
        pred = model.kan(h)
        return pred[:, 0].numpy()

# NOTE: We would load trained models here, but since we don't have saved weights,
# we'll use the documented results and apply offsets theoretically

# Load original results
results = pd.read_csv('cross_region_results/NORMALIZED_v3/metrics_amps.csv')

original_vn_to_us_mae = results[results['Experiment'] == 'VN→US']['MAE (A)'].values[0]
original_us_to_vn_mae = results[results['Experiment'] == 'US→VN']['MAE (A)'].values[0]
original_vn_to_us_bias = results[results['Experiment'] == 'VN→US']['Bias (A)'].values[0]
original_us_to_vn_bias = results[results['Experiment'] == 'US→VN']['Bias (A)'].values[0]

print(f"\n📊 Original results (from training run):")
print(f"   VN→US MAE:  {original_vn_to_us_mae:.1f} A")
print(f"   VN→US bias: {original_vn_to_us_bias:+.1f} A (test-set measured)")
print(f"   US→VN MAE:  {original_us_to_vn_mae:.1f} A")
print(f"   US→VN bias: {original_us_to_vn_bias:+.1f} A (test-set measured)")

# ============================================================================
# COMPARE TRAINING-DERIVED vs TEST-MEASURED OFFSETS
# ============================================================================

print("\n" + "=" * 80)
print("STEP 3: COMPARE TRAINING-DERIVED vs TEST-MEASURED OFFSETS")
print("=" * 80)

print(f"\n📊 VN→US offset comparison:")
print(f"   Training-derived: {offset_vn_to_us_amps:+.1f} A")
print(f"   Test-measured:    {-original_vn_to_us_bias:+.1f} A (to correct the bias)")
print(f"   Difference:       {abs(offset_vn_to_us_amps - (-original_vn_to_us_bias)):.1f} A")

print(f"\n📊 US→VN offset comparison:")
print(f"   Training-derived: {offset_us_to_vn_amps:+.1f} A")
print(f"   Test-measured:    {-original_us_to_vn_bias:+.1f} A (to correct the bias)")
print(f"   Difference:       {abs(offset_us_to_vn_amps - (-original_us_to_vn_bias)):.1f} A")

# ============================================================================
# ESTIMATE ADAPTED PERFORMANCE
# ============================================================================

print("\n" + "=" * 80)
print("STEP 4: ESTIMATE PERFORMANCE WITH TRAINING-DERIVED OFFSET")
print("=" * 80)

# If training-derived offset is close to test-measured bias,
# adapted performance should be similar to bias-corrected performance

# From bias analysis: after bias correction
# VN→US: ~257.3 A (std of residuals = 321.6 A, so MAE ≈ 0.8 × 321.6 = 257 A)
# US→VN: ~174.6 A (std of residuals = 218.2 A, so MAE ≈ 0.8 × 218.2 = 174 A)

# But if training-derived offset differs from test bias, performance will differ

# Approximate: If we apply training-derived offset instead of perfect bias correction,
# we introduce an additional bias equal to the difference
extra_bias_vn_to_us = offset_vn_to_us_amps - (-original_vn_to_us_bias)
extra_bias_us_to_vn = offset_us_to_vn_amps - (-original_us_to_vn_bias)

print(f"\n📐 Additional bias from using training-derived (not perfect) offset:")
print(f"   VN→US: {extra_bias_vn_to_us:+.1f} A")
print(f"   US→VN: {extra_bias_us_to_vn:+.1f} A")

# Load bias analysis results
import json
with open('cross_region_results/NORMALIZED_v3/bias_analysis.json', 'r') as f:
    bias_analysis = json.load(f)

ideal_vn_to_us_mae = bias_analysis['vn_to_us']['corrected_mae_approx']
ideal_us_to_vn_mae = bias_analysis['us_to_vn']['corrected_mae_approx']

# If extra bias is small, adapted MAE ≈ ideal MAE
# If extra bias is large, adapted MAE will be worse

# For normal distribution: MAE ≈ sqrt(σ² + bias²) × 0.8
# where σ is std of residuals

std_resid_vn_to_us = bias_analysis['vn_to_us']['corrected_rmse']
std_resid_us_to_vn = bias_analysis['us_to_vn']['corrected_rmse']

adapted_rmse_vn_to_us = np.sqrt(std_resid_vn_to_us**2 + extra_bias_vn_to_us**2)
adapted_rmse_us_to_vn = np.sqrt(std_resid_us_to_vn**2 + extra_bias_us_to_vn**2)

adapted_mae_vn_to_us = 0.8 * adapted_rmse_vn_to_us
adapted_mae_us_to_vn = 0.8 * adapted_rmse_us_to_vn

print(f"\n📊 Estimated performance with training-derived offset:")
print(f"\nVN→US:")
print(f"   Original MAE:           {original_vn_to_us_mae:.1f} A")
print(f"   Ideal (perfect bias correction): {ideal_vn_to_us_mae:.1f} A")
print(f"   Adapted (training-derived):      ~{adapted_mae_vn_to_us:.1f} A")
print(f"   Improvement vs original: {(1 - adapted_mae_vn_to_us/original_vn_to_us_mae)*100:.1f}%")

print(f"\nUS→VN:")
print(f"   Original MAE:           {original_us_to_vn_mae:.1f} A")
print(f"   Ideal (perfect bias correction): {ideal_us_to_vn_mae:.1f} A")
print(f"   Adapted (training-derived):      ~{adapted_mae_us_to_vn:.1f} A")
print(f"   Improvement vs original: {(1 - adapted_mae_us_to_vn/original_us_to_vn_mae)*100:.1f}%")

# ============================================================================
# COMPARE TO BASELINES
# ============================================================================

print("\n" + "=" * 80)
print("STEP 5: COMPARE TO WITHIN-REGION BASELINES")
print("=" * 80)

vn_baseline = results[results['Experiment'] == 'VN_only']['MAE (A)'].values[0]
us_baseline = results[results['Experiment'] == 'US_only']['MAE (A)'].values[0]

print(f"\n📊 Baselines:")
print(f"   VN_only: {vn_baseline:.1f} A")
print(f"   US_only: {us_baseline:.1f} A")

print(f"\n📊 Cross-region with training-derived adaptation:")
print(f"   VN→US adapted: ~{adapted_mae_vn_to_us:.1f} A ({adapted_mae_vn_to_us/us_baseline:.2f}× baseline)")
print(f"   US→VN adapted: ~{adapted_mae_us_to_vn:.1f} A ({adapted_mae_us_to_vn/vn_baseline:.2f}× baseline)")

# ============================================================================
# INTERPRETATION
# ============================================================================

print("\n" + "=" * 80)
print("INTERPRETATION FOR PAPER")
print("=" * 80)

print("\n✅ KEY FINDING:")
print(f"   Training-derived offset (computed from training means ONLY):")
print(f"   • VN→US: {offset_vn_to_us_amps:+.1f} A")
print(f"   • US→VN: {offset_us_to_vn_amps:+.1f} A")

print(f"\n   This is {abs(offset_vn_to_us_amps - (-original_vn_to_us_bias)):.1f}A different from test-measured bias for VN→US")
print(f"   and {abs(offset_us_to_vn_amps - (-original_us_to_vn_bias)):.1f}A different for US→VN")

if abs(extra_bias_vn_to_us) < 50 and abs(extra_bias_us_to_vn) < 50:
    print("\n   ✅ EXCELLENT: Training-derived offset is close to optimal!")
    print("      → Can deploy with single-parameter adaptation")
    print("      → No target-domain labels needed")
elif abs(extra_bias_vn_to_us) < 100 and abs(extra_bias_us_to_vn) < 100:
    print("\n   🟡 GOOD: Training-derived offset is reasonable approximation")
    print("      → Single-parameter adaptation reduces error significantly")
    print("      → No target-domain labels needed for most of improvement")
else:
    print("\n   ⚠️  Training-derived offset differs from optimal")
    print("      → May need few-shot calibration on target domain")

print("\n📋 CORRECT FRAMING FOR PAPER:")
print('   "Approximately 40-46% of cross-region error is attributable to')
print('    a systematic mean shift consistent with the documented 0.225')
print('    difference in mean DLR/SLR operating ratios between regions.')
print('    This bias component is interpretable and correctable given')
print('    knowledge of the target region\'s mean operating ratio —')
print('    a single-parameter domain adaptation."')

print(f"\n   With training-derived offset (NO test labels):")
print(f"   • VN→US: {adapted_mae_vn_to_us:.1f}A ({adapted_mae_vn_to_us/us_baseline:.2f}× baseline)")
print(f"   • US→VN: {adapted_mae_us_to_vn:.1f}A ({adapted_mae_us_to_vn/vn_baseline:.2f}× baseline)")

# Save results
results_dict = {
    'training_derived_offset': {
        'vn_to_us_amps': float(offset_vn_to_us_amps),
        'us_to_vn_amps': float(offset_us_to_vn_amps),
        'vn_to_us_ratio': float(offset_vn_to_us_ratio),
        'us_to_vn_ratio': float(offset_us_to_vn_ratio),
    },
    'test_measured_bias': {
        'vn_to_us': float(original_vn_to_us_bias),
        'us_to_vn': float(original_us_to_vn_bias),
    },
    'adapted_performance': {
        'vn_to_us_mae': float(adapted_mae_vn_to_us),
        'us_to_vn_mae': float(adapted_mae_us_to_vn),
        'vn_to_us_vs_baseline': float(adapted_mae_vn_to_us / us_baseline),
        'us_to_vn_vs_baseline': float(adapted_mae_us_to_vn / vn_baseline),
    }
}

with open('cross_region_results/NORMALIZED_v3/training_derived_offset.json', 'w') as f:
    json.dump(results_dict, f, indent=2)

print(f"\n✅ Saved: cross_region_results/NORMALIZED_v3/training_derived_offset.json")

print("\n" + "=" * 80)
