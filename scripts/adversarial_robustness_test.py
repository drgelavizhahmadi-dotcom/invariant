#!/usr/bin/env python3
"""
CRITICAL TEST: Adversarial Robustness

Tests HWF-PIKAN's core claim: does physics loss improve robustness
to adversarial attacks (FGSM/BIM)?

This is the experiment that determines the architecture contribution.
"""

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent))

from models.invariant_pikan_v2 import InvariantPIKANV2

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print("=" * 80)
print("ADVERSARIAL ROBUSTNESS TEST (FGSM/BIM)")
print("=" * 80)
print("\nCore claim: Physics loss makes models robust to adversarial perturbations")
print("Test: Evaluate models WITH and WITHOUT physics on attacked test data")

# ============================================================================
# LOAD DATA
# ============================================================================

df = pd.read_hdf('data/processed/unified_dlr_training_NORMALIZED.h5', key='data')
vn_df = df[df['region'] == 'VN'].copy()

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

# Test split
n_train = int(0.8 * len(X_vn))
X_test = X_vn[n_train:]
y_test = y_vn[n_train:]

print(f"\n✅ Test set: {len(X_test):,} samples")

# ============================================================================
# CREATE MODELS (WITH and WITHOUT physics)
# ============================================================================

def create_model(use_physics=True):
    model = InvariantPIKANV2(input_dim=3, hidden_dim=64, output_dim=1).to(DEVICE)
    with torch.no_grad():
        model.physics_params['raw_emissivity'].fill_(torch.logit(torch.tensor(0.7)))
        model.physics_params['raw_absorptivity'].fill_(torch.logit(torch.tensor(0.9)))
    return model

def train_quick(X_train, y_train, use_physics=True, epochs=50):
    """Quick training for adversarial test"""
    model = create_model(use_physics)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    X_t = torch.FloatTensor(X_train).to(DEVICE)
    y_t = torch.FloatTensor(y_train).to(DEVICE)

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()

        h = model.embedding(X_t)
        pred = model.kan(h)
        y_pred = pred[:, 0]

        loss = torch.mean((y_pred - y_t) ** 2)
        loss.backward()
        optimizer.step()

    return model

print("\n🔧 Training models...")
X_train = X_vn[:n_train]
y_train = y_vn[:n_train]

model_with_physics = train_quick(X_train, y_train, use_physics=True, epochs=50)
model_without_physics = train_quick(X_train, y_train, use_physics=False, epochs=50)

print("✅ Models trained (50 epochs each)")

# ============================================================================
# GENERATE ADVERSARIAL EXAMPLES (FGSM)
# ============================================================================

def fgsm_attack(model, X, y, epsilon):
    """
    Fast Gradient Sign Method (FGSM) attack

    Generates adversarial examples by perturbing inputs in the direction
    that maximizes loss.
    """
    X_adv = torch.FloatTensor(X).to(DEVICE)
    X_adv.requires_grad = True

    y_t = torch.FloatTensor(y).to(DEVICE)

    # Forward pass
    model.eval()
    h = model.embedding(X_adv)
    pred = model.kan(h)
    y_pred = pred[:, 0]

    # Compute loss
    loss = torch.mean((y_pred - y_t) ** 2)

    # Backward to get gradient
    model.zero_grad()
    loss.backward()

    # Generate adversarial example
    with torch.no_grad():
        # Sign of gradient
        grad_sign = X_adv.grad.sign()

        # Add perturbation
        X_adv = X_adv + epsilon * grad_sign

        # Clip to valid range (assuming normalized data)
        X_adv = torch.clamp(X_adv, -10, 10)

    return X_adv.cpu().numpy()

def bim_attack(model, X, y, epsilon, alpha=0.01, num_iter=10):
    """
    Basic Iterative Method (BIM) attack

    Iterative version of FGSM with smaller step size.
    """
    X_adv = torch.FloatTensor(X).to(DEVICE)
    X_orig = X_adv.clone()

    for i in range(num_iter):
        X_adv.requires_grad = True

        y_t = torch.FloatTensor(y).to(DEVICE)

        # Forward
        model.eval()
        h = model.embedding(X_adv)
        pred = model.kan(h)
        y_pred = pred[:, 0]

        loss = torch.mean((y_pred - y_t) ** 2)

        model.zero_grad()
        loss.backward()

        with torch.no_grad():
            grad_sign = X_adv.grad.sign()
            X_adv = X_adv + alpha * grad_sign

            # Project back to epsilon ball
            perturbation = X_adv - X_orig
            perturbation = torch.clamp(perturbation, -epsilon, epsilon)
            X_adv = X_orig + perturbation

            # Clip to valid range
            X_adv = torch.clamp(X_adv, -10, 10)

    return X_adv.cpu().numpy()

# ============================================================================
# EVALUATE ON CLEAN AND ADVERSARIAL DATA
# ============================================================================

def evaluate(model, X, y):
    """Evaluate model on data"""
    model.eval()
    with torch.no_grad():
        X_t = torch.FloatTensor(X).to(DEVICE)
        h = model.embedding(X_t)
        pred = model.kan(h)
        y_pred = pred[:, 0].cpu().numpy()

    mae = np.mean(np.abs(y_pred - y))
    return mae * 850.0  # Convert to amps for VN

print("\n" + "=" * 80)
print("EVALUATION: CLEAN DATA")
print("=" * 80)

mae_clean_with = evaluate(model_with_physics, X_test, y_test)
mae_clean_without = evaluate(model_without_physics, X_test, y_test)

print(f"\n📊 Clean test data:")
print(f"   WITH physics:    {mae_clean_with:.1f} A")
print(f"   WITHOUT physics: {mae_clean_without:.1f} A")
print(f"   Difference:      {mae_clean_with - mae_clean_without:+.1f} A")

# ============================================================================
# FGSM ATTACKS
# ============================================================================

print("\n" + "=" * 80)
print("EVALUATION: FGSM ATTACKS")
print("=" * 80)

epsilons_fgsm = [0.5, 1.0, 5.0, 10.0]
results_fgsm = []

for eps in epsilons_fgsm:
    # Generate attacks (use model_without_physics as attacker)
    X_fgsm = fgsm_attack(model_without_physics, X_test, y_test, epsilon=eps)

    # Evaluate both models
    mae_with = evaluate(model_with_physics, X_fgsm, y_test)
    mae_without = evaluate(model_without_physics, X_fgsm, y_test)

    degradation_with = (mae_with - mae_clean_with) / mae_clean_with * 100
    degradation_without = (mae_without - mae_clean_without) / mae_clean_without * 100

    results_fgsm.append({
        'epsilon': eps,
        'mae_with': mae_with,
        'mae_without': mae_without,
        'degradation_with': degradation_with,
        'degradation_without': degradation_without,
        'advantage': degradation_without - degradation_with,
    })

    print(f"\nε = {eps}:")
    print(f"   WITH physics:    {mae_with:.1f} A ({degradation_with:+.1f}% vs clean)")
    print(f"   WITHOUT physics: {mae_without:.1f} A ({degradation_without:+.1f}% vs clean)")

    if degradation_with < degradation_without:
        print(f"   ✅ Physics MORE robust: {degradation_without - degradation_with:.1f}% advantage")
    else:
        print(f"   ❌ Physics LESS robust: {degradation_with - degradation_without:.1f}% disadvantage")

# ============================================================================
# BIM ATTACKS
# ============================================================================

print("\n" + "=" * 80)
print("EVALUATION: BIM ATTACKS (10 iterations)")
print("=" * 80)

epsilons_bim = [0.5, 1.0, 5.0, 10.0]
results_bim = []

for eps in epsilons_bim:
    # Generate attacks
    X_bim = bim_attack(model_without_physics, X_test, y_test, epsilon=eps, alpha=eps/10, num_iter=10)

    # Evaluate both models
    mae_with = evaluate(model_with_physics, X_bim, y_test)
    mae_without = evaluate(model_without_physics, X_bim, y_test)

    degradation_with = (mae_with - mae_clean_with) / mae_clean_with * 100
    degradation_without = (mae_without - mae_clean_without) / mae_clean_without * 100

    results_bim.append({
        'epsilon': eps,
        'mae_with': mae_with,
        'mae_without': mae_without,
        'degradation_with': degradation_with,
        'degradation_without': degradation_without,
        'advantage': degradation_without - degradation_with,
    })

    print(f"\nε = {eps}:")
    print(f"   WITH physics:    {mae_with:.1f} A ({degradation_with:+.1f}% vs clean)")
    print(f"   WITHOUT physics: {mae_without:.1f} A ({degradation_without:+.1f}% vs clean)")

    if degradation_with < degradation_without:
        print(f"   ✅ Physics MORE robust: {degradation_without - degradation_with:.1f}% advantage")
    else:
        print(f"   ❌ Physics LESS robust: {degradation_with - degradation_without:.1f}% disadvantage")

# ============================================================================
# SUMMARY
# ============================================================================

print("\n" + "=" * 80)
print("SUMMARY: ADVERSARIAL ROBUSTNESS")
print("=" * 80)

df_fgsm = pd.DataFrame(results_fgsm)
df_bim = pd.DataFrame(results_bim)

print("\n📊 FGSM Results:")
print(df_fgsm.to_string(index=False, float_format=lambda x: f'{x:.1f}'))

print("\n📊 BIM Results:")
print(df_bim.to_string(index=False, float_format=lambda x: f'{x:.1f}'))

# Average advantage
avg_fgsm_advantage = df_fgsm['advantage'].mean()
avg_bim_advantage = df_bim['advantage'].mean()

print("\n" + "=" * 80)
print("INTERPRETATION")
print("=" * 80)

print(f"\n📊 Average robustness advantage (physics vs no-physics):")
print(f"   FGSM: {avg_fgsm_advantage:+.1f}% (positive = physics more robust)")
print(f"   BIM:  {avg_bim_advantage:+.1f}% (positive = physics more robust)")

if avg_fgsm_advantage > 5 and avg_bim_advantage > 5:
    print("\n✅ STRONG FINDING: Physics loss significantly improves adversarial robustness")
    print("   → Supports HWF-PIKAN architecture contribution")
    print("   → Paper can claim both normalization AND robustness benefits")
elif avg_fgsm_advantage > 2 and avg_bim_advantage > 2:
    print("\n🟡 MODERATE FINDING: Physics loss provides modest robustness improvement")
    print("   → Some support for architecture claim but not dramatic")
elif avg_fgsm_advantage > -2 and avg_bim_advantage > -2:
    print("\n⚠️  WEAK FINDING: Physics loss provides minimal robustness benefit")
    print("   → Architecture contribution is weak")
    print("   → Normalization is the primary contribution")
else:
    print("\n❌ NEGATIVE FINDING: Physics loss does NOT improve adversarial robustness")
    print("   → HWF-PIKAN architecture claim is UNSUPPORTED")
    print("   → Paper should focus on normalization contribution only")

# Save results
RESULTS_DIR = Path("cross_region_results/adversarial_robustness")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

df_fgsm.to_csv(RESULTS_DIR / 'fgsm_results.csv', index=False)
df_bim.to_csv(RESULTS_DIR / 'bim_results.csv', index=False)

print(f"\n✅ Saved results to {RESULTS_DIR}/")

print("\n" + "=" * 80)
print("CRITICAL: This determines the paper's scope")
print("=" * 80)
