#!/usr/bin/env python3
"""
EXPERIMENT 2: Baseline Comparison for AI Act Vision

Trains 3 models on identical data splits and compares:
1. Prediction accuracy (MAE, RMSE)
2. Physics consistency (residual magnitude)
3. Explainability (heat balance decomposition availability)

Models:
- VanillaMLP: Pure data-driven, no physics, no explainability
- VanillaPINN: MLP + physics loss + learnable params + explainability
- HWF-PIKAN:  KAN + wavelet-Fourier + physics loss + explainability

All models:
- Matched capacity (~12k params)
- Same train/val/test split (fixed seed=42)
- Same training procedure (100 epochs, AdamW, lr=1e-3)
- Evaluated on Vietnam test set

Output:
- Table 3: Accuracy + physics consistency + explainability comparison
- Learning curves figure
- Explainability audit sample
"""

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
import sys
import json

sys.path.append(str(Path(__file__).parent.parent))

from models.vanilla_mlp import VanillaMLP
from models.vanilla_pinn import VanillaPINN
from models.invariant_pikan_v2 import InvariantPIKANV2

# ============================================================================
# CONFIGURATION
# ============================================================================

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
EPOCHS = 100
LR = 1e-3
PHYSICS_WEIGHT = 0.05

FEATURE_COLS = ['temperature', 'wind_speed', 'solar_irradiance']
TARGET_COL = 'target_ratio'
VN_SLR = 850.0
US_SLR = 1000.0

RESULTS_DIR = Path("cross_region_results/baseline_comparison")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

print(f"Device: {DEVICE}")
print("=" * 80)
print("EXPERIMENT 2: BASELINE COMPARISON")
print("=" * 80)

# ============================================================================
# LOAD DATA
# ============================================================================

print("\nLoading normalized dataset...")
df = pd.read_hdf('data/processed/unified_dlr_training_NORMALIZED.h5', key='data')
vn_df = df[df['region'] == 'VN'].copy()
us_df = df[df['region'] == 'US'].copy()
print(f"   Vietnam: {len(vn_df):,} samples")
print(f"   US:      {len(us_df):,} samples")

# ============================================================================
# CREATE SPLITS (FIXED SEED)
# ============================================================================

def create_splits(df, feature_cols, target_col, test_size=0.1, val_size=0.1, seed=42):
    """Create train/val/test splits with fixed seed. Normalize on train only."""
    rng = np.random.RandomState(seed)
    n = len(df)
    indices = rng.permutation(n)

    n_test = int(n * test_size)
    n_val = int(n * val_size)

    train_idx = indices[:n - n_test - n_val]
    val_idx = indices[n - n_test - n_val:n - n_test]
    test_idx = indices[n - n_test:]

    X = df[feature_cols].values.astype(np.float32)
    y = df[target_col].values.astype(np.float32)

    # Fit normalization on train only
    X_mean = X[train_idx].mean(axis=0)
    X_std = X[train_idx].std(axis=0)
    X_norm = (X - X_mean) / (X_std + 1e-8)

    return {
        'X_train': X_norm[train_idx], 'y_train': y[train_idx],
        'X_val': X_norm[val_idx],     'y_val': y[val_idx],
        'X_test': X_norm[test_idx],   'y_test': y[test_idx],
        'X_mean': X_mean, 'X_std': X_std,
        'X_test_raw': X[test_idx],  # raw features for explain_prediction
    }

print("\nCreating splits (seed=42)...")
vn = create_splits(vn_df, FEATURE_COLS, TARGET_COL, seed=SEED)
us = create_splits(us_df, FEATURE_COLS, TARGET_COL, seed=SEED)
print(f"   VN - Train: {len(vn['X_train']):,}, Val: {len(vn['X_val']):,}, Test: {len(vn['X_test']):,}")
print(f"   US - Train: {len(us['X_train']):,}, Val: {len(us['X_val']):,}, Test: {len(us['X_test']):,}")


# ============================================================================
# HWF-PIKAN WRAPPER (adapts dict-output API to tensor-output API)
# ============================================================================

class HWFPIKANWrapper(nn.Module):
    """
    Wraps InvariantPIKANV2 to match VanillaMLP/VanillaPINN forward API.

    InvariantPIKANV2.forward(x, weather_dict) -> dict
    HWFPIKANWrapper.forward(x) -> tensor (batch, 1)

    All other methods (explain_prediction, get_physics_params) delegate
    to the underlying InvariantPIKANV2 instance.
    """

    def __init__(self, input_dim=3, hidden_dim=64, output_dim=1,
                 emissivity=0.7, absorptivity=0.9):
        super().__init__()
        self.model = InvariantPIKANV2(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            fourier_bands=16,
            wavelet_scales=8,
        )
        # Set physics params from DynaLiRD Table 2
        with torch.no_grad():
            self.model.physics_params['raw_emissivity'].fill_(
                torch.logit(torch.tensor(emissivity))
            )
            self.model.physics_params['raw_absorptivity'].fill_(
                torch.logit(torch.tensor(absorptivity))
            )

        self.param_count = sum(p.numel() for p in self.parameters())

    def forward(self, x, line_ids=None):
        """Simplified forward: embedding -> KAN -> first output."""
        h = self.model.embedding(x)
        predictions = self.model.kan(h)
        return predictions[:, 0:1]  # (batch, 1)

    def compute_physics_residual(self, x, y_pred):
        """Physics residual using same simplified formulation as VanillaPINN."""
        params = self.model.get_physics_params()
        T_amb = x[:, 0]
        V_wind = x[:, 1]
        Q_solar = x[:, 2]

        import math
        D = 0.02814
        R_ref = 7.283e-5

        wind_factor = torch.clamp(V_wind, min=0.1).abs() ** 0.5
        q_conv = 2.5 * wind_factor * (D * math.pi)
        q_solar = Q_solar.abs() * params['absorptivity'] * (D * math.pi)
        q_resist = y_pred ** 2 * R_ref * params['resistance_factor']

        return torch.abs(q_conv - q_solar - q_resist)

    def explain_prediction(self, x, I_pred, T_conductor=75.0):
        """Delegates to InvariantPIKANV2.explain_prediction."""
        return self.model.explain_prediction(x, I_pred, T_conductor)

    def get_physics_params(self):
        return self.model.get_physics_params()


# ============================================================================
# TRAINING FUNCTION
# ============================================================================

def train_model(model, X_train, y_train, X_val, y_val,
                model_name, epochs=EPOCHS, lr=LR, physics_weight=0.0):
    """
    Train model. Returns trained model and loss history.
    All models use same API: model(x) -> (batch, 1) tensor.
    """
    model = model.to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    X_train_t = torch.FloatTensor(X_train).to(DEVICE)
    y_train_t = torch.FloatTensor(y_train).to(DEVICE)
    X_val_t = torch.FloatTensor(X_val).to(DEVICE)
    y_val_t = torch.FloatTensor(y_val).to(DEVICE)

    history = {'train_loss': [], 'val_loss': [], 'physics_loss': []}

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()

        y_pred = model(X_train_t).squeeze()
        mse_loss = torch.mean((y_pred - y_train_t) ** 2)

        # Physics loss (for PINN and HWF-PIKAN)
        if physics_weight > 0 and hasattr(model, 'compute_physics_residual'):
            physics_res = model.compute_physics_residual(X_train_t, y_pred)
            physics_loss = physics_res.mean()
            total_loss = mse_loss + physics_weight * physics_loss
            history['physics_loss'].append(physics_loss.item())
        else:
            total_loss = mse_loss
            history['physics_loss'].append(0.0)

        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        # Validation
        model.eval()
        with torch.no_grad():
            y_val_pred = model(X_val_t).squeeze()
            val_mse = torch.mean((y_val_pred - y_val_t) ** 2).item()

        history['train_loss'].append(total_loss.item())
        history['val_loss'].append(val_mse)

        if (epoch + 1) % 25 == 0:
            phys_str = f", phys={history['physics_loss'][-1]:.4f}" if physics_weight > 0 else ""
            print(f"  [{model_name}] Epoch {epoch+1:3d}/{epochs}: "
                  f"train={total_loss.item():.4f}, val={val_mse:.4f}{phys_str}")

    return model, history


# ============================================================================
# EVALUATION FUNCTION
# ============================================================================

def evaluate_model(model, X_test, y_test, slr, X_test_raw=None):
    """Evaluate model. Returns metrics in amps."""
    model.eval()
    with torch.no_grad():
        X_t = torch.FloatTensor(X_test).to(DEVICE)
        y_pred = model(X_t).squeeze().cpu().numpy()

    y_pred_amps = y_pred * slr
    y_test_amps = y_test * slr

    errors = y_pred_amps - y_test_amps
    mae = np.mean(np.abs(errors))
    rmse = np.sqrt(np.mean(errors ** 2))
    bias = np.mean(errors)

    # Physics residual
    with torch.no_grad():
        if hasattr(model, 'compute_physics_residual'):
            res = model.compute_physics_residual(X_t, torch.FloatTensor(y_pred).to(DEVICE))
            mean_res = res.mean().item()
        else:
            mean_res = float('nan')

    # Explainability check
    has_explain = hasattr(model, 'explain_prediction')
    explanation = None
    if has_explain and X_test_raw is not None:
        X_raw_t = torch.FloatTensor(X_test_raw[:5]).to(DEVICE)
        y_sample = torch.FloatTensor(y_pred_amps[:5]).to(DEVICE)
        explanation = model.explain_prediction(X_raw_t, y_sample)

    return {
        'mae_amps': mae,
        'rmse_amps': rmse,
        'bias_amps': bias,
        'physics_residual': mean_res,
        'explainable': has_explain and explanation is not None,
        'explanation_sample': explanation,
        'predictions_amps': y_pred_amps,
        'actuals_amps': y_test_amps,
    }


# ============================================================================
# TRAIN ALL MODELS ON VIETNAM
# ============================================================================

print("\n" + "=" * 80)
print("TRAINING ON VIETNAM DATA")
print("=" * 80)

models = {}
histories = {}

# 1. VanillaMLP
print("\n[1/3] VanillaMLP (no physics, no explainability)")
m = VanillaMLP(input_dim=3, hidden_dim=64, n_layers=4)
print(f"   Params: {m.param_count:,}")
m, h = train_model(m, vn['X_train'], vn['y_train'], vn['X_val'], vn['y_val'],
                    "VanillaMLP", physics_weight=0.0)
models['VanillaMLP'] = m
histories['VanillaMLP'] = h

# 2. VanillaPINN
print("\n[2/3] VanillaPINN (physics loss + explainability)")
m = VanillaPINN(input_dim=3, hidden_dim=64, n_layers=4)
print(f"   Params: {m.param_count:,}")
m, h = train_model(m, vn['X_train'], vn['y_train'], vn['X_val'], vn['y_val'],
                    "VanillaPINN", physics_weight=PHYSICS_WEIGHT)
models['VanillaPINN'] = m
histories['VanillaPINN'] = h

# 3. HWF-PIKAN
print("\n[3/3] HWF-PIKAN (KAN + wavelet-Fourier + physics + explainability)")
m = HWFPIKANWrapper(input_dim=3, hidden_dim=64, output_dim=1,
                     emissivity=0.7, absorptivity=0.9)
print(f"   Params: {m.param_count:,}")
m, h = train_model(m, vn['X_train'], vn['y_train'], vn['X_val'], vn['y_val'],
                    "HWF-PIKAN", physics_weight=PHYSICS_WEIGHT)
models['HWF-PIKAN'] = m
histories['HWF-PIKAN'] = h

# ============================================================================
# EVALUATE ON VIETNAM TEST SET
# ============================================================================

print("\n" + "=" * 80)
print("EVALUATION: VIETNAM TEST SET")
print("=" * 80)

results_rows = []

for name, model in models.items():
    metrics = evaluate_model(model, vn['X_test'], vn['y_test'], VN_SLR,
                             X_test_raw=vn.get('X_test_raw'))
    metrics['model'] = name
    results_rows.append(metrics)

    print(f"\n{name}:")
    print(f"   MAE:     {metrics['mae_amps']:.1f} A")
    print(f"   RMSE:    {metrics['rmse_amps']:.1f} A")
    print(f"   Bias:    {metrics['bias_amps']:+.1f} A")
    print(f"   Physics: {metrics['physics_residual']:.4f}")
    print(f"   Explain: {'YES' if metrics['explainable'] else 'NO'}")

# ============================================================================
# EVALUATE CROSS-REGION: VN-TRAINED MODELS ON US TEST SET
# ============================================================================

print("\n" + "=" * 80)
print("CROSS-REGION: VN-TRAINED -> US TEST SET")
print("=" * 80)

for name, model in models.items():
    metrics = evaluate_model(model, us['X_test'], us['y_test'], US_SLR,
                             X_test_raw=us.get('X_test_raw'))

    print(f"\n{name} (VN->US):")
    print(f"   MAE:     {metrics['mae_amps']:.1f} A")
    print(f"   RMSE:    {metrics['rmse_amps']:.1f} A")
    print(f"   Bias:    {metrics['bias_amps']:+.1f} A")

# ============================================================================
# TABLE 3: SUMMARY
# ============================================================================

print("\n" + "=" * 80)
print("TABLE 3: MODEL COMPARISON (Vietnam within-region)")
print("=" * 80)

print(f"\n{'Model':<20} {'Params':>8} {'MAE(A)':>8} {'RMSE(A)':>9} "
      f"{'Bias(A)':>9} {'PhysRes':>9} {'Explain':>8}")
print("-" * 80)

for r in results_rows:
    name = r['model']
    m = models[name]
    p = sum(pp.numel() for pp in m.parameters())
    print(f"{name:<20} {p:>8,} {r['mae_amps']:>8.1f} {r['rmse_amps']:>9.1f} "
          f"{r['bias_amps']:>+9.1f} {r['physics_residual']:>9.4f} "
          f"{'YES' if r['explainable'] else 'NO':>8}")

# ============================================================================
# EXPLAINABILITY AUDIT SAMPLE
# ============================================================================

print("\n" + "=" * 80)
print("EXPLAINABILITY AUDIT: Sample predictions with IEEE 738 decomposition")
print("=" * 80)

for name in ['VanillaPINN', 'HWF-PIKAN']:
    model = models[name]
    r = [x for x in results_rows if x['model'] == name][0]
    expl = r.get('explanation_sample')

    if expl is not None:
        print(f"\n--- {name} ---")
        for i in range(min(3, len(expl['q_convective']))):
            bc = "CONVECTION-LIMITED" if expl['binding_constraint'][i].item() < 0.5 else "RADIATION-LIMITED"
            print(f"  Sample {i+1}:")
            print(f"    q_convective:  {expl['q_convective'][i].item():8.3f} W/m")
            print(f"    q_radiative:   {expl['q_radiative'][i].item():8.3f} W/m")
            print(f"    q_solar:       {expl['q_solar'][i].item():8.3f} W/m")
            print(f"    q_joule:       {expl['q_joule'][i].item():8.3f} W/m")
            print(f"    conv_fraction: {expl['convective_fraction'][i].item():8.3f}")
            print(f"    BINDING:       {bc}")

# ============================================================================
# LEARNED PHYSICS PARAMETERS
# ============================================================================

print("\n" + "=" * 80)
print("LEARNED PHYSICS PARAMETERS (after training)")
print("=" * 80)

for name in ['VanillaPINN', 'HWF-PIKAN']:
    model = models[name]
    params = model.get_physics_params()
    print(f"\n{name}:")
    for k, v in params.items():
        print(f"   {k}: {v.item():.4f}")

# ============================================================================
# SAVE RESULTS
# ============================================================================

# CSV (aggregated metrics)
df_results = pd.DataFrame([{
    'model': r['model'],
    'mae_amps': r['mae_amps'],
    'rmse_amps': r['rmse_amps'],
    'bias_amps': r['bias_amps'],
    'physics_residual': r['physics_residual'],
    'explainable': r['explainable'],
} for r in results_rows])
df_results.to_csv(RESULTS_DIR / 'table3_baseline_comparison.csv', index=False)

# Per-sample predictions (needed for true paired t-tests in statistical_significance.py)
for r in results_rows:
    model_key = r['model'].lower().replace('-', '_').replace(' ', '_')
    np.save(RESULTS_DIR / f'{model_key}_pred_amps.npy', r['predictions_amps'])
np.save(RESULTS_DIR / 'true_amps.npy', results_rows[0]['actuals_amps'])
print(f"Saved: per-sample prediction arrays to {RESULTS_DIR}/")

# Learning curves figure
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

ax = axes[0]
for name, hist in histories.items():
    ax.plot(hist['val_loss'], label=name, linewidth=2)
ax.set_xlabel('Epoch')
ax.set_ylabel('Validation Loss (MSE)')
ax.set_yscale('log')
ax.legend()
ax.grid(True, alpha=0.3)
ax.set_title('Validation Loss')

ax = axes[1]
for name, hist in histories.items():
    if any(v > 0 for v in hist['physics_loss']):
        ax.plot(hist['physics_loss'], label=name, linewidth=2)
ax.set_xlabel('Epoch')
ax.set_ylabel('Physics Loss')
ax.legend()
ax.grid(True, alpha=0.3)
ax.set_title('Physics Constraint Violation')

plt.tight_layout()
plt.savefig(RESULTS_DIR / 'learning_curves.png', dpi=150)
print(f"\nSaved: {RESULTS_DIR}/learning_curves.png")
print(f"Saved: {RESULTS_DIR}/table3_baseline_comparison.csv")

print("\n" + "=" * 80)
print("EXPERIMENT 2 COMPLETE")
print("=" * 80)
