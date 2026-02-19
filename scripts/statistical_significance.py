#!/usr/bin/env python3
"""
Statistical Significance Analysis for Paper Revision

Preferred path: loads per-sample prediction arrays saved by experiment2_baseline_comparison.py
and runs true paired t-tests (scipy.stats.ttest_rel).

Fallback: if .npy files are not yet available, uses the delta-method approximation
with Var(|error|) = RMSE² - MAE² and assumed inter-model correlation rho = 0.9.

Results will be added to Table 1 with asterisks indicating significance levels:
* p < 0.05
** p < 0.01
*** p < 0.001
"""

import numpy as np
from pathlib import Path
from scipy import stats

SEED = 42
PRED_DIR = Path("cross_region_results/baseline_comparison")

# Summary statistics (from Experiment 2, N=2,626 samples)
# Used only when per-sample arrays are unavailable.
SUMMARY = {
    'VanillaMLP': {'mae': 100.0, 'rmse': 122.0, 'n': 1312},
    'VanillaPINN': {'mae': 98.3,  'rmse': 119.8, 'n': 1312},
    'HWF-PIKAN':  {'mae': 106.1, 'rmse': 133.8, 'n': 1312},
}

MODEL_KEYS = {
    'VanillaMLP': 'vanillamlp',
    'VanillaPINN': 'vanillapinn',
    'HWF-PIKAN': 'hwf_pikan',
}


# ─── helpers ─────────────────────────────────────────────────────────────────

def sig_marker(p):
    if p < 0.001: return "***"
    if p < 0.01:  return "**"
    if p < 0.05:  return "*"
    return "ns"


def fmt_p(p):
    if p < 1e-10: return "< 1e-10"
    if p < 0.001: return f"{p:.2e}"
    return f"{p:.4f}"


def mae_ci_delta(mae, rmse, n, z=1.96):
    """95% CI for MAE via CLT: SE(MAE) = sqrt(Var(|error|) / n),
    where Var(|error|) = RMSE² - MAE²  (exact when E[error]=0)."""
    var_abs = max(rmse**2 - mae**2, 0)
    se = np.sqrt(var_abs / n)
    return mae - z * se, mae + z * se


def mae_ci_bootstrap(abs_errors, n_boot=10_000, z=1.96, seed=SEED):
    """95% CI for MAE via nonparametric bootstrap on actual per-sample |errors|."""
    rng = np.random.default_rng(seed)
    n = len(abs_errors)
    boots = rng.choice(abs_errors, size=(n_boot, n), replace=True).mean(axis=1)
    return np.percentile(boots, [2.5, 97.5])


def paired_approx_ttest(mae1, rmse1, mae2, rmse2, n, rho=0.9):
    """Approximate paired t-test from summary statistics only.
    Var(|e1|-|e2|) = Var(|e1|) + Var(|e2|) - 2*rho*sqrt(Var(|e1|)*Var(|e2|))"""
    var1 = max(rmse1**2 - mae1**2, mae1**2 * 0.01)
    var2 = max(rmse2**2 - mae2**2, mae2**2 * 0.01)
    var_diff = var1 + var2 - 2 * rho * np.sqrt(var1 * var2)
    se = np.sqrt(max(var_diff, 0) / n)
    t = (mae1 - mae2) / se if se > 0 else np.inf
    p = 2 * (1 - stats.t.cdf(abs(t), df=n - 1))
    return t, p


# ─── load per-sample arrays if available ─────────────────────────────────────

errors = {}   # model_name -> abs error array
have_arrays = False

true_path = PRED_DIR / 'true_amps.npy'
if true_path.exists():
    true_amps = np.load(true_path)
    missing = []
    for name, key in MODEL_KEYS.items():
        pred_path = PRED_DIR / f'{key}_pred_amps.npy'
        if pred_path.exists():
            errors[name] = np.abs(np.load(pred_path) - true_amps)
        else:
            missing.append(name)
    if not missing:
        have_arrays = True

print("=" * 80)
print("STATISTICAL SIGNIFICANCE ANALYSIS")
if have_arrays:
    n_samples = len(true_amps)
    print(f"Method: TRUE paired t-test + bootstrap CIs  |  N={n_samples:,}")
else:
    print("Method: delta-method CIs + approximate paired t-test (no per-sample arrays)")
    print("        Run experiment2_baseline_comparison.py first for exact results.")
print("=" * 80)

# ─── 1. Confidence intervals ─────────────────────────────────────────────────

print("\n1. 95% CONFIDENCE INTERVALS")
print("-" * 80)

ci = {}
for model, m in SUMMARY.items():
    if have_arrays:
        lo, hi = mae_ci_bootstrap(errors[model])
        method = "bootstrap"
    else:
        lo, hi = mae_ci_delta(m['mae'], m['rmse'], m['n'])
        method = "delta method"
    ci[model] = (lo, hi)
    print(f"  {model:<14}  MAE = {m['mae']:.1f}  95% CI: [{lo:.1f}, {hi:.1f}]  "
          f"(±{(hi-lo)/2:.1f} A)  [{method}]")

# ─── 2. Pairwise significance tests ─────────────────────────────────────────

print("\n\n2. PAIRWISE PAIRED T-TESTS")
print("-" * 80)
if have_arrays:
    print("  Using scipy.stats.ttest_rel on actual per-sample |errors|\n")
else:
    print("  Approximation: rho = 0.9 assumed between |error| vectors\n")

comparisons = [
    ('VanillaMLP',  'VanillaPINN', 'VanillaMLP vs VanillaPINN'),
    ('HWF-PIKAN',   'VanillaPINN', 'HWF-PIKAN  vs VanillaPINN'),
    ('VanillaMLP',  'HWF-PIKAN',  'VanillaMLP vs HWF-PIKAN  '),
]

pvals = {}
for m1_key, m2_key, label in comparisons:
    m1, m2 = SUMMARY[m1_key], SUMMARY[m2_key]
    diff = m1['mae'] - m2['mae']

    if have_arrays:
        t, p = stats.ttest_rel(errors[m1_key], errors[m2_key])
    else:
        t, p = paired_approx_ttest(m1['mae'], m1['rmse'], m2['mae'], m2['rmse'], m1['n'])

    pvals[(m1_key, m2_key)] = p
    print(f"  {label}:  ΔMAE = {diff:+.1f} A  |  t = {t:.2f}  |  "
          f"p = {fmt_p(p)}  |  {sig_marker(p)}")

# ─── 3. Table ────────────────────────────────────────────────────────────────

print("\n\n3. TABLE 1 WITH SIGNIFICANCE MARKERS")
print("=" * 80)

header = f"{'Model':<14} {'MAE (A)':<26} {'RMSE (A)':<12} {'vs VanillaPINN'}"
print(header)
print("-" * len(header))

for model, m in SUMMARY.items():
    lo, hi = ci[model]
    mae_str = f"{m['mae']:.1f} [{lo:.1f}, {hi:.1f}]"

    if model == 'VanillaPINN':
        sig_str = "— (reference)"
    else:
        key = (model, 'VanillaPINN')
        alt = ('VanillaPINN', model)
        p = pvals[key] if key in pvals else pvals.get(alt)
        sig_str = f"{sig_marker(p)} (p={fmt_p(p)})" if p is not None else "?"

    print(f"  {model:<14} {mae_str:<26} {m['rmse']:<12.1f} {sig_str}")

# ─── 4. Paper text ───────────────────────────────────────────────────────────

mlp_p   = pvals[('VanillaMLP', 'VanillaPINN')]
pikan_p = pvals[('HWF-PIKAN',  'VanillaPINN')]
lo_pinn, hi_pinn   = ci['VanillaPINN']
lo_pikan, hi_pikan = ci['HWF-PIKAN']
mlp_sig = "statistically significant" if mlp_p < 0.05 else "not statistically significant"
test_label = "paired t-test" if have_arrays else "approximate paired t-test"
n_label = len(true_amps) if have_arrays else SUMMARY['VanillaMLP']['n']

print(f"""

4. SUGGESTED PAPER TEXT (Section 5.1.1)
{'=' * 80}

"VanillaPINN achieves 98.3 A MAE (95% CI: [{lo_pinn:.1f}, {hi_pinn:.1f}] A),
outperforming VanillaMLP by 1.7 A; this difference is {mlp_sig}
({sig_marker(mlp_p)}, p={fmt_p(mlp_p)}, {test_label}, n={n_label:,}). HWF-PIKAN's
106.1 A MAE (95% CI: [{lo_pikan:.1f}, {hi_pikan:.1f}] A) represents a 7.8 A penalty
relative to VanillaPINN ({sig_marker(pikan_p)}, p={fmt_p(pikan_p)}). The accuracy cost for
parameter auditability is robust and not attributable to sampling variation."
""")

if not have_arrays:
    print(f"NOTE: Run experiment2_baseline_comparison.py to generate per-sample arrays in")
    print(f"      {PRED_DIR}/  — rerun this script for exact (non-approximate) results.")
    print(f"      The MLP vs PINN p-value ({fmt_p(mlp_p)}) is sensitive to the rho=0.9 assumption.")

ci_note = "nonparametric bootstrap" if have_arrays else "delta method (Var(|e|)=RMSE²−MAE²)"
test_note = "scipy.stats.ttest_rel on per-sample |errors|" if have_arrays else "approximated from summary statistics, rho=0.9"
print(f"""
Footnote: CIs via {ci_note}. Paired t-test: {test_note}.
""")

print("=" * 80)
print("DONE.")
