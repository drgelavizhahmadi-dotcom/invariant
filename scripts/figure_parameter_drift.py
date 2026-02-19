#!/usr/bin/env python3
"""
Figure: Physics Parameter Drift Comparison

Shows how VanillaPINN's learned absorptivity drifts 21% from the documented
prior (DynaLiRD Table 2) while HWF-PIKAN stays within 1%.

This is the AI Act auditability argument: a regulator can verify that the
model's internal physics parameters match documented conductor specifications.
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

RESULTS_DIR = Path("cross_region_results/baseline_comparison")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Documented priors from DynaLiRD Table 2
PRIOR_ABSORPTIVITY = 0.90
PRIOR_EMISSIVITY = 0.70
PRIOR_RESISTANCE = 1.00  # factor = 1.0 means no scaling

# Learned values from Experiment 2
learned = {
    'VanillaPINN': {
        'absorptivity': 0.7139,
        'emissivity': 0.6900,
        'resistance_factor': 1.1178,
    },
    'HWF-PIKAN': {
        'absorptivity': 0.9101,
        'emissivity': 0.7000,
        'resistance_factor': 1.1206,
    },
}

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# --- Panel 1: Absorptivity ---
ax = axes[0]
prior = PRIOR_ABSORPTIVITY
zone = 0.10  # ±10% compliance zone

ax.axhline(prior, color='#2ecc71', linestyle='--', linewidth=2, label=f'Documented prior ({prior})')
ax.axhspan(prior * (1 - zone), prior * (1 + zone), alpha=0.15, color='#2ecc71',
           label=f'±10% compliance zone')

pinn_val = learned['VanillaPINN']['absorptivity']
hwf_val = learned['HWF-PIKAN']['absorptivity']
pinn_drift = abs(pinn_val - prior) / prior * 100
hwf_drift = abs(hwf_val - prior) / prior * 100

ax.bar(0, pinn_val, width=0.6, color='#e74c3c', alpha=0.8, edgecolor='black', linewidth=1.2)
ax.bar(1, hwf_val, width=0.6, color='#3498db', alpha=0.8, edgecolor='black', linewidth=1.2)

ax.text(0, pinn_val + 0.01, f'{pinn_drift:.0f}% drift', ha='center', fontsize=11,
        fontweight='bold', color='#e74c3c')
ax.text(1, hwf_val + 0.01, f'{hwf_drift:.0f}% drift', ha='center', fontsize=11,
        fontweight='bold', color='#3498db')

ax.set_xticks([0, 1])
ax.set_xticklabels(['VanillaPINN', 'HWF-PIKAN'], fontsize=11)
ax.set_ylabel('Learned Absorptivity', fontsize=12)
ax.set_title('Absorptivity', fontsize=13, fontweight='bold')
ax.set_ylim(0.55, 1.05)
ax.legend(loc='lower left', fontsize=9)
ax.grid(axis='y', alpha=0.3)

# --- Panel 2: Emissivity ---
ax = axes[1]
prior = PRIOR_EMISSIVITY

ax.axhline(prior, color='#2ecc71', linestyle='--', linewidth=2, label=f'Documented prior ({prior})')
ax.axhspan(prior * (1 - zone), prior * (1 + zone), alpha=0.15, color='#2ecc71',
           label=f'±10% compliance zone')

pinn_val = learned['VanillaPINN']['emissivity']
hwf_val = learned['HWF-PIKAN']['emissivity']
pinn_drift = abs(pinn_val - prior) / prior * 100
hwf_drift = abs(hwf_val - prior) / prior * 100

ax.bar(0, pinn_val, width=0.6, color='#e74c3c', alpha=0.8, edgecolor='black', linewidth=1.2)
ax.bar(1, hwf_val, width=0.6, color='#3498db', alpha=0.8, edgecolor='black', linewidth=1.2)

ax.text(0, pinn_val + 0.01, f'{pinn_drift:.1f}% drift', ha='center', fontsize=11,
        fontweight='bold', color='#e74c3c')
ax.text(1, hwf_val + 0.01, f'{hwf_drift:.1f}% drift', ha='center', fontsize=11,
        fontweight='bold', color='#3498db')

ax.set_xticks([0, 1])
ax.set_xticklabels(['VanillaPINN', 'HWF-PIKAN'], fontsize=11)
ax.set_ylabel('Learned Emissivity', fontsize=12)
ax.set_title('Emissivity', fontsize=13, fontweight='bold')
ax.set_ylim(0.50, 0.90)
ax.legend(loc='lower left', fontsize=9)
ax.grid(axis='y', alpha=0.3)

# --- Panel 3: Resistance Factor ---
ax = axes[2]
prior = PRIOR_RESISTANCE

ax.axhline(prior, color='#2ecc71', linestyle='--', linewidth=2, label=f'Documented prior ({prior})')
ax.axhspan(prior * (1 - zone), prior * (1 + zone), alpha=0.15, color='#2ecc71',
           label=f'±10% compliance zone')

pinn_val = learned['VanillaPINN']['resistance_factor']
hwf_val = learned['HWF-PIKAN']['resistance_factor']
pinn_drift = abs(pinn_val - prior) / prior * 100
hwf_drift = abs(hwf_val - prior) / prior * 100

ax.bar(0, pinn_val, width=0.6, color='#e74c3c', alpha=0.8, edgecolor='black', linewidth=1.2)
ax.bar(1, hwf_val, width=0.6, color='#3498db', alpha=0.8, edgecolor='black', linewidth=1.2)

ax.text(0, pinn_val + 0.01, f'{pinn_drift:.0f}% drift', ha='center', fontsize=11,
        fontweight='bold', color='#e74c3c')
ax.text(1, hwf_val + 0.01, f'{hwf_drift:.0f}% drift', ha='center', fontsize=11,
        fontweight='bold', color='#3498db')

ax.set_xticks([0, 1])
ax.set_xticklabels(['VanillaPINN', 'HWF-PIKAN'], fontsize=11)
ax.set_ylabel('Learned Resistance Factor', fontsize=12)
ax.set_title('Resistance Factor', fontsize=13, fontweight='bold')
ax.set_ylim(0.80, 1.30)
ax.legend(loc='lower left', fontsize=9)
ax.grid(axis='y', alpha=0.3)

fig.suptitle('Physics Parameter Auditability: Drift from Documented Priors\n'
             '(DynaLiRD Table 2 / NREL specifications)',
             fontsize=14, fontweight='bold', y=1.02)

plt.tight_layout()
plt.savefig(RESULTS_DIR / 'parameter_drift_comparison.png', dpi=300, bbox_inches='tight')
print(f"Saved: {RESULTS_DIR}/parameter_drift_comparison.png")

# Print summary
print("\nParameter Drift Summary:")
print(f"{'Parameter':<20} {'Prior':>8} {'VanillaPINN':>14} {'HWF-PIKAN':>12} {'Winner':>10}")
print("-" * 70)
for param in ['absorptivity', 'emissivity', 'resistance_factor']:
    prior_val = {'absorptivity': PRIOR_ABSORPTIVITY, 'emissivity': PRIOR_EMISSIVITY,
                 'resistance_factor': PRIOR_RESISTANCE}[param]
    pinn = learned['VanillaPINN'][param]
    hwf = learned['HWF-PIKAN'][param]
    pinn_d = abs(pinn - prior_val) / prior_val * 100
    hwf_d = abs(hwf - prior_val) / prior_val * 100
    winner = 'HWF-PIKAN' if hwf_d < pinn_d else 'VanillaPINN'
    print(f"{param:<20} {prior_val:>8.3f} {pinn:>8.4f} ({pinn_d:>4.1f}%) "
          f"{hwf:>7.4f} ({hwf_d:>4.1f}%) {winner:>10}")
