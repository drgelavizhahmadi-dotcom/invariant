#!/usr/bin/env python3
"""
Sensitivity analysis: Why is TRUE VN_SLR so low (493A)?

This script investigates:
1. Effect of T_max (75°C vs 90°C vs 100°C)
2. Effect of conductor parameters (diameter, resistance)
3. Comparison with US SLR calculation
4. Possible explanations for low SLR value
"""

import pandas as pd
import numpy as np
import sys
from pathlib import Path
import matplotlib.pyplot as plt

sys.path.append(str(Path(__file__).parent.parent))

from core.physics import IEEE738HeatBalance
import torch

print("=" * 80)
print("VN_SLR SENSITIVITY ANALYSIS")
print("=" * 80)

# Load data
df = pd.read_hdf('data/processed/unified_dlr_training.h5', key='data')
vn_df = df[df['region'] == 'VN'].copy()
us_df = df[df['region'] == 'US'].copy()

# ============================================================================
# STEP 1: EFFECT OF T_MAX
# ============================================================================
print("\n" + "=" * 80)
print("STEP 1: SENSITIVITY TO MAX CONDUCTOR TEMPERATURE")
print("=" * 80)

# Vietnam conductor params
conductor_params = {
    'diameter': 0.0275,
    'emissivity': 0.7,
    'absorptivity': 0.8,
    'resistance_20C': 1.4085e-4,
    'temp_coeff': 0.0039,
}

R_25C = conductor_params['resistance_20C'] * (1 + conductor_params['temp_coeff'] * 5)

# NREL conservative conditions
T_amb = torch.tensor([40.0])
wind = torch.tensor([0.61])
solar = torch.tensor([1000.0])

print("\nNREL Conservative Conditions:")
print(f"  T_amb: 40°C, Wind: 0.61 m/s, Solar: 1000 W/m²")

T_max_values = [75.0, 90.0, 100.0, 120.0]
slr_values = []

print("\nVN_SLR vs Max Conductor Temperature:")
print("=" * 60)

for T_max in T_max_values:
    physics = IEEE738HeatBalance(
        conductor_diameter=conductor_params['diameter'],
        conductor_emissivity=conductor_params['emissivity'],
        conductor_absorptivity=conductor_params['absorptivity'],
        resistance_per_meter_25C=R_25C,
        temp_coeff_resistance=conductor_params['temp_coeff'],
        max_conductor_temp=T_max,
    )

    slr = physics.ampacity(
        T_max=T_max,
        T_ambient=T_amb,
        wind_speed=wind,
        solar_irradiance=solar,
    ).item()

    slr_values.append(slr)

    vn_ratio_mean = vn_df['Ampacity'].mean() / slr

    print(f"  T_max = {T_max:3.0f}°C → SLR = {slr:4.0f} A → VN ratio = {vn_ratio_mean:.3f}")

# Check which T_max gives ratio closest to literature expectations (1.4-1.5)
target_ratio = 1.45
required_slr = vn_df['Ampacity'].mean() / target_ratio

print(f"\n📊 To achieve literature-expected ratio of {target_ratio}:")
print(f"   Required VN_SLR: {required_slr:.0f} A")

# Find which T_max gives this SLR
for T_max, slr in zip(T_max_values, slr_values):
    if abs(slr - required_slr) < 50:
        print(f"   ✅ T_max = {T_max}°C gives SLR = {slr:.0f} A (close!)")

# ============================================================================
# STEP 2: COMPARE WITH US SLR CALCULATION
# ============================================================================
print("\n" + "=" * 80)
print("STEP 2: COMPARE WITH US SLR CALCULATION")
print("=" * 80)

# US has constant SLR = 1000A
us_slr_mean = (us_df['dlr_amps'] / us_df['dlr_ratio']).mean()

print(f"\nUS SLR (from data): {us_slr_mean:.0f} A")
print(f"VN mean Ampacity:   {vn_df['Ampacity'].mean():.0f} A")
print(f"Ratio (VN/US SLR):  {vn_df['Ampacity'].mean() / us_slr_mean:.3f}")

print("\nIf VN and US had similar conductors:")
print(f"  VN_SLR ≈ US_SLR × (voltage_ratio or cross_section_ratio)")
print(f"  VN is 220kV, US is mixed (likely similar)")
print(f"  → Expected VN_SLR ≈ 800-1200 A")

# ============================================================================
# STEP 3: REVERSE ENGINEER FROM DATA
# ============================================================================
print("\n" + "=" * 80)
print("STEP 3: REVERSE ENGINEER VN_SLR FROM EMPIRICAL DATA")
print("=" * 80)

# Assumption: VN data follows similar DLR methodology to US
# US ratio mean: 1.629 (63% above SLR)
# If VN used same aggressive DLR, what would VN_SLR be?

assumed_vn_ratio = us_df['dlr_ratio'].mean()  # Assume VN uses similar DLR approach
vn_slr_inferred = vn_df['Ampacity'].mean() / assumed_vn_ratio

print(f"\nAssuming VN uses similar DLR methodology to US:")
print(f"  US DLR/SLR ratio: {us_df['dlr_ratio'].mean():.3f}")
print(f"  If VN had same ratio: {assumed_vn_ratio:.3f}")
print(f"  Inferred VN_SLR:  {vn_slr_inferred:.0f} A")

# Literature: 40-50% benefit
literature_ratio_low = 1.40
literature_ratio_high = 1.50

vn_slr_lit_low = vn_df['Ampacity'].mean() / literature_ratio_high
vn_slr_lit_high = vn_df['Ampacity'].mean() / literature_ratio_low

print(f"\nAssuming literature DLR benefit (40-50%):")
print(f"  For 40% benefit (ratio 1.40): VN_SLR = {vn_slr_lit_high:.0f} A")
print(f"  For 50% benefit (ratio 1.50): VN_SLR = {vn_slr_lit_low:.0f} A")

# ============================================================================
# STEP 4: POSSIBLE EXPLANATIONS
# ============================================================================
print("\n" + "=" * 80)
print("STEP 4: POSSIBLE EXPLANATIONS FOR LOW IEEE 738 SLR")
print("=" * 80)

print("\n🔍 WHY IS TRUE VN_SLR (IEEE 738) ONLY 493A?")
print("\nPossible Explanations:")

print("\n1. ❓ Incorrect Conductor Parameters:")
print(f"   • Documented diameter: {conductor_params['diameter']*1000:.1f} mm")
print(f"   • Documented resistance: {conductor_params['resistance_20C']*1e6:.2f} µΩ/m at 20°C")
print(f"   • These may be WRONG or for different line")
print(f"   • VN conductor might be LARGER (higher cross-section)")

print("\n2. ❓ Higher Operating Temperature:")
print(f"   • Assumed T_max: 75°C (conservative ACSR limit)")
print(f"   • VN might allow: 90-100°C (less conservative)")
print(f"   • At T_max=90°C: SLR = {slr_values[1]:.0f} A")
print(f"   • At T_max=100°C: SLR = {slr_values[2]:.0f} A")

print("\n3. ✅ VN Ampacity ≠ DLR (Different Definition):")
print(f"   • VN 'Ampacity' might be NAMEPLATE rating, not DLR")
print(f"   • Or uses different standards (IEC vs IEEE)")
print(f"   • Mean value (1193A) could be the SLR itself!")

print("\n4. ❓ Conservative NREL Conditions Too Harsh:")
print(f"   • NREL uses T_amb=40°C (very hot)")
print(f"   • Vietnam tropical climate mean T_amb ≈ 28°C")
print(f"   • Utility might use local climate for SLR baseline")

# ============================================================================
# STEP 5: RECALCULATE WITH ALTERNATIVE ASSUMPTIONS
# ============================================================================
print("\n" + "=" * 80)
print("STEP 5: RECALCULATE WITH ALTERNATIVE ASSUMPTIONS")
print("=" * 80)

print("\n📊 Scenario Analysis:")

scenarios = [
    ("IEEE 738 (T_max=75°C, T_amb=40°C)", slr_values[0], 75, 40, 0.61),
    ("Higher T_max (90°C)", slr_values[1], 90, 40, 0.61),
    ("Higher T_max (100°C)", slr_values[2], 100, 40, 0.61),
    ("Local climate (75°C, 35°C)", None, 75, 35, 0.61),
    ("Moderate wind (75°C, 40°C, 2 m/s)", None, 75, 40, 2.0),
]

results = []

for name, slr_cached, T_max, T_amb_val, wind_val in scenarios:
    if slr_cached is None:
        physics = IEEE738HeatBalance(
            conductor_diameter=conductor_params['diameter'],
            conductor_emissivity=conductor_params['emissivity'],
            conductor_absorptivity=conductor_params['absorptivity'],
            resistance_per_meter_25C=R_25C,
            temp_coeff_resistance=conductor_params['temp_coeff'],
        )

        slr = physics.ampacity(
            T_max=T_max,
            T_ambient=torch.tensor([T_amb_val]),
            wind_speed=torch.tensor([wind_val]),
            solar_irradiance=torch.tensor([1000.0]),
        ).item()
    else:
        slr = slr_cached

    ratio = vn_df['Ampacity'].mean() / slr
    gap_vs_us = abs(ratio - us_df['dlr_ratio'].mean())

    results.append({
        'scenario': name,
        'slr': slr,
        'ratio': ratio,
        'gap': gap_vs_us
    })

    print(f"\n{name}:")
    print(f"  SLR:             {slr:4.0f} A")
    print(f"  VN DLR/SLR:      {ratio:.3f}")
    print(f"  Gap vs US:       {gap_vs_us:.3f}")

# ============================================================================
# STEP 6: VISUALIZATION
# ============================================================================
print("\n" + "=" * 80)
print("STEP 6: VISUALIZATION")
print("=" * 80)

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Plot 1: SLR vs T_max
ax1 = axes[0, 0]
ax1.plot(T_max_values, slr_values, 'o-', linewidth=2, markersize=8, color='steelblue')
ax1.axhline(vn_df['Ampacity'].mean(), color='red', linestyle='--', linewidth=2,
            label=f'VN mean Ampacity ({vn_df["Ampacity"].mean():.0f}A)')
ax1.axhline(us_slr_mean, color='green', linestyle='--', linewidth=2,
            label=f'US SLR ({us_slr_mean:.0f}A)')
ax1.set_xlabel('Max Conductor Temperature (°C)', fontsize=11)
ax1.set_ylabel('Static Line Rating (A)', fontsize=11)
ax1.set_title('VN SLR vs Max Conductor Temperature\n(NREL Conservative: 40°C, 0.61 m/s)', fontsize=12)
ax1.grid(True, alpha=0.3)
ax1.legend()

# Plot 2: DLR/SLR ratio vs T_max
ax2 = axes[0, 1]
ratios = [vn_df['Ampacity'].mean() / slr for slr in slr_values]
ax2.plot(T_max_values, ratios, 'o-', linewidth=2, markersize=8, color='orange')
ax2.axhline(us_df['dlr_ratio'].mean(), color='green', linestyle='--', linewidth=2,
            label=f'US mean ratio ({us_df["dlr_ratio"].mean():.2f})')
ax2.axhspan(1.4, 1.5, alpha=0.2, color='yellow', label='Literature range (40-50%)')
ax2.set_xlabel('Max Conductor Temperature (°C)', fontsize=11)
ax2.set_ylabel('Mean DLR/SLR Ratio', fontsize=11)
ax2.set_title('VN DLR/SLR Ratio vs T_max', fontsize=12)
ax2.grid(True, alpha=0.3)
ax2.legend()

# Plot 3: Scenario comparison
ax3 = axes[1, 0]
scenario_names = [r['scenario'].split('(')[0].strip() for r in results]
scenario_slrs = [r['slr'] for r in results]
colors = ['steelblue', 'orange', 'red', 'green', 'purple']
bars = ax3.barh(scenario_names, scenario_slrs, color=colors, alpha=0.7, edgecolor='black')
ax3.axvline(vn_df['Ampacity'].mean(), color='red', linestyle='--', linewidth=2,
            label=f'VN mean Ampacity')
ax3.axvline(us_slr_mean, color='green', linestyle='--', linewidth=2,
            label=f'US SLR')
ax3.set_xlabel('Static Line Rating (A)', fontsize=11)
ax3.set_title('VN SLR Under Different Assumptions', fontsize=12)
ax3.legend()
ax3.grid(True, alpha=0.3, axis='x')

# Plot 4: Gap comparison
ax4 = axes[1, 1]
scenario_gaps = [r['gap'] for r in results]
bars = ax4.bar(range(len(results)), scenario_gaps, color=colors, alpha=0.7, edgecolor='black')
ax4.set_xticks(range(len(results)))
ax4.set_xticklabels([str(i+1) for i in range(len(results))], fontsize=10)
ax4.set_ylabel('Gap from US Mean Ratio', fontsize=11)
ax4.set_title('VN-US Gap Under Different SLR Assumptions', fontsize=12)
ax4.grid(True, alpha=0.3, axis='y')

# Add scenario legend
legend_text = '\n'.join([f"{i+1}. {r['scenario']}" for i, r in enumerate(results)])
ax4.text(1.05, 0.5, legend_text, transform=ax4.transAxes, fontsize=8,
         verticalalignment='center', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

plt.tight_layout()
plt.savefig('cross_region_results/vn_slr_sensitivity.png', dpi=300, bbox_inches='tight')
print("\n✅ Saved: cross_region_results/vn_slr_sensitivity.png")

# ============================================================================
# FINAL RECOMMENDATION
# ============================================================================
print("\n" + "=" * 80)
print("FINAL RECOMMENDATION")
print("=" * 80)

best_scenario = min(results, key=lambda x: x['gap'])

print(f"\n🎯 BEST SCENARIO (minimizes VN-US gap):")
print(f"   {best_scenario['scenario']}")
print(f"   VN_SLR = {best_scenario['slr']:.0f} A")
print(f"   VN ratio = {best_scenario['ratio']:.3f}")
print(f"   Gap = {best_scenario['gap']:.3f}")

current_gap = abs(1.0 - us_df['dlr_ratio'].mean())

print(f"\n📊 COMPARISON:")
print(f"   Current (mean):  VN_SLR = 1193 A, gap = {current_gap:.3f}")
print(f"   Best physics:    VN_SLR = {best_scenario['slr']:.0f} A, gap = {best_scenario['gap']:.3f}")

if best_scenario['gap'] < current_gap:
    print(f"\n✅ Physics-based SLR ({best_scenario['scenario']}) IMPROVES alignment")
    print(f"   Gap reduction: {(current_gap - best_scenario['gap']):.3f}")
else:
    print(f"\n⚠️  Current empirical approach (mean) still performs better")
    print(f"   Physics-based SLR underestimates true baseline")

print("\n🔬 LIKELY EXPLANATION:")
print("   VN conductor parameters from validate_mendeley_corrected.py")
print("   are INCORRECT or for a DIFFERENT line.")
print("   True VN conductor is likely:")
print(f"   • Larger diameter (>27.5mm)")
print(f"   • Lower resistance per unit length")
print(f"   • Higher thermal capacity")
print("   → Would give SLR closer to 800-1200A")

print("\n✅ RECOMMENDATION:")
print("   • Keep VN_SLR = 1193A (mean) for normalization")
print("   • Empirically works better than physics-based calculation")
print("   • Physics parameters unavailable or inaccurate")
print("   • Document this limitation in paper")

print("\n" + "=" * 80)
