#!/usr/bin/env python3
"""
Verify target variable mismatch between Vietnam and US datasets
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

print("=" * 70)
print("TARGET VARIABLE VERIFICATION")
print("=" * 70)

# Load data
df = pd.read_hdf('data/processed/unified_dlr_training.h5', key='data')
vn_df = df[df['region'] == 'VN'].copy()
us_df = df[df['region'] == 'US'].copy()

print(f"\nDataset sizes:")
print(f"  Vietnam: {len(vn_df):,} samples")
print(f"  US:      {len(us_df):,} samples")

# Show all columns
print("\n" + "=" * 70)
print("VIETNAM COLUMNS")
print("=" * 70)
print(vn_df.columns.tolist())

print("\n" + "=" * 70)
print("US COLUMNS")
print("=" * 70)
print(us_df.columns.tolist())

# Find target column
vn_target_candidates = ['Ampacity', 'ampacity', 'target', 'dlr_ampacity', 'actual']
us_target_candidates = ['actual', 'dlr_ampacity', 'Ampacity', 'ampacity', 'target']

vn_target_col = None
us_target_col = None

for col in vn_target_candidates:
    if col in vn_df.columns:
        vn_target_col = col
        break

for col in us_target_candidates:
    if col in us_df.columns:
        us_target_col = col
        break

if vn_target_col is None or us_target_col is None:
    print("\n❌ ERROR: Could not find target columns!")
    print(f"   Vietnam candidates checked: {vn_target_candidates}")
    print(f"   US candidates checked: {us_target_candidates}")
    exit(1)

print("\n" + "=" * 70)
print("TARGET COLUMN IDENTIFICATION")
print("=" * 70)
print(f"Vietnam target column: '{vn_target_col}'")
print(f"US target column:      '{us_target_col}'")

if vn_target_col != us_target_col:
    print("\n⚠️  WARNING: Different column names used for targets!")
    print(f"   This is a strong indicator of semantic mismatch.")

# Get target data
vn_target = vn_df[vn_target_col].dropna()
us_target = us_df[us_target_col].dropna()

# Vietnam statistics
print("\n" + "=" * 70)
print("VIETNAM TARGET STATISTICS")
print("=" * 70)
print(f"Column: '{vn_target_col}'")
print(f"Count:  {len(vn_target):,} samples")
print(f"Mean:   {vn_target.mean():.1f} A")
print(f"Median: {vn_target.median():.1f} A")
print(f"Std:    {vn_target.std():.1f} A")
print(f"Min:    {vn_target.min():.1f} A")
print(f"Max:    {vn_target.max():.1f} A")
print(f"25%:    {vn_target.quantile(0.25):.1f} A")
print(f"75%:    {vn_target.quantile(0.75):.1f} A")

# US statistics
print("\n" + "=" * 70)
print("US TARGET STATISTICS")
print("=" * 70)
print(f"Column: '{us_target_col}'")
print(f"Count:  {len(us_target):,} samples")
print(f"Mean:   {us_target.mean():.1f} A")
print(f"Median: {us_target.median():.1f} A")
print(f"Std:    {us_target.std():.1f} A")
print(f"Min:    {us_target.min():.1f} A")
print(f"Max:    {us_target.max():.1f} A")
print(f"25%:    {us_target.quantile(0.25):.1f} A")
print(f"75%:    {us_target.quantile(0.75):.1f} A")

# Comparison
print("\n" + "=" * 70)
print("COMPARISON")
print("=" * 70)

mean_diff = us_target.mean() - vn_target.mean()
mean_pct = (mean_diff / vn_target.mean()) * 100

print(f"\nMean difference:")
print(f"  US - VN: {mean_diff:+.1f} A ({mean_pct:+.1f}%)")

median_diff = us_target.median() - vn_target.median()
median_pct = (median_diff / vn_target.median()) * 100

print(f"\nMedian difference:")
print(f"  US - VN: {median_diff:+.1f} A ({median_pct:+.1f}%)")

# Check for overlap
vn_range = (vn_target.min(), vn_target.max())
us_range = (us_target.min(), us_target.max())

overlap_min = max(vn_range[0], us_range[0])
overlap_max = min(vn_range[1], us_range[1])

if overlap_min < overlap_max:
    overlap_pct_vn = ((overlap_max - overlap_min) / (vn_range[1] - vn_range[0])) * 100
    overlap_pct_us = ((overlap_max - overlap_min) / (us_range[1] - us_range[0])) * 100
    print(f"\nRange overlap:")
    print(f"  VN range: [{vn_range[0]:.1f}, {vn_range[1]:.1f}] A")
    print(f"  US range: [{us_range[0]:.1f}, {us_range[1]:.1f}] A")
    print(f"  Overlap:  [{overlap_min:.1f}, {overlap_max:.1f}] A")
    print(f"  Overlap coverage: VN {overlap_pct_vn:.1f}%, US {overlap_pct_us:.1f}%")
else:
    print(f"\n❌ NO OVERLAP in target ranges!")
    print(f"  VN range: [{vn_range[0]:.1f}, {vn_range[1]:.1f}] A")
    print(f"  US range: [{us_range[0]:.1f}, {us_range[1]:.1f}] A")

# Interpretation
print("\n" + "=" * 70)
print("INTERPRETATION")
print("=" * 70)

if abs(mean_pct) > 20:
    print(f"\n🔴 SEVERE TARGET MISMATCH DETECTED!")
    print(f"   US targets are {abs(mean_pct):.1f}% {'higher' if mean_pct > 0 else 'lower'} than VN targets")
    print("\n   This strongly suggests:")
    if mean_pct > 0:
        print("   • US and VN measure different quantities (e.g., actual vs rating)")
        print("   • US uses different conductor types with higher capacity")
        print("   • US applies different safety margins or standards")
    else:
        print("   • VN uses more conservative ratings")
        print("   • Different measurement methodologies")
    print("\n   ❌ Cross-region transfer is IMPOSSIBLE without target normalization")
elif abs(mean_pct) > 10:
    print(f"\n🟠 MODERATE TARGET MISMATCH")
    print(f"   {abs(mean_pct):.1f}% difference may cause transfer issues")
else:
    print(f"\n✅ TARGETS ARE COMPARABLE")
    print(f"   Only {abs(mean_pct):.1f}% difference - within acceptable range")

# Visualizations
print("\n" + "=" * 70)
print("GENERATING VISUALIZATIONS")
print("=" * 70)

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Histogram comparison
ax1 = axes[0, 0]
vn_target.hist(bins=50, ax=ax1, color='blue', alpha=0.6, label=f'Vietnam ({vn_target_col})', density=True)
us_target.hist(bins=50, ax=ax1, color='red', alpha=0.6, label=f'US ({us_target_col})', density=True)
ax1.axvline(vn_target.mean(), color='blue', linestyle='--', linewidth=2, label=f'VN mean: {vn_target.mean():.0f}A')
ax1.axvline(us_target.mean(), color='red', linestyle='--', linewidth=2, label=f'US mean: {us_target.mean():.0f}A')
ax1.set_title('Target Distribution Comparison', fontsize=14, fontweight='bold')
ax1.set_xlabel('Current (A)')
ax1.set_ylabel('Density')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Box plot comparison
ax2 = axes[0, 1]
bp_data = [vn_target, us_target]
bp = ax2.boxplot(bp_data, labels=['Vietnam', 'US'], patch_artist=True)
bp['boxes'][0].set_facecolor('blue')
bp['boxes'][1].set_facecolor('red')
for box in bp['boxes']:
    box.set_alpha(0.6)
ax2.set_title('Target Range Comparison', fontsize=14, fontweight='bold')
ax2.set_ylabel('Current (A)')
ax2.grid(True, alpha=0.3, axis='y')

# CDF comparison
ax3 = axes[1, 0]
vn_sorted = np.sort(vn_target)
us_sorted = np.sort(us_target)
vn_cdf = np.arange(1, len(vn_sorted) + 1) / len(vn_sorted)
us_cdf = np.arange(1, len(us_sorted) + 1) / len(us_sorted)
ax3.plot(vn_sorted, vn_cdf, color='blue', linewidth=2, label='Vietnam', alpha=0.7)
ax3.plot(us_sorted, us_cdf, color='red', linewidth=2, label='US', alpha=0.7)
ax3.set_title('Cumulative Distribution Function', fontsize=14, fontweight='bold')
ax3.set_xlabel('Current (A)')
ax3.set_ylabel('Cumulative Probability')
ax3.legend()
ax3.grid(True, alpha=0.3)

# Scatter: percentile comparison
ax4 = axes[1, 1]
percentiles = np.arange(0, 101, 5)
vn_percentiles = np.percentile(vn_target, percentiles)
us_percentiles = np.percentile(us_target, percentiles)
ax4.scatter(vn_percentiles, us_percentiles, alpha=0.6, s=50, color='purple')
# Add diagonal line (what we'd see if distributions matched)
min_val = min(vn_percentiles.min(), us_percentiles.min())
max_val = max(vn_percentiles.max(), us_percentiles.max())
ax4.plot([min_val, max_val], [min_val, max_val], 'k--', linewidth=2, alpha=0.5, label='Perfect match')
ax4.set_title('Percentile-to-Percentile Comparison', fontsize=14, fontweight='bold')
ax4.set_xlabel('Vietnam Percentile Value (A)')
ax4.set_ylabel('US Percentile Value (A)')
ax4.legend()
ax4.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('target_variable_comparison.png', dpi=300, bbox_inches='tight')
print("✅ Saved: target_variable_comparison.png")

# Summary statistics table
print("\n" + "=" * 70)
print("SUMMARY TABLE")
print("=" * 70)
print("\n| Statistic | Vietnam | US | Difference | Pct Diff |")
print("|-----------|---------|-----|------------|----------|")
print(f"| Mean | {vn_target.mean():.1f} A | {us_target.mean():.1f} A | {mean_diff:+.1f} A | {mean_pct:+.1f}% |")
print(f"| Median | {vn_target.median():.1f} A | {us_target.median():.1f} A | {median_diff:+.1f} A | {median_pct:+.1f}% |")
print(f"| Std Dev | {vn_target.std():.1f} A | {us_target.std():.1f} A | {us_target.std()-vn_target.std():+.1f} A | {((us_target.std()-vn_target.std())/vn_target.std()*100):+.1f}% |")
print(f"| Min | {vn_target.min():.1f} A | {us_target.min():.1f} A | {us_target.min()-vn_target.min():+.1f} A | {((us_target.min()-vn_target.min())/vn_target.min()*100):+.1f}% |")
print(f"| Max | {vn_target.max():.1f} A | {us_target.max():.1f} A | {us_target.max()-vn_target.max():+.1f} A | {((us_target.max()-vn_target.max())/vn_target.max()*100):+.1f}% |")

print("\n" + "=" * 70)
print("CONCLUSION")
print("=" * 70)

if abs(mean_pct) > 20:
    print("\n❌ DATASETS ARE INCOMPATIBLE FOR CROSS-REGION TRANSFER")
    print(f"\n{abs(mean_pct):.1f}% target difference is too large to ignore.")
    print("\nRecommendations:")
    print("  1. Normalize targets by conductor baseline rating")
    print("  2. Add conductor capacity as a model input feature")
    print("  3. Train with region-specific output scaling layers")
    print("  4. Investigate what each dataset actually measures")
else:
    print("\n✅ Datasets may be compatible for cross-region transfer")

print("\n" + "=" * 70)
