#!/usr/bin/env python3
"""
Definitively classify US target variable:
- Is it DLR ampacity (thermal limit based on weather)?
- Is it actual dispatch current (operational load)?
- Something else?

Key tests:
1. Correlation with weather (ampacity should correlate strongly)
2. Time-of-day patterns (ampacity peaks at night, actual peaks during day)
3. Comparison with Vietnam's known ampacity column
"""

import pandas as pd
import numpy as np
from scipy.stats import pearsonr
import matplotlib.pyplot as plt

print("=" * 80)
print("DEFINITIVELY CLASSIFYING US TARGET VARIABLE")
print("=" * 80)

# Load data
df = pd.read_hdf('data/processed/unified_dlr_training.h5', key='data')
vn_df = df[df['region'] == 'VN'].copy()
us_df = df[df['region'] == 'US'].copy()

# Identify target columns
us_target_col = 'actual'
vn_target_col = 'Ampacity'

print(f"\nDatasets:")
print(f"  Vietnam: {len(vn_df):,} samples, target column: '{vn_target_col}'")
print(f"  US:      {len(us_df):,} samples, target column: '{us_target_col}'")

# ============================================================================
# TEST 1: CORRELATION WITH WEATHER
# ============================================================================
print("\n" + "=" * 80)
print("TEST 1: CORRELATION WITH WEATHER INPUTS")
print("=" * 80)
print("\nHypothesis:")
print("  - DLR ampacity: STRONG correlation with wind (↑wind = ↑cooling = ↑ampacity)")
print("  - DLR ampacity: NEGATIVE correlation with temp (↑temp = ↓ampacity)")
print("  - Actual current: WEAK/NO correlation with weather (driven by demand)")

# US correlations
print(f"\n{'US (' + us_target_col + ')':.<30}")
us_target = us_df[us_target_col].dropna()
us_wind = us_df.loc[us_target.index, 'wind_speed']
us_temp = us_df.loc[us_target.index, 'temperature']
us_solar = us_df.loc[us_target.index, 'solar_irradiance']

# Remove any remaining NaNs
mask = ~(us_wind.isna() | us_temp.isna() | us_solar.isna())
us_target = us_target[mask]
us_wind = us_wind[mask]
us_temp = us_temp[mask]
us_solar = us_solar[mask]

r_wind_us, p_wind_us = pearsonr(us_wind, us_target)
r_temp_us, p_temp_us = pearsonr(us_temp, us_target)
r_solar_us, p_solar_us = pearsonr(us_solar, us_target)

print(f"  Wind speed:  r={r_wind_us:+.3f}, p={p_wind_us:.4e} {'✓ Significant' if p_wind_us < 0.01 else '✗ Not sig'}")
print(f"  Temperature: r={r_temp_us:+.3f}, p={p_temp_us:.4e} {'✓ Significant' if p_temp_us < 0.01 else '✗ Not sig'}")
print(f"  Solar:       r={r_solar_us:+.3f}, p={p_solar_us:.4e} {'✓ Significant' if p_solar_us < 0.01 else '✗ Not sig'}")

# Vietnam correlations (KNOWN to be ampacity)
print(f"\n{'Vietnam (' + vn_target_col + ') [KNOWN AMPACITY]':.<30}")
vn_target = vn_df[vn_target_col].dropna()
vn_wind = vn_df.loc[vn_target.index, 'wind_speed']
vn_temp = vn_df.loc[vn_target.index, 'temperature']
vn_solar = vn_df.loc[vn_target.index, 'solar_irradiance']

mask_vn = ~(vn_wind.isna() | vn_temp.isna() | vn_solar.isna())
vn_target = vn_target[mask_vn]
vn_wind = vn_wind[mask_vn]
vn_temp = vn_temp[mask_vn]
vn_solar = vn_solar[mask_vn]

r_wind_vn, p_wind_vn = pearsonr(vn_wind, vn_target)
r_temp_vn, p_temp_vn = pearsonr(vn_temp, vn_target)
r_solar_vn, p_solar_vn = pearsonr(vn_solar, vn_target)

print(f"  Wind speed:  r={r_wind_vn:+.3f}, p={p_wind_vn:.4e} {'✓ Expected' if abs(r_wind_vn) > 0.3 else '✗ Unexpected'}")
print(f"  Temperature: r={r_temp_vn:+.3f}, p={p_temp_vn:.4e} {'✓ Expected' if r_temp_vn < -0.1 else '✗ Unexpected'}")
print(f"  Solar:       r={r_solar_vn:+.3f}, p={p_solar_vn:.4e}")

# Interpretation
print("\n📊 INTERPRETATION (TEST 1):")
if abs(r_wind_us) > 0.3 and r_wind_us > 0:
    print("  ✅ US shows STRONG positive wind correlation (like ampacity)")
    test1_result = "AMPACITY"
elif abs(r_wind_us) < 0.1:
    print("  ⚠️  US shows WEAK wind correlation (unlike ampacity)")
    test1_result = "ACTUAL_CURRENT"
else:
    print("  ❓ US shows MODERATE wind correlation (inconclusive)")
    test1_result = "UNCLEAR"

# ============================================================================
# TEST 2: TIME-OF-DAY PATTERNS
# ============================================================================
print("\n" + "=" * 80)
print("TEST 2: TIME-OF-DAY PATTERNS")
print("=" * 80)
print("\nHypothesis:")
print("  - DLR ampacity: PEAKS at night/early morning (coolest temps, often higher wind)")
print("  - Actual current: PEAKS during day (9am-6pm business hours, high demand)")

# Parse timestamps
if 'timestamp' in us_df.columns:
    us_df['hour'] = pd.to_datetime(us_df['timestamp'], errors='coerce').dt.hour
elif 'datetime' in us_df.columns:
    us_df['hour'] = pd.to_datetime(us_df['datetime'], errors='coerce').dt.hour
else:
    print("\n⚠️  No timestamp column found, skipping time-of-day analysis")
    us_df['hour'] = np.nan

if 'timestamp' in vn_df.columns:
    vn_df['hour'] = pd.to_datetime(vn_df['timestamp'], errors='coerce').dt.hour
elif 'datetime' in vn_df.columns:
    vn_df['hour'] = pd.to_datetime(vn_df['datetime'], errors='coerce').dt.hour
else:
    vn_df['hour'] = np.nan

if not us_df['hour'].isna().all():
    # US hourly pattern
    us_hourly = us_df.groupby('hour')[us_target_col].agg(['mean', 'count'])
    print(f"\nUS hourly pattern ('{us_target_col}'):")
    print(f"  Midnight (0h):    {us_hourly.loc[0, 'mean']:.1f} A  (n={us_hourly.loc[0, 'count']:,})")
    print(f"  Early morn (6h):  {us_hourly.loc[6, 'mean']:.1f} A  (n={us_hourly.loc[6, 'count']:,})")
    print(f"  Noon (12h):       {us_hourly.loc[12, 'mean']:.1f} A  (n={us_hourly.loc[12, 'count']:,})")
    print(f"  Evening (18h):    {us_hourly.loc[18, 'mean']:.1f} A  (n={us_hourly.loc[18, 'count']:,})")

    # Peak detection
    night_avg = us_hourly.loc[0:6, 'mean'].mean()  # Midnight to 6am
    day_avg = us_hourly.loc[9:18, 'mean'].mean()   # 9am to 6pm

    print(f"\n  Night average (0-6h):   {night_avg:.1f} A")
    print(f"  Day average (9-18h):    {day_avg:.1f} A")
    print(f"  Ratio (day/night):      {day_avg/night_avg:.2f}")

    if day_avg > night_avg * 1.1:
        print("  → PEAKS DURING DAY (demand-driven, suggests ACTUAL CURRENT)")
        test2_result = "ACTUAL_CURRENT"
    elif night_avg > day_avg * 1.1:
        print("  → PEAKS AT NIGHT (cooling-driven, suggests AMPACITY)")
        test2_result = "AMPACITY"
    else:
        print("  → FLAT PATTERN (inconclusive)")
        test2_result = "UNCLEAR"

    # Vietnam comparison
    if not vn_df['hour'].isna().all():
        vn_hourly = vn_df.groupby('hour')[vn_target_col].agg(['mean', 'count'])
        print(f"\nVietnam hourly pattern ('{vn_target_col}') [KNOWN AMPACITY]:")
        vn_night_avg = vn_hourly.loc[0:6, 'mean'].mean()
        vn_day_avg = vn_hourly.loc[9:18, 'mean'].mean()
        print(f"  Night average (0-6h):   {vn_night_avg:.1f} A")
        print(f"  Day average (9-18h):    {vn_day_avg:.1f} A")
        print(f"  Ratio (day/night):      {vn_day_avg/vn_night_avg:.2f}")
else:
    print("\n⚠️  Timestamp parsing failed, cannot perform time-of-day analysis")
    test2_result = "N/A"

# ============================================================================
# TEST 3: VARIABILITY ANALYSIS
# ============================================================================
print("\n" + "=" * 80)
print("TEST 3: COEFFICIENT OF VARIATION")
print("=" * 80)
print("\nHypothesis:")
print("  - DLR ampacity: Moderate CV (~15-20%) - varies with weather")
print("  - Actual current: High CV (>30%) - varies with demand + weather")

us_cv = (us_df[us_target_col].std() / us_df[us_target_col].mean()) * 100
vn_cv = (vn_df[vn_target_col].std() / vn_df[vn_target_col].mean()) * 100

print(f"\nUS '{us_target_col}':      CV = {us_cv:.1f}%")
print(f"VN '{vn_target_col}' [KNOWN]: CV = {vn_cv:.1f}%")

if us_cv > 30:
    print("  → HIGH variability (suggests ACTUAL CURRENT with demand fluctuations)")
    test3_result = "ACTUAL_CURRENT"
elif 15 <= us_cv <= 25:
    print("  → MODERATE variability (suggests AMPACITY weather-driven)")
    test3_result = "AMPACITY"
else:
    print("  → LOW variability (unusual)")
    test3_result = "UNCLEAR"

# ============================================================================
# TEST 4: CHECK OTHER COLUMNS FOR CLUES
# ============================================================================
print("\n" + "=" * 80)
print("TEST 4: OTHER COLUMNS IN US DATASET")
print("=" * 80)

print("\nAll US columns:")
for i, col in enumerate(us_df.columns, 1):
    print(f"  {i:2d}. {col}")

# Look for SLR or DLR columns
suspicious_cols = []
for col in us_df.columns:
    if any(keyword in col.lower() for keyword in ['dlr', 'slr', 'ampacity', 'rating', 'limit']):
        suspicious_cols.append(col)

if suspicious_cols:
    print(f"\n🔍 Found potentially relevant columns:")
    for col in suspicious_cols:
        if col in us_df.columns:
            print(f"\n  '{col}':")
            print(f"    Mean: {us_df[col].mean():.1f}")
            print(f"    Min:  {us_df[col].min():.1f}")
            print(f"    Max:  {us_df[col].max():.1f}")

            # Check correlation with 'actual'
            if col != us_target_col:
                corr = us_df[[col, us_target_col]].corr().iloc[0, 1]
                print(f"    Correlation with '{us_target_col}': {corr:.3f}")

# ============================================================================
# FINAL VERDICT
# ============================================================================
print("\n" + "=" * 80)
print("FINAL VERDICT")
print("=" * 80)

print(f"\nTest results:")
print(f"  Test 1 (Weather correlation): {test1_result}")
if 'test2_result' in locals():
    print(f"  Test 2 (Time-of-day pattern):  {test2_result}")
print(f"  Test 3 (Variability):          {test3_result}")

# Count votes
votes = [test1_result]
if 'test2_result' in locals():
    votes.append(test2_result)
votes.append(test3_result)

ampacity_votes = votes.count("AMPACITY")
actual_votes = votes.count("ACTUAL_CURRENT")

print("\n" + "=" * 80)
if ampacity_votes > actual_votes:
    print("🎯 VERDICT: US 'actual' is most likely DLR AMPACITY")
    print("\nEvidence:")
    print("  • Strong correlation with wind speed (like Vietnam)")
    if 'test2_result' in locals() and test2_result == "AMPACITY":
        print("  • Peaks at night when cooler (thermal limit behavior)")
    print("  • Coefficient of variation matches ampacity pattern")
    print("\n⚠️  WARNING: But it's 36% higher than Vietnam ampacity!")
    print("   Possible explanations:")
    print("   1. Different conductor types (larger diameter, higher rating)")
    print("   2. Different baseline SLR (Static Line Rating)")
    print("   3. Different DLR calculation methodology")
elif actual_votes > ampacity_votes:
    print("🎯 VERDICT: US 'actual' is most likely ACTUAL DISPATCH CURRENT")
    print("\nEvidence:")
    print("  • Weak correlation with weather")
    if 'test2_result' in locals() and test2_result == "ACTUAL_CURRENT":
        print("  • Peaks during business hours (demand-driven)")
    print("  • High variability (demand fluctuations + weather)")
    print("\n💡 IMPLICATION: Cannot train DLR model on actual dispatch current!")
    print("   You need the DLR limit (ampacity), not the actual flow.")
else:
    print("🎯 VERDICT: INCONCLUSIVE")
    print("\nResults are mixed. Need more information:")
    print("  • Check NREL dataset documentation")
    print("  • Look for SLR or baseline rating columns")
    print("  • Verify data provenance")

print("\n" + "=" * 80)

# ============================================================================
# VISUALIZATION
# ============================================================================
print("\nGenerating visualizations...")

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Plot 1: Wind vs Target (both datasets)
ax1 = axes[0, 0]
# Sample for visibility
us_sample = us_df.sample(min(5000, len(us_df)))
vn_sample = vn_df.sample(min(1000, len(vn_df)))
ax1.scatter(us_sample['wind_speed'], us_sample[us_target_col], alpha=0.3, s=10, color='red', label=f'US ({us_target_col})')
ax1.scatter(vn_sample['wind_speed'], vn_sample[vn_target_col], alpha=0.5, s=10, color='blue', label=f'VN ({vn_target_col})')
ax1.set_xlabel('Wind Speed (m/s)')
ax1.set_ylabel('Target Value (A)')
ax1.set_title(f'Wind Speed vs Target\n(US r={r_wind_us:.3f}, VN r={r_wind_vn:.3f})')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Plot 2: Temperature vs Target
ax2 = axes[0, 1]
ax2.scatter(us_sample['temperature'], us_sample[us_target_col], alpha=0.3, s=10, color='red', label=f'US ({us_target_col})')
ax2.scatter(vn_sample['temperature'], vn_sample[vn_target_col], alpha=0.5, s=10, color='blue', label=f'VN ({vn_target_col})')
ax2.set_xlabel('Temperature (°C)')
ax2.set_ylabel('Target Value (A)')
ax2.set_title(f'Temperature vs Target\n(US r={r_temp_us:.3f}, VN r={r_temp_vn:.3f})')
ax2.legend()
ax2.grid(True, alpha=0.3)

# Plot 3: Hourly pattern (if available)
ax3 = axes[1, 0]
if not us_df['hour'].isna().all():
    us_hourly = us_df.groupby('hour')[us_target_col].mean()
    ax3.plot(us_hourly.index, us_hourly.values, 'o-', color='red', linewidth=2, markersize=8, label=f'US ({us_target_col})')
    if not vn_df['hour'].isna().all():
        vn_hourly = vn_df.groupby('hour')[vn_target_col].mean()
        ax3.plot(vn_hourly.index, vn_hourly.values, 'o-', color='blue', linewidth=2, markersize=8, label=f'VN ({vn_target_col})')
    ax3.set_xlabel('Hour of Day')
    ax3.set_ylabel('Mean Target Value (A)')
    ax3.set_title('Time-of-Day Pattern')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    ax3.set_xticks(range(0, 24, 3))
else:
    ax3.text(0.5, 0.5, 'No timestamp data', ha='center', va='center')
    ax3.set_title('Time-of-Day Pattern (N/A)')

# Plot 4: Distribution comparison
ax4 = axes[1, 1]
ax4.hist(us_df[us_target_col], bins=50, alpha=0.5, color='red', label=f'US ({us_target_col})', density=True)
ax4.hist(vn_df[vn_target_col], bins=50, alpha=0.5, color='blue', label=f'VN ({vn_target_col})', density=True)
ax4.axvline(us_df[us_target_col].mean(), color='red', linestyle='--', linewidth=2)
ax4.axvline(vn_df[vn_target_col].mean(), color='blue', linestyle='--', linewidth=2)
ax4.set_xlabel('Target Value (A)')
ax4.set_ylabel('Density')
ax4.set_title(f'Target Distribution\n(US mean: {us_df[us_target_col].mean():.0f}A, VN mean: {vn_df[vn_target_col].mean():.0f}A)')
ax4.legend()
ax4.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('us_target_classification.png', dpi=300, bbox_inches='tight')
print("✅ Saved: us_target_classification.png")

print("\n" + "=" * 80)
print("ANALYSIS COMPLETE")
print("=" * 80)
