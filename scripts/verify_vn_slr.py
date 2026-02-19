#!/usr/bin/env python3
"""
Verify VN_SLR calculation method
"""

import pandas as pd
import json
import numpy as np
from scipy.stats import pearsonr

# Load data
df = pd.read_hdf('data/processed/unified_dlr_training.h5', key='data')
vn_df = df[df['region'] == 'VN'].copy()
us_df = df[df['region'] == 'US'].copy()

# Load normalization params
with open('data/processed/normalization_params.json', 'r') as f:
    normalization_params = json.load(f)

print("=" * 80)
print("VN_SLR VERIFICATION")
print("=" * 80)

print(f"\nVN_SLR used: {normalization_params['VN_SLR']:.1f} A")
print(f"VN mean ampacity: {vn_df['Ampacity'].mean():.1f} A")
print(f"Are they equal? {abs(normalization_params['VN_SLR'] - vn_df['Ampacity'].mean()) < 1}")

if abs(normalization_params['VN_SLR'] - vn_df['Ampacity'].mean()) < 1:
    print("\n✅ CONFIRMED: VN_SLR was set to mean(VN Ampacity)")
    print("   NOT calculated from IEEE 738 with conservative defaults")
    print("\n   This means:")
    print("   • Normalized VN ratios will average exactly 1.0")
    print("   • No baseline shift correction applied")
    print("   • Assumes VN Ampacity mean represents SLR baseline")

# Analyze VN ampacity variation
print("\n" + "=" * 80)
print("VN AMPACITY VARIATION ANALYSIS")
print("=" * 80)

print(f"\nVN Ampacity statistics:")
print(f"  Min:    {vn_df['Ampacity'].min():.0f} A")
print(f"  Max:    {vn_df['Ampacity'].max():.0f} A")
print(f"  Mean:   {vn_df['Ampacity'].mean():.0f} A")
print(f"  Median: {vn_df['Ampacity'].median():.0f} A")
print(f"  Std:    {vn_df['Ampacity'].std():.0f} A")
print(f"  Range:  {vn_df['Ampacity'].max() - vn_df['Ampacity'].min():.0f} A")

vn_cv = (vn_df['Ampacity'].std() / vn_df['Ampacity'].mean()) * 100
us_cv = (us_df['dlr_ratio'].std() / us_df['dlr_ratio'].mean()) * 100

print(f"\nCoefficient of Variation:")
print(f"  VN Ampacity:  {vn_cv:.1f}%")
print(f"  US DLR ratio: {us_cv:.1f}%")

# Interpretation
print("\n" + "=" * 80)
print("INTERPRETATION")
print("=" * 80)

if vn_cv < 15:
    print(f"\n✅ VN shows LOW variability ({vn_cv:.1f}% CV)")
    print("   • Ampacity is relatively constant")
    print("   • Minimal weather-dependent adjustment")
    print("   • Behaves like STATIC RATING (SLR)")
    print("\n   → Setting VN_SLR = mean(Ampacity) is reasonable!")
elif vn_cv > 25:
    print(f"\n⚠️  VN shows HIGH variability ({vn_cv:.1f}% CV)")
    print("   • Ampacity varies significantly")
    print("   • Strong weather-dependent effects")
    print("   • Behaves like DYNAMIC RATING (DLR)")
    print("\n   → Setting VN_SLR = mean(Ampacity) may be incorrect!")
else:
    print(f"\n🟡 VN shows MODERATE variability ({vn_cv:.1f}% CV)")
    print("   • Some weather-dependent variation")
    print("   • Between static and fully dynamic")

# Check correlation with weather
print("\n" + "=" * 80)
print("WEATHER CORRELATION ANALYSIS")
print("=" * 80)

# Clean data
vn_clean = vn_df.dropna(subset=['Ampacity', 'wind_speed', 'temperature', 'solar_irradiance'])

r_wind, _ = pearsonr(vn_clean['wind_speed'], vn_clean['Ampacity'])
r_temp, _ = pearsonr(vn_clean['temperature'], vn_clean['Ampacity'])
r_solar, _ = pearsonr(vn_clean['solar_irradiance'], vn_clean['Ampacity'])

print(f"\nVN Ampacity correlation with weather:")
print(f"  Wind speed:  r = {r_wind:+.3f}")
print(f"  Temperature: r = {r_temp:+.3f}")
print(f"  Solar:       r = {r_solar:+.3f}")

if abs(r_wind) > 0.5:
    print(f"\n  ✅ STRONG wind correlation ({r_wind:.3f})")
    print("     Ampacity responds significantly to wind")
else:
    print(f"\n  ⚠️  WEAK wind correlation ({r_wind:.3f})")
    print("     Ampacity does not respond strongly to wind")

# Compare magnitude of weather effects
print("\n" + "=" * 80)
print("MAGNITUDE OF WEATHER EFFECTS")
print("=" * 80)

# Bin by wind speed
vn_clean['wind_bin'] = pd.cut(vn_clean['wind_speed'], bins=[0, 5, 10, 15, 100], labels=['Low', 'Med', 'High', 'Very High'])
wind_effect = vn_clean.groupby('wind_bin')['Ampacity'].mean()

print(f"\nVN Ampacity by wind speed:")
for bin_name, amp in wind_effect.items():
    print(f"  {bin_name:10s}: {amp:6.0f} A")

if wind_effect.max() - wind_effect.min() > 200:
    print(f"\n  ✅ LARGE wind effect: {wind_effect.max() - wind_effect.min():.0f} A range")
    print("     Ampacity varies significantly with wind")
else:
    print(f"\n  ⚠️  SMALL wind effect: {wind_effect.max() - wind_effect.min():.0f} A range")
    print("     Ampacity relatively constant despite wind changes")

# Final verdict
print("\n" + "=" * 80)
print("FINAL VERDICT")
print("=" * 80)

print("\n🔍 IS VN_SLR = mean(Ampacity) CORRECT?")

if vn_cv < 15 and (wind_effect.max() - wind_effect.min()) < 200:
    print("\n✅ YES, setting VN_SLR = mean(Ampacity) is REASONABLE")
    print("\nReasoning:")
    print(f"  • Low variability (CV = {vn_cv:.1f}%)")
    print(f"  • Small wind effect ({wind_effect.max() - wind_effect.min():.0f} A range)")
    print("  • VN Ampacity behaves like static/conservative rating")
    print("  • Mean is good approximation of baseline SLR")
elif abs(r_wind) > 0.7 and (wind_effect.max() - wind_effect.min()) > 300:
    print("\n⚠️  QUESTIONABLE - VN Ampacity shows DLR characteristics")
    print("\nReasoning:")
    print(f"  • Strong wind correlation (r = {r_wind:.3f})")
    print(f"  • Large wind effect ({wind_effect.max() - wind_effect.min():.0f} A range)")
    print("  • VN might already be reporting DLR, not SLR")
    print("\n  BETTER APPROACH:")
    print("  • Calculate true SLR from IEEE 738 conservative defaults")
    print("  • Or use minimum observed Ampacity as SLR estimate")
else:
    print("\n🟡 ACCEPTABLE but could be improved")
    print("\nReasoning:")
    print(f"  • Moderate variability (CV = {vn_cv:.1f}%)")
    print(f"  • Moderate wind effect ({wind_effect.max() - wind_effect.min():.0f} A range)")
    print("  • VN shows some DLR characteristics but not full dynamic rating")

# Alternative SLR estimates
print("\n" + "=" * 80)
print("ALTERNATIVE VN_SLR ESTIMATES")
print("=" * 80)

print("\nOption 1: Use minimum observed Ampacity")
vn_slr_min = vn_df['Ampacity'].min()
print(f"  VN_SLR_min = {vn_slr_min:.0f} A")
print(f"  → Assumes worst-case conditions represent SLR baseline")
print(f"  → VN ratios would range: {vn_df['Ampacity'].min()/vn_slr_min:.2f} to {vn_df['Ampacity'].max()/vn_slr_min:.2f}")

print("\nOption 2: Use 25th percentile")
vn_slr_p25 = vn_df['Ampacity'].quantile(0.25)
print(f"  VN_SLR_p25 = {vn_slr_p25:.0f} A")
print(f"  → Conservative estimate excluding top 75%")
print(f"  → VN ratios would range: {vn_df['Ampacity'].min()/vn_slr_p25:.2f} to {vn_df['Ampacity'].max()/vn_slr_p25:.2f}")

print("\nOption 3: Use mean (CURRENT)")
vn_slr_mean = vn_df['Ampacity'].mean()
print(f"  VN_SLR_mean = {vn_slr_mean:.0f} A")
print(f"  → Assumes symmetric variation around baseline")
print(f"  → VN ratios would range: {vn_df['Ampacity'].min()/vn_slr_mean:.2f} to {vn_df['Ampacity'].max()/vn_slr_mean:.2f}")

print("\nOption 4: Calculate from IEEE 738")
print("  Would need conductor specifications:")
print("  • Diameter (mm)")
print("  • Resistance (Ω/km)")
print("  • Emissivity, absorptivity")
print("  → Most accurate but requires conductor metadata")

print("\n" + "=" * 80)
print("RECOMMENDATION")
print("=" * 80)

print("\n📋 CURRENT APPROACH (mean) is:")
if vn_cv < 15:
    print("  ✅ ACCEPTABLE for initial analysis")
    print("\n  • VN Ampacity is relatively static (CV < 15%)")
    print("  • Mean represents central tendency well")
    print("  • Good enough for proof-of-concept")
    print("\n  NEXT STEPS:")
    print("  • Retrain with current normalization")
    print("  • If results are good, keep current approach")
    print("  • If results are poor, try Option 1 or 2 (lower SLR baseline)")
else:
    print("  ⚠️  POTENTIALLY SUBOPTIMAL")
    print("\n  • VN shows more variation than expected for SLR")
    print("  • May need lower baseline (min or p25)")
    print("\n  RECOMMENDATION:")
    print("  • Try Option 1 (min) or Option 2 (p25)")
    print("  • This would increase VN ratios, closer to US range")

print("\n" + "=" * 80)
