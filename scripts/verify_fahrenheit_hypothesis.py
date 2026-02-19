#!/usr/bin/env python3
"""
Verify if US temperature is in Fahrenheit instead of Celsius
"""

import pandas as pd
import os

print("="*70)
print("FAHRENHEIT HYPOTHESIS TEST")
print("="*70)

# Load US data
df = pd.read_hdf('data/processed/unified_dlr_training.h5', key='data')
us_df = df[df['region'] == 'US'].copy()

us_mean = us_df['temperature'].mean()
us_min = us_df['temperature'].min()
us_max = us_df['temperature'].max()
us_median = us_df['temperature'].median()

print("\nIf US temperature is in FAHRENHEIT:")
print("-" * 70)
print(f"  Mean:   {us_mean:.1f}°F = {(us_mean-32)*5/9:.1f}°C")
print(f"  Median: {us_median:.1f}°F = {(us_median-32)*5/9:.1f}°C")
print(f"  Max:    {us_max:.1f}°F = {(us_max-32)*5/9:.1f}°C")
print(f"  Min:    {us_min:.1f}°F = {(us_min-32)*5/9:.1f}°C")

# Check specific meaningful temperatures
print("\nKey temperature conversions:")
print("-" * 70)
print(f"  0°F (cold winter)     = {(0-32)*5/9:.1f}°C")
print(f"  32°F (freezing point) = {(32-32)*5/9:.1f}°C")
print(f"  75°F (mild)           = {(75-32)*5/9:.1f}°C")
print(f"  100°F (hot summer)    = {(100-32)*5/9:.1f}°C")
print(f"  150°F (very hot)      = {(150-32)*5/9:.1f}°C")

# Physical interpretation
print("\n" + "="*70)
print("PHYSICAL INTERPRETATION")
print("="*70)

mean_celsius = (us_mean - 32) * 5/9
max_celsius = (us_max - 32) * 5/9
median_celsius = (us_median - 32) * 5/9

if 15 <= mean_celsius <= 35:
    print("✅ FAHRENHEIT HYPOTHESIS LIKELY CORRECT!")
    print(f"\n   Mean of {mean_celsius:.1f}°C is in typical AMBIENT temperature range")
    print(f"   Median of {median_celsius:.1f}°C is reasonable for US climate")
    print(f"   Max of {max_celsius:.1f}°C ({us_max:.1f}°F) is hot but plausible ambient")
    print("\n   This matches typical weather conditions, NOT conductor temperature!")
elif 50 <= mean_celsius <= 100:
    print("❌ FAHRENHEIT HYPOTHESIS REJECTED")
    print(f"\n   Even converted, mean of {mean_celsius:.1f}°C is too high for ambient")
    print("   Original interpretation (conductor temp in Celsius) is correct")
else:
    print("⚠️  UNCLEAR - needs further investigation")

# Check distribution of converted values
print("\n" + "="*70)
print("DISTRIBUTION ANALYSIS (if Fahrenheit)")
print("="*70)

us_temp_celsius = (us_df['temperature'] - 32) * 5/9

print(f"\nConverted to Celsius:")
print(f"  Below 0°C (freezing):        {(us_temp_celsius < 0).sum():,} samples ({(us_temp_celsius < 0).sum()/len(us_temp_celsius)*100:.1f}%)")
print(f"  0-15°C (cold):               {((us_temp_celsius >= 0) & (us_temp_celsius < 15)).sum():,} samples ({((us_temp_celsius >= 0) & (us_temp_celsius < 15)).sum()/len(us_temp_celsius)*100:.1f}%)")
print(f"  15-25°C (mild):              {((us_temp_celsius >= 15) & (us_temp_celsius < 25)).sum():,} samples ({((us_temp_celsius >= 15) & (us_temp_celsius < 25)).sum()/len(us_temp_celsius)*100:.1f}%)")
print(f"  25-35°C (warm):              {((us_temp_celsius >= 25) & (us_temp_celsius < 35)).sum():,} samples ({((us_temp_celsius >= 25) & (us_temp_celsius < 35)).sum()/len(us_temp_celsius)*100:.1f}%)")
print(f"  Above 35°C (hot):            {(us_temp_celsius >= 35).sum():,} samples ({(us_temp_celsius >= 35).sum()/len(us_temp_celsius)*100:.1f}%)")

# Compare with Vietnam (known to be Celsius)
vn_df = df[df['region'] == 'VN'].copy()
print("\n" + "="*70)
print("COMPARISON WITH VIETNAM (known Celsius)")
print("="*70)
print(f"\nVietnam (original):       Mean={vn_df['temperature'].mean():.1f}°C, Range=[{vn_df['temperature'].min():.1f}, {vn_df['temperature'].max():.1f}]°C")
print(f"US (if Fahrenheit→°C):    Mean={mean_celsius:.1f}°C, Range=[{(us_min-32)*5/9:.1f}, {(us_max-32)*5/9:.1f}]°C")
print(f"US (original, as-is):     Mean={us_mean:.1f}°C(?), Range=[{us_min:.1f}, {us_max:.1f}]°C(?)")

diff_original = abs(us_mean - vn_df['temperature'].mean())
diff_converted = abs(mean_celsius - vn_df['temperature'].mean())

print(f"\nMean temperature difference:")
print(f"  Original (no conversion): {diff_original:.1f}°C")
print(f"  Converted from °F to °C:  {diff_converted:.1f}°C")

if diff_converted < diff_original:
    print(f"\n✅ Converting from Fahrenheit REDUCES the gap by {diff_original - diff_converted:.1f}°C")
    print("   This suggests US data IS in Fahrenheit!")
else:
    print(f"\n❌ Converting from Fahrenheit INCREASES the gap")
    print("   US data is NOT in Fahrenheit")

# Check NREL source documentation
print("\n" + "="*70)
print("CHECKING FOR DOCUMENTATION")
print("="*70)

readme_paths = [
    'data/README.txt',
    'data/README.md',
    'data/raw/us/README.txt',
    'data/raw/us/README.md',
    'US_DLR_VALIDATION_GUIDE.md',
    'US_TRAINING_DATA_GUIDE.md',
]

found_docs = []
for path in readme_paths:
    if os.path.exists(path):
        found_docs.append(path)
        print(f"\n📄 Found: {path}")
        with open(path) as f:
            content = f.read()
            # Look for temperature-related info
            for line in content.split('\n'):
                if any(keyword in line.lower() for keyword in ['temperature', 'fahrenheit', 'celsius', 'unit', '°f', '°c']):
                    print(f"   → {line.strip()}")

if not found_docs:
    print("\n⚠️  No README files found - cannot verify units from documentation")

# Sample 10 rows with context
print("\n" + "="*70)
print("SAMPLE DATA WITH CONTEXT")
print("="*70)
print("\n10 Random samples (showing temp in both interpretations):")
print("-" * 70)

sample = us_df[['temperature', 'wind_speed', 'solar_irradiance']].sample(10)
for idx, row in sample.iterrows():
    temp_f = row['temperature']
    temp_c = (temp_f - 32) * 5/9
    print(f"  Temp: {temp_f:6.1f}°F = {temp_c:6.1f}°C  |  Wind: {row['wind_speed']:5.1f} m/s  |  Solar: {row['solar_irradiance']:6.1f} W/m²")

print("\n" + "="*70)
print("FINAL VERDICT")
print("="*70)

# Check the peak at 150
peak_150_count = (us_df['temperature'] == 150).sum()
peak_150_celsius = (150 - 32) * 5/9

print(f"\nThe suspicious 150 peak:")
print(f"  150°F = {peak_150_celsius:.1f}°C")
print(f"  Samples at exactly 150: {peak_150_count:,} ({peak_150_count/len(us_df)*100:.1f}%)")

if peak_150_celsius < 70:
    print(f"\n  ✅ {peak_150_celsius:.1f}°C is a reasonable hot summer day maximum")
    print("     This makes MUCH more sense than 150°C conductor temperature!")
else:
    print(f"\n  ⚠️  Even at {peak_150_celsius:.1f}°C, this is quite hot")

# Final decision
if 15 <= mean_celsius <= 35 and diff_converted < 20:
    print("\n" + "="*70)
    print("🎯 CONCLUSION: US temperature is VERY LIKELY in FAHRENHEIT")
    print("="*70)
    print("\nEvidence:")
    print(f"  1. Mean converts to {mean_celsius:.1f}°C (reasonable ambient)")
    print(f"  2. Range [{(us_min-32)*5/9:.1f}, {(us_max-32)*5/9:.1f}]°C is plausible for US climate")
    print(f"  3. Gap with Vietnam reduces from {diff_original:.1f}°C to {diff_converted:.1f}°C")
    print(f"  4. Peak at 150°F = {peak_150_celsius:.1f}°C (hot but realistic)")
    print("\n✅ RECOMMENDATION: Convert all US temperatures from °F to °C before training!")
    print("   Formula: T_celsius = (T_fahrenheit - 32) * 5/9")
else:
    print("\n" + "="*70)
    print("❌ CONCLUSION: US temperature is NOT in Fahrenheit")
    print("="*70)
    print(f"\n  Converted mean ({mean_celsius:.1f}°C) doesn't match expected range")
    print("  Original interpretation stands")

print("\n" + "="*70)
