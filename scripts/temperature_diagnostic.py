#!/usr/bin/env python3
"""
Quick diagnostic: What is the temperature column in the US dataset?
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load unified data
print("Loading unified dataset...")
df = pd.read_hdf('data/processed/unified_dlr_training.h5', key='data')

# Filter for US data only
us_df = df[df['region'] == 'US'].copy()
print(f"Found {len(us_df):,} US samples")

# Step 1: Raw statistics BEFORE any normalization
print("\n" + "="*70)
print("RAW TEMPERATURE STATISTICS (US DATA)")
print("="*70)
print(us_df['temperature'].describe())
print(f"\nMin: {us_df['temperature'].min():.1f}°C")
print(f"Max: {us_df['temperature'].max():.1f}°C")
print(f"Mean: {us_df['temperature'].mean():.1f}°C")
print(f"Median: {us_df['temperature'].median():.1f}°C")
print(f"Std: {us_df['temperature'].std():.1f}°C")
print(f"\nValues below 0°C: {(us_df['temperature'] < 0).sum():,} rows ({(us_df['temperature'] < 0).sum()/len(us_df)*100:.1f}%)")
print(f"Values 0-40°C (typical ambient): {((us_df['temperature'] >= 0) & (us_df['temperature'] <= 40)).sum():,} rows ({((us_df['temperature'] >= 0) & (us_df['temperature'] <= 40)).sum()/len(us_df)*100:.1f}%)")
print(f"Values 40-80°C (warm conductor): {((us_df['temperature'] > 40) & (us_df['temperature'] <= 80)).sum():,} rows ({((us_df['temperature'] > 40) & (us_df['temperature'] <= 80)).sum()/len(us_df)*100:.1f}%)")
print(f"Values above 80°C (hot conductor): {(us_df['temperature'] > 80).sum():,} rows ({(us_df['temperature'] > 80).sum()/len(us_df)*100:.1f}%)")

# Step 2: Quick histogram
fig, axes = plt.subplots(2, 1, figsize=(10, 8))

# Top plot: Full distribution
ax1 = axes[0]
us_df['temperature'].hist(bins=100, ax=ax1, color='steelblue', edgecolor='black', alpha=0.7)
ax1.set_xlabel('Temperature (°C)', fontsize=12)
ax1.set_ylabel('Count', fontsize=12)
ax1.set_title('US Dataset Temperature Distribution (Full Range)', fontsize=14, fontweight='bold')
ax1.axvline(x=0, color='blue', linestyle='--', linewidth=2, label='0°C (freezing)')
ax1.axvline(x=40, color='orange', linestyle='--', linewidth=2, label='40°C (hot ambient)')
ax1.axvline(x=75, color='red', linestyle='--', linewidth=2, label='75°C (conductor max)')
ax1.axvline(x=us_df['temperature'].mean(), color='green', linestyle='-', linewidth=2, label=f'Mean: {us_df["temperature"].mean():.1f}°C')
ax1.legend(fontsize=10)
ax1.grid(True, alpha=0.3)

# Bottom plot: Zoomed to 0-120°C range
ax2 = axes[1]
temp_filtered = us_df['temperature'][(us_df['temperature'] >= 0) & (us_df['temperature'] <= 120)]
temp_filtered.hist(bins=60, ax=ax2, color='steelblue', edgecolor='black', alpha=0.7)
ax2.set_xlabel('Temperature (°C)', fontsize=12)
ax2.set_ylabel('Count', fontsize=12)
ax2.set_title('US Dataset Temperature Distribution (0-120°C Zoom)', fontsize=14, fontweight='bold')
ax2.axvline(x=0, color='blue', linestyle='--', linewidth=2, label='0°C (freezing)')
ax2.axvline(x=40, color='orange', linestyle='--', linewidth=2, label='40°C (hot ambient)')
ax2.axvline(x=75, color='red', linestyle='--', linewidth=2, label='75°C (conductor max)')
ax2.axvline(x=us_df['temperature'].mean(), color='green', linestyle='-', linewidth=2, label=f'Mean: {us_df["temperature"].mean():.1f}°C')
ax2.legend(fontsize=10)
ax2.grid(True, alpha=0.3)
ax2.set_xlim(0, 120)

plt.tight_layout()
plt.savefig('temperature_diagnostic.png', dpi=300, bbox_inches='tight')
print("\n✅ Saved: temperature_diagnostic.png")

# Step 3: Print the column names so we know what we're working with
print("\n" + "="*70)
print("ALL COLUMN NAMES IN US DATASET")
print("="*70)
print(us_df.columns.tolist())

# Step 4: Sample 5 rows to see raw values in context
print("\n" + "="*70)
print("5 RANDOM SAMPLE ROWS")
print("="*70)
cols_to_show = ['temperature', 'wind_speed', 'solar_irradiance']
if 'actual' in us_df.columns:
    cols_to_show.append('actual')
if 'dlr_ampacity' in us_df.columns:
    cols_to_show.append('dlr_ampacity')

print(us_df[cols_to_show].sample(min(5, len(us_df))))

# Step 5: Compare with Vietnam to see if temperature means the same thing
print("\n" + "="*70)
print("COMPARISON: US vs VIETNAM TEMPERATURE")
print("="*70)
vn_df = df[df['region'] == 'VN'].copy()
print(f"\nVietnam temperature:")
print(f"  Mean: {vn_df['temperature'].mean():.1f}°C")
print(f"  Min: {vn_df['temperature'].min():.1f}°C")
print(f"  Max: {vn_df['temperature'].max():.1f}°C")
print(f"\nUS temperature:")
print(f"  Mean: {us_df['temperature'].mean():.1f}°C")
print(f"  Min: {us_df['temperature'].min():.1f}°C")
print(f"  Max: {us_df['temperature'].max():.1f}°C")

# Interpretation
print("\n" + "="*70)
print("INTERPRETATION")
print("="*70)
us_mean = us_df['temperature'].mean()
vn_mean = vn_df['temperature'].mean()

if us_mean > 60:
    print("🔴 US temperature is likely CONDUCTOR TEMPERATURE:")
    print(f"   - Mean of {us_mean:.1f}°C is far above ambient range")
    print("   - This represents the actual conductor surface temperature")
    print("   - Physical interpretation: How hot the transmission line wire is")
elif us_mean > 40:
    print("🟠 US temperature is likely ELEVATED TEMPERATURE:")
    print(f"   - Mean of {us_mean:.1f}°C is above typical ambient")
    print("   - Could be conductor temp, ambient + solar heating, or calibrated value")
else:
    print("🟢 US temperature is likely AMBIENT TEMPERATURE:")
    print(f"   - Mean of {us_mean:.1f}°C is within typical ambient range")
    print("   - This is the air temperature around the conductor")

if vn_mean < 40:
    print(f"\n✅ Vietnam temperature (mean {vn_mean:.1f}°C) appears to be AMBIENT TEMPERATURE")
else:
    print(f"\n⚠️ Vietnam temperature (mean {vn_mean:.1f}°C) may also be conductor temperature")

print("\n" + "="*70)
print("CONCLUSION")
print("="*70)
if abs(us_mean - vn_mean) > 30:
    print("❌ CRITICAL MISMATCH: US and Vietnam temperature columns measure DIFFERENT things!")
    print(f"   US mean ({us_mean:.1f}°C) vs VN mean ({vn_mean:.1f}°C) = {abs(us_mean - vn_mean):.1f}°C difference")
    print("\n   This explains the cross-region transfer failure:")
    print("   • Model trained on VN (ambient ~28°C) cannot predict on US (conductor ~84°C)")
    print("   • Inputs are not comparable between datasets")
    print("\n   RECOMMENDATION: Re-process datasets to use consistent temperature definition")
else:
    print("✅ US and Vietnam temperatures appear to use the same measurement standard")
    print(f"   Difference: {abs(us_mean - vn_mean):.1f}°C")

print("\n" + "="*70)
