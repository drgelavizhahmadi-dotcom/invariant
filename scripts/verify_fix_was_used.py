#!/usr/bin/env python3
"""
Verify that the cross-region experiments used the FIXED temperature data
"""

import pandas as pd

# Load the dataset that the experiments used
df = pd.read_hdf('data/processed/unified_dlr_training.h5', key='data')

# Check US temperatures
us_df = df[df['region'] == 'US']
us_mean = us_df['temperature'].mean()
us_max = us_df['temperature'].max()

print("=" * 70)
print("VERIFICATION: Did experiments use FIXED data?")
print("=" * 70)

print(f"\nUS Temperature in current unified dataset:")
print(f"  Mean: {us_mean:.1f}")
print(f"  Max:  {us_max:.1f}")

if us_mean > 60:
    print("\n❌ PROBLEM: US temperatures are still in FAHRENHEIT!")
    print(f"   Mean of {us_mean:.1f} suggests Fahrenheit (not Celsius)")
    print("\n   The experiments used the OLD (buggy) data!")
    print("   The fix may not have been applied correctly.")
elif us_mean > 40:
    print("\n⚠️  WARNING: US temperatures seem high")
    print(f"   Mean of {us_mean:.1f}°C is above typical ambient range")
else:
    print("\n✅ CORRECT: US temperatures are in CELSIUS")
    print(f"   Mean of {us_mean:.1f}°C is reasonable ambient temperature")
    print("\n   The experiments used the FIXED data correctly!")

print("\n" + "=" * 70)
