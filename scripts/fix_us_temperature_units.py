#!/usr/bin/env python3
"""
CRITICAL BUG FIX: Convert US temperatures from Fahrenheit to Celsius

The unified dataset contains US temperatures in Fahrenheit (not Celsius as documented).
This script:
1. Loads the unified dataset
2. Converts US temperature from °F to °C
3. Validates the conversion
4. Saves the corrected dataset with backup

Author: Gelavizh Ahmadi / Invariant Research
Date: 2026-02-17
"""

import pandas as pd
import numpy as np
import h5py
from pathlib import Path
import shutil
from datetime import datetime

# Paths
UNIFIED_PATH = Path('data/processed/unified_dlr_training.h5')
BACKUP_PATH = Path('data/processed/unified_dlr_training_BACKUP_fahrenheit.h5')
FIXED_PATH = Path('data/processed/unified_dlr_training.h5')

def backup_original():
    """Create backup of original file"""
    if UNIFIED_PATH.exists() and not BACKUP_PATH.exists():
        print(f"📦 Creating backup: {BACKUP_PATH}")
        shutil.copy2(UNIFIED_PATH, BACKUP_PATH)
        print("   ✅ Backup created")
    elif BACKUP_PATH.exists():
        print(f"   ℹ️  Backup already exists: {BACKUP_PATH}")
    else:
        print("   ⚠️  Original file not found")

def load_data():
    """Load unified dataset"""
    print(f"\n📊 Loading unified dataset from: {UNIFIED_PATH}")
    df = pd.read_hdf(UNIFIED_PATH, key='data')
    print(f"   ✅ Loaded {len(df):,} total samples")
    print(f"   Regions: {df['region'].value_counts().to_dict()}")
    return df

def diagnose_temperature(df):
    """Diagnose temperature units for each region"""
    print("\n🔍 TEMPERATURE DIAGNOSTIC")
    print("=" * 70)

    for region in df['region'].unique():
        region_df = df[df['region'] == region]
        temp = region_df['temperature']

        print(f"\n{region} Region:")
        print(f"  Samples: {len(region_df):,}")
        print(f"  Temperature statistics:")
        print(f"    Mean:   {temp.mean():6.1f}")
        print(f"    Median: {temp.median():6.1f}")
        print(f"    Min:    {temp.min():6.1f}")
        print(f"    Max:    {temp.max():6.1f}")
        print(f"    Std:    {temp.std():6.1f}")

        # Hypothesis test
        if temp.mean() > 60:
            print(f"  🔴 LIKELY FAHRENHEIT (mean = {temp.mean():.1f})")
            print(f"     Converted: {(temp.mean()-32)*5/9:.1f}°C")
        elif 20 <= temp.mean() <= 40:
            print(f"  ✅ LIKELY CELSIUS (mean = {temp.mean():.1f}°C)")
        else:
            print(f"  ⚠️  UNCERTAIN (mean = {temp.mean():.1f})")

def fix_us_temperature(df):
    """Convert US temperature from Fahrenheit to Celsius"""
    print("\n🔧 APPLYING FIX: US temperature °F → °C")
    print("=" * 70)

    # Separate US and non-US data
    us_mask = df['region'] == 'US'
    n_us = us_mask.sum()

    print(f"\nUS samples to fix: {n_us:,}")

    # Store original for comparison
    original_us_temp = df.loc[us_mask, 'temperature'].copy()

    print("\nBEFORE conversion:")
    print(f"  Mean:   {original_us_temp.mean():.1f}°F")
    print(f"  Median: {original_us_temp.median():.1f}°F")
    print(f"  Min:    {original_us_temp.min():.1f}°F")
    print(f"  Max:    {original_us_temp.max():.1f}°F")
    print(f"  Samples at 150: {(original_us_temp == 150).sum():,} ({(original_us_temp == 150).sum()/len(original_us_temp)*100:.1f}%)")

    # Apply conversion: °C = (°F - 32) × 5/9
    df.loc[us_mask, 'temperature'] = (df.loc[us_mask, 'temperature'] - 32) * 5/9

    converted_us_temp = df.loc[us_mask, 'temperature'].copy()

    print("\nAFTER conversion:")
    print(f"  Mean:   {converted_us_temp.mean():.1f}°C")
    print(f"  Median: {converted_us_temp.median():.1f}°C")
    print(f"  Min:    {converted_us_temp.min():.1f}°C")
    print(f"  Max:    {converted_us_temp.max():.1f}°C")
    print(f"  Samples at 65.6°C (was 150°F): {np.abs(converted_us_temp - 65.56).lt(0.1).sum():,}")

    return df

def validate_fix(df):
    """Validate the temperature conversion"""
    print("\n✅ VALIDATION")
    print("=" * 70)

    # Check Vietnam (should be unchanged)
    vn_df = df[df['region'] == 'VN']
    print(f"\nVietnam (should be unchanged):")
    print(f"  Mean: {vn_df['temperature'].mean():.1f}°C")
    print(f"  Expected: ~28.2°C")
    if 27 <= vn_df['temperature'].mean() <= 29:
        print("  ✅ PASS: Vietnam unchanged")
    else:
        print("  ❌ FAIL: Vietnam was modified!")
        raise ValueError("Vietnam temperatures were incorrectly modified")

    # Check US (should now be reasonable Celsius)
    us_df = df[df['region'] == 'US']
    us_mean = us_df['temperature'].mean()
    us_min = us_df['temperature'].min()
    us_max = us_df['temperature'].max()

    print(f"\nUS (should now be Celsius):")
    print(f"  Mean: {us_mean:.1f}°C")
    print(f"  Min:  {us_min:.1f}°C")
    print(f"  Max:  {us_max:.1f}°C")

    # Assertions
    try:
        assert us_max < 80, f"Temperature too high ({us_max:.1f}°C) - possible Fahrenheit data still present"
        assert us_min > -60, f"Temperature too low ({us_min:.1f}°C) - check units"
        assert 20 <= us_mean <= 40, f"Mean temperature ({us_mean:.1f}°C) outside expected range"
        print("  ✅ PASS: All assertions passed")
    except AssertionError as e:
        print(f"  ❌ FAIL: {e}")
        raise

    # Compare US and Vietnam
    vn_mean = vn_df['temperature'].mean()
    diff = abs(us_mean - vn_mean)
    print(f"\nCross-region comparison:")
    print(f"  US mean:      {us_mean:.1f}°C")
    print(f"  Vietnam mean: {vn_mean:.1f}°C")
    print(f"  Difference:   {diff:.1f}°C ({diff/vn_mean*100:.1f}%)")

    if diff < 10:
        print(f"  ✅ EXCELLENT: Temperatures now comparable (< 10°C difference)")
    elif diff < 20:
        print(f"  ✅ GOOD: Reasonable temperature difference")
    else:
        print(f"  ⚠️  WARNING: Large temperature difference still exists")

def save_fixed_data(df):
    """Save the corrected dataset"""
    print("\n💾 SAVING CORRECTED DATASET")
    print("=" * 70)

    # Add metadata about the fix
    fix_metadata = {
        'fix_applied': True,
        'fix_date': datetime.now().isoformat(),
        'fix_description': 'Converted US temperature from Fahrenheit to Celsius',
        'conversion_formula': '(T_F - 32) * 5/9',
        'backup_path': str(BACKUP_PATH)
    }

    # Fix mixed-type columns for HDF5 serialization
    print("\nPreparing data for HDF5...")
    df_fixed = df.copy()

    # Convert problematic columns to string
    if 'line_id' in df_fixed.columns:
        df_fixed['line_id'] = df_fixed['line_id'].astype(str)
    if 'conductor_type' in df_fixed.columns:
        df_fixed['conductor_type'] = df_fixed['conductor_type'].astype(str)

    print(f"Saving to: {FIXED_PATH}")
    # Use pickle protocol for complex dtypes
    df_fixed.to_hdf(FIXED_PATH, key='data', mode='w', complevel=9)

    # Save metadata separately
    metadata_path = FIXED_PATH.parent / 'unified_dlr_training_metadata.json'
    import json
    with open(metadata_path, 'w') as f:
        json.dump(fix_metadata, f, indent=2)

    print(f"✅ Saved {len(df):,} samples")
    print(f"✅ Metadata saved to: {metadata_path}")

def compare_before_after():
    """Generate comparison report"""
    print("\n📊 BEFORE/AFTER COMPARISON")
    print("=" * 70)

    # Load original
    df_original = pd.read_hdf(BACKUP_PATH, key='data')
    df_fixed = pd.read_hdf(FIXED_PATH, key='data')

    print("\nUS Temperature Statistics:")
    print("-" * 70)
    print(f"{'Metric':<20} {'Before (°F?)':<20} {'After (°C)':<20} {'Change':<15}")
    print("-" * 70)

    us_orig = df_original[df_original['region'] == 'US']['temperature']
    us_fixed = df_fixed[df_fixed['region'] == 'US']['temperature']

    metrics = {
        'Mean': (us_orig.mean(), us_fixed.mean()),
        'Median': (us_orig.median(), us_fixed.median()),
        'Min': (us_orig.min(), us_fixed.min()),
        'Max': (us_orig.max(), us_fixed.max()),
        'Std': (us_orig.std(), us_fixed.std())
    }

    for metric, (before, after) in metrics.items():
        change = after - before
        print(f"{metric:<20} {before:>8.1f} {'':<10} {after:>8.1f} {'':<10} {change:>8.1f}")

    print("\n" + "=" * 70)
    print("🎉 FIX COMPLETE!")
    print("=" * 70)
    print("\nNext steps:")
    print("  1. ✅ Backup created at:", BACKUP_PATH)
    print("  2. ✅ Fixed dataset at:", FIXED_PATH)
    print("  3. ⚠️  MUST RERUN all experiments trained on US data!")
    print("  4. ⚠️  MUST RERUN all cross-region experiments!")
    print("\nRecommended commands:")
    print("  python scripts/cross_region_validation_v2.py")
    print("  python scripts/train_invariant_pikan_production.py --us-data")

def main():
    """Main execution"""
    print("\n" + "=" * 70)
    print("CRITICAL BUG FIX: US Temperature Units (°F → °C)")
    print("=" * 70)

    # Step 1: Backup
    backup_original()

    # Step 2: Load
    df = load_data()

    # Step 3: Diagnose
    diagnose_temperature(df)

    # Step 4: Fix
    df = fix_us_temperature(df)

    # Step 5: Validate
    validate_fix(df)

    # Step 6: Save
    save_fixed_data(df)

    # Step 7: Compare
    compare_before_after()

if __name__ == '__main__':
    main()
