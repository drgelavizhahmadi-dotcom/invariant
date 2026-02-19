#!/usr/bin/env python3
"""
Cross-Region Transfer Failure Diagnostic Analysis

This script provides detailed analysis of why cross-region transfer fails:
1. Feature distribution comparison (domain gap)
2. Error analysis by weather conditions
3. Bias direction analysis
4. Prediction vs actual comparison

Author: Code Assistant
Copyright (C) 2025 Gelavizh Ahmadi / Invariant Research
"""

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import json
import sys
from typing import Dict, List, Tuple

sys.path.append(str(Path(__file__).parent.parent))


def load_checkpoint(ckpt_path: str) -> Dict:
    """Load model checkpoint"""
    return torch.load(ckpt_path, map_location='cpu', weights_only=False)


def load_unified_data() -> pd.DataFrame:
    """Load unified dataset"""
    print("📊 Loading unified dataset...")
    df = pd.read_hdf('data/processed/unified_dlr_training.h5', key='data')
    return df


def analyze_feature_distributions(df: pd.DataFrame, output_dir: Path):
    """
    Analysis 2: Compare input feature distributions between Vietnam and US
    """
    print("\n" + "="*70)
    print("ANALYSIS 2: Input Feature Distribution Comparison")
    print("="*70)
    
    # Get region-specific data
    vn_df = df[df['region'] == 'VN'].copy()
    us_df = df[df['region'] == 'US'].copy()
    
    features = {
        'temperature': {'label': 'Temperature (°C)', 'xlim': None},
        'wind_speed': {'label': 'Wind Speed (m/s)', 'xlim': None},
        'solar_irradiance': {'label': 'Solar Irradiance (W/m²)', 'xlim': None},
    }
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    fig.suptitle('Input Feature Distributions: Vietnam vs US', fontsize=14, fontweight='bold')
    
    stats_table = []
    
    for idx, (feature, config) in enumerate(features.items()):
        ax = axes[idx]
        
        # Get data (drop NaN)
        vn_data = vn_df[feature].dropna()
        us_data = us_df[feature].dropna()
        
        # Plot histograms
        ax.hist(vn_data, bins=50, alpha=0.6, label=f'Vietnam (n={len(vn_data):,})', 
                color='blue', density=True)
        ax.hist(us_data, bins=50, alpha=0.6, label=f'US (n={len(us_data):,})', 
                color='red', density=True)
        
        ax.set_xlabel(config['label'])
        ax.set_ylabel('Density')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Calculate statistics
        stats = {
            'Feature': feature,
            'VN_mean': vn_data.mean(),
            'VN_std': vn_data.std(),
            'VN_min': vn_data.min(),
            'VN_max': vn_data.max(),
            'US_mean': us_data.mean(),
            'US_std': us_data.std(),
            'US_min': us_data.min(),
            'US_max': us_data.max(),
            'Mean_Diff_%': abs(vn_data.mean() - us_data.mean()) / us_data.mean() * 100
        }
        stats_table.append(stats)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'feature_distributions_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Saved: feature_distributions_comparison.png")
    
    # Print statistics table
    print("\n📈 Feature Statistics:")
    print("-" * 100)
    print(f"{'Feature':<20} {'VN Mean':<12} {'VN Std':<10} {'US Mean':<12} {'US Std':<10} {'Mean Diff %':<12}")
    print("-" * 100)
    for s in stats_table:
        print(f"{s['Feature']:<20} {s['VN_mean']:<12.2f} {s['VN_std']:<10.2f} "
              f"{s['US_mean']:<12.2f} {s['US_std']:<10.2f} {s['Mean_Diff_%']:<12.1f}")
    print("-" * 100)
    
    return stats_table


def analyze_vn_to_us_errors(output_dir: Path):
    """
    Analysis 3: Where does VN→US fail worst?
    Bin errors by temperature and wind speed
    """
    print("\n" + "="*70)
    print("ANALYSIS 3: VN→US Error Analysis by Weather Condition")
    print("="*70)
    
    # Load the VN_to_US checkpoint with predictions
    ckpt = load_checkpoint('cross_region_results/model_VN_to_US.pt')
    metrics = ckpt['metrics']
    
    # We need to reload the test data to get features
    df = load_unified_data()
    
    # Get US test data (same sampling as in training script)
    us_df = df[df['region'] == 'US'].copy()
    us_test = us_df.sample(n=min(10000, len(us_df)), random_state=42)
    us_test = us_test.dropna(subset=['temperature', 'wind_speed', 'solar_irradiance', 'actual'])
    
    # Reconstruct predictions (we need to run inference again)
    from cross_region_validation_v2 import SimpleDLRModel, CrossRegionNormalizer
    
    # Load VN normalizer (used for training)
    vn_ckpt = load_checkpoint('cross_region_results/model_VN_only.pt')
    normalizer = CrossRegionNormalizer()
    normalizer.mean = vn_ckpt['normalizer_mean']
    normalizer.std = vn_ckpt['normalizer_std']
    
    # Load VN_to_US model
    model = SimpleDLRModel(input_dim=3, hidden_dims=[128, 128, 64], dropout=0.1)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    
    # Prepare test data
    X = us_test[['temperature', 'wind_speed', 'solar_irradiance']].values.astype(np.float32)
    X_norm = normalizer.transform(X)
    y_true = us_test['actual'].values
    
    # Run inference
    with torch.no_grad():
        X_tensor = torch.tensor(X_norm, dtype=torch.float32)
        y_pred = model(X_tensor).numpy().flatten()
    
    # Calculate errors
    errors = y_pred - y_true
    abs_errors = np.abs(errors)
    
    # Create results dataframe
    results_df = pd.DataFrame({
        'temperature': X[:, 0],
        'wind_speed': X[:, 1],
        'solar_irradiance': X[:, 2],
        'actual': y_true,
        'predicted': y_pred,
        'error': errors,
        'abs_error': abs_errors
    })
    
    print(f"\n📊 Total test samples: {len(results_df)}")
    
    # Bin by temperature
    print("\n🌡️  Errors by Temperature Range:")
    print("-" * 60)
    
    temp_bins = [
        ('Cold (< 0°C)', results_df['temperature'] < 0),
        ('Mild (0-20°C)', (results_df['temperature'] >= 0) & (results_df['temperature'] <= 20)),
        ('Hot (> 20°C)', results_df['temperature'] > 20)
    ]
    
    temp_table = []
    for name, mask in temp_bins:
        subset = results_df[mask]
        if len(subset) > 0:
            mae = subset['abs_error'].mean()
            bias = subset['error'].mean()
            print(f"  {name:<20} | Count: {len(subset):>5} | MAE: {mae:>7.1f}A | Bias: {bias:>+7.1f}A")
            temp_table.append({
                'Condition': name,
                'Sample Count': len(subset),
                'MAE (A)': round(mae, 1),
                'Bias (A)': round(bias, 1)
            })
    print("-" * 60)
    
    # Bin by wind speed
    print("\n💨 Errors by Wind Speed:")
    print("-" * 60)
    
    wind_bins = [
        ('Calm (< 2 m/s)', results_df['wind_speed'] < 2),
        ('Moderate (2-8 m/s)', (results_df['wind_speed'] >= 2) & (results_df['wind_speed'] <= 8)),
        ('High (> 8 m/s)', results_df['wind_speed'] > 8)
    ]
    
    wind_table = []
    for name, mask in wind_bins:
        subset = results_df[mask]
        if len(subset) > 0:
            mae = subset['abs_error'].mean()
            bias = subset['error'].mean()
            print(f"  {name:<20} | Count: {len(subset):>5} | MAE: {mae:>7.1f}A | Bias: {bias:>+7.1f}A")
            wind_table.append({
                'Condition': name,
                'Sample Count': len(subset),
                'MAE (A)': round(mae, 1),
                'Bias (A)': round(bias, 1)
            })
    print("-" * 60)
    
    # Find worst conditions
    worst_temp = max(temp_table, key=lambda x: x['MAE (A)'])
    worst_wind = max(wind_table, key=lambda x: x['MAE (A)'])
    
    print(f"\n🔴 Worst condition by temperature: {worst_temp['Condition']} (MAE: {worst_temp['MAE (A)']}A)")
    print(f"🔴 Worst condition by wind speed: {worst_wind['Condition']} (MAE: {worst_wind['MAE (A)']}A)")
    
    return temp_table, wind_table, results_df


def analyze_bias_direction(output_dir: Path):
    """
    Analysis 4: Error bias direction for all cross-region combinations
    """
    print("\n" + "="*70)
    print("ANALYSIS 4: Error Bias Direction")
    print("="*70)
    
    experiments = [
        ('VN_to_US', 'Vietnam → US'),
        ('US_to_VN', 'US → Vietnam')
    ]
    
    print("\n📊 Bias Analysis (bias = mean(predicted - actual)):")
    print("-" * 80)
    print(f"{'Experiment':<20} {'Overall Bias':<15} {'Interpretation'}")
    print("-" * 80)
    
    bias_table = []
    
    for exp_name, exp_label in experiments:
        ckpt = load_checkpoint(f'cross_region_results/model_{exp_name}.pt')
        metrics = ckpt['metrics']
        bias = metrics['bias']
        
        # Interpretation
        if bias > 200:
            interpretation = "⚠️ OVER-PREDICTING (predicts higher ampacity than actual)"
            safety = "DANGEROUS - May exceed thermal limits"
        elif bias < -200:
            interpretation = "✅ UNDER-PREDICTING (conservative, predicts lower ampacity)"
            safety = "SAFE - Built-in safety margin"
        else:
            interpretation = "✓ Balanced predictions"
            safety = "Acceptable"
        
        print(f"{exp_label:<20} {bias:>+10.1f}A    {interpretation}")
        print(f"{'':<20} {'':<15}     → {safety}")
        print()
        
        bias_table.append({
            'Experiment': exp_label,
            'Bias (A)': round(bias, 1),
            'Interpretation': interpretation,
            'Safety': safety
        })
    
    print("-" * 80)
    
    # Additional: Compare ampacity ranges
    print("\n📈 Ampacity Range Comparison:")
    print("-" * 60)
    
    df = load_unified_data()
    vn_amp = df[df['region'] == 'VN']['Ampacity'].dropna()
    us_amp = df[df['region'] == 'US']['actual'].dropna()
    
    print(f"Vietnam Ampacity:  mean={vn_amp.mean():.1f}A, std={vn_amp.std():.1f}A, "
          f"range=[{vn_amp.min():.0f}, {vn_amp.max():.0f}]")
    print(f"US Ampacity:       mean={us_amp.mean():.1f}A, std={us_amp.std():.1f}A, "
          f"range=[{us_amp.min():.0f}, {us_amp.max():.0f}]")
    print(f"Mean difference:   {us_amp.mean() - vn_amp.mean():.1f}A "
          f"({(us_amp.mean() / vn_amp.mean() - 1) * 100:.1f}% higher in US)")
    print("-" * 60)
    
    return bias_table


def create_error_visualization(results_df: pd.DataFrame, output_dir: Path):
    """Create visualization of errors by conditions"""
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('VN→US Cross-Region Error Analysis', fontsize=14, fontweight='bold')
    
    # 1. Error vs Temperature
    ax = axes[0, 0]
    ax.scatter(results_df['temperature'], results_df['error'], alpha=0.3, s=5)
    ax.axhline(y=0, color='red', linestyle='--', linewidth=2)
    ax.set_xlabel('Temperature (°C)')
    ax.set_ylabel('Prediction Error (A)')
    ax.set_title('Error vs Temperature')
    ax.grid(True, alpha=0.3)
    
    # Add trend line
    z = np.polyfit(results_df['temperature'], results_df['error'], 1)
    p = np.poly1d(z)
    temp_range = np.linspace(results_df['temperature'].min(), results_df['temperature'].max(), 100)
    ax.plot(temp_range, p(temp_range), "r-", alpha=0.8, linewidth=2, label='Trend')
    ax.legend()
    
    # 2. Error vs Wind Speed
    ax = axes[0, 1]
    ax.scatter(results_df['wind_speed'], results_df['error'], alpha=0.3, s=5)
    ax.axhline(y=0, color='red', linestyle='--', linewidth=2)
    ax.set_xlabel('Wind Speed (m/s)')
    ax.set_ylabel('Prediction Error (A)')
    ax.set_title('Error vs Wind Speed')
    ax.grid(True, alpha=0.3)
    
    # 3. Predicted vs Actual
    ax = axes[1, 0]
    ax.scatter(results_df['actual'], results_df['predicted'], alpha=0.3, s=5)
    min_val = min(results_df['actual'].min(), results_df['predicted'].min())
    max_val = max(results_df['actual'].max(), results_df['predicted'].max())
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2)
    ax.set_xlabel('Actual Ampacity (A)')
    ax.set_ylabel('Predicted Ampacity (A)')
    ax.set_title('Predicted vs Actual (VN→US)')
    ax.grid(True, alpha=0.3)
    
    # 4. Error distribution
    ax = axes[1, 1]
    ax.hist(results_df['error'], bins=100, alpha=0.7, edgecolor='black')
    ax.axvline(x=0, color='red', linestyle='--', linewidth=2)
    ax.axvline(x=results_df['error'].mean(), color='green', linestyle='-', linewidth=2, 
               label=f'Mean bias: {results_df["error"].mean():.1f}A')
    ax.set_xlabel('Prediction Error (A)')
    ax.set_ylabel('Frequency')
    ax.set_title('Error Distribution')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'vn_to_us_error_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Saved: vn_to_us_error_analysis.png")


def create_summary_report(output_dir: Path, feature_stats: List[Dict], 
                         temp_table: List[Dict], wind_table: List[Dict],
                         bias_table: List[Dict]):
    """Create a summary report for the paper"""
    
    report = []
    report.append("="*80)
    report.append("CROSS-REGION TRANSFER FAILURE DIAGNOSTIC REPORT")
    report.append("="*80)
    report.append("")
    
    # Section 1: Feature Distribution Gap
    report.append("1. FEATURE DISTRIBUTION GAP (Domain Shift)")
    report.append("-" * 80)
    report.append(f"{'Feature':<20} {'VN Mean':<12} {'US Mean':<12} {'Difference %':<15}")
    report.append("-" * 80)
    for s in feature_stats:
        report.append(f"{s['Feature']:<20} {s['VN_mean']:<12.2f} {s['US_mean']:<12.2f} "
                     f"{s['Mean_Diff_%']:<15.1f}%")
    report.append("-" * 80)
    report.append("")
    
    # Section 2: Error by Condition
    report.append("2. VN→US ERRORS BY WEATHER CONDITION")
    report.append("-" * 80)
    report.append("Temperature Conditions:")
    report.append(f"{'Condition':<25} {'Count':<10} {'MAE (A)':<12} {'Bias (A)'}")
    report.append("-" * 80)
    for row in temp_table:
        report.append(f"{row['Condition']:<25} {row['Sample Count']:<10} "
                     f"{row['MAE (A)']:<12} {row['Bias (A)']:+.1f}")
    report.append("")
    report.append("Wind Speed Conditions:")
    report.append(f"{'Condition':<25} {'Count':<10} {'MAE (A)':<12} {'Bias (A)'}")
    report.append("-" * 80)
    for row in wind_table:
        report.append(f"{row['Condition']:<25} {row['Sample Count']:<10} "
                     f"{row['MAE (A)']:<12} {row['Bias (A)']:+.1f}")
    report.append("-" * 80)
    report.append("")
    
    # Section 3: Bias Direction
    report.append("3. BIAS DIRECTION ANALYSIS")
    report.append("-" * 80)
    report.append(f"{'Experiment':<20} {'Bias (A)':<15} {'Interpretation'}")
    report.append("-" * 80)
    for row in bias_table:
        report.append(f"{row['Experiment']:<20} {row['Bias (A)']:<+15.1f} {row['Interpretation']}")
    report.append("-" * 80)
    report.append("")
    
    # Conclusions
    report.append("4. CONCLUSIONS")
    report.append("-" * 80)
    report.append("Key findings explaining cross-region transfer failure:")
    report.append("")
    
    # Find the largest mean difference
    max_diff_feature = max(feature_stats, key=lambda x: x['Mean_Diff_%'])
    report.append(f"• Domain Gap: {max_diff_feature['Feature']} shows {max_diff_feature['Mean_Diff_%']:.1f}% "
                  f"difference between regions")
    
    worst_temp = max(temp_table, key=lambda x: x['MAE (A)'])
    report.append(f"• Worst Condition: {worst_temp['Condition']} causes MAE of {worst_temp['MAE (A)']}A")
    
    vn_to_us_bias = next((b for b in bias_table if 'Vietnam → US' in b['Experiment']), None)
    if vn_to_us_bias:
        if vn_to_us_bias['Bias (A)'] < -200:
            report.append(f"• Safety: VN→US model under-predicts by {abs(vn_to_us_bias['Bias (A)']):.0f}A "
                         f"(conservative/safe)")
        elif vn_to_us_bias['Bias (A)'] > 200:
            report.append(f"• Safety: VN→US model over-predicts by {vn_to_us_bias['Bias (A)']:.0f}A "
                         f"(DANGEROUS - may exceed thermal limits)")
    
    report.append("")
    report.append("Recommendation: The ~600A MAE gap is due to:")
    report.append("  1. Different conductor types/ratings between regions")
    report.append("  2. Domain shift in weather distributions")
    report.append("  3. Lack of per-line physics adaptation in cross-region setting")
    report.append("")
    report.append("="*80)
    
    report_text = "\n".join(report)
    
    # Save to file
    with open(output_dir / 'diagnostic_report.txt', 'w') as f:
        f.write(report_text)
    
    print(report_text)
    
    return report_text


def main():
    output_dir = Path('cross_region_results')
    output_dir.mkdir(exist_ok=True)
    
    print("="*80)
    print("CROSS-REGION TRANSFER FAILURE DIAGNOSTIC")
    print("="*80)
    
    # Load data
    df = load_unified_data()
    
    # Run analyses
    feature_stats = analyze_feature_distributions(df, output_dir)
    temp_table, wind_table, results_df = analyze_vn_to_us_errors(output_dir)
    bias_table = analyze_bias_direction(output_dir)
    
    # Create visualizations
    create_error_visualization(results_df, output_dir)
    
    # Create summary report
    create_summary_report(output_dir, feature_stats, temp_table, wind_table, bias_table)
    
    print("\n" + "="*80)
    print("✅ DIAGNOSTIC ANALYSIS COMPLETE")
    print("="*80)
    print(f"\nOutput files saved to: {output_dir}")
    print("  - feature_distributions_comparison.png")
    print("  - vn_to_us_error_analysis.png")
    print("  - diagnostic_report.txt")


if __name__ == "__main__":
    main()
