"""
Invariant-PIKAN: Adversarially-Robust Physics-Informed Neural Networks for Dynamic Line Rating
Copyright (C) 2025 Gelavizh Ahmadi / Invariant Research

This software is licensed under the Business Source License 1.1 (BSL 1.1).
Commercial production use requires a separate license agreement.
See LICENSE.txt for full terms.

"""

#!/usr/bin/env python3
"""
Batch test of safety-buffered model against multiple adversarial datasets.

Evaluates model robustness across different attack strengths:
- 20% vs 50% adversarial samples
- epsilon values: 0.5, 1.0, 10.0
- BIM (Basic Iterative Method) attacks

Expected files in data/mendeley/:
  - "20% BIM e=0.5.csv"
  - "20% BIM e=1.csv"
  - "20% BIM e=10.csv"
  - "50% BIM e=0.5.csv"
  - "50% BIM e=1.csv"
  - "50% BIM e=10.csv"
"""

import torch
import pandas as pd
import numpy as np
from pathlib import Path
import sys
import json
import re
import matplotlib.pyplot as plt

sys.path.append(str(Path(__file__).parent.parent))

from scripts.create_safety_buffered_model import SafetyBufferedModel
from core.data import VietnamDataset


def evaluate_on_file(model, csv_path, device='cpu', batch_size=256, region='VN'):
    """
    Evaluate model on a CSV file.
    
    Returns dict with metrics or None if file doesn't exist.
    """
    if not Path(csv_path).exists():
        return None
    
    print(f"  Loading {Path(csv_path).name}...")
    
    try:
        dataset = VietnamDataset(str(csv_path))
        loader = torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, shuffle=False, num_workers=0
        )
    except Exception as e:
        print(f"    Error loading dataset: {e}")
        return None
    
    model.eval()
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            
            # Extract weather features
            weather = x[:, :4]
            weather_dict = {
                'T_amb': weather[:, 0],
                'wind_speed': weather[:, 1],
                'solar': weather[:, 3]
            }
            
            # Forward through safety-buffered model
            output = model(weather, weather_dict, region=region)
            
            all_preds.extend(output['ampacity'].cpu().numpy())
            all_targets.extend(y[:, 1].cpu().numpy())
    
    preds = np.array(all_preds).flatten()
    targets = np.array(all_targets).flatten()
    
    # Calculate metrics
    errors = preds - targets
    mae = np.mean(np.abs(errors))
    rmse = np.sqrt(np.mean(errors**2))
    bias = np.mean(errors)
    
    # Percentage within various thresholds
    pct_within_5 = np.mean(np.abs(errors) / targets < 0.05) * 100
    pct_within_10 = np.mean(np.abs(errors) / targets < 0.10) * 100
    pct_within_20 = np.mean(np.abs(errors) / targets < 0.20) * 100
    
    return {
        'mae': mae,
        'rmse': rmse,
        'bias': bias,
        'pct_within_5': pct_within_5,
        'pct_within_10': pct_within_10,
        'pct_within_20': pct_within_20,
        'samples': len(preds)
    }


def parse_filename(filename):
    """
    Extract attack parameters from filename.
    
    Pattern: "{percentage}% BIM e={epsilon}.csv"
    Example: "20% BIM e=0.5.csv" -> {'percentage': 20, 'method': 'BIM', 'epsilon': 0.5}
    """
    # Pattern: digits%, space, method, space, e=, number (handle trailing dots)
    pattern = r'(\d+)%\s*([A-Za-z]+)\s*e=([0-9.]+)'
    match = re.search(pattern, filename)
    
    if match:
        epsilon_str = match.group(3).rstrip('.')  # Remove trailing dots
        return {
            'percentage': int(match.group(1)),
            'method': match.group(2),
            'epsilon': float(epsilon_str)
        }
    
    # Try alternative patterns
    # Some files might use different naming
    alt_pattern = r'([A-Za-z]+)_(\d+)pct_eps([0-9.]+)'
    match = re.search(alt_pattern, filename)
    if match:
        epsilon_str = match.group(3).rstrip('.')
        return {
            'method': match.group(1),
            'percentage': int(match.group(2)),
            'epsilon': float(epsilon_str)
        }
    
    return {'percentage': None, 'method': None, 'epsilon': None}


def plot_results(df, clean_mae, output_dir):
    """Generate visualization of adversarial robustness."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 1. MAE vs Epsilon
    ax = axes[0, 0]
    for pct in df['percentage'].unique():
        if pd.isna(pct):
            continue
        subset = df[df['percentage'] == pct].sort_values('epsilon')
        ax.plot(subset['epsilon'], subset['mae'], marker='o', label=f'{pct}% adversarial')
    
    ax.axhline(y=clean_mae, color='green', linestyle='--', label='Clean baseline')
    ax.set_xlabel('Epsilon (attack strength)')
    ax.set_ylabel('MAE (Amps)')
    ax.set_title('MAE vs Attack Strength')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xscale('log')
    
    # 2. Degradation percentage
    ax = axes[0, 1]
    for pct in df['percentage'].unique():
        if pd.isna(pct):
            continue
        subset = df[df['percentage'] == pct].sort_values('epsilon')
        ax.plot(subset['epsilon'], subset['mae_degrad_pct'], marker='s', label=f'{pct}% adversarial')
    
    ax.axhline(y=10, color='orange', linestyle='--', label='10% threshold (good)')
    ax.axhline(y=20, color='red', linestyle='--', label='20% threshold (moderate)')
    ax.set_xlabel('Epsilon (attack strength)')
    ax.set_ylabel('MAE Degradation (%)')
    ax.set_title('Performance Degradation vs Attack Strength')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xscale('log')
    
    # 3. Bias comparison
    ax = axes[1, 0]
    x_pos = np.arange(len(df))
    width = 0.35
    
    colors = ['lightblue' if p == 20 else 'lightcoral' for p in df['percentage']]
    ax.bar(x_pos, df['bias'], color=colors)
    ax.set_xticks(x_pos)
    ax.set_xticklabels([f"{r['percentage']}%\nε={r['epsilon']}" for _, r in df.iterrows()], 
                       rotation=45, ha='right')
    ax.set_ylabel('Bias (Amps)')
    ax.set_title('Prediction Bias by Attack Configuration')
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax.grid(True, alpha=0.3, axis='y')
    
    # 4. Accuracy within thresholds
    ax = axes[1, 1]
    width = 0.25
    x = np.arange(len(df))
    
    ax.bar(x - width, df['pct_within_5'], width, label='±5%', alpha=0.8)
    ax.bar(x, df['pct_within_10'], width, label='±10%', alpha=0.8)
    ax.bar(x + width, df['pct_within_20'], width, label='±20%', alpha=0.8)
    
    ax.set_xticks(x)
    ax.set_xticklabels([f"{r['percentage']}%\nε={r['epsilon']}" for _, r in df.iterrows()],
                       rotation=45, ha='right')
    ax.set_ylabel('Percentage of Predictions (%)')
    ax.set_title('Prediction Accuracy by Threshold')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim([0, 100])
    
    plt.tight_layout()
    plot_path = output_dir / 'adversarial_robustness_plot.png'
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"\n📊 Plot saved to {plot_path}")
    plt.close()


def main():
    print("="*70)
    print("ADVERSARIAL ROBUSTNESS BATCH TEST")
    print("="*70)
    
    # Device
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"\n🔧 Device: {device}")
    
    # Load model
    model_path = 'models/safety_buffered_model.pt'
    print(f"\n📦 Loading model from {model_path}")
    model = SafetyBufferedModel.load(model_path, device=device)
    model = model.to(device)
    
    # Clean data baseline
    clean_path = Path('data/mendeley/One_and_half_year_data.csv')
    print(f"\n📊 Baseline: Evaluating on clean data...")
    clean_metrics = evaluate_on_file(model, clean_path, device, region='VN')
    
    if clean_metrics is None:
        print("❌ Clean data not found!")
        return
    
    print(f"\n✅ Clean Data Results:")
    print(f"  MAE: {clean_metrics['mae']:.2f} A")
    print(f"  RMSE: {clean_metrics['rmse']:.2f} A")
    print(f"  Bias: {clean_metrics['bias']:.2f} A")
    print(f"  Within ±5%: {clean_metrics['pct_within_5']:.1f}%")
    print(f"  Within ±10%: {clean_metrics['pct_within_10']:.1f}%")
    print(f"  Within ±20%: {clean_metrics['pct_within_20']:.1f}%")
    print(f"  Samples: {clean_metrics['samples']}")
    
    # Find all adversarial CSVs
    adv_dir = Path('data/mendeley')
    
    # Expected adversarial files (BIM and FGSM)
    expected_files = [
        # BIM attacks
        "20% BIM e=0.5.csv",
        "20% BIM e=1.csv",
        "20% BIM e=10.csv",
        "50% BIM e=0.5.csv",
        "50% BIM e=1.csv",
        "50% BIM e=10.csv",
        # FGSM attacks
        "20% FGSM e=0.5.csv",
        "20% FGSM e=1.csv",
        "20% FGSM e=10.csv",
        "50% FGSM e=0.5.csv",
        "50% FGSM e=1.csv",
        "50% FGSM e=10.csv",
    ]
    
    print(f"\n🔍 Looking for adversarial files in {adv_dir}...")
    
    results = []
    found_files = []
    
    for filename in expected_files:
        filepath = adv_dir / filename
        if filepath.exists():
            found_files.append(filename)
            print(f"\n⚠️  Evaluating {filename}...")
            
            # Parse filename
            info = parse_filename(filename)
            print(f"    Attack: {info['method']}, {info['percentage']}% samples, ε={info['epsilon']}")
            
            # Evaluate
            metrics = evaluate_on_file(model, filepath, device, region='VN')
            
            if metrics:
                # Add attack parameters
                metrics.update(info)
                metrics['file'] = filename
                
                # Calculate degradation
                metrics['mae_degrad'] = metrics['mae'] - clean_metrics['mae']
                metrics['mae_degrad_pct'] = (metrics['mae_degrad'] / clean_metrics['mae']) * 100
                metrics['rmse_degrad'] = metrics['rmse'] - clean_metrics['rmse']
                metrics['bias_shift'] = metrics['bias'] - clean_metrics['bias']
                
                results.append(metrics)
                
                print(f"    MAE: {metrics['mae']:.2f} A (degradation: {metrics['mae_degrad_pct']:+.1f}%)")
                print(f"    Bias: {metrics['bias']:.2f} A (shift: {metrics['bias_shift']:+.2f})")
        else:
            print(f"\n❌ Not found: {filename}")
    
    if not results:
        print("\n" + "="*70)
        print("❌ No adversarial files found!")
        print("="*70)
        print("\nPlease download the adversarial datasets from Mendeley:")
        print("  https://data.mendeley.com/datasets/xrhwdj7m7z/")
        print("\nExpected files:")
        for f in expected_files:
            print(f"  - {f}")
        print(f"\nPlace them in: {adv_dir}/")
        return
    
    # Create DataFrame
    df = pd.DataFrame(results)
    
    # Sort for display
    df = df.sort_values(['percentage', 'epsilon'])
    
    # Summary table
    print("\n" + "="*70)
    print("ADVERSARIAL ROBUSTNESS SUMMARY")
    print("="*70)
    
    display_cols = ['percentage', 'epsilon', 'mae', 'mae_degrad_pct', 
                   'bias', 'pct_within_10', 'pct_within_20']
    
    print("\n" + df[display_cols].to_string(index=False))
    
    # Robustness assessment
    print("\n" + "="*70)
    print("ROBUSTNESS ASSESSMENT")
    print("="*70)
    
    for _, row in df.iterrows():
        degrad = row['mae_degrad_pct']
        
        if degrad < 10:
            status = "✅ EXCELLENT"
        elif degrad < 20:
            status = "✅ GOOD"
        elif degrad < 50:
            status = "⚠️  MODERATE"
        else:
            status = "❌ POOR"
        
        print(f"{row['percentage']}% BIM ε={row['epsilon']}: {status} ({degrad:+.1f}% degradation)")
    
    # Overall assessment
    avg_degrad = df['mae_degrad_pct'].mean()
    max_degrad = df['mae_degrad_pct'].max()
    
    print(f"\nOverall Statistics:")
    print(f"  Average degradation: {avg_degrad:.1f}%")
    print(f"  Maximum degradation: {max_degrad:.1f}%")
    
    if max_degrad < 20:
        overall = "✅ ROBUST - Model handles adversarial attacks well"
    elif max_degrad < 50:
        overall = "⚠️  MODERATE - Some vulnerability to strong attacks"
    else:
        overall = "❌ VULNERABLE - Significant degradation under attack"
    
    print(f"\n{overall}")
    
    # Save results
    out_dir = Path('validation_results')
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # CSV
    csv_path = out_dir / 'adversarial_results.csv'
    df.to_csv(csv_path, index=False)
    print(f"\n💾 CSV saved to {csv_path}")
    
    # JSON
    json_path = out_dir / 'adversarial_results.json'
    
    # Convert numpy types to Python native types
    def convert_to_native(obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: convert_to_native(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_native(i) for i in obj]
        return obj
    
    with open(json_path, 'w') as f:
        json.dump({
            'clean_metrics': convert_to_native(clean_metrics),
            'adversarial_results': convert_to_native(results),
            'summary': {
                'average_degradation_pct': float(avg_degrad),
                'maximum_degradation_pct': float(max_degrad),
                'overall_assessment': overall,
                'files_tested': found_files
            }
        }, f, indent=2)
    print(f"💾 JSON saved to {json_path}")
    
    # Plot
    plot_results(df, clean_metrics['mae'], out_dir)
    
    print("\n" + "="*70)
    print("✅ Batch test complete!")
    print("="*70)


if __name__ == '__main__':
    main()
