"""
Invariant-PIKAN: Adversarially-Robust Physics-Informed Neural Networks for Dynamic Line Rating
Copyright (C) 2025 Gelavizh Ahmadi / Invariant Research

This software is licensed under the Business Source License 1.1 (BSL 1.1).
Commercial production use requires a separate license agreement.
See LICENSE.txt for full terms.

DISCLAIMER: This implementation is independent of concurrent academic work on
HWF-PIKAN for plasma physics (Heravifard et al., Sharif University, 2025).
"""

#!/usr/bin/env python3
"""
Test model robustness on adversarial Vietnam DLR data.

Compares performance on clean vs adversarial (perturbed) data.
Expected: MAE should not degrade significantly (<20% increase).
"""

import torch
import pandas as pd
import numpy as np
from pathlib import Path
import sys
import argparse

sys.path.append(str(Path(__file__).parent.parent))

from scripts.create_safety_buffered_model import SafetyBufferedModel
from core.data import VietnamDataset


def evaluate_on_data(model, data_path, device='cpu', region='VN'):
    """
    Evaluate model on a dataset.
    
    Args:
        model: SafetyBufferedModel instance
        data_path: Path to CSV file
        device: torch device
        region: 'VN' or 'US' for bias correction
        
    Returns:
        dict with metrics
    """
    if not Path(data_path).exists():
        return None
    
    print(f"\n📊 Evaluating on {Path(data_path).name}...")
    
    # Load dataset
    try:
        dataset = VietnamDataset(data_path)
        loader = torch.utils.data.DataLoader(
            dataset, batch_size=256, shuffle=False, num_workers=0
        )
    except Exception as e:
        print(f"  Error loading dataset: {e}")
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
            
            # Forward pass
            output = model(weather, weather_dict, region=region)
            
            all_preds.extend(output['ampacity'].cpu().numpy())
            all_targets.extend(y[:, 1].cpu().numpy())
    
    preds = np.array(all_preds).flatten()
    targets = np.array(all_targets).flatten()
    
    # Calculate metrics
    mae = np.mean(np.abs(preds - targets))
    rmse = np.sqrt(np.mean((preds - targets)**2))
    bias = np.mean(preds - targets)
    
    # Percentage within ±10% of target
    pct_within_10 = np.mean(np.abs(preds - targets) / targets < 0.1) * 100
    
    return {
        'mae': mae,
        'rmse': rmse,
        'bias': bias,
        'pct_within_10': pct_within_10,
        'n_samples': len(preds)
    }


def compare_clean_vs_adversarial(model_path, device='cpu'):
    """
    Compare model performance on clean vs adversarial data.
    """
    print("="*70)
    print("ADVERSARIAL ROBUSTNESS TEST")
    print("="*70)
    
    # Load model
    print(f"\n📦 Loading model from {model_path}")
    model = SafetyBufferedModel.load(model_path, device=device)
    model = model.to(device)
    
    # Paths
    clean_path = Path('data/mendeley/One_and_half_year_data.csv')
    adv_path = Path('data/mendeley/One_and_half_year_data_adversarial.csv')
    
    # Evaluate on clean data
    clean_results = evaluate_on_data(model, str(clean_path), device, region='VN')
    
    if clean_results is None:
        print("❌ Clean data not found!")
        return
    
    print(f"\n✅ Clean Data Results:")
    print(f"  MAE: {clean_results['mae']:.2f} A")
    print(f"  RMSE: {clean_results['rmse']:.2f} A")
    print(f"  Bias: {clean_results['bias']:.2f} A")
    print(f"  Within ±10%: {clean_results['pct_within_10']:.1f}%")
    print(f"  Samples: {clean_results['n_samples']}")
    
    # Evaluate on adversarial data
    adv_results = evaluate_on_data(model, str(adv_path), device, region='VN')
    
    if adv_results is None:
        print(f"\n⚠️  Adversarial data not found at {adv_path}")
        print("\nPlease download the adversarial dataset from Mendeley:")
        print("  https://data.mendeley.com/datasets/xrhwdj7m7z/")
        print("\nLook for a file named:")
        print("  - One_and_half_year_data_adversarial.csv")
        print("  - One_and_half_year_data_adv.csv")
        print("  - Or similar with 'adversarial' in the name")
        print(f"\nPlace it in: {adv_path.parent}/")
        return
    
    print(f"\n⚠️  Adversarial Data Results:")
    print(f"  MAE: {adv_results['mae']:.2f} A")
    print(f"  RMSE: {adv_results['rmse']:.2f} A")
    print(f"  Bias: {adv_results['bias']:.2f} A")
    print(f"  Within ±10%: {adv_results['pct_within_10']:.1f}%")
    print(f"  Samples: {adv_results['n_samples']}")
    
    # Compare
    print(f"\n" + "="*70)
    print("COMPARISON")
    print("="*70)
    
    mae_degradation = adv_results['mae'] - clean_results['mae']
    mae_degradation_pct = (mae_degradation / clean_results['mae']) * 100
    
    rmse_degradation = adv_results['rmse'] - clean_results['rmse']
    rmse_degradation_pct = (rmse_degradation / clean_results['rmse']) * 100
    
    print(f"\nMAE Degradation: {mae_degradation:.2f} A ({mae_degradation_pct:+.1f}%)")
    print(f"RMSE Degradation: {rmse_degradation:.2f} A ({rmse_degradation_pct:+.1f}%)")
    
    # Robustness assessment
    print(f"\n" + "="*70)
    print("ROBUSTNESS ASSESSMENT")
    print("="*70)
    
    if mae_degradation_pct < 10:
        robustness = "✅ EXCELLENT - Model is highly robust"
    elif mae_degradation_pct < 20:
        robustness = "✅ GOOD - Minor degradation, acceptable for production"
    elif mae_degradation_pct < 50:
        robustness = "⚠️  MODERATE - Consider additional regularization"
    else:
        robustness = "❌ POOR - Model vulnerable to adversarial perturbations"
    
    print(f"\n{robustness}")
    
    # Save results
    results = {
        'clean': clean_results,
        'adversarial': adv_results,
        'degradation': {
            'mae_abs': float(mae_degradation),
            'mae_pct': float(mae_degradation_pct),
            'rmse_abs': float(rmse_degradation),
            'rmse_pct': float(rmse_degradation_pct)
        },
        'assessment': robustness
    }
    
    output_path = Path('validation_results/adversarial_test.json')
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n💾 Results saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Test model robustness on adversarial data'
    )
    parser.add_argument('--model', type=str, 
                       default='models/safety_buffered_model.pt',
                       help='Path to safety-buffered model')
    parser.add_argument('--device', type=str, default='auto')
    
    args = parser.parse_args()
    
    # Device
    if args.device == 'auto':
        device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    
    compare_clean_vs_adversarial(args.model, device)


if __name__ == '__main__':
    main()
