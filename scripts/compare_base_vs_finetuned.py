"""
Invariant-PIKAN: Adversarially-Robust Physics-Informed Neural Networks for Dynamic Line Rating
Copyright (C) 2025 Gelavizh Ahmadi / Invariant Research

This software is licensed under the Business Source License 1.1 (BSL 1.1).
Commercial production use requires a separate license agreement.
See LICENSE.txt for full terms.

HWF-PIKAN for plasma physics (Heravifard et al., Sharif University, 2025).
"""

#!/usr/bin/env python3
"""
Compare base model vs fine-tuned model on US test data.
"""

import torch
import pandas as pd
import numpy as np
from pathlib import Path
import sys
import argparse

sys.path.append(str(Path(__file__).parent.parent))

from models.invariant_pikan_v2 import create_invariant_pikan_v2
from core.data import VietnamDataset
from torch.utils.data import DataLoader, Subset


def compute_metrics(y_true, y_pred):
    errors = y_pred - y_true
    mae = np.mean(np.abs(errors))
    rmse = np.sqrt(np.mean(errors**2))
    bias = np.mean(errors)
    return {'mae': mae, 'rmse': rmse, 'bias': bias}


def evaluate_model(model, test_loader, device):
    """Evaluate model on test set."""
    all_preds = []
    all_targets = []
    
    model.eval()
    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(device)
            weather = x[:, :4]
            weather_dict = {
                'T_amb': weather[:, 0],
                'wind_speed': weather[:, 1],
                'solar': weather[:, 3]
            }
            
            output = model(weather, weather_dict)
            preds = output['ampacity'].cpu().numpy()
            targets = y[:, 1].cpu().numpy()
            
            all_preds.extend(preds.flatten())
            all_targets.extend(targets.flatten())
    
    return np.array(all_preds), np.array(all_targets)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--base-model', required=True, help='Path to base model')
    parser.add_argument('--finetuned-model', required=True, help='Path to fine-tuned model')
    parser.add_argument('--test-data', required=True, help='Path to test data CSV')
    parser.add_argument('--test-idx', required=True, help='Path to test indices')
    parser.add_argument('--device', default='auto')
    args = parser.parse_args()
    
    # Device
    if args.device == 'auto':
        device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    print(f"Device: {device}")
    
    # Load test indices
    test_idx = torch.load(args.test_idx)
    print(f"Test samples: {len(test_idx)}")
    
    # Load dataset (test_data.csv already contains only test samples)
    dataset = VietnamDataset(args.test_data)
    test_loader = DataLoader(dataset, batch_size=256, shuffle=False)
    
    # Get region info
    test_df = pd.read_csv(args.test_data)
    # test_data.csv already contains only test samples, so use directly
    if len(test_df) == len(test_idx):
        pass  # Already filtered
    else:
        test_df = test_df.iloc[test_idx]
    
    # Load base model
    print(f"\nLoading base model: {args.base_model}")
    base_ckpt = torch.load(args.base_model, map_location=device, weights_only=False)
    model_cfg = base_ckpt.get('config', {}).get('model', None) if isinstance(base_ckpt, dict) and 'config' in base_ckpt else {}
    base_model = create_invariant_pikan_v2(config=model_cfg)
    base_model.load_state_dict(base_ckpt['model_state_dict'])
    base_model = base_model.to(device)
    
    # Load fine-tuned model
    print(f"Loading fine-tuned model: {args.finetuned_model}")
    ft_ckpt = torch.load(args.finetuned_model, map_location=device, weights_only=False)
    ft_model = create_invariant_pikan_v2(config=model_cfg)
    ft_model.load_state_dict(ft_ckpt['model_state_dict'])
    ft_model = ft_model.to(device)
    
    # Evaluate base model
    print("\nEvaluating base model...")
    base_preds, targets = evaluate_model(base_model, test_loader, device)
    base_metrics = compute_metrics(targets, base_preds)
    
    # Evaluate fine-tuned model
    print("Evaluating fine-tuned model...")
    ft_preds, _ = evaluate_model(ft_model, test_loader, device)
    ft_metrics = compute_metrics(targets, ft_preds)
    
    # Overall results
    print("\n" + "="*70)
    print("OVERALL RESULTS")
    print("="*70)
    print(f"{'Model':<20} {'MAE':<12} {'RMSE':<12} {'Bias':<12}")
    print("-"*70)
    print(f"{'Base Model':<20} {base_metrics['mae']:<12.2f} {base_metrics['rmse']:<12.2f} {base_metrics['bias']:<12.2f}")
    print(f"{'Fine-tuned (US)':<20} {ft_metrics['mae']:<12.2f} {ft_metrics['rmse']:<12.2f} {ft_metrics['bias']:<12.2f}")
    
    mae_change = base_metrics['mae'] - ft_metrics['mae']
    bias_change = abs(base_metrics['bias']) - abs(ft_metrics['bias'])
    print(f"\nImprovement:")
    print(f"  MAE: {mae_change:+.2f} A ({'better' if mae_change > 0 else 'worse'})")
    print(f"  Bias reduction: {bias_change:+.2f} A ({'better' if bias_change > 0 else 'worse'})")
    
    # US-specific results
    if 'region' in test_df.columns:
        us_mask = test_df['region'].values == 'US'
        us_targets = targets[us_mask]
        us_base_preds = base_preds[us_mask]
        us_ft_preds = ft_preds[us_mask]
        
        us_base_metrics = compute_metrics(us_targets, us_base_preds)
        us_ft_metrics = compute_metrics(us_targets, us_ft_preds)
        
        print("\n" + "="*70)
        print(f"US REGION RESULTS ({us_mask.sum()} samples)")
        print("="*70)
        print(f"{'Model':<20} {'MAE':<12} {'RMSE':<12} {'Bias':<12}")
        print("-"*70)
        print(f"{'Base Model':<20} {us_base_metrics['mae']:<12.2f} {us_base_metrics['rmse']:<12.2f} {us_base_metrics['bias']:<12.2f}")
        print(f"{'Fine-tuned (US)':<20} {us_ft_metrics['mae']:<12.2f} {us_ft_metrics['rmse']:<12.2f} {us_ft_metrics['bias']:<12.2f}")
        
        us_mae_change = us_base_metrics['mae'] - us_ft_metrics['mae']
        us_bias_change = abs(us_base_metrics['bias']) - abs(us_ft_metrics['bias'])
        print(f"\nUS Improvement:")
        print(f"  MAE: {us_mae_change:+.2f} A ({'better' if us_mae_change > 0 else 'worse'})")
        print(f"  Bias reduction: {us_bias_change:+.2f} A ({'better' if us_bias_change > 0 else 'worse'})")


if __name__ == '__main__':
    main()
