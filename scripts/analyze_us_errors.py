"""
Invariant-PIKAN: Adversarially-Robust Physics-Informed Neural Networks for Dynamic Line Rating
Copyright (C) 2025 Gelavizh Ahmadi / Invariant Research

This software is licensed under the Business Source License 1.1 (BSL 1.1).
Commercial production use requires a separate license agreement.
See LICENSE.txt for full terms.

"""

#!/usr/bin/env python
"""
Analyze US test errors by condition (wind speed, temperature, etc.)
Usage: python scripts/analyze_us_errors.py --run-dir <path> [--output-dir <path>]
"""

import argparse
import json
import torch
import pandas as pd
import numpy as np
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent.parent))

from models.invariant_pikan_v2 import create_invariant_pikan_v2
from core.data import VietnamDataset, USDataset
from torch.utils.data import Subset, DataLoader

def compute_metrics(y_true, y_pred):
    errors = y_pred - y_true
    mae = np.mean(np.abs(errors))
    rmse = np.sqrt(np.mean(errors**2))
    bias = np.mean(errors)
    return mae, rmse, bias

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-dir', required=True)
    parser.add_argument('--output-dir', default='us_error_analysis')
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--device', default='cpu')
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load test indices
    test_idx = torch.load(run_dir / 'test_indices.pt')
    print(f"Loaded {len(test_idx)} test indices")

    # Load config to get data path and dataset type
    with open(run_dir / 'config.json') as f:
        config = json.load(f)
    
    # Resolve device
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu'))
    else:
        device = torch.device(args.device)

    # Reconstruct dataset used for training
    dataset = None
    raw_df_for_meta = None
    temp_csv = run_dir / 'temp_unified_data.csv'
    
    if temp_csv.exists():
        dataset = VietnamDataset(str(temp_csv))
        raw_df_for_meta = pd.read_csv(temp_csv)
    else:
        data_path = config.get('data_path')
        us_data = config.get('us_data_path')
        if us_data:
            dataset = USDataset(us_data, normalizer=None)
        elif data_path:
            dataset = VietnamDataset(data_path)
            # Try to load full dataframe for metadata
            if data_path.endswith('.h5') or data_path.endswith('.hdf5'):
                raw_df_for_meta = pd.read_hdf(data_path, key='data')
            else:
                raw_df_for_meta = pd.read_csv(data_path)
        else:
            raise RuntimeError('Could not determine dataset path from run config')

    test_dataset = Subset(dataset, test_idx)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    
    print(f"Test dataset size: {len(test_dataset)}")

    # Load model
    ckpt_path = run_dir / 'best_model.pt'
    if not ckpt_path.exists():
        ckpt_path = run_dir / 'final_model.pt'
    
    ckpt = torch.load(ckpt_path, map_location='cpu')
    model_cfg = None
    if isinstance(ckpt, dict) and 'config' in ckpt and isinstance(ckpt['config'], dict):
        model_cfg = ckpt['config'].get('model', None)

    model = create_invariant_pikan_v2(config=model_cfg) if model_cfg is not None else create_invariant_pikan_v2({})
    model.load_state_dict(ckpt['model_state_dict'])
    model.to(device)
    model.eval()

    # Run inference
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for x, y in test_loader:
            weather = x[:, :4].to(device)
            current = x[:, 4:5].to(device)
            weather_dict = {'T_amb': weather[:, 0], 'wind_speed': weather[:, 1], 'solar': weather[:, 3]}
            
            out = model(weather, weather_dict)
            preds = out['ampacity'].detach().cpu().numpy()
            targets = y[:, 1].cpu().numpy()
            
            all_preds.append(preds)
            all_targets.append(targets)

    preds = np.concatenate(all_preds)
    targets = np.concatenate(all_targets)
    
    print(f"Total predictions: {len(preds)}")

    # Create results DataFrame with metadata if available
    if raw_df_for_meta is not None:
        test_meta = raw_df_for_meta.iloc[test_idx].copy()
        test_meta['pred'] = preds
        test_meta['target'] = targets
        test_meta['error'] = preds - targets
        test_meta['abs_error'] = np.abs(test_meta['error'])
    else:
        # Create minimal dataframe
        test_meta = pd.DataFrame({
            'pred': preds,
            'target': targets,
            'error': preds - targets,
            'abs_error': np.abs(preds - targets),
            'region': 'UNKNOWN'
        })

    # Filter US if region column exists
    if 'region' in test_meta.columns:
        us_df = test_meta[test_meta['region'] == 'US'].copy()
    else:
        us_df = test_meta.copy()
    
    print(f"US test samples: {len(us_df)}")
    
    if len(us_df) == 0:
        print("No US samples found. Available regions:", test_meta['region'].unique() if 'region' in test_meta.columns else 'N/A')
        return

    # Define bins
    wind_bins = [(0, 2), (2, 5), (5, 10), (10, 100)]
    temp_bins = [(-100, 0), (0, 15), (15, 25), (25, 100)]

    # Need to get original feature values for binning
    # If we have raw_df_for_meta, use it; otherwise we need to extract from dataset
    if raw_df_for_meta is not None and 'wind_speed' in raw_df_for_meta.columns:
        us_meta = raw_df_for_meta.iloc[test_idx].copy()
        us_meta = us_meta[us_meta['region'] == 'US'] if 'region' in us_meta.columns else us_meta
        us_meta['pred'] = us_df['pred'].values
        us_meta['target'] = us_df['target'].values
        us_meta['error'] = us_df['error'].values
        us_meta['abs_error'] = us_df['abs_error'].values
    else:
        us_meta = us_df

    # Map column names - the unified data uses different column names
    wind_col = 'Wind1' if 'Wind1' in us_meta.columns else ('wind_speed' if 'wind_speed' in us_meta.columns else None)
    temp_col = 'temp' if 'temp' in us_meta.columns else ('temperature' if 'temperature' in us_meta.columns else ('T_amb' if 'T_amb' in us_meta.columns else None))
    
    print(f"Using columns: wind={wind_col}, temp={temp_col}")
    print(f"Available columns: {list(us_meta.columns)}")

    # Per wind speed bin
    wind_stats = []
    if wind_col:
        for low, high in wind_bins:
            mask = (us_meta[wind_col] >= low) & (us_meta[wind_col] < high)
            sub = us_meta[mask]
            if len(sub) > 0:
                mae, rmse, bias = compute_metrics(sub['target'].values, sub['pred'].values)
                wind_stats.append({
                    'wind_range': f"{low}-{high} m/s",
                    'count': len(sub),
                    'mae': mae,
                    'rmse': rmse,
                    'bias': bias
                })
        
        if wind_stats:
            wind_df = pd.DataFrame(wind_stats)
            wind_df.to_csv(output_dir / 'us_errors_by_wind.csv', index=False)
            print("\nSaved US errors by wind to", output_dir / 'us_errors_by_wind.csv')
            print(wind_df.to_string(index=False))

    # Per temperature bin
    temp_stats = []
    if temp_col:
        for low, high in temp_bins:
            mask = (us_meta[temp_col] >= low) & (us_meta[temp_col] < high)
            sub = us_meta[mask]
            if len(sub) > 0:
                mae, rmse, bias = compute_metrics(sub['target'].values, sub['pred'].values)
                temp_stats.append({
                    'temp_range': f"{low}-{high} °C",
                    'count': len(sub),
                    'mae': mae,
                    'rmse': rmse,
                    'bias': bias
                })
        
        if temp_stats:
            temp_df = pd.DataFrame(temp_stats)
            temp_df.to_csv(output_dir / 'us_errors_by_temp.csv', index=False)
            print("\nSaved US errors by temperature to", output_dir / 'us_errors_by_temp.csv')
            print(temp_df.to_string(index=False))

    # Also save full US results
    us_df.to_csv(output_dir / 'us_test_predictions.csv', index=False)
    print(f"\nSaved full US predictions to {output_dir / 'us_test_predictions.csv'}")

    # Overall US metrics
    overall_mae, overall_rmse, overall_bias = compute_metrics(us_df['target'].values, us_df['pred'].values)
    print(f"\n=== Overall US Metrics ===")
    print(f"MAE: {overall_mae:.2f} A")
    print(f"RMSE: {overall_rmse:.2f} A")
    print(f"Bias: {overall_bias:.2f} A")
    print(f"Samples: {len(us_df)}")

    print("\nAnalysis complete.")

if __name__ == '__main__':
    main()
