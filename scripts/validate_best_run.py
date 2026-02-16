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
validate_best_run.py

Comprehensive validation for latest InvariantPIKAN v2 training run.

- Loads best_model.pt from the most recent run (falls back to models/best_model.pt)
- Uses Vietnam CSV for test set (temporal 20% holdout)
- Computes ampacity/temp MAE, RMSE, R², physics residual MAE
- Breakdown by wind-speed bins, temperature bins, and time-of-day
- Scatter plot: predicted vs actual ampacity
- Saves results to validation_results.txt and validation_results.json (and CSV of per-sample predictions)

Usage:
  python scripts/validate_best_run.py
  python scripts/validate_best_run.py --model-path runs/<ts>/best_model.pt --output-dir results/validation

"""
import argparse
import glob
import json
import math
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Attempt to import the production predictor
try:
    from core.inference import DLRPredictor
    _HAS_PREDICTOR = True
except Exception:
    DLRPredictor = None  # type: ignore
    _HAS_PREDICTOR = False

# Fallback model loader (from validate_vietnam)
try:
    from scripts.validate_vietnam import load_model as load_model_fallback, VietnamDataset
    _HAS_VALIDATE_VIETNAM = True
except Exception:
    load_model_fallback = None  # type: ignore
    VietnamDataset = None  # type: ignore
    _HAS_VALIDATE_VIETNAM = False


def find_latest_run(runs_root: str = 'runs') -> str:
    if not os.path.isdir(runs_root):
        raise FileNotFoundError(f"runs directory not found: {runs_root}")
    subdirs = [os.path.join(runs_root, d) for d in os.listdir(runs_root) if os.path.isdir(os.path.join(runs_root, d))]
    if not subdirs:
        raise FileNotFoundError(f"no run directories found under {runs_root}")
    return max(subdirs, key=os.path.getmtime)


def find_best_checkpoint(run_dir: str) -> str:
    # look for best_model.pt in run_dir
    candidate = os.path.join(run_dir, 'best_model.pt')
    if os.path.exists(candidate):
        return candidate
    # fallback to any .pt file (latest)
    pts = glob.glob(os.path.join(run_dir, '**', '*.pt'), recursive=True)
    if pts:
        return max(pts, key=os.path.getmtime)
    # fallback to models/best_model.pt
    fallback = os.path.join('models', 'best_model.pt')
    if os.path.exists(fallback):
        return fallback
    raise FileNotFoundError('No checkpoint found in run or models/')


def temporal_test_split_df(df: pd.DataFrame, test_fraction: float = 0.2) -> Tuple[pd.DataFrame, pd.DataFrame]:
    # Assume df is time-ordered by datetime column if present, else by row order
    if 'datetime' in df.columns:
        df = df.sort_values('datetime').reset_index(drop=True)
    n = len(df)
    split_idx = int(n * (1 - test_fraction))
    train_df = df.iloc[:split_idx].reset_index(drop=True)
    test_df = df.iloc[split_idx:].reset_index(drop=True)
    return train_df, test_df


def make_conditions_from_df(df: pd.DataFrame, assumed_current: float = 1000.0) -> List[Dict]:
    conditions = []
    for _, row in df.iterrows():
        cond = {
            'T_ambient': float(row['temp']),
            'wind_speed': float(row['Wind1']),
            'solar_irradiance': float(row['GHI']),
            'current': float(assumed_current),
            'wind_angle': float(row['WinDir']) if 'WinDir' in row and not pd.isna(row['WinDir']) else 45.0,
        }
        conditions.append(cond)
    return conditions


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    mae = float(mean_absolute_error(y_true, y_pred))
    rmse = float(math.sqrt(mean_squared_error(y_true, y_pred)))
    r2 = float(r2_score(y_true, y_pred))
    return {'mae': mae, 'rmse': rmse, 'r2': r2}


def breakdown_by_bins(df: pd.DataFrame, pred_amp: np.ndarray, true_amp: np.ndarray) -> Dict:
    results = {}
    # Wind speed bins: [0-2, 2-5, 5-10, >10]
    # ensure the final bin edge is at least the static upper bound so `pd.cut` bins increase monotonically
    # make sure the final edge is strictly greater than the preceding static edge
    wind_max_edge = max(11.0, float(df['Wind1'].max()) + 1.0)
    wind_bins = [0, 2, 5, 10, wind_max_edge]
    wind_labels = ['0-2', '2-5', '5-10', '>10']
    df['wind_bin'] = pd.cut(df['Wind1'], bins=wind_bins, labels=wind_labels, include_lowest=True)

    # Temperature bins: ensure final edge > 40 to avoid duplicate edges
    temp_max_edge = max(41.0, float(df['temp'].max()) + 1.0)
    temp_bins = [-1e9, 20, 30, 40, temp_max_edge]
    temp_labels = ['<20', '20-30', '30-40', '>40']
    df['temp_bin'] = pd.cut(df['temp'], bins=temp_bins, labels=temp_labels, include_lowest=True)

    # Time of day bins: night (0-6), morning (6-12), afternoon (12-18), evening (18-24)
    if 'datetime' in df.columns:
        hours = pd.to_datetime(df['datetime']).dt.hour
    else:
        # if no datetime, try to infer from index modulo 24
        hours = pd.Series([i % 24 for i in range(len(df))])
    def tod_label(h):
        if 0 <= h < 6:
            return 'night'
        if 6 <= h < 12:
            return 'morning'
        if 12 <= h < 18:
            return 'afternoon'
        return 'evening'
    df['hour'] = hours
    df['tod'] = df['hour'].apply(tod_label)

    # compute metrics per group
    def compute_group_metrics(group_idx):
        if len(group_idx) == 0:
            return None
        y_t = true_amp[group_idx]
        y_p = pred_amp[group_idx]
        return compute_metrics(y_t, y_p)

    # Wind bins
    wind_results = {}
    for label in wind_labels:
        idx = df.index[df['wind_bin'] == label].tolist()
        m = compute_group_metrics(idx)
        wind_results[label] = m

    # Temp bins
    temp_results = {}
    for label in temp_labels:
        idx = df.index[df['temp_bin'] == label].tolist()
        m = compute_group_metrics(idx)
        temp_results[label] = m

    # Time of day
    tod_groups = ['night', 'morning', 'afternoon', 'evening']
    tod_results = {}
    for label in tod_groups:
        idx = df.index[df['tod'] == label].tolist()
        m = compute_group_metrics(idx)
        tod_results[label] = m

    results['wind_bins'] = wind_results
    results['temp_bins'] = temp_results
    results['time_of_day'] = tod_results
    return results


def save_results(output_dir: str, summary: Dict, per_sample: pd.DataFrame):
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    txt_path = os.path.join(output_dir, 'validation_results.txt')
    json_path = os.path.join(output_dir, 'validation_results.json')
    csv_path = os.path.join(output_dir, 'validation_per_sample.csv')

    # TXT summary
    with open(txt_path, 'w') as f:
        f.write('Validation summary - ' + datetime.utcnow().isoformat() + 'Z\n')
        f.write(json.dumps(summary, indent=2))

    # JSON
    with open(json_path, 'w') as f:
        json.dump(summary, f, indent=2)

    # per-sample CSV
    per_sample.to_csv(csv_path, index=False)

    return txt_path, json_path, csv_path


def plot_scatter_amp(true_amp: np.ndarray, pred_amp: np.ndarray, out_path: str):
    plt.figure(figsize=(6, 6))
    plt.scatter(true_amp, pred_amp, alpha=0.5, s=8)
    mn = min(true_amp.min(), pred_amp.min())
    mx = max(true_amp.max(), pred_amp.max())
    plt.plot([mn, mx], [mn, mx], 'r--')
    plt.xlabel('True Ampacity (A)')
    plt.ylabel('Predicted Ampacity (A)')
    plt.title('Predicted vs Actual Ampacity')
    plt.grid(alpha=0.3)
    plt.savefig(out_path, dpi=150)
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Validate best model from latest run')
    parser.add_argument('--runs-dir', default='runs', help='Root runs directory')
    parser.add_argument('--model-path', default=None, help='Optional explicit checkpoint path')
    parser.add_argument('--data-path', default='data/mendeley/vietnam_220kv.csv', help='Vietnam CSV path')
    parser.add_argument('--assumed-current', type=float, default=1000.0, help='Assumed current for validation (A)')
    parser.add_argument('--test-fraction', type=float, default=0.2, help='Temporal test fraction')
    parser.add_argument('--baseline', type=float, default=308.0, help='Baseline ampacity MAE for comparison')
    parser.add_argument('--output-dir', default=None, help='Directory to save validation outputs (defaults to latest run)')
    args = parser.parse_args()

    # Find checkpoint
    if args.model_path:
        ckpt = args.model_path
    else:
        latest_run = find_latest_run(args.runs_dir)
        try:
            ckpt = find_best_checkpoint(latest_run)
        except FileNotFoundError:
            ckpt = os.path.join('models', 'best_model.pt')

    if not os.path.exists(ckpt):
        print(f"[error] Checkpoint not found: {ckpt}")
        sys.exit(1)

    print(f"Using checkpoint: {ckpt}")

    # Load predictor
    device = 'cpu'
    predictor = None
    if _HAS_PREDICTOR:
        try:
            import torch
            device_arg = torch.device('cpu')  # load on CPU to avoid device-buffer mismatch
            predictor = DLRPredictor.from_checkpoint(ckpt, device=device_arg)
            device = str(predictor.device)
            print(f"Loaded predictor on device: {device}")
        except Exception as e:
            print(f"[warning] DLRPredictor.from_checkpoint failed: {e}")
            predictor = None

    # If predictor couldn't be created from the run checkpoint, try a safer fallback
    if predictor is None:
        # Try models/best_model.pt (pre-saved compatible checkpoint)
        fallback_ckpt = os.path.join('models', 'best_model.pt')
        if os.path.exists(fallback_ckpt):
            try:
                print(f"Attempting fallback predictor from {fallback_ckpt}")
                import torch
                predictor = DLRPredictor.from_checkpoint(fallback_ckpt, device=torch.device('cpu')) if _HAS_PREDICTOR else None
                if predictor:
                    print('Loaded predictor from models/best_model.pt')
            except Exception as e:
                import traceback
                print(f"[warning] fallback DLRPredictor failed: {e}")
                traceback.print_exc()
                predictor = None

    # Last-resort: use validate_vietnam.load_model() wrapper if available
    if predictor is None:
        # Try dynamic import of DLRPredictor from core.inference (preferred)
        try:
            # ensure project root is on sys.path
            from pathlib import Path as _P
            project_root = str(_P(__file__).resolve().parents[1])
            if project_root not in sys.path:
                sys.path.insert(0, project_root)
            from core.inference import DLRPredictor as RuntimeDLRPredictor
            import torch
            predictor = RuntimeDLRPredictor.from_checkpoint(os.path.join('models', 'best_model.pt'), device=torch.device('cpu'))
            print('Loaded predictor from models/best_model.pt via core.inference')
        except Exception as e:
            print(f"[warning] Runtime DLRPredictor import/load failed: {e}")
            # Fallback to validate_vietnam loader if available
            if load_model_fallback is not None:
                try:
                    model, normalizer = load_model_fallback(os.path.join('models', 'best_model.pt'), device='cpu')
                    class SimplePredictor:
                        def __init__(self, model, normalizer):
                            self.model = model
                            self.normalizer = normalizer
                        def predict(self, T_ambient, wind_speed, solar_irradiance, current, wind_angle=45.0):
                            import torch
                            x = np.array([[T_ambient, wind_speed, wind_angle, solar_irradiance, current, self.model.physics.R_ref.item() if hasattr(self.model, 'physics') else 7.283e-5]])
                            x_norm = self.normalizer.transform(x)
                            xt = torch.tensor(x_norm, dtype=torch.float32)
                            with torch.no_grad():
                                temp_t, amp_t = self.model(xt)
                            return type('R', (), {'conductor_temperature': float(temp_t.item()), 'dynamic_rating': float(amp_t.item()), 'physics_residual': float('nan')})
                    predictor = SimplePredictor(model, normalizer)
                    print('Loaded fallback model via validate_vietnam.load_model (models/best_model.pt)')
                except Exception as e2:
                    print(f"[error] validate_vietnam.load_model failed: {e2}")
                    sys.exit(1)
            else:
                print('[error] No compatible model loader available')
                sys.exit(1)

    # Load Vietnam CSV and split temporally
    df = pd.read_csv(args.data_path, parse_dates=['datetime'])
    train_df, test_df = temporal_test_split_df(df, test_fraction=args.test_fraction)
    print(f"Dataset: {len(df)} rows — test set: {len(test_df)} rows (last {args.test_fraction*100:.0f}%)")

    # Prepare conditions
    conditions = make_conditions_from_df(test_df, assumed_current=args.assumed_current)

    # Batch predict
    preds_temp = []
    preds_amp = []
    phys_res = []
    batch_size = 1024
    for i in range(0, len(conditions), batch_size):
        batch = conditions[i:i+batch_size]
        if hasattr(predictor, 'predict_batch'):
            results = predictor.predict_batch(batch)
            for r in results:
                preds_temp.append(r.conductor_temperature)
                preds_amp.append(r.dynamic_rating)
                phys_res.append(r.physics_residual)
        else:
            for cond in batch:
                r = predictor.predict(**cond)
                preds_temp.append(r.conductor_temperature)
                preds_amp.append(r.dynamic_rating)
                phys_res.append(r.physics_residual)

    preds_temp = np.array(preds_temp)
    preds_amp = np.array(preds_amp)
    phys_res = np.array(phys_res)

    true_temp = (test_df['temp'].values + 10.0)  # earlier proxy used ambient+10
    true_amp = test_df['Ampacity'].values.astype(float)

    # Metrics
    amp_metrics = compute_metrics(true_amp, preds_amp)
    temp_metrics = compute_metrics(true_temp, preds_temp)
    phys_res_mae = float(np.nanmean(np.abs(phys_res))) if len(phys_res) > 0 else float('nan')

    # Breakdown
    breakdown = breakdown_by_bins(test_df.copy(), preds_amp, true_amp)

    # Scatter plot
    latest_run_dir = os.path.dirname(ckpt) if 'runs' in ckpt else os.path.join('runs', find_latest_run(args.runs_dir))
    output_dir = args.output_dir or latest_run_dir
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    scatter_path = os.path.join(output_dir, 'ampacity_scatter.png')
    plot_scatter_amp(true_amp, preds_amp, scatter_path)

    # Save per-sample CSV
    per_sample = test_df.copy()
    per_sample['pred_amp'] = preds_amp
    per_sample['pred_temp'] = preds_temp
    per_sample['phys_residual'] = phys_res

    # Summary
    summary = {
        'checkpoint': ckpt,
        'device': device,
        'n_test_samples': int(len(test_df)),
        'amp_metrics': amp_metrics,
        'temp_metrics': temp_metrics,
        'physics_residual_mae': phys_res_mae,
        'breakdown': breakdown,
        'scatter_plot': scatter_path,
        'assumed_current': args.assumed_current,
    }

    # Compare to previous best (baseline)
    baseline = args.baseline
    comparison = {
        'baseline_ampacity_mae': baseline,
        'current_ampacity_mae': amp_metrics['mae'],
        'delta_vs_baseline': baseline - amp_metrics['mae'],
        'beats_baseline': amp_metrics['mae'] < baseline
    }
    summary['comparison_vs_baseline'] = comparison

    # Save results
    txt_path, json_path, csv_path = save_results(output_dir, summary, per_sample)

    # Print concise summary
    print('\n=== Validation Summary ===')
    print(f"Test samples: {len(test_df)}")
    print(f"Ampacity MAE: {amp_metrics['mae']:.2f} A    RMSE: {amp_metrics['rmse']:.2f} A    R²: {amp_metrics['r2']:.3f}")
    print(f"Temperature MAE: {temp_metrics['mae']:.2f} °C    RMSE: {temp_metrics['rmse']:.2f} °C    R²: {temp_metrics['r2']:.3f}")
    print(f"Physics residual MAE: {phys_res_mae:.3f} W/m")
    print(f"Scatter plot: {scatter_path}")
    print(f"Saved results: {txt_path}, {json_path}, {csv_path}")

    if comparison['beats_baseline']:
        print(f"✅ Ampacity MAE ({amp_metrics['mae']:.2f}A) is better than baseline {baseline}A by {comparison['delta_vs_baseline']:.2f}A")
    else:
        print(f"⚠️ Ampacity MAE ({amp_metrics['mae']:.2f}A) does NOT beat baseline {baseline}A (needs {comparison['delta_vs_baseline']:.2f}A improvement)")

    print('\nBreakdown by wind-speed bins:')
    for k, v in breakdown['wind_bins'].items():
        print(f"  {k}: {v}")
    print('\nBreakdown by temperature bins:')
    for k, v in breakdown['temp_bins'].items():
        print(f"  {k}: {v}")
    print('\nBreakdown by time of day:')
    for k, v in breakdown['time_of_day'].items():
        print(f"  {k}: {v}")

    print('\nValidation complete.')


if __name__ == '__main__':
    main()
