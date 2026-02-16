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
"""Evaluate a saved model on the held-out test split created by
`scripts/train_invariant_pikan_production.py --save-test`.

Usage:
  python scripts/evaluate_test_split.py --run-dir runs/invariant_pikan_production_YYYYMMDD_HHMMSS

The script expects `test_indices.pt` to exist in the run directory (saved by training when
`--save-test` was used). For unified/HDF5 runs the training also saves `temp_unified_data.csv`
into the same run dir; when present this file is used to provide `region`/metadata for
per-region breakdowns.
"""
import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Subset

# ensure project package imports work when running as a script
sys.path.append(str(Path(__file__).resolve().parents[1]))

from models.invariant_pikan_v2 import create_invariant_pikan_v2
from core.data import VietnamDataset, USDataset


def compute_metrics_np(targets: np.ndarray, preds: np.ndarray):
    err = preds - targets
    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(err ** 2)))
    bias = float(np.mean(err))
    return {'mae': mae, 'rmse': rmse, 'bias': bias}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--run-dir', required=True, help='Path to training run directory')
    p.add_argument('--batch-size', type=int, default=64)
    p.add_argument('--device', default='auto')
    args = p.parse_args()

    run_dir = Path(args.run_dir)
    assert run_dir.exists(), f"Run dir not found: {run_dir}"

    # load config.json if present to reconstruct dataset path / device
    cfg_path = run_dir / 'config.json'
    config = {}
    if cfg_path.exists():
        config = json.loads(cfg_path.read_text())

    # resolve device
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu'))
    else:
        device = torch.device(args.device)

    # load test indices
    idx_path = run_dir / 'test_indices.pt'
    if not idx_path.exists():
        raise FileNotFoundError(f"test_indices.pt not found in {run_dir}. Run training with --save-test to produce it.")
    test_idx = torch.load(idx_path)
    if isinstance(test_idx, (list, tuple)):
        test_idx = list(test_idx)

    # Reconstruct dataset used for training
    # Priority: use run_dir/temp_unified_data.csv if present, else use config['data_path'] or config['us_data_path']
    dataset = None
    temp_csv = run_dir / 'temp_unified_data.csv'
    if temp_csv.exists():
        dataset = VietnamDataset(str(temp_csv))
        raw_df_for_meta = pd.read_csv(temp_csv)
    else:
        data_path = config.get('data_path')
        us_data = config.get('us_data_path')
        if us_data:
            dataset = USDataset(us_data, normalizer=None)
            raw_df_for_meta = None
        elif data_path:
            # use VietnamDataset for CSV path
            dataset = VietnamDataset(data_path)
            raw_df_for_meta = None
        else:
            raise RuntimeError('Could not determine dataset path from run config or run_dir')

    test_dataset = Subset(dataset, test_idx)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)

    # Load model checkpoint (prefer best_model.pt, fallback to final_model.pt)
    ckpt_path = run_dir / 'best_model.pt'
    if not ckpt_path.exists():
        ckpt_path = run_dir / 'final_model.pt'
    assert ckpt_path.exists(), f"No checkpoint found in {run_dir} (expected best_model.pt or final_model.pt)"

    ckpt = torch.load(ckpt_path, map_location='cpu')
    # attempt to recover model config saved with checkpoint
    model_cfg = None
    if isinstance(ckpt, dict) and 'config' in ckpt and isinstance(ckpt['config'], dict):
        model_cfg = ckpt['config'].get('model', None)

    model = create_invariant_pikan_v2(config=model_cfg) if model_cfg is not None else create_invariant_pikan_v2({})
    model.load_state_dict(ckpt['model_state_dict'])
    model.to(device)
    model.eval()

    all_preds = []
    all_targets = []
    all_meta_regions = []

    with torch.no_grad():
        for x, y in test_loader:
            # x: normalized inputs (VietnamDataset.__getitem__ preserves same ordering used during training)
            weather = x[:, :4].to(device)
            current = x[:, 4:5].to(device)
            weather_dict = {'T_amb': weather[:, 0], 'wind_speed': weather[:, 1], 'solar': weather[:, 3]}

            out = model(weather, weather_dict)
            preds = out['ampacity'].detach().cpu().numpy()
            targets = y[:, 1].cpu().numpy()

            all_preds.append(preds)
            all_targets.append(targets)

    preds = np.concatenate(all_preds, axis=0)
    targets = np.concatenate(all_targets, axis=0)

    overall = compute_metrics_np(targets, preds)
    report = {'samples': int(len(preds)), **overall}

    # If test_data.csv / temp_unified_data.csv present, compute per-region breakdown
    region_report = None
    if temp_csv.exists() and 'region' in raw_df_for_meta.columns:
        region_series = raw_df_for_meta.loc[test_idx, 'region'].fillna('UNKNOWN').astype(str).values
        df_meta = pd.DataFrame({'region': region_series, 'target': targets, 'pred': preds})
        by_region = df_meta.groupby('region').apply(lambda g: pd.Series(compute_metrics_np(g['target'].values, g['pred'].values))).reset_index()
        by_region.to_csv(run_dir / 'test_eval_by_region.csv', index=False)
        region_report = by_region.to_dict(orient='records')
        report['by_region'] = region_report

    # Save JSON report
    out_json = run_dir / 'test_eval.json'
    out = {'run_dir': str(run_dir), 'checkpoint': str(ckpt_path.name), 'report': report}
    out_json.write_text(json.dumps(out, indent=2))

    # Print a concise summary
    print(f"Test samples: {len(preds)}")
    print(f"MAE: {overall['mae']:.3f} A  RMSE: {overall['rmse']:.3f} A  Bias: {overall['bias']:.3f} A")
    if region_report is not None:
        print(f"Per-region breakdown saved to {run_dir / 'test_eval_by_region.csv'}")

    print(f"Saved test evaluation -> {out_json}")


if __name__ == '__main__':
    main()
