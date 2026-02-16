#!/usr/bin/env python3
"""
training_watchdog.py

Monitors the latest HWF-PIKAN v2 training run for common issues and alerts
when red flags are detected.

Checks (best-effort, relies on history.csv or TensorBoard events):
 - Loss increasing for N consecutive epochs (default 5)
 - Gradient norms near zero (< 1e-6) [if logged]
 - Physics weight stuck at extremes (<0.1 or >0.9)
 - No improvement in ampacity MAE for M epochs (default 20)
 - NaN values in any metric

Actions on detection:
 - Print warning to console
 - Append an entry to `training_warnings.log` in the run directory
 - Send macOS desktop notification (osascript)
 - Suggest remedial actions

Usage:
    python -m scripts.training_watchdog         # runs continuously (10-min checks)
    python -m scripts.training_watchdog --once  # run single check and exit

"""
import argparse
import glob
import json
import math
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# TensorBoard EventAccumulator (optional)
try:
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    _HAS_EA = True
except Exception:
    EventAccumulator = None  # type: ignore
    _HAS_EA = False


def find_latest_run(runs_root: str = "runs") -> str:
    if not os.path.isdir(runs_root):
        raise FileNotFoundError(f"runs directory not found: {runs_root}")
    subdirs = [os.path.join(runs_root, d) for d in os.listdir(runs_root) if os.path.isdir(os.path.join(runs_root, d))]
    if not subdirs:
        raise FileNotFoundError(f"no run directories found under {runs_root}")
    return max(subdirs, key=os.path.getmtime)


def read_history(run_dir: str) -> Dict[str, List[float]]:
    """Return history-like dict. Prefer history.csv, fallback to TB events.

    Keys normalized to: train_loss, val_amp_mae, weights_physics, grad_norm (if available), epochs
    """
    history: Dict[str, List[float]] = {}

    # 1) history.csv if present
    hist_csv = os.path.join(run_dir, "history.csv")
    if os.path.exists(hist_csv):
        try:
            df = pd.read_csv(hist_csv)
            # normalize common keys
            if 'epoch' in df.columns:
                history['epoch'] = df['epoch'].tolist()
            # try several possible column names
            for col_map in [
                ('train_loss', ['train_loss', 'Loss/train']),
                ('val_amp_mae', ['val_amp_mae', 'val_amp_mae', 'val_amp', 'val_ampicity_mae']),
                ('weights_physics', ['weights_physics', 'Weights/physics']),
                ('grad_norm', ['grad_norm', 'gradient_norm']),
                ('val_temp_mae', ['val_temp_mae', 'val_temp_mae'])
            ]:
                out_key, candidates = col_map[0], col_map[1]
                for c in candidates:
                    if c in df.columns:
                        history[out_key] = df[c].astype(float).tolist()
                        break
            # also expose any column that looks like amp mae or physics
            if 'val_amp_mae' not in history:
                for c in df.columns:
                    if 'amp' in c and 'mae' in c:
                        history['val_amp_mae'] = df[c].astype(float).tolist()
                        break
            return history
        except Exception:
            pass

    # 2) TensorBoard event files
    if _HAS_EA:
        ev_files = glob.glob(os.path.join(run_dir, '**', 'events.out.tfevents.*'), recursive=True)
        if not ev_files:
            return history
        ev = max(ev_files, key=os.path.getmtime)
        try:
            ea = EventAccumulator(ev, size_guidance={'scalars': 0})
            ea.Reload()
            tags = ea.Tags().get('scalars', [])
            # helper to read a scalar tag -> dict(step->value)
            def read_tag(tag: str) -> Dict[int, float]:
                if tag not in tags:
                    return {}
                vals = ea.Scalars(tag)
                return {int(v.step): float(v.value) for v in vals}

            tag_map = {
                'train_loss': ['Loss/train'],
                'val_amp_mae': ['Metrics/amp_mae', 'val_amp_mae'],
                'weights_physics': ['Weights/physics'],
                'grad_norm': ['Gradients/norm', 'Gradients/total_norm', 'Grad/norm']
            }
            series = {}
            steps = set()
            for k, tag_candidates in tag_map.items():
                for t in tag_candidates:
                    d = read_tag(t)
                    if d:
                        series[k] = d
                        steps.update(d.keys())
                        break
            steps_sorted = sorted(steps)
            for k, d in series.items():
                history[k] = [d.get(s, float('nan')) for s in steps_sorted]
            if steps_sorted:
                history['epoch'] = steps_sorted
        except Exception:
            pass

    return history


def vals_last(history: Dict[str, List[float]], key: str, n: int = 5) -> List[float]:
    arr = history.get(key, [])
    if not arr:
        return []
    return [float(x) for x in arr[-n:]]


def check_loss_increasing(history: Dict[str, List[float]], window: int = 5) -> Optional[Dict[str, Any]]:
    vals = vals_last(history, 'train_loss', n=window)
    if len(vals) < window:
        return None
    # check strictly increasing sequence
    increasing = all(vals[i] < vals[i+1] for i in range(len(vals)-1))
    if increasing:
        return {'type': 'loss_increasing', 'window': window, 'values': vals}
    return None


def check_gradients_vanishing(history: Dict[str, List[float]], threshold: float = 1e-6) -> Optional[Dict[str, Any]]:
    vals = vals_last(history, 'grad_norm', n=3)
    if not vals:
        return None  # grad norms not available / not logged
    if all(abs(v) < threshold for v in vals):
        return {'type': 'grad_vanishing', 'threshold': threshold, 'recent': vals}
    return None


def check_physics_weight_extreme(history: Dict[str, List[float]], window: int = 5) -> Optional[Dict[str, Any]]:
    vals = vals_last(history, 'weights_physics', n=window)
    if len(vals) < window:
        return None
    if all(v < 0.1 for v in vals):
        return {'type': 'physics_weight_low', 'window': window, 'recent': vals}
    if all(v > 0.9 for v in vals):
        return {'type': 'physics_weight_high', 'window': window, 'recent': vals}
    return None


def check_no_improvement_amp(history: Dict[str, List[float]], patience: int = 20) -> Optional[Dict[str, Any]]:
    arr = history.get('val_amp_mae', [])
    if len(arr) < patience + 1:
        return None
    # find index of last improvement (new global minimum)
    best_idx = None
    best_val = float('inf')
    for i, v in enumerate(arr):
        if not math.isnan(v) and v < best_val - 1e-9:
            best_val = v
            best_idx = i
    if best_idx is None:
        return None
    if (len(arr) - 1 - best_idx) >= patience:
        return {'type': 'no_improvement_amp', 'patience': patience, 'best_epoch': best_idx, 'best_value': best_val, 'current_epoch': len(arr)-1}
    return None


def check_nan_metrics(history: Dict[str, List[float]]) -> Optional[Dict[str, Any]]:
    for k, v in history.items():
        if k == 'epoch':
            continue
        for i, x in enumerate(v):
            if x is None:
                return {'type': 'nan_metric', 'metric': k, 'index': i}
            try:
                if math.isnan(float(x)):
                    return {'type': 'nan_metric', 'metric': k, 'index': i}
            except Exception:
                continue
    return None


def notify_macos(title: str, message: str):
    try:
        esc_msg = message.replace('"', '\\"')
        esc_title = title.replace('"', '\\"')
        cmd = ['osascript', '-e', f'display notification "{esc_msg}" with title "{esc_title}"']
        subprocess.run(cmd, check=True)
    except Exception as e:
        print(f"[warning] macOS notification failed: {e}")


SUGGESTIONS = {
    'loss_increasing': 'Reduce learning rate',
    'grad_vanishing': 'Gradients vanishing: check for dead neurons / ReLU saturation / try different initialization or optimizer',
    'physics_weight_low': 'Physics weight very small: consider increasing lambda_physics or check physics loss scale',
    'physics_weight_high': 'Physics weight very large: reduce lambda_physics or inspect physics target scaling',
    'no_improvement_amp': 'No improvement: consider early stopping, adjust learning rate or regularization',
    'nan_metric': 'NaN encountered: check for exploding gradients, numerical stability, clamp inputs'
}


def log_warning(run_dir: str, entry: Dict[str, Any], log_name: str = 'training_warnings.log'):
    path = os.path.join(run_dir, log_name)
    ts = datetime.utcnow().isoformat() + 'Z'
    line = {'timestamp': ts, 'warning': entry}
    try:
        with open(path, 'a') as f:
            f.write(json.dumps(line) + '\n')
    except Exception as e:
        print(f"[error] failed to write warning log: {e}")


def format_entry(entry: Dict[str, Any]) -> str:
    t = entry.get('type', 'unknown')
    suggestion = SUGGESTIONS.get(t, '')
    return f"[{t}] details={entry} -- suggestion: {suggestion}"


def single_check(runs_dir: str = 'runs') -> List[Dict[str, Any]]:
    try:
        latest = find_latest_run(runs_dir)
    except Exception as e:
        print(f"[error] cannot find latest run: {e}")
        return []

    history = read_history(latest)
    issues: List[Dict[str, Any]] = []

    # perform checks
    checks = [
        check_loss_increasing(history, window=5),
        check_gradients_vanishing(history, threshold=1e-6),
        check_physics_weight_extreme(history, window=5),
        check_no_improvement_amp(history, patience=20),
        check_nan_metrics(history)
    ]

    for c in checks:
        if c is not None:
            issues.append(c)
            # print + log + notify
            msg = format_entry(c)
            print(f"[warning] {msg}")
            log_warning(latest, c)
            notify_macos('Training Watchdog Alert', msg)

    if not issues:
        print(f"[{datetime.utcnow().isoformat()}] No issues detected in latest run: {latest}")
    return issues


def main():
    parser = argparse.ArgumentParser(description='Training watchdog for HWF-PIKAN v2')
    parser.add_argument('--runs-dir', default='runs')
    parser.add_argument('--interval', type=int, default=600, help='Check interval in seconds (default 600s)')
    parser.add_argument('--once', action='store_true', help='Run single check and exit')
    args = parser.parse_args()

    if args.once:
        single_check(args.runs_dir)
        return

    print(f"Starting training watchdog (check every {args.interval}s). Press Ctrl-C to stop.")
    try:
        while True:
            single_check(args.runs_dir)
            time.sleep(args.interval)
    except KeyboardInterrupt:
        print('\nWatchdog stopped by user')


if __name__ == '__main__':
    main()
