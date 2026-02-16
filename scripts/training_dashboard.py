"""
Invariant-PIKAN: Adversarially-Robust Physics-Informed Neural Networks for Dynamic Line Rating
Copyright (C) 2025 Gelavizh Ahmadi / Invariant Research

This software is licensed under the Business Source License 1.1 (BSL 1.1).
Commercial production use requires a separate license agreement.
See LICENSE.txt for full terms.

"""

#!/usr/bin/env python3
"""
training_dashboard.py

Real-time training dashboard for InvariantPIKAN v2.
- Reads latest checkpoint or history.csv from `runs/`
- Plots 4 subplots (train loss, val amp MAE, physics weight, temp MAE)
- Adds annotations (best MAE, improvement vs baseline, ETA)
- Saves `training_progress.png` in the latest run folder and project root
- Can run in --loop mode to refresh every N seconds (default 300s)

Usage:
  python scripts/training_dashboard.py         # one-shot
  python scripts/training_dashboard.py --loop  # refresh every 5 minutes

"""
import argparse
import glob
import math
import os
import sys
import time
from datetime import datetime, timedelta

# plotting
import matplotlib.pyplot as plt
import numpy as np

# try optional deps
try:
    import torch
    _HAS_TORCH = True
except Exception:
    torch = None  # type: ignore
    _HAS_TORCH = False

try:
    import pandas as pd
    _HAS_PANDAS = True
except Exception:
    pd = None  # type: ignore
    _HAS_PANDAS = False

try:
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    _HAS_EVENT_ACC = True
except Exception:
    EventAccumulator = None  # type: ignore
    _HAS_EVENT_ACC = False


def find_latest_run(runs_root: str = 'runs'):
    if not os.path.isdir(runs_root):
        raise FileNotFoundError(f"Runs directory not found: {runs_root}")
    subdirs = [os.path.join(runs_root, d) for d in os.listdir(runs_root) if os.path.isdir(os.path.join(runs_root, d))]
    if not subdirs:
        raise FileNotFoundError(f"No run directories found under {runs_root}")
    return max(subdirs, key=os.path.getmtime)


def find_latest_checkpoint(runs_root: str = 'runs'):
    # search for .pt files under runs_root
    pts = glob.glob(os.path.join(runs_root, '**', '*.pt'), recursive=True)
    if not pts:
        return None
    return max(pts, key=os.path.getmtime)


def load_history_from_checkpoint(ckpt_path: str):
    if not _HAS_TORCH:
        return None
    try:
        ckpt = torch.load(ckpt_path, map_location='cpu')
        history = ckpt.get('history')
        config = ckpt.get('config')
        val_metrics = ckpt.get('val_metrics') or ckpt.get('val_metric')
        return {'history': history, 'config': config, 'val_metrics': val_metrics, 'ckpt': ckpt}
    except Exception:
        return None


def read_history_csv(run_dir: str):
    path = os.path.join(run_dir, 'history.csv')
    if not os.path.exists(path):
        return None
    try:
        if _HAS_PANDAS:
            df = pd.read_csv(path)
            return df
        else:
            import csv
            rows = []
            with open(path, 'r') as f:
                r = csv.DictReader(f)
                for row in r:
                    rows.append(row)
            return rows
    except Exception:
        return None


def read_scalar_from_event(event_file: str, tag: str, n: int = 10000):
    """Return list of (step, value, wall_time) for the given tag using EventAccumulator."""
    if not _HAS_EVENT_ACC:
        return []
    try:
        ea = EventAccumulator(event_file, size_guidance={
            'scalars': 0,
        })
        ea.Reload()
        if tag not in ea.Tags().get('scalars', []):
            return []
        vals = ea.Scalars(tag)
        return [(int(v.step), float(v.value), float(v.wall_time)) for v in vals]
    except Exception:
        return []


def seconds_per_epoch_from_event(event_file: str, tag: str = 'Loss/train'):
    series = read_scalar_from_event(event_file, tag, n=1000)
    if len(series) < 2:
        return None
    steps = np.array([s[0] for s in series])
    times = np.array([s[2] for s in series])
    # use first and last to estimate seconds per step
    dsteps = steps[-1] - steps[0]
    dtime = times[-1] - times[0]
    if dsteps <= 0 or dtime <= 0:
        return None
    return dtime / float(dsteps)


def fmt_seconds(s):
    if s is None:
        return 'unknown'
    return str(timedelta(seconds=int(round(s))))


def safe_get_list(history_obj, key_names):
    """Return first found list-like value for any key in key_names from history dict or DataFrame."""
    if history_obj is None:
        return None
    # if pandas DataFrame
    if _HAS_PANDAS and isinstance(history_obj, pd.DataFrame):
        for k in key_names:
            if k in history_obj.columns:
                return list(history_obj[k].values)
        return None
    # assume dict-like
    if isinstance(history_obj, dict):
        for k in key_names:
            if k in history_obj and history_obj[k] is not None:
                return history_obj[k]
    return None


def plot_dashboard(run_dir: str, baseline: float = 308.0, save_copy_to_cwd: bool = True):
    # Load checkpoint/history
    ckpt_path = find_latest_checkpoint(run_dir)
    history_obj = None
    config = None
    if ckpt_path and _HAS_TORCH:
        ckpt_info = load_history_from_checkpoint(ckpt_path)
        if ckpt_info and ckpt_info.get('history'):
            history_obj = ckpt_info['history']
            config = ckpt_info.get('config')
    # fallback to history.csv
    if history_obj is None:
        hist_csv = read_history_csv(run_dir)
        if hist_csv is not None:
            history_obj = hist_csv

    # try to find event file
    event_files = glob.glob(os.path.join(run_dir, '**', 'events.out.tfevents.*'), recursive=True)
    event_file = event_files[-1] if event_files else None

    # extract series
    epochs = safe_get_list(history_obj, ['epoch', 'epochs']) or []
    train_loss = safe_get_list(history_obj, ['train_loss', 'loss', 'train_loss']) or []
    val_amp = safe_get_list(history_obj, ['val_amp_mae', 'val_amp', 'amp_mae']) or []
    val_temp = safe_get_list(history_obj, ['val_temp_mae', 'val_temp', 'temp_mae']) or []

    # physics weight: try history keys then event file
    phys_weight = safe_get_list(history_obj, ['physics_weight', 'physics_weights', 'train_physics_weight'])
    # history might have 'weights' as list-of-lists
    if phys_weight is None and isinstance(history_obj, dict) and 'weights' in history_obj:
        w = history_obj['weights']
        # w could be list-of-lists or list of 3 floats per epoch
        try:
            phys_weight = [ww[2] for ww in w]
        except Exception:
            phys_weight = None

    if phys_weight is None and event_file:
        ev = read_scalar_from_event(event_file, 'Weights/physics', n=10000)
        phys_weight = [v for (_, v, _) in ev] if ev else None

    # fallback: try to infer epoch indices from event steps
    if (not epochs or len(epochs) == 0) and event_file:
        ev_loss = read_scalar_from_event(event_file, 'Loss/train', n=1000)
        epochs = [s for (s, _, _) in ev_loss]
        train_loss = [v for (_, v, _) in ev_loss]

    # determine plotting x-axis
    x = epochs if epochs else list(range(1, len(train_loss) + 1))

    # compute current best amp MAE
    best_amp = None
    if val_amp:
        try:
            best_amp = float(min(val_amp))
        except Exception:
            best_amp = float(val_amp[-1])
    elif ckpt_path and _HAS_TORCH:
        # try to read best value from checkpoint
        try:
            ckpt = torch.load(ckpt_path, map_location='cpu')
            vm = ckpt.get('val_metrics') or ckpt.get('val_metric')
            if isinstance(vm, dict):
                for k in ('amp_mae', 'val_amp_mae', 'val_amp'):
                    if k in vm:
                        best_amp = float(vm[k])
                        break
        except Exception:
            pass

    current_epoch = x[-1] if x else (len(train_loss) or None)

    # time estimation
    total_epochs = None
    if ckpt_path and _HAS_TORCH:
        try:
            ckpt = torch.load(ckpt_path, map_location='cpu')
            cfg = ckpt.get('config')
            if isinstance(cfg, dict) and 'epochs' in cfg:
                total_epochs = int(cfg['epochs'])
        except Exception:
            total_epochs = None
    # fallback: check history_obj for a 'epochs' key
    if total_epochs is None and isinstance(history_obj, dict):
        total_epochs = history_obj.get('total_epochs') or history_obj.get('epochs')

    remaining_epochs = None
    if total_epochs and current_epoch:
        remaining_epochs = max(0, int(total_epochs) - int(current_epoch))

    sec_per_epoch = None
    if event_file:
        sec_per_epoch = seconds_per_epoch_from_event(event_file, tag='Loss/train')

    eta_seconds = None
    if remaining_epochs is not None and sec_per_epoch is not None:
        eta_seconds = remaining_epochs * sec_per_epoch

    # Start plotting
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    plt.suptitle('InvariantPIKAN v2 — Training Progress', fontsize=16)

    # Top-left: Training loss
    ax = axes[0, 0]
    if train_loss and x:
        ax.plot(x[:len(train_loss)], train_loss, 'b-', label='train_loss')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Train loss')
        ax.grid(alpha=0.3)
    else:
        ax.text(0.5, 0.5, 'Train loss not available', ha='center', va='center')
        ax.set_axis_off()

    ax.set_title('Training loss')

    # Top-right: Validation ampacity MAE with baseline
    ax = axes[0, 1]
    if val_amp and x:
        ax.plot(x[:len(val_amp)], val_amp, 'r-o', label='val_amp_mae')
        ax.axhline(baseline, color='k', linestyle='--', label=f'baseline {baseline}A')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Amp MAE (A)')
        ax.grid(alpha=0.3)
        ax.legend()
    else:
        ax.text(0.5, 0.5, 'Validation ampacity MAE not available', ha='center', va='center')
        ax.set_axis_off()
    ax.set_title('Validation Ampacity MAE')

    # Bottom-left: Physics weight evolution
    ax = axes[1, 0]
    if phys_weight and x:
        ax.plot(x[:len(phys_weight)], phys_weight, 'g-', marker='s')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Physics weight')
        ax.grid(alpha=0.3)
    else:
        ax.text(0.5, 0.5, 'Physics weight not available', ha='center', va='center')
        ax.set_axis_off()
    ax.set_title('Physics weight evolution')

    # Bottom-right: Temperature MAE
    ax = axes[1, 1]
    if val_temp and x:
        ax.plot(x[:len(val_temp)], val_temp, 'm-', label='val_temp_mae')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Temp MAE (°C)')
        ax.grid(alpha=0.3)
    else:
        ax.text(0.5, 0.5, 'Temperature MAE not available', ha='center', va='center')
        ax.set_axis_off()
    ax.set_title('Validation Temperature MAE')

    # Annotations (use top-right area)
    ann_x = 0.01
    ann_y = 0.99
    lines = []
    if best_amp is not None:
        lines.append(f"Current best amp MAE: {best_amp:.2f} A")
        improvement = baseline - best_amp
        lines.append(f"Improvement vs baseline: {improvement:.2f} A ({(improvement / baseline * 100) if baseline else 0:.2f}% )")
    else:
        lines.append('Current best amp MAE: N/A')
        lines.append('Improvement vs baseline: N/A')

    if eta_seconds is not None:
        lines.append(f"ETA to completion: {fmt_seconds(eta_seconds)}")
    elif remaining_epochs is not None:
        lines.append(f"Remaining epochs: {remaining_epochs} (time/epoch unknown)")
    else:
        lines.append('ETA to completion: unknown')

    # draw annotation box on figure
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
    fig.text(0.02, 0.02, '\n'.join(lines), fontsize=10, bbox=props)

    # Save figure
    out_path = os.path.join(run_dir, 'training_progress.png')
    try:
        fig.tight_layout(rect=[0, 0.03, 1, 0.95])
        fig.savefig(out_path, dpi=150)
        if save_copy_to_cwd:
            try:
                fig.savefig(os.path.join(os.getcwd(), 'training_progress.png'), dpi=150)
            except Exception:
                pass
    except Exception as e:
        print(f"[warning] failed to save figure: {e}")

    # Display
    try:
        plt.show(block=False)
        plt.pause(0.1)
    except Exception:
        pass

    print(f"Saved dashboard to: {out_path}")


def main():
    parser = argparse.ArgumentParser(description='Real-time training dashboard for InvariantPIKAN v2')
    parser.add_argument('--runs-dir', default='runs', help='Root runs directory')
    parser.add_argument('--baseline', type=float, default=308.0, help='Ampacity MAE baseline (A)')
    parser.add_argument('--interval', type=int, default=300, help='Refresh interval in seconds when looping')
    parser.add_argument('--loop', action='store_true', help='Run in continuous refresh mode')
    args = parser.parse_args()

    try:
        latest = find_latest_run(args.runs_dir)
    except Exception as e:
        print(f"[error] {e}")
        sys.exit(1)

    if not args.loop:
        plot_dashboard(latest, baseline=args.baseline)
        return 0

    print(f"Entering loop mode — refreshing every {args.interval}s. Ctrl-C to exit.")
    try:
        while True:
            plot_dashboard(latest, baseline=args.baseline)
            time.sleep(args.interval)
    except KeyboardInterrupt:
        print('\nExiting loop.')
        return 0


if __name__ == '__main__':
    main()
