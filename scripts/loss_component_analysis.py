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
loss_component_analysis.py

- Loads training history from the latest checkpoint (or history.csv)
- Plots train_temp_loss, train_amp_loss, train_physics_loss on a log scale
- Calculates pairwise ratios and dominance (ratio > threshold)
- Plots adaptive loss-balancer weights (if available via history or TensorBoard)
- Prints recommendations based on dominance
- Saves figure to `loss_component_analysis.png` in latest run and CWD

Usage:
  python scripts/loss_component_analysis.py
  python scripts/loss_component_analysis.py --show --threshold 10

"""
import argparse
import glob
import math
import os
import sys
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np

# optional deps
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


def find_latest_run(runs_root: str = 'runs') -> str:
    if not os.path.isdir(runs_root):
        raise FileNotFoundError(f"runs directory not found: {runs_root}")
    subdirs = [os.path.join(runs_root, d) for d in os.listdir(runs_root) if os.path.isdir(os.path.join(runs_root, d))]
    if not subdirs:
        raise FileNotFoundError(f"no run subdirectories found under {runs_root}")
    return max(subdirs, key=os.path.getmtime)


def find_latest_checkpoint(run_dir: str) -> Optional[str]:
    pts = glob.glob(os.path.join(run_dir, '**', '*.pt'), recursive=True)
    if not pts:
        return None
    return max(pts, key=os.path.getmtime)


def load_history_from_checkpoint(ckpt_path: str) -> Optional[Dict]:
    if not _HAS_TORCH:
        return None
    try:
        data = torch.load(ckpt_path, map_location='cpu')
        hist = data.get('history')
        if isinstance(hist, dict):
            return hist
        # sometimes history saved as DataFrame
        if isinstance(hist, list):
            # list-of-dicts
            return {'_from_list': hist}
    except Exception:
        return None
    return None


def read_history_csv(run_dir: str):
    path = os.path.join(run_dir, 'history.csv')
    if not os.path.exists(path):
        return None
    if _HAS_PANDAS:
        try:
            return pd.read_csv(path)
        except Exception:
            return None
    # fallback simple CSV reader
    try:
        import csv
        rows = []
        with open(path, 'r') as f:
            r = csv.DictReader(f)
            for row in r:
                rows.append(row)
        return rows
    except Exception:
        return None


def read_weights_from_event(event_file: str) -> Optional[Dict[str, List[float]]]:
    if not _HAS_EVENT_ACC:
        return None
    tags = ['Weights/temp', 'Weights/amp', 'Weights/physics']
    ea = EventAccumulator(event_file, size_guidance={'scalars': 0})
    try:
        ea.Reload()
    except Exception:
        return None
    out = {}
    for tag in tags:
        if tag in ea.Tags().get('scalars', []):
            vals = ea.Scalars(tag)
            out[tag] = [float(v.value) for v in vals]
    return out if out else None


def read_scalar_from_event(event_file: str, tag: str) -> List[Tuple[int, float, float]]:
    """Read scalar series (step, value, wall_time) for `tag` using EventAccumulator."""
    if not _HAS_EVENT_ACC:
        return []
    try:
        ea = EventAccumulator(event_file, size_guidance={'scalars': 0})
        ea.Reload()
        if tag not in ea.Tags().get('scalars', []):
            return []
        vals = ea.Scalars(tag)
        return [(int(v.step), float(v.value), float(v.wall_time)) for v in vals]
    except Exception:
        return []


def extract_losses(history_obj) -> Tuple[List[float], List[float], List[float], List[int]]:
    # history_obj may be dict (from checkpoint) or pandas DataFrame or list-of-dicts
    # Try multiple common key names
    temp_keys = ['train_temp_loss', 'loss_temp', 'train_temp']
    amp_keys = ['train_amp_loss', 'loss_amp', 'train_amp']
    phys_keys = ['train_physics_loss', 'loss_physics', 'train_physics']
    epoch_keys = ['epoch', 'epochs', 'step']

    def find_series(obj, keys):
        if obj is None:
            return None
        # DataFrame
        if _HAS_PANDAS and isinstance(obj, pd.DataFrame):
            for k in keys:
                if k in obj.columns:
                    return list(obj[k].astype(float).values)
            return None
        # dict-like
        if isinstance(obj, dict):
            for k in keys:
                if k in obj and obj[k] is not None:
                    return obj[k]
        # list-of-dicts
        if isinstance(obj, list):
            for k in keys:
                try:
                    vals = [float(r[k]) for r in obj if k in r and r[k] != '']
                    if vals:
                        return vals
                except Exception:
                    continue
        return None

    t = find_series(history_obj, temp_keys) or []
    a = find_series(history_obj, amp_keys) or []
    p = find_series(history_obj, phys_keys) or []
    ep = find_series(history_obj, epoch_keys) or []

    # If epochs missing, build a 1-based index based on length of longest series
    maxlen = max(len(t), len(a), len(p))
    if not ep:
        ep = list(range(1, maxlen + 1))
    return t, a, p, ep


def align_by_epoch(series_dict: Dict[int, float], epochs: List[int]) -> List[Optional[float]]:
    return [series_dict.get(e) for e in epochs]


def detect_dominance(temp, amp, phys, threshold: float = 10.0) -> Dict[str, bool]:
    # For each epoch, compute max_component / median(other_components)
    n = max(len(temp), len(amp), len(phys))
    dom_counts = {'temp': 0, 'amp': 0, 'phys': 0}
    for i in range(n):
        vals = []
        for arr in (temp, amp, phys):
            vals.append(float(arr[i]) if i < len(arr) and arr[i] is not None else float('nan'))
        if any(math.isnan(v) for v in vals):
            continue
        comps = {'temp': vals[0], 'amp': vals[1], 'phys': vals[2]}
        max_k = max(comps, key=lambda k: comps[k])
        others = [v for k, v in comps.items() if k != max_k]
        median_others = np.median(others) if others else 0.0
        if median_others <= 0:
            continue
        ratio = comps[max_k] / median_others
        if ratio > threshold:
            dom_counts[max_k] += 1
    # Decide domination if it appears in the final epoch or appears in >5% of epochs
    domination = {k: False for k in dom_counts}
    total_epochs = max(len(temp), len(amp), len(phys))
    for k, count in dom_counts.items():
        if count > 0:
            # check last epoch
            last_vals = []
            try:
                last_idx = total_epochs - 1
                last_vals = [temp[last_idx] if last_idx < len(temp) else None,
                             amp[last_idx] if last_idx < len(amp) else None,
                             phys[last_idx] if last_idx < len(phys) else None]
            except Exception:
                last_vals = []
            if last_vals and all(v is not None for v in last_vals):
                comps = {'temp': float(last_vals[0]), 'amp': float(last_vals[1]), 'phys': float(last_vals[2])}
                max_k = max(comps, key=lambda k: comps[k])
                others = [v for kk, v in comps.items() if kk != max_k]
                median_others = np.median(others)
                if median_others > 0 and (comps[max_k] / median_others) > threshold:
                    domination[max_k] = True
                    continue
            # or frequent dominance (>5% epochs)
            if total_epochs > 0 and (count / total_epochs) > 0.05:
                domination[k] = True
    return domination


def main():
    parser = argparse.ArgumentParser(description='Analyze loss components and adaptive weights')
    parser.add_argument('--runs-dir', default='runs', help='Root runs directory')
    parser.add_argument('--threshold', type=float, default=10.0, help='Dominance ratio threshold (default 10)')
    parser.add_argument('--show', action='store_true', help='Show plot interactively')
    parser.add_argument('--save-path', type=str, default=None, help='Optional output path for PNG')
    args = parser.parse_args()

    try:
        latest = find_latest_run(args.runs_dir)
    except Exception as e:
        print(f"[error] {e}")
        sys.exit(1)

    # try checkpoint
    ckpt = find_latest_checkpoint(latest)
    history_obj = None
    if ckpt and _HAS_TORCH:
        history_obj = load_history_from_checkpoint(ckpt)

    # fallback to history.csv
    event_file = None
    if history_obj is None:
        hist_csv = read_history_csv(latest)
        if hist_csv is not None:
            history_obj = hist_csv
        else:
            # try TensorBoard event file as a last resort
            event_files = glob.glob(os.path.join(latest, '**', 'events.out.tfevents.*'), recursive=True)
            if event_files and _HAS_EVENT_ACC:
                event_file = event_files[-1]
                print(f"No checkpoint/history.csv — using TensorBoard event file: {event_file}")
            else:
                print('[error] No history found (checkpoint/history.csv/event file missing)')
                sys.exit(1)

    # extract losses either from history_obj OR from event file
    temp, amp, phys, epochs = [], [], [], []
    if history_obj is not None:
        temp, amp, phys, epochs = extract_losses(history_obj)
    else:
        # read from TensorBoard event file
        def ev_series(tag):
            vals = read_scalar_from_event(event_file, tag)
            return ([float(v) for (_, v, _) in vals], [int(s) for (s, _, _) in vals])

        t_vals, t_steps = ev_series('Loss/temp')
        a_vals, a_steps = ev_series('Loss/amp')
        p_vals, p_steps = ev_series('Loss/physics')
        # choose common epoch axis (union of steps)
        steps = sorted(set(t_steps + a_steps + p_steps))
        epochs = steps
        # map step->value
        t_map = {s: v for s, v in zip(t_steps, t_vals)}
        a_map = {s: v for s, v in zip(a_steps, a_vals)}
        p_map = {s: v for s, v in zip(p_steps, p_vals)}
        temp = [t_map.get(s, float('nan')) for s in steps]
        amp = [a_map.get(s, float('nan')) for s in steps]
        phys = [p_map.get(s, float('nan')) for s in steps]

    if not (temp or amp or phys):
        print('[error] Could not find loss component series in history or events')
        sys.exit(1)

    # attempt to get adaptive weights: check history dict fields first
    weights_series = None
    # if history_obj is dict and contains 'weights'
    if isinstance(history_obj, dict) and 'weights' in history_obj:
        try:
            weights_series = np.array(history_obj['weights']).T.tolist()  # expecting list-of-lists
        except Exception:
            weights_series = None
    # else check CSV columns
    if weights_series is None and _HAS_PANDAS and isinstance(history_obj, pd.DataFrame):
        # possible columns: 'weights' (json-like), or 'weight_temp', 'weight_amp', 'weight_phys'
        df = history_obj
        if 'weights' in df.columns:
            try:
                # weights stored as stringified list per epoch
                import ast
                w = [ast.literal_eval(x) if isinstance(x, str) else x for x in df['weights'].fillna('[]')]
                weights_series = np.array(w).T.tolist()
            except Exception:
                weights_series = None
        else:
            # check for separate columns
            cols = []
            for c in ['Weights/temp', 'Weights/amp', 'Weights/physics', 'weight_temp', 'weight_amp', 'weight_phys']:
                if c in df.columns:
                    cols.append(c)
            if cols:
                try:
                    ws = [list(df[c].astype(float).values) for c in cols]
                    weights_series = ws
                except Exception:
                    weights_series = None

    # fallback to TensorBoard events for adaptive weights if not present
    if weights_series is None and _HAS_EVENT_ACC:
        event_files = glob.glob(os.path.join(latest, '**', 'events.out.tfevents.*'), recursive=True)
        if event_files:
            ev = event_files[-1]
            w_ev = read_weights_from_event(ev)
            if w_ev:
                # order: temp, amp, physics
                temp_w = w_ev.get('Weights/temp') or w_ev.get('weight_temp') or []
                amp_w = w_ev.get('Weights/amp') or w_ev.get('weight_amp') or []
                phys_w = w_ev.get('Weights/physics') or w_ev.get('weight_phys') or []
                if temp_w or amp_w or phys_w:
                    # pad shorter arrays to match length
                    maxlen = max(len(temp_w), len(amp_w), len(phys_w))
                    def pad(a):
                        return list(a) + [math.nan] * (maxlen - len(a))
                    weights_series = [pad(temp_w), pad(amp_w), pad(phys_w)]

    # Prepare arrays for plotting (convert to numpy, handle missing lengths)
    maxlen = max(len(temp), len(amp), len(phys))
    def pad_to(a, n):
        a = list(a)
        return a + [np.nan] * (n - len(a))
    temp_arr = np.array(pad_to(temp, maxlen), dtype=float)
    amp_arr = np.array(pad_to(amp, maxlen), dtype=float)
    phys_arr = np.array(pad_to(phys, maxlen), dtype=float)
    epochs_x = list(epochs)[:maxlen]

    # compute ratios
    # ratio matrix: each component / median(of others)
    ratios = np.zeros((maxlen, 3))  # columns: temp, amp, phys
    for i in range(maxlen):
        comps = np.array([temp_arr[i], amp_arr[i], phys_arr[i]])
        # avoid zeros
        others_median = [np.nanmedian(np.delete(comps, j)) for j in range(3)]
        with np.errstate(divide='ignore', invalid='ignore'):
            ratios[i, 0] = comps[0] / others_median[0] if others_median[0] != 0 else np.nan
            ratios[i, 1] = comps[1] / others_median[1] if others_median[1] != 0 else np.nan
            ratios[i, 2] = comps[2] / others_median[2] if others_median[2] != 0 else np.nan

    domination = detect_dominance(list(temp_arr), list(amp_arr), list(phys_arr), threshold=args.threshold)

    # Prepare figure
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 9), sharex=True)

    # Top: loss components (log scale)
    ax1.plot(epochs_x, temp_arr, label='train_temp_loss', linewidth=2)
    ax1.plot(epochs_x, amp_arr, label='train_amp_loss', linewidth=2)
    ax1.plot(epochs_x, phys_arr, label='train_physics_loss', linewidth=2)
    ax1.set_yscale('log')
    ax1.set_ylabel('Loss (log scale)')
    ax1.set_title('Loss components over epochs')
    ax1.grid(True, which='both', alpha=0.3)
    ax1.legend()

    # Show dominance markers on top plot (if any)
    for k, dom in domination.items():
        if dom:
            ax1.text(0.99, 0.95 - (0.05 * list(domination.keys()).index(k)), f'{k.upper()} dominating', transform=ax1.transAxes, ha='right', color='red', fontsize=10, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.6))

    # Add small inset showing ratios (log scale)
    ax1_rat = ax1.twinx()
    ax1_rat.plot(epochs_x, ratios[:, 0], '--', color='C3', alpha=0.6, label='ratio_temp')
    ax1_rat.plot(epochs_x, ratios[:, 1], ':', color='C4', alpha=0.6, label='ratio_amp')
    ax1_rat.plot(epochs_x, ratios[:, 2], '-.', color='C5', alpha=0.6, label='ratio_phys')
    ax1_rat.set_yscale('log')
    ax1_rat.set_ylabel('Dominance ratio (log)')
    # combine legends
    lines, labels = ax1.get_legend_handles_labels()
    l2, lab2 = ax1_rat.get_legend_handles_labels()
    ax1.legend(lines + l2, labels + lab2, loc='upper right', fontsize='small')

    # Bottom: adaptive weights (if available)
    ax2.set_title('Adaptive loss-balancer weights')
    if weights_series:
        # weights_series expected as [temp_list, amp_list, phys_list]
        try:
            w_temp, w_amp, w_phys = weights_series
            xw = list(range(1, len(w_temp) + 1))
            ax2.plot(xw[:len(w_temp)], w_temp, label='temp weight', marker='o')
            ax2.plot(xw[:len(w_amp)], w_amp, label='amp weight', marker='s')
            ax2.plot(xw[:len(w_phys)], w_phys, label='phys weight', marker='^')
            ax2.set_xlabel('Epoch')
            ax2.set_ylabel('Adaptive weight')
            ax2.grid(True, alpha=0.3)
            ax2.legend()
        except Exception:
            ax2.text(0.5, 0.5, 'Adaptive weights found but failed to plot', ha='center', va='center')
    else:
        ax2.text(0.5, 0.5, 'Adaptive weights not available (check history or TensorBoard)', ha='center', va='center')
        ax2.set_axis_off()

    plt.tight_layout()

    # Save figure
    out_path = args.save_path or os.path.join(latest, 'loss_component_analysis.png')
    try:
        fig.savefig(out_path, dpi=150)
        # also copy to cwd
        try:
            fig.savefig(os.path.join(os.getcwd(), 'loss_component_analysis.png'), dpi=150)
        except Exception:
            pass
        print(f"Saved plot to: {out_path}")
    except Exception as e:
        print(f"[warning] failed to save figure: {e}")

    if args.show:
        plt.show()

    # Analyze domination and print recommendations
    recs = []
    if domination.get('phys'):
        recs.append('Physics loss dominating -> Reduce lambda_physics in config')
    if domination.get('amp'):
        recs.append('Ampacity loss dominating -> Check if ampacity targets are scaled correctly')
    if not any(domination.values()):
        recs.append('Weights are balanced -> Optimal training configuration')

    print('\n=== Dominance summary ===')
    for k, v in domination.items():
        print(f"{k}: {'DOMINATING' if v else 'balanced/ok'}")

    print('\n=== Recommendations ===')
    for r in recs:
        print('- ' + r)


if __name__ == '__main__':
    main()
