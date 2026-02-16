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
monitor_training.py

Quick monitor utility for InvariantPIKAN v2 training runs.

Features:
  1. Check whether the training process is running (by name).
  2. Find the most recent run directory / log file under `runs/`.
  3. Show the last N recorded epochs + train loss (from history.csv or TensorBoard events).
  4. Display current best validation Amp MAE from `best_model.pt` (or history.csv fallback).

Usage:
  python scripts/monitor_training.py
  python scripts/monitor_training.py --runs-dir runs --proc-name train_invariant_pikan_v2 --lines 10

Designed to be robust (falls back to history.csv, event files, checkpoint contents).
"""
import os
import sys
import glob
import argparse
import subprocess
import time
from datetime import datetime

try:
    import torch
    _HAS_TORCH = True
except Exception:
    _HAS_TORCH = False

try:
    # tensorboard event reader
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    _HAS_TB_ACC = True
except Exception:
    _HAS_TB_ACC = False

try:
    import pandas as pd
    _HAS_PANDAS = True
except Exception:
    _HAS_PANDAS = False


def is_process_running(name: str):
    """Return list of PIDs matching process name (uses pgrep -f then ps fallback)."""
    pids = []
    # Try pgrep -f (available on macOS)
    try:
        out = subprocess.check_output(['pgrep', '-f', name], stderr=subprocess.DEVNULL).decode().strip()
        if out:
            pids = [int(x) for x in out.split()]
            return pids
    except Exception:
        pass

    # Fallback to ps + grep
    try:
        out = subprocess.check_output(['ps', '-ax', '-o', 'pid=,command='], stderr=subprocess.DEVNULL).decode()
        for line in out.splitlines():
            if name in line and 'grep' not in line:
                parts = line.strip().split(None, 1)
                if parts:
                    try:
                        pid = int(parts[0])
                        pids.append(pid)
                    except Exception:
                        continue
    except Exception:
        pass

    return pids


def latest_run_dir(runs_root: str = 'runs'):
    """Return the most recently modified subdirectory under runs_root."""
    if not os.path.isdir(runs_root):
        return None
    subdirs = [os.path.join(runs_root, d) for d in os.listdir(runs_root) if os.path.isdir(os.path.join(runs_root, d))]
    if not subdirs:
        return None
    latest = max(subdirs, key=os.path.getmtime)
    return latest


def most_recent_file_in_dir(path: str):
    files = []
    for root, _, fnames in os.walk(path):
        for f in fnames:
            files.append(os.path.join(root, f))
    if not files:
        return None
    return max(files, key=os.path.getmtime)


def tail_text_file(path: str, n: int = 10):
    """Return last n lines of a text file (safe)."""
    try:
        with open(path, 'rb') as f:
            f.seek(0, os.SEEK_END)
            end = f.tell()
            size = 1024
            data = b''
            while end > 0 and data.count(b'\n') <= n:
                read_size = min(size, end)
                f.seek(end - read_size)
                data = f.read(read_size) + data
                end -= read_size
            lines = data.splitlines()[-n:]
            return [ln.decode(errors='replace') for ln in lines]
    except Exception:
        return None


def read_history_csv(run_dir: str):
    path = os.path.join(run_dir, 'history.csv')
    if not os.path.exists(path):
        return None
    if _HAS_PANDAS:
        try:
            df = pd.read_csv(path)
            return df
        except Exception:
            pass
    # fallback simple CSV parser
    rows = []
    try:
        import csv
        with open(path, 'r') as f:
            reader = csv.DictReader(f)
            for r in reader:
                rows.append(r)
        return rows
    except Exception:
        return None


def read_tb_scalars(event_file: str, tags: list, n: int = 10):
    if not _HAS_TB_ACC:
        return None
    try:
        ea = EventAccumulator(event_file, size_guidance={
            'scalars': 0,
        })
        ea.Reload()
        out = {}
        for tag in tags:
            if tag in ea.Tags().get('scalars', []):
                vals = ea.Scalars(tag)
                out[tag] = vals[-n:]
        return out
    except Exception:
        return None


def best_val_from_checkpoint(run_dir: str):
    # Prefer best_model.pt
    ckpt_path = os.path.join(run_dir, 'best_model.pt')
    if os.path.exists(ckpt_path) and _HAS_TORCH:
        try:
            ckpt = torch.load(ckpt_path, map_location='cpu')
            vm = ckpt.get('val_metrics') or ckpt.get('val_metric') or ckpt.get('results')
            if isinstance(vm, dict):
                # prefer amp MAE
                for k in ('amp_mae', 'val_amp_mae', 'ampacity_mae', 'ampacity'):
                    if k in vm:
                        try:
                            return float(vm[k])
                        except Exception:
                            pass
            # fallback to history inside checkpoint
            hist = ckpt.get('history')
            if hist and isinstance(hist, dict):
                vals = hist.get('val_amp_mae') or hist.get('val_amp')
                if vals:
                    try:
                        return float(min(vals)) if isinstance(vals, list) else float(vals)
                    except Exception:
                        pass
        except Exception:
            pass
    # fallback to history.csv
    df = read_history_csv(run_dir)
    if df is None:
        return None
    try:
        if _HAS_PANDAS and isinstance(df, type(pd.DataFrame())):
            if 'val_amp_mae' in df.columns:
                return float(df['val_amp_mae'].min())
            # try other names
            for name in ['val_amp', 'amp_mae', 'ampacity']:
                if name in df.columns:
                    return float(df[name].min())
            # return last val if min not available
            for name in ['val_amp_mae', 'val_amp', 'amp_mae']:
                if name in df.columns:
                    return float(df[name].iloc[-1])
        else:
            # list-of-dicts
            rows = df
            best = None
            for r in rows:
                for key in ('val_amp_mae', 'val_amp', 'amp_mae'):
                    if key in r and r[key] != '':
                        try:
                            v = float(r[key])
                            best = v if best is None else min(best, v)
                        except Exception:
                            pass
            return best
    except Exception:
        return None


def main(args):
    # 1) Is training running?
    pids = is_process_running(args.proc_name)
    running = len(pids) > 0

    print('\n== Training process status ==')
    if running:
        print(f"Training process '{args.proc_name}' is RUNNING (PIDs: {pids})")
    else:
        print(f"Training process '{args.proc_name}' is NOT running")

    # 2) Latest run dir / most recent log file
    run_dir = args.runs_dir if os.path.isdir(args.runs_dir) else 'runs'
    latest = latest_run_dir(run_dir)
    if not latest:
        print('\nNo run directories found under', run_dir)
        return 0

    print('\n== Latest run directory ==')
    print(latest)
    most_recent_file = most_recent_file_in_dir(latest)
    if most_recent_file:
        mtime = datetime.fromtimestamp(os.path.getmtime(most_recent_file)).isoformat()
        print(f"Most recent file: {most_recent_file}  (modified: {mtime})")
    else:
        print('No files found in latest run directory')

    # 3) Parse last N lines / last N epochs
    print('\n== Recent training entries (last {} rows) =='.format(args.lines))

    # Prefer history.csv
    df = read_history_csv(latest)
    if df is not None:
        if _HAS_PANDAS and isinstance(df, type(pd.DataFrame())):
            tail = df.tail(args.lines)
            # Print selected columns if available
            cols = [c for c in ['epoch', 'train_loss', 'train_temp_loss', 'train_amp_loss', 'train_physics_loss', 'val_amp_mae', 'val_physics_mae'] if c in tail.columns]
            print(tail[cols].to_string(index=False))
        else:
            # list-of-dicts
            rows = df[-args.lines:]
            for r in rows:
                ep = r.get('epoch') or r.get('Epoch')
                tl = r.get('train_loss') or r.get('loss')
                va = r.get('val_amp_mae') or r.get('val_amp')
                print(f"epoch={ep}  train_loss={tl}  val_amp_mae={va}")
    else:
        # Try TensorBoard events
        ev_files = glob.glob(os.path.join(latest, '**', 'events.out.tfevents.*'), recursive=True)
        ev_file = ev_files[-1] if ev_files else None
        if ev_file and _HAS_TB_ACC:
            tags = ['Loss/train', 'Metrics/amp_mae', 'Weights/physics']
            scalars = read_tb_scalars(ev_file, tags, n=args.lines)
            if scalars:
                for tag, values in scalars.items():
                    print(f"\nTag: {tag}")
                    for v in values:
                        # v has (wall_time, step, value) depending on backend; event_acc returns namedtuple (wall_time, step, value)
                        print(f" step={v.step:4d}  value={v.value:.6g}")
            else:
                print('No readable TensorBoard scalars found in event file')
        else:
            print('No history.csv or tensorboard event file available to show recent epochs')

    # 4) Best validation MAE from latest checkpoint
    best_val = best_val_from_checkpoint(latest)
    print('\n== Best validation amp MAE (from checkpoint/history) ==')
    if best_val is not None:
        print(f"Best validation Amp MAE: {best_val:.3f} A")
    else:
        print('No checkpoint/history providing best validation MAE found')

    print('\nDone.')
    return 0


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Monitor InvariantPIKAN v2 training run')
    parser.add_argument('--runs-dir', default='runs', help='Root runs directory')
    parser.add_argument('--proc-name', default='train_invariant_pikan_v2', help='Process name to check')
    parser.add_argument('--lines', type=int, default=10, help='Number of recent entries/lines to show')
    args = parser.parse_args()
    sys.exit(main(args))
