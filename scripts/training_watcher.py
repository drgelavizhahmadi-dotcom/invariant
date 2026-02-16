"""
Invariant-PIKAN: Adversarially-Robust Physics-Informed Neural Networks for Dynamic Line Rating
Copyright (C) 2025 Gelavizh Ahmadi / Invariant Research

This software is licensed under the Business Source License 1.1 (BSL 1.1).
Commercial production use requires a separate license agreement.
See LICENSE.txt for full terms.

"""

#!/usr/bin/env python3
"""
training_watcher.py

Monitors the latest training run under `runs/` and notifies/logs when:
 - epochs remaining < --warn-epochs (default 20)
 - ampacity MAE drops below --stretch-target (default 280A)
 - training completes

Writes alerts to `training_alerts.log` in the repository root.

Usage:
  python scripts/training_watcher.py           # runs continuously (default interval 300s)
  python scripts/training_watcher.py --once    # single check and exit

"""
import argparse
import glob
import os
import sys
import time
import math
import subprocess
from datetime import datetime, timedelta

# Optional dependencies
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

LOG_PATH = 'training_alerts.log'
DEFAULT_TOTAL_EPOCHS = 200


def find_latest_run(runs_root: str = 'runs'):
    if not os.path.isdir(runs_root):
        raise FileNotFoundError(f"Runs directory not found: {runs_root}")
    subdirs = [os.path.join(runs_root, d) for d in os.listdir(runs_root) if os.path.isdir(os.path.join(runs_root, d))]
    if not subdirs:
        raise FileNotFoundError(f"No run directories found under {runs_root}")
    return max(subdirs, key=os.path.getmtime)


def find_latest_checkpoint(run_dir: str):
    pts = glob.glob(os.path.join(run_dir, '**', '*.pt'), recursive=True)
    if not pts:
        return None
    return max(pts, key=os.path.getmtime)


def read_history_csv(run_dir: str):
    path = os.path.join(run_dir, 'history.csv')
    if not os.path.exists(path):
        return None
    if _HAS_PANDAS:
        try:
            return pd.read_csv(path)
        except Exception:
            return None
    # simple fallback
    try:
        import csv
        rows = []
        with open(path, 'r') as f:
            reader = csv.DictReader(f)
            for r in reader:
                rows.append(r)
        return rows
    except Exception:
        return None


def read_scalar_from_event(event_file: str, tag: str):
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


def seconds_per_epoch_from_event(event_file: str, tag: str = 'Loss/train'):
    series = read_scalar_from_event(event_file, tag)
    if len(series) < 2:
        return None
    steps = [s[0] for s in series]
    times = [s[2] for s in series]
    dsteps = steps[-1] - steps[0]
    dtime = times[-1] - times[0]
    if dsteps <= 0 or dtime <= 0:
        return None
    return dtime / float(dsteps)


def send_macos_notification(title: str, message: str):
    try:
        esc_msg = message.replace('"', '\\"')
        esc_title = title.replace('"', '\\"')
        subprocess.run(['osascript', '-e', f'display notification "{esc_msg}" with title "{esc_title}"'], check=False)
    except Exception as e:
        print(f"[warning] notification failed: {e}")


def append_log(text: str):
    ts = datetime.utcnow().isoformat() + 'Z'
    line = f"{ts} | {text}\n"
    with open(LOG_PATH, 'a') as f:
        f.write(line)


def last_log_lines(n=200):
    if not os.path.exists(LOG_PATH):
        return []
    with open(LOG_PATH, 'rb') as f:
        f.seek(0, os.SEEK_END)
        end = f.tell()
        size = 4096
        data = b''
        while end > 0 and data.count(b'\n') <= n:
            read_size = min(size, end)
            f.seek(end - read_size)
            data = f.read(read_size) + data
            end -= read_size
        lines = data.splitlines()[-n:]
        return [ln.decode(errors='replace') for ln in lines]


def get_current_state(run_dir: str, default_total_epochs: int = DEFAULT_TOTAL_EPOCHS):
    # total epochs
    total_epochs = default_total_epochs
    ckpt = find_latest_checkpoint(run_dir)
    history = read_history_csv(run_dir)

    if ckpt and _HAS_TORCH:
        try:
            data = torch.load(ckpt, map_location='cpu')
            cfg = data.get('config')
            if isinstance(cfg, dict) and 'epochs' in cfg:
                total_epochs = int(cfg['epochs'])
        except Exception:
            pass

    if history is not None:
        # pandas DataFrame
        if _HAS_PANDAS and isinstance(history, pd.DataFrame):
            if 'epoch' in history.columns and len(history['epoch']) > 0:
                current_epoch = int(history['epoch'].iloc[-1])
            else:
                current_epoch = None
            # total_epochs fallback
            if 'epochs' in history.columns:
                try:
                    total_epochs = int(history['epochs'].iloc[-1])
                except Exception:
                    pass
        else:
            # list-of-dicts fallback
            try:
                current_epoch = int(history[-1].get('epoch') or history[-1].get('Epoch'))
            except Exception:
                current_epoch = None
    else:
        current_epoch = None

    # checkpoint 'epoch' field fallback
    if current_epoch is None and ckpt and _HAS_TORCH:
        try:
            d = torch.load(ckpt, map_location='cpu')
            if 'epoch' in d:
                current_epoch = int(d['epoch'])
        except Exception:
            current_epoch = None

    # event-file step fallback
    event_files = glob.glob(os.path.join(run_dir, '**', 'events.out.tfevents.*'), recursive=True)
    event_file = event_files[-1] if event_files else None
    if current_epoch is None and event_file:
        series = read_scalar_from_event(event_file, 'Loss/train')
        if series:
            current_epoch = int(series[-1][0])

    # best validation MAE
    best_val = None
    if history is not None:
        if _HAS_PANDAS and isinstance(history, pd.DataFrame):
            for col in ['val_amp_mae', 'val_amp', 'amp_mae']:
                if col in history.columns:
                    try:
                        best_val = float(history[col].min())
                        break
                    except Exception:
                        continue
        else:
            # list-of-dicts
            vals = []
            for r in history:
                for key in ('val_amp_mae', 'val_amp', 'amp_mae'):
                    if r.get(key):
                        try:
                            vals.append(float(r[key]))
                        except Exception:
                            pass
            if vals:
                best_val = min(vals)

    if best_val is None and ckpt and _HAS_TORCH:
        try:
            d = torch.load(ckpt, map_location='cpu')
            vm = d.get('val_metrics') or d.get('val_metric')
            if isinstance(vm, dict):
                for k in ('amp_mae', 'val_amp_mae', 'val_amp'):
                    if k in vm:
                        best_val = float(vm[k])
                        break
        except Exception:
            pass

    # seconds per epoch
    sec_per_epoch = None
    if event_file:
        sec_per_epoch = seconds_per_epoch_from_event(event_file)

    return {
        'run_dir': run_dir,
        'checkpoint': ckpt,
        'current_epoch': current_epoch,
        'total_epochs': total_epochs,
        'best_val_amp_mae': best_val,
        'sec_per_epoch': sec_per_epoch,
        'event_file': event_file
    }


def should_notify(state, warn_epochs_threshold, stretch_target, alerted):
    alerts = []
    ce = state['current_epoch']
    te = state['total_epochs']
    best = state['best_val_amp_mae']

    epochs_remaining = None
    if ce is not None and te is not None:
        epochs_remaining = te - ce

    # approaching completion
    if epochs_remaining is not None and epochs_remaining <= warn_epochs_threshold and 'near_completion' not in alerted:
        alerts.append(('near_completion', f'Epochs remaining: {epochs_remaining}'))

    # stretch target reached
    if best is not None and best < stretch_target and 'stretch_target' not in alerted:
        alerts.append(('stretch_target', f'Best amp MAE {best:.2f}A < stretch target {stretch_target}A'))

    # training complete
    if epochs_remaining is not None and epochs_remaining <= 0 and 'completed' not in alerted:
        alerts.append(('completed', 'Training reported complete'))

    return alerts


def main():
    parser = argparse.ArgumentParser(description='Watch training progress and send alerts')
    parser.add_argument('--runs-dir', default='runs', help='Root runs directory')
    parser.add_argument('--warn-epochs', type=int, default=20, help='Notify when epochs remaining is below this')
    parser.add_argument('--stretch-target', type=float, default=280.0, help='Stretch target for ampacity MAE (A)')
    parser.add_argument('--interval', type=int, default=300, help='Polling interval in seconds')
    parser.add_argument('--once', action='store_true', help='Run a single check and exit')
    args = parser.parse_args()

    alerted_for_run = {}  # run_dir -> set(alert_keys)

    try:
        latest = find_latest_run(args.runs_dir)
    except Exception as e:
        print(f"[error] {e}")
        sys.exit(1)

    print(f"Monitoring latest run: {latest}")

    def do_check():
        state = get_current_state(latest)
        ce = state['current_epoch']
        te = state['total_epochs']
        best = state['best_val_amp_mae']
        sec_epoch = state['sec_per_epoch']

        epochs_remaining = None
        if ce is not None and te is not None:
            epochs_remaining = te - ce

        eta = None
        if epochs_remaining is not None and sec_epoch is not None:
            eta = timedelta(seconds=int(round(epochs_remaining * sec_epoch)))

        print('\n=== Training watcher status ===')
        print(f"Run dir: {latest}")
        print(f"Current epoch: {ce} / {te}")
        print(f"Epochs remaining: {epochs_remaining}")
        print(f"Seconds/epoch (est): {sec_epoch if sec_epoch else 'unknown'}")
        print(f"ETA: {eta if eta else 'unknown'}")
        print(f"Best validation Amp MAE: {best if best is not None else 'N/A'}")

        # prepare alerted set for this run
        alerted = alerted_for_run.setdefault(latest, set())

        # determine alerts to fire
        alerts = should_notify(state, args.warn_epochs, args.stretch_target, alerted)
        for key, reason in alerts:
            title = 'Training alert — InvariantPIKAN v2'
            msg = f"{key}: {reason} | epoch {ce}/{te} | best_amp_mae={best}" 
            send_macos_notification(title, msg)
            append_log(f"{key.upper()} | run={latest} | epoch={ce}/{te} | best_amp_mae={best} | reason={reason}")
            print(f"[alert sent] {key} — {reason}")
            alerted.add(key)

        # log a checkpoint line for monitoring
        append_log(f"CHECK | run={latest} | epoch={ce}/{te} | remaining={epochs_remaining} | best_amp_mae={best} | eta={eta}")

        # final completion handling
        if epochs_remaining is not None and epochs_remaining <= 0:
            print('Training appears complete.')
            return True
        return False

    # initial check
    finished = do_check()
    if args.once:
        return 0

    # loop
    try:
        while not finished:
            time.sleep(args.interval)
            finished = do_check()
            # refresh latest run if new runs appear
            try:
                latest = find_latest_run(args.runs_dir)
            except Exception:
                pass
    except KeyboardInterrupt:
        print('\nWatcher stopped by user')

    return 0

if __name__ == '__main__':
    main()
