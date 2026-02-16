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
tb_summary.py

Fetch key scalars from TensorBoard (prefers `tensorboard.data.experimental` when
available; falls back to reading event files). Prints a short progress summary and
(optionally) sends a macOS desktop notification when ampacity MAE drops below a
baseline threshold.

Features requested by user:
 - Uses `from tensorboard.data import experimental as tb_data` when available
 - Fetches latest values for Metrics/amp_mae, Metrics/temp_mae, Weights/physics,
   Loss/train
 - Calculates rate of improvement over last N epochs (default 10)
 - Prints current epoch, ampacity MAE (+target), projected epoch to beat target,
   and physics-weight trend
 - Sends macOS notification if ampacity MAE < baseline

Works on macOS and includes clear error handling.
"""
import argparse
import glob
import math
import os
import subprocess
import sys
from datetime import datetime

# Prefer tensorboard.data.experimental if available (user asked for this)
try:
    from tensorboard.data import experimental as tb_data  # type: ignore
    _HAS_TB_DATA = True
except Exception:
    tb_data = None  # type: ignore
    _HAS_TB_DATA = False

# Fallback: use EventAccumulator to read event files
try:
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    _HAS_EA = True
except Exception:
    EventAccumulator = None  # type: ignore
    _HAS_EA = False


def find_latest_event_file(runs_dir: str):
    """Return most recent events.out.tfevents.* file under runs_dir/latest."""
    if not os.path.isdir(runs_dir):
        raise FileNotFoundError(f"runs directory not found: {runs_dir}")
    subdirs = [os.path.join(runs_dir, d) for d in os.listdir(runs_dir) if os.path.isdir(os.path.join(runs_dir, d))]
    if not subdirs:
        raise FileNotFoundError(f"no run subdirectories found under {runs_dir}")
    latest = max(subdirs, key=os.path.getmtime)
    ev_files = glob.glob(os.path.join(latest, '**', 'events.out.tfevents.*'), recursive=True)
    if not ev_files:
        raise FileNotFoundError(f"no TensorBoard event files found in latest run dir: {latest}")
    return latest, max(ev_files, key=os.path.getmtime)


def read_scalars_from_event(event_file: str, tag: str, n: int = 10):
    """Return list of (step, value) for the last n scalar events for tag."""
    if not _HAS_EA:
        raise RuntimeError('tensorboard EventAccumulator not available in this environment')
    ea = EventAccumulator(event_file, size_guidance={
        'scalars': 0,
        'histograms': 0,
        'images': 0,
        'tensors': 0,
    })
    ea.Reload()
    tags = ea.Tags().get('scalars', [])
    if tag not in tags:
        return []
    vals = ea.Scalars(tag)
    if not vals:
        return []
    last = vals[-n:]
    return [(int(v.step), float(v.value)) for v in last]


def rate_of_change(series):
    """Compute per-step rate (linear) for a series of numeric values. Returns slope."""
    if not series or len(series) < 2:
        return None
    x0 = 0
    x1 = len(series) - 1
    y0 = series[0]
    y1 = series[-1]
    slope = (y1 - y0) / (x1 - x0)
    return slope


def trend_label(values, rel_threshold=0.01):
    """Return 'increasing'/'decreasing'/'stable' for a numeric sequence."""
    if not values or len(values) < 2:
        return 'unknown'
    first = values[0]
    last = values[-1]
    if abs(first) < 1e-8:
        diff = last - first
        if abs(diff) < 1e-6:
            return 'stable'
        return 'increasing' if diff > 0 else 'decreasing'
    pct = (last - first) / abs(first)
    if abs(pct) < rel_threshold:
        return 'stable'
    return 'increasing' if pct > 0 else 'decreasing'


def notify_macos(title: str, message: str):
    """Send macOS desktop notification using osascript (AppleScript)."""
    try:
        # Escape double quotes in message
        esc_msg = message.replace('"', '\\"')
        esc_title = title.replace('"', '\\"')
        cmd = ['osascript', '-e', f'display notification "{esc_msg}" with title "{esc_title}"']
        subprocess.run(cmd, check=True)
    except Exception as e:
        print(f"[warning] macOS notification failed: {e}")


def fetch_latest_scalars_via_tbdata(tb_server_url: str, tag: str, limit: int = 10):
    """Try to fetch scalars via tensorboard.data.experimental (best-effort).

    Note: API surface for tb_data.experimental varies by TensorBoard version; this
    function will attempt common call patterns and gracefully fail back to None.
    """
    if not _HAS_TB_DATA:
        return None
    try:
        # Several TB versions expose a `MultiplexerDataProvider` or `DataProvider`.
        # We'll attempt a few common entrypoints (best-effort). If none exist, return None.
        # The goal here is to *use* tb_data.experimental when possible, but the script
        # also supports reading event files directly (fallback below).
        if hasattr(tb_data, 'read_scalars'):
            # hypothetical convenience function
            return tb_data.read_scalars(tag_name=tag, limit=limit)
        if hasattr(tb_data, 'data_provider'):
            provider = tb_data.data_provider.DataProvider()  # type: ignore
            # try provider.read_scalars signature
            if hasattr(provider, 'read_scalars'):
                return provider.read_scalars(tag, limit=limit)  # type: ignore
        # No supported API found — return None to signal fallback
        return None
    except Exception:
        return None


def main():
    parser = argparse.ArgumentParser(description='Fetch scalar summary from TensorBoard')
    parser.add_argument('--tb-url', default='http://localhost:6007', help='TensorBoard server URL')
    parser.add_argument('--runs-dir', default='runs', help='Local runs directory (fallback)')
    parser.add_argument('--window', type=int, default=10, help='How many latest epochs to use for rate calc')
    parser.add_argument('--baseline', type=float, default=308.0, help='Ampacity MAE baseline to beat (A)')
    parser.add_argument('--notify', action='store_true', help='Send macOS notification if baseline is beaten')
    parser.add_argument('--use-tb-data', action='store_true', default=True, help='Attempt to use tensorboard.data.experimental (optional)')
    args = parser.parse_args()

    tags = {
        'amp': 'Metrics/amp_mae',
        'temp': 'Metrics/temp_mae',
        'phys_w': 'Weights/physics',
        'train_loss': 'Loss/train'
    }

    # First try tb_data.experimental if requested
    tb_results = {}
    if args.use_tb_data and _HAS_TB_DATA:
        for k, tag in tags.items():
            try:
                val = fetch_latest_scalars_via_tbdata(args.tb_url, tag, limit=args.window)
                tb_results[k] = val
            except Exception:
                tb_results[k] = None

    # If tb_data not available or returned None, fall back to reading local event files
    latest_run = None
    event_file = None
    try:
        latest_run, event_file = find_latest_event_file(args.runs_dir)
    except FileNotFoundError as e:
        print(f"[error] {e}")
        if not (_HAS_TB_DATA and args.use_tb_data):
            sys.exit(1)

    # For each tag, get last `window` points
    data = {}
    for k, tag in tags.items():
        series = None
        # Prefer tb_data results if they look usable
        tb_val = tb_results.get(k)
        if tb_val:
            # tb_data return format unknown — attempt to coerce
            try:
                # if tb_val is sequence-like of (step, value), use it
                if isinstance(tb_val, (list, tuple)) and tb_val:
                    coerced = []
                    for entry in tb_val[-args.window:]:
                        # entry might be an object or dict
                        if isinstance(entry, dict) and 'value' in entry and 'step' in entry:
                            coerced.append((int(entry['step']), float(entry['value'])))
                        elif hasattr(entry, 'step') and hasattr(entry, 'value'):
                            coerced.append((int(entry.step), float(entry.value)))
                    if coerced:
                        series = coerced
            except Exception:
                series = None
        # Fallback to EventAccumulator
        if series is None:
            try:
                series = read_scalars_from_event(event_file, tag, n=args.window)
            except Exception as e:
                print(f"[warning] couldn't read tag {tag}: {e}")
                series = []
        data[k] = series

    # Helper to extract numeric-only series (values) and get current step
    def vals_and_step(series):
        if not series:
            return [], None
        steps, vals = zip(*series)
        return list(vals), int(steps[-1])

    amp_vals, amp_step = vals_and_step(data['amp'])
    temp_vals, temp_step = vals_and_step(data['temp'])
    phys_vals, phys_step = vals_and_step(data['phys_w'])
    loss_vals, loss_step = vals_and_step(data['train_loss'])

    # Current values (most recent)
    amp_current = amp_vals[-1] if amp_vals else None
    temp_current = temp_vals[-1] if temp_vals else None
    phys_current = phys_vals[-1] if phys_vals else None
    loss_current = loss_vals[-1] if loss_vals else None
    current_epoch = amp_step or temp_step or loss_step or phys_step

    # Rate of improvement over window
    amp_slope = rate_of_change(amp_vals)
    temp_slope = rate_of_change(temp_vals)
    loss_slope = rate_of_change(loss_vals)
    phys_slope = rate_of_change(phys_vals)

    # Project epoch to beat baseline (linear extrapolation on amp MAE if improving)
    projected_epoch = None
    if amp_current is not None and amp_slope is not None and amp_slope < 0:
        improvement_per_epoch = -amp_slope
        if improvement_per_epoch > 0:
            epochs_needed = (amp_current - args.baseline) / improvement_per_epoch
            if epochs_needed <= 0:
                projected_epoch = current_epoch
            else:
                projected_epoch = int(math.ceil(current_epoch + epochs_needed))

    # Physics weight trend label
    phys_trend = trend_label(phys_vals, rel_threshold=0.01) if phys_vals else 'unknown'

    # Print summary
    print('\n=== TensorBoard training summary ===')
    print(f"Latest run dir: {latest_run}")
    print(f"Current epoch (approx): {current_epoch}")
    if amp_current is not None:
        print(f"Ampacity MAE: {amp_current:.3f} A  (target: {args.baseline:.1f} A)")
    else:
        print("Ampacity MAE: not available")
    if temp_current is not None:
        print(f"Temperature MAE: {temp_current:.3f} °C")
    else:
        print("Temperature MAE: not available")
    if phys_current is not None:
        print(f"Physics weight: {phys_current:.3f}   (trend: {phys_trend})")
    else:
        print("Physics weight: not available")
    if loss_current is not None:
        print(f"Training loss: {loss_current:.3g}   (rate per epoch: {loss_slope:+.3g})")
    else:
        print("Training loss: not available")

    # Improvement rates (negative slope for MAE means improvement)
    if amp_slope is not None:
        print(f"Amp MAE rate (per epoch): {amp_slope:+.6f} (negative = improving)")
    if temp_slope is not None:
        print(f"Temp MAE rate (per epoch): {temp_slope:+.6f}")

    if projected_epoch:
        print(f"Projected epoch to beat {args.baseline:.1f}A: {projected_epoch}")
    else:
        if amp_current is None:
            print("Projected epoch: insufficient ampacity MAE data")
        elif amp_slope is None or amp_slope >= 0:
            print("Projected epoch: cannot project (no improving trend detected)")
        else:
            print("Projected epoch: unknown")

    # Notification on macOS if requested and baseline beaten
    if args.notify and amp_current is not None and amp_current < args.baseline:
        msg = f"Amp MAE {amp_current:.1f}A dropped below target {args.baseline:.1f}A at epoch {current_epoch}"
        notify_macos('InvariantPIKAN Training', msg)
        print('\n[notification sent] ' + msg)

    print('\n(If some scalars are missing, ensure TensorBoard logged those tags or use --runs-dir to point to local logs)')


if __name__ == '__main__':
    main()
