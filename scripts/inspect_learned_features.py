#!/usr/bin/env python3
"""
inspect_learned_features.py

Examine learned multi-resolution features from the latest HWF-PIKAN v2 checkpoint.

- Loads model from the latest `runs/*/best_model.pt` (falls back to `models/best_model.pt`).
- Extracts embedding parameters: `scales`, `freqs`, `wavelet_weights`.
- Attempts to read adaptive physics weight from TensorBoard (`Weights/physics`).
- Produces visualizations and a short textual analysis comparing learned scales
  to theoretical expectations (temperature: daily/weekly; wind: minute-level; solar: hourly).

Saves `learned_features.png` in the run directory and prints a brief summary.
"""
import glob
import math
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np

import torch

# Try to import model factory
try:
    from models.hwf_pikan_v2 import create_hwf_pikan_v2
except Exception:
    # ensure project root on path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from models.hwf_pikan_v2 import create_hwf_pikan_v2

# TensorBoard EventAccumulator for reading Weights/physics
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


def find_best_checkpoint(run_dir: str) -> str:
    candidate = os.path.join(run_dir, "best_model.pt")
    if os.path.exists(candidate):
        return candidate
    pts = glob.glob(os.path.join(run_dir, "**", "*.pt"), recursive=True)
    if pts:
        return max(pts, key=os.path.getmtime)
    fallback = os.path.join("models", "best_model.pt")
    if os.path.exists(fallback):
        return fallback
    raise FileNotFoundError("No checkpoint found in run or models/")


def load_model_from_checkpoint(ckpt_path: str):
    """Attempt several safe fallbacks to obtain a HWF-PIKAN v2 model instance.

    Order:
      1) torch.load(ckpt_path, weights_only=False)
      2) torch.load(models/best_model.pt, weights_only=False)
      3) raise RuntimeError
    """
    fallback_ckpt = os.path.join('models', 'best_model.pt')

    def _load(path: str):
        try:
            return torch.load(path, map_location='cpu', weights_only=False)
        except TypeError:
            # older torch versions may not accept weights_only kwarg
            return torch.load(path, map_location='cpu')

    data = None
    # 1) try the requested checkpoint
    try:
        data = _load(ckpt_path)
    except Exception as e:
        print(f"[warning] loading checkpoint {ckpt_path} failed: {e}")
        data = None

    # 2) try global fallback models/best_model.pt
    if data is None and os.path.exists(fallback_ckpt):
        try:
            data = _load(fallback_ckpt)
            print(f"Loaded fallback checkpoint: {fallback_ckpt}")
        except Exception as e:
            print(f"[warning] loading fallback checkpoint {fallback_ckpt} failed: {e}")
            data = None

    if data is None:
        raise RuntimeError('Unable to load checkpoint (checked run and models/best_model.pt)')

    # Instantiate model using config if present (use only model-related keys)
    cfg = data.get('config', None) if isinstance(data, dict) else None
    model_cfg = {}
    if isinstance(cfg, dict):
        if 'model_config' in cfg and isinstance(cfg['model_config'], dict):
            model_cfg = cfg['model_config']
        else:
            # pick keys that match create_hwf_pikan_v2 signature
            allowed = {'input_dim', 'hidden_dim', 'output_dim', 'fourier_bands', 'wavelet_scales', 'kan_grid', 'kan_k'}
            model_cfg = {k: cfg[k] for k in allowed & set(cfg.keys())}
    model = create_hwf_pikan_v2(config=model_cfg or {})

    # determine state_dict
    if isinstance(data, dict) and 'model_state_dict' in data:
        state = data['model_state_dict']
    elif isinstance(data, dict) and all(k.startswith('embedding') or k.startswith('kan') or k.startswith('physics_params') for k in data.keys()):
        state = data
    else:
        state = data

    model.load_state_dict(state)
    model.eval()
    return model, data


def read_physics_weight_from_events(run_dir: str) -> Optional[float]:
    if not _HAS_EA:
        return None
    ev_files = glob.glob(os.path.join(run_dir, '**', 'events.out.tfevents.*'), recursive=True)
    if not ev_files:
        return None
    ev = max(ev_files, key=os.path.getmtime)
    try:
        ea = EventAccumulator(ev, size_guidance={'scalars': 0})
        ea.Reload()
        tags = ea.Tags().get('scalars', [])
        if 'Weights/physics' not in tags:
            return None
        vals = ea.Scalars('Weights/physics')
        if not vals:
            return None
        return float(vals[-1].value)
    except Exception:
        return None


def representative_scale_per_dim(scales: np.ndarray, weights: np.ndarray) -> np.ndarray:
    # weights: [n_scales, input_dim], scales: [n_scales]
    # Use magnitude-weighted average of scales per input dim
    abs_w = np.abs(weights)
    denom = abs_w.sum(axis=0)
    # avoid division by zero
    denom_safe = np.where(denom == 0, 1.0, denom)
    rep = (scales[:, None] * abs_w).sum(axis=0) / denom_safe
    # for dims with zero total weight, fallback to median scale
    rep = np.where(denom == 0, np.median(scales), rep)
    return rep


def categorize_time_scale(seconds: float) -> str:
    if seconds < 60:
        return f"sub-minute (~{seconds:.0f}s)"
    if seconds < 3600:
        return f"minute-level (~{seconds/60:.1f} min / {seconds:.0f}s)"
    if seconds < 86400:
        return f"hourly (~{seconds/3600:.2f} h / {seconds:.0f}s)"
    return f"daily/weekly (~{seconds/86400:.2f} d / {seconds:.0f}s)"


def main(runs_dir: str = 'runs') -> None:
    latest = find_latest_run(runs_dir)
    print(f"Inspecting latest run: {latest}")

    ckpt = find_best_checkpoint(latest)
    print(f"Loading checkpoint: {ckpt}")

    model, ckpt_data = load_model_from_checkpoint(ckpt)

    emb = getattr(model, 'embedding', None)
    if emb is None:
        raise RuntimeError('Model embedding not found')

    scales = emb.scales.detach().cpu().numpy()          # [n_scales]
    freqs = emb.freqs.detach().cpu().numpy()            # [input_dim, n_bands]
    wavelet_w = emb.wavelet_weights.detach().cpu().numpy()  # [n_scales, input_dim]

    n_scales = scales.shape[0]
    input_dim = wavelet_w.shape[1]

    # Infer input labels (prefer checkpoint normalizer length -> VietnamDataset order)
    default_labels = ['T_ambient', 'wind_speed', 'wind_angle', 'solar_irradiance', 'current', 'resistance']
    normalizer = None
    if isinstance(ckpt_data, dict) and 'normalizer' in ckpt_data:
        mean = ckpt_data['normalizer'].get('mean')
        if mean is not None and len(mean) == input_dim:
            labels = default_labels[:input_dim]
        else:
            labels = default_labels[:input_dim]
    else:
        labels = default_labels[:input_dim]

    # Representative scale per input-dimension (assume scales unit: hours)
    rep_scales = representative_scale_per_dim(scales, wavelet_w)  # in "scale units"
    # ASSUMPTION: scales are in hours (see model init: logspace(0,2) -> 1..100). Convert to seconds.
    rep_seconds = rep_scales * 3600.0

    # Fourier frequencies: flatten and compute dominant period per-dim (if freq>0)
    freqs_flat = freqs.flatten()
    # compute periods assuming freqs are angular frequency (rad/hour) -> period_hours = 2*pi / freq
    # Try safer heuristic: if median(freqs) > 1e-6 treat as rad/hour else treat as cycles/hour
    median_freq = np.median(freqs_flat)
    if median_freq > 1e-6:
        # assume freqs are in rad/hour -> period_hours = 2*pi / freq
        periods_hours = 2 * np.pi / (freqs + 1e-12)
    else:
        # fallback: treat freqs as cycles per hour -> period_hours = 1 / freq
        periods_hours = 1.0 / (freqs + 1e-12)

    # For per-dimension dominant period (hours)
    dominant_periods_h = periods_hours.min(axis=1)  # smallest period => highest frequency
    dominant_periods_s = dominant_periods_h * 3600.0

    # Physics weight: try TensorBoard first, else check checkpoint config/history
    phys_weight = read_physics_weight_from_events(latest)
    if phys_weight is None:
        # try history or config
        hist = None
        if isinstance(ckpt_data, dict) and 'history' in ckpt_data:
            hist = ckpt_data['history']
        if hist and 'physics_weight' in hist:
            try:
                phys_weight = float(hist['physics_weight'][-1])
            except Exception:
                phys_weight = None
        elif isinstance(ckpt_data, dict) and 'config' in ckpt_data and 'lambda_physics' in ckpt_data['config']:
            phys_weight = float(ckpt_data['config']['lambda_physics'])
        else:
            phys_weight = None

    # Plotting
    out_png = os.path.join(latest, 'learned_features.png')
    fig = plt.figure(constrained_layout=True, figsize=(14, 8))
    gs = fig.add_gridspec(2, 3)

    # 1) Bar chart: representative wavelet scale per dimension (log scale)
    ax1 = fig.add_subplot(gs[0, 0])
    x = np.arange(input_dim)
    ax1.bar(x, rep_scales)
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, rotation=45, ha='right')
    ax1.set_yscale('log')
    ax1.set_ylabel('Wavelet scale (hours, log scale)')
    ax1.set_title('Representative wavelet scale per input dimension')

    # 2) Histogram of Fourier frequencies (flattened)
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.hist(freqs_flat, bins=30, color='C1', edgecolor='k', alpha=0.8)
    ax2.set_xlabel('Learned Fourier frequencies (raw units)')
    ax2.set_title('Histogram of learned Fourier frequencies')

    # 3) Heatmap of wavelet weights [scales x dims]
    ax3 = fig.add_subplot(gs[0, 2])
    im = ax3.imshow(wavelet_w, aspect='auto', cmap='bwr', interpolation='nearest')
    ax3.set_yticks(np.arange(n_scales))
    ax3.set_yticklabels([f"scale_{i+1}={scales[i]:.2f}" for i in range(n_scales)])
    ax3.set_xticks(np.arange(input_dim))
    ax3.set_xticklabels(labels, rotation=45, ha='right')
    ax3.set_title('Wavelet weights (scales x input dimensions)')
    fig.colorbar(im, ax=ax3, orientation='vertical', fraction=0.046)

    # 4) Per-dimension dominant Fourier period (hours) as bar chart
    ax4 = fig.add_subplot(gs[1, 0])
    ax4.bar(np.arange(len(dominant_periods_h)), dominant_periods_h, color='C2')
    ax4.set_xticks(np.arange(len(dominant_periods_h)))
    ax4.set_xticklabels(labels, rotation=45, ha='right')
    ax4.set_ylabel('Dominant Fourier period (hours)')
    ax4.set_title('Dominant Fourier period per input dimension')

    # 5) Scatter: rep wavelet scale vs dominant Fourier period
    ax5 = fig.add_subplot(gs[1, 1:])
    ax5.scatter(rep_scales, dominant_periods_h)
    for i, lab in enumerate(labels):
        ax5.annotate(lab, (rep_scales[i], dominant_periods_h[i]))
    ax5.set_xscale('log')
    ax5.set_yscale('log')
    ax5.set_xlabel('Representative wavelet scale (hours)')
    ax5.set_ylabel('Dominant Fourier period (hours)')
    ax5.set_title('Wavelet scale vs Fourier period (per-dimension)')

    plt.suptitle('Learned multi-resolution features (HWF-PIKAN v2)')
    plt.savefig(out_png, dpi=200)
    plt.close(fig)

    # Analysis sentences
    # Map dimension names to expected categories
    expectations = {
        'T_ambient': 'daily/weekly',
        'temp': 'daily/weekly',
        'wind_speed': 'minute-level',
        'Wind1': 'minute-level',
        'solar_irradiance': 'hourly',
        'GHI': 'hourly'
    }

    # pick wind and temperature dims if present
    temp_idx = None
    wind_idx = None
    solar_idx = None
    for i, lab in enumerate(labels):
        if 'temp' in lab.lower() or 'ambient' in lab.lower():
            temp_idx = i
        if 'wind' in lab.lower() and 'angle' not in lab.lower():
            wind_idx = i
        if 'solar' in lab.lower() or 'ghi' in lab.lower():
            solar_idx = i

    # Prepare readable resolutions (seconds)
    temp_res_s = int(rep_seconds[temp_idx]) if temp_idx is not None else None
    wind_res_s = int(rep_seconds[wind_idx]) if wind_idx is not None else None
    solar_res_s = int(rep_seconds[solar_idx]) if solar_idx is not None else None

    def humanize(sec: Optional[int]) -> str:
        if sec is None:
            return 'unknown'
        if sec < 60:
            return f"{sec}s"
        if sec < 3600:
            return f"{sec//60}m ({sec}s)"
        if sec < 86400:
            return f"{sec//3600}h ({sec}s)"
        return f"{sec//86400}d ({sec}s)"

    # Print summary
    print('\n=== Learned features summary ===')
    print(f"Wavelet scales (n={n_scales}): {np.array2string(scales, precision=3)}")
    print(f"Representative scales per-dim (hours): {np.array2string(rep_scales, precision=3)}")
    print(f"Fourier frequencies shape: {freqs.shape}; sample bands (first dim): {freqs[0,:5]}")
    print(f"Physics weight (adaptive / lambda fallback): {phys_weight}")

    # Comparative statements
    temp_msg = f"temperature needs {humanize(temp_res_s)} resolution" if temp_res_s else "temperature resolution unknown"
    wind_msg = f"wind requires {humanize(wind_res_s)} resolution" if wind_res_s else "wind resolution unknown"
    solar_msg = f"solar favors {humanize(solar_res_s)} resolution" if solar_res_s else "solar resolution unknown"

    print(f"The model learned that wind requires {humanize(wind_res_s)} resolution while temperature needs {humanize(temp_res_s)} resolution.")
    print(f"(Solar: {solar_msg})")
    print(f"Visualization saved to: {out_png}")


if __name__ == '__main__':
    main()
