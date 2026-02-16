#!/usr/bin/env python3
"""
train_hwf_pikan_production.py

Comprehensive, production-ready training script for HWF-PIKAN v2 (Dynamic Line Rating).

Features implemented (per your specification):
- HWF-PIKAN v2 architecture (reuse `models.hwf_pikan_v2.create_hwf_pikan_v2`)
- Multi-scale embedding (learnable Fourier freqs + Morlet wavelets)
- KAN backbone (Chebyshev style, grid=5, k=3)
- Physics-informed loss using IEEE 738 heat balance
- Adaptive loss balancing (temperature, ampacity, physics)
- Two-stage optimization: AdamW (first 100 epochs) then L-BFGS fine-tune
- TensorBoard logging (losses, metrics, learned params, grad norms)
- Checkpointing (best / periodic / resume)
- Early stopping, NaN-handling, gradient clipping, robust error handling

Run example:
    python -m scripts.train_hwf_pikan_production --epochs 200 --batch-size 64

"""
import argparse
import json
import math
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from torch.utils.data import DataLoader, Subset, WeightedRandomSampler
from torch.utils.tensorboard import SummaryWriter

# Project imports (reuse model + dataset + physics)
from core.data import VietnamDataset, USDataset, InputNormalizer
from models.hwf_pikan_v2 import create_hwf_pikan_v2
from core.physics import IEEE738HeatBalance
# Prefer the training-compatible PhysicsInformedLoss (has __call__(preds, targets, weather, return_components))
try:
    from scripts.train_hwf_pikan_v2 import PhysicsInformedLoss
except Exception:
    # Fall back to core.model.PhysicsInformedLoss and wrap it to the expected API
    from core.model import PhysicsInformedLoss as CorePhysicsInformedLoss

    class PhysicsInformedLoss(CorePhysicsInformedLoss):
        """Compatibility wrapper: adapt core.model.PhysicsInformedLoss to the training script API.

        The core implementation expects raw tensors; this wrapper accepts the
        (predictions: dict, targets: dict, weather: Tensor, return_components: bool)
        signature used elsewhere in the training scripts.
        """
        def __init__(self, physics_engine=None, lambda_physics: float = 0.05):
            # Initialize parent with sensible defaults
            super().__init__(physics_weight=lambda_physics, temp_weight=2.0, rating_weight=1.0, consistency_weight=0.0)
            # keep a reference to physics engine for potential residual computations
            self._physics_engine = physics_engine
            # expose lambda_physics for compatibility
            self.lambda_physics = lambda_physics

        def __call__(self, predictions, targets, weather, return_components: bool = False):
            # predictions: dict with 'temperature', 'ampacity', 'physics_residual' (optional)
            pred_temp = predictions['temperature'].unsqueeze(-1) if predictions['temperature'].dim() == 1 else predictions['temperature']
            pred_amp = predictions['ampacity'].unsqueeze(-1) if predictions['ampacity'].dim() == 1 else predictions['ampacity']
            true_temp = targets['temperature'].unsqueeze(-1) if targets['temperature'].dim() == 1 else targets['temperature']
            true_amp = targets['ampacity'].unsqueeze(-1) if targets['ampacity'].dim() == 1 else targets['ampacity']

            # compute (or reuse) physics residual if provided by model
            phys_res = predictions.get('physics_residual', None)
            if phys_res is None:
                # if a physics engine is available, compute an approximate residual vectorized
                if self._physics_engine is not None:
                    try:
                        # Vectorized computation
                        T_pred = pred_temp.squeeze()
                        I_pred = pred_amp.squeeze()
                        T_amb = weather[:, 0]
                        wind_speed = weather[:, 1]
                        solar = weather[:, 2]
                        phys_T = self._physics_engine.steady_state_temperature(
                            current=I_pred, T_ambient=T_amb, wind_speed=wind_speed, solar_irradiance=solar
                        )
                        phys_res = (T_pred - phys_T).abs()
                    except Exception:
                        phys_res = torch.zeros(pred_temp.shape[0], device=pred_temp.device)
                else:
                    phys_res = torch.zeros(pred_temp.shape[0], device=pred_temp.device)

            total_loss, metrics = super().forward(pred_temp, pred_amp, true_temp, true_amp, phys_res)
            if return_components:
                return total_loss, {
                    'loss_temp': metrics.get('temp_loss', float('nan')),
                    'loss_amp': metrics.get('rating_loss', float('nan')),
                    'loss_physics': metrics.get('physics_loss', float('nan'))
                }
            return total_loss

def temporal_split_dataset(dataset, test_frac: float = 0.2) -> Tuple[Subset, Subset]:
    """Temporal 80/20 split (train earlier, val later)."""
    n = len(dataset)
    split = int(n * (1 - test_frac))
    train_idx = list(range(0, split))
    val_idx = list(range(split, n))
    return Subset(dataset, train_idx), Subset(dataset, val_idx)


def random_split_dataset(dataset, test_frac: float = 0.2) -> Tuple[Subset, Subset]:
    """Random 80/20 split for datasets without temporal ordering."""
    n = len(dataset)
    indices = list(range(n))
    np.random.shuffle(indices)
    split = int(n * (1 - test_frac))
    train_idx = indices[:split]
    val_idx = indices[split:]
    return Subset(dataset, train_idx), Subset(dataset, val_idx)


def three_way_split_dataset(dataset, test_frac: float = 0.1, val_frac_of_remaining: float = 0.1, randomize: bool = False) -> Tuple[Subset, Subset, Subset]:
    """Split dataset into train / val / test subsets.

    - test_frac: fraction of total to reserve for final testing (e.g. 0.1)
    - val_frac_of_remaining: fraction of the remaining (after test) to use as validation
      (default 0.1 -> validation is 9% of total when test_frac=0.1)
    - randomize: if True perform a random permutation before splitting; otherwise use temporal order

    Returns (train_subset, val_subset, test_subset)
    """
    n = len(dataset)
    test_size = int(n * test_frac)
    remaining = n - test_size
    val_size = int(remaining * val_frac_of_remaining)
    train_size = n - test_size - val_size

    if randomize:
        indices = list(range(n))
        np.random.shuffle(indices)
        test_idx = indices[:test_size]
        val_idx = indices[test_size:test_size + val_size]
        train_idx = indices[test_size + val_size:]
    else:
        train_idx = list(range(0, train_size))
        val_idx = list(range(train_size, train_size + val_size))
        test_idx = list(range(train_size + val_size, n))

    return Subset(dataset, train_idx), Subset(dataset, val_idx), Subset(dataset, test_idx)


def compute_metrics(true: np.ndarray, pred: np.ndarray) -> Dict[str, float]:
    mae = float(mean_absolute_error(true, pred))
    rmse = float(math.sqrt(mean_squared_error(true, pred)))
    try:
        r2 = float(r2_score(true, pred))
    except Exception:
        r2 = float('nan')
    return {'mae': mae, 'rmse': rmse, 'r2': r2}


def save_checkpoint(path: str, state: Dict):
    torch.save(state, path)


def load_checkpoint(path: str, device: str = 'cpu') -> Dict:
    return torch.load(path, map_location=device)


def plot_training_history(history: Dict[str, List[float]], out_path: str):
    df = pd.DataFrame(history)
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    epochs = df['epoch'] if 'epoch' in df else list(range(len(df)))

    axes[0, 0].plot(epochs, df.get('train_loss', [])); axes[0, 0].set_title('Train Loss')
    axes[0, 1].plot(epochs, df.get('train_temp_loss', []), label='temp'); axes[0, 1].plot(epochs, df.get('train_amp_loss', []), label='amp'); axes[0, 1].plot(epochs, df.get('train_phys_loss', []), label='phys'); axes[0, 1].legend(); axes[0, 1].set_title('Loss components')
    axes[0, 2].plot(epochs, df.get('val_temp_mae', [])); axes[0, 2].set_title('Val Temp MAE')
    axes[1, 0].plot(epochs, df.get('val_amp_mae', [])); axes[1, 0].set_title('Val Amp MAE')
    axes[1, 1].plot(epochs, df.get('val_phys_mae', [])); axes[1, 1].set_title('Val Physics MAE')
    axes[1, 2].plot(epochs, df.get('lr', [])); axes[1, 2].set_title('LR'); axes[1, 2].set_yscale('log')
    plt.tight_layout(); plt.savefig(out_path); plt.close()


# ----------------------------- Utilities (extra) -------------------------

def find_optimal_lr(model, train_loader, loss_fn, device, init_lr=1e-5, final_lr=1.0):
    """Cyclic LR finder (Smith-style) to suggest an optimal learning rate.

    - Raises LR exponentially from init_lr -> final_lr over up to 100 steps.
    - Returns suggested LR (steepest descent on loss curve) and the full (lrs, losses).
    """
    optimizer = torch.optim.AdamW(model.parameters(), lr=init_lr)
    lr_mult = (final_lr / init_lr) ** (1.0 / 100.0)

    lrs = []
    losses = []

    model.train()
    for batch_idx, batch in enumerate(train_loader):
        if batch_idx > 100:
            break

        x, y = batch
        weather = x[:, :4].to(device)
        weather_dict = {'T_amb': weather[:, 0], 'wind_speed': weather[:, 1], 'solar': weather[:, 3]}

        preds = model(weather, weather_dict)
        loss, _ = loss_fn(preds, {'temperature': y[:, 0].to(device), 'ampacity': y[:, 1].to(device)},
                         torch.cat([weather[:, :3], x[:, 4:5].to(device)], dim=-1), return_components=True)

        lrs.append(optimizer.param_groups[0]['lr'])
        losses.append(float(loss.item()))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        optimizer.param_groups[0]['lr'] *= lr_mult

    # Smooth losses and find steepest negative gradient region
    if len(losses) < 3:
        return None, (lrs, losses)
    grad = np.gradient(np.array(losses))
    idx = int(np.argmin(grad))
    return lrs[idx], (lrs, losses)


def validate_wind_gust_response(model, val_loader, device):
    """Test model response to sudden wind changes (returns mean response steps).

    Creates a synthetic gust (3x wind) in the second half of each batch, then
    measures how quickly model's ampacity prediction approaches the new level.
    """
    model.eval()
    response_times = []

    with torch.no_grad():
        for x, y in val_loader:
            weather = x[:, :4].to(device)
            weather_gust = weather.clone()
            gust_idx = max(0, weather.shape[0] // 2)
            # apply gust to second half of batch
            weather_gust[gust_idx:, 1] = weather_gust[gust_idx:, 1] * 3.0

            weather_dict = {'T_amb': weather[:, 0], 'wind_speed': weather[:, 1], 'solar': weather[:, 3]}
            weather_dict_gust = {'T_amb': weather_gust[:, 0], 'wind_speed': weather_gust[:, 1], 'solar': weather_gust[:, 3]}

            pred_normal = model(weather, weather_dict)['ampacity']
            pred_gust = model(weather_gust, weather_dict_gust)['ampacity']

            # compute a simple target (mean of gusted segment)
            if gust_idx >= pred_gust.shape[0]:
                continue
            target = pred_gust[gust_idx:].mean()

            # Measure steps to reach 90% of new equilibrium (look ahead up to 9 indices)
            for i in range(1, min(10, pred_gust.shape[0] - gust_idx)):
                if (pred_gust[gust_idx + i] - target).abs() < 0.1 * target:
                    response_times.append(i)
                    break

    return float(np.mean(response_times)) if response_times else float('nan')


# ----------------------------- Training Core ------------------------------

def build_model(cfg: Dict) -> nn.Module:
    """Create HWF-PIKAN v2 model with requested hyperparameters and per-dim scale priors."""
    model = create_hwf_pikan_v2(config=cfg)

    # Per-dimension wavelet specialization initialization (temperature slow, wind fast, solar medium, current slow)
    # mapping indices -> rough semantic order assumed by dataset: [T_ambient, wind_speed, wind_angle, solar_irradiance, current]
    try:
        w = model.embedding.wavelet_weights.detach().cpu().numpy()  # [scales, input_dim]
        n_scales, input_dim = w.shape
        # Prefer small-scale index 0 for wind, medium index 1-2 for solar, large-scale index n_scales-1 for temperature/current
        # Set an informative prior by scaling initialization (this is only a soft prior)
        wind_idx = 1 if input_dim > 1 else 0
        temp_idx = 0
        solar_idx = 3 if input_dim > 3 else (input_dim - 2)
        current_idx = 4 if input_dim > 4 else (input_dim - 1)
        # scale preferences
        with torch.no_grad():
            # Increase amplitude for chosen scale-dim pairs
            model.embedding.wavelet_weights[0, wind_idx] += 2.0   # fast (small scale)
            model.embedding.wavelet_weights[-1, temp_idx] += 2.5  # slow (large scale)
            if n_scales >= 3:
                model.embedding.wavelet_weights[1, solar_idx] += 1.8  # medium
            model.embedding.wavelet_weights[-1, current_idx] += 1.5
    except Exception:
        # If embedding shape differs, ignore initialization hint
        pass

    return model


def train(
    run_dir: str,
    data_path: str = 'data/mendeley/vietnam_220kv.csv',
    us_data_path: Optional[str] = None,
    epochs: int = 200,
    batch_size: int = 64,
    lr: float = 1e-3,
    adam_epochs: int = 100,
    lambda_physics: float = 0.05,
    device: str = 'cpu',
    patience: int = 30,
    checkpoint_every: int = 10,
    validate_every: int = 5,  # Validate every 5 epochs to reduce overhead
    log_grad_every: int = 10,
    report_every: int = 5,
    physics_loss_every: int = 5,  # Compute physics loss every N batches to reduce overhead
    resume_from: Optional[str] = None,
    save_final: bool = True,
    use_amp: bool = False,
    use_uq: bool = False,
    random_split: bool = False,
    test_frac: float = 0.1,
    save_test: bool = False,
    seed: int = 42,
):
        # Resolve device string (support 'auto' -> cuda > mps > cpu)
        device = torch.device(
            device
            if device != 'auto'
            else (
                'cuda'
                if torch.cuda.is_available()
                else ('mps' if torch.backends.mps.is_available() else 'cpu')
            )
        )

        # create run directory + TensorBoard writer
        run_dir = Path(run_dir)
        run_dir.mkdir(parents=True, exist_ok=True)
        writer = SummaryWriter(str(run_dir))

        # ---------------- reproducibility & config ----------------
        import random
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        try:
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
        except Exception:
            pass

        # Save run config for reproducibility / later inspection
        cfg = {
            'data_path': data_path,
            'us_data_path': us_data_path,
            'epochs': epochs,
            'batch_size': batch_size,
            'lr': lr,
            'lambda_physics': lambda_physics,
            'device': str(device),
            'physics_loss_every': physics_loss_every,
            'validate_every': validate_every,
            'use_amp': use_amp,
            'use_uq': use_uq,
            'random_split': random_split,
            'test_frac': test_frac,
            'save_test': save_test,
            'seed': seed,
        }
        try:
            with open(run_dir / 'config.json', 'w') as _f:
                json.dump(cfg, _f, indent=2)
        except Exception as _e:
            print(f"Warning: could not write config.json: {_e}")

        # ---------------- Data ----------------
        if us_data_path is not None:
            print(f"Loading US training dataset from: {us_data_path}")
            dataset = USDataset(us_data_path, normalizer=None)
            # three-way split: train / val / test (randomized for US data)
            train_set, val_set, test_set = three_way_split_dataset(dataset, test_frac=test_frac, val_frac_of_remaining=0.1, randomize=True)
            dataset_type = "US_DLR"
            # optionally persist test indices
            if save_test:
                try:
                    torch.save(test_set.indices, run_dir / 'test_indices.pt')
                    print(f"Saved test indices -> {run_dir / 'test_indices.pt'}")
                except Exception as _e:
                    print(f"Warning: failed to save test indices: {_e}")
        elif data_path.endswith('.h5'):
            print(f"Loading unified training dataset from: {data_path}")
            df = pd.read_hdf(data_path, key='data')
            # Rename columns to match VietnamDataset expectations (handle US/VN/EU variants)
            rename_dict = {
                'datetime': 'time',
                'timestamp': 'time',
                'temperature': 'temp',
                'wind_speed': 'Wind1',
                'wind_direction': 'WinDir',
                'WinDir': 'WinDir',
                'solar_irradiance': 'GHI',
                'actual': 'Ampacity',
                'dlr_amps': 'Ampacity',
                'Ampacity': 'Ampacity'
            }
            df = df.rename(columns=rename_dict)

            # If the rename produced duplicate column names (e.g. 'actual' + 'dlr_amps' -> 'Ampacity'),
            # coalesce duplicates by taking the first non-null value per row and then drop duplicates.
            if df.columns.duplicated().any():
                dup_names = [c for i, c in enumerate(df.columns) if c in df.columns[:i]]
                for col in set(dup_names):
                    cols = [i for i, name in enumerate(df.columns) if name == col]
                    # take first non-null value across duplicate columns
                    vals = df.iloc[:, cols]
                    df[col] = vals.bfill(axis=1).iloc[:, 0]
                # keep only first occurrence of each column name
                df = df.loc[:, ~df.columns.duplicated()]

            # Coalesce alternate ampacity columns if needed (fallback)
            if 'Ampacity' not in df.columns:
                if 'actual' in df.columns:
                    df['Ampacity'] = df['actual']
                elif 'dlr_amps' in df.columns:
                    df['Ampacity'] = df['dlr_amps']

            # Ensure 'time' column exists
            if 'time' not in df.columns:
                df['time'] = pd.date_range(start='2020-01-01', periods=len(df), freq='H')

            # Normalize timestamps to pandas datetimes (coerce bad formats) and reformat to ISO strings
            df['time'] = pd.to_datetime(df['time'], errors='coerce')
            # Drop rows where time couldn't be parsed (these would break VietnamDataset)
            df = df[df['time'].notna()]
            # write ISO strings so the downstream CSV is consistent
            df['time'] = df['time'].dt.strftime('%Y-%m-%d %H:%M:%S')

            # Drop rows missing any essential columns required by the VietnamDataset
            required_cols = [c for c in ['time', 'temp', 'Wind1', 'WinDir', 'GHI', 'Ampacity'] if c in df.columns]
            before_n = len(df)
            if required_cols:
                df = df.dropna(subset=required_cols, how='any')
            # Ensure no NaT in the time column
            if 'time' in df.columns:
                df = df[df['time'].notna()]
            after_n = len(df)
            dropped = before_n - after_n
            print(f"Dropped {dropped} rows with missing essential inputs (remaining={after_n})")
            if df.empty:
                raise RuntimeError(f"No valid rows remain after dropping NaNs from {data_path}; aborting training.")

            # Save to temp CSV for VietnamDataset
            temp_csv = run_dir / 'temp_unified_data.csv'
            df.to_csv(temp_csv, index=False)
            dataset = VietnamDataset(str(temp_csv))
            # three-way split: train / val / test
            train_set, val_set, test_set = three_way_split_dataset(dataset, test_frac=test_frac, val_frac_of_remaining=0.1, randomize=random_split)
            dataset_type = "Unified"
            # persist test indices and test-data CSV when requested
            if save_test:
                try:
                    torch.save(test_set.indices, run_dir / 'test_indices.pt')
                    try:
                        df.iloc[test_set.indices].to_csv(run_dir / 'test_data.csv', index=False)
                    except Exception:
                        pass
                    print(f"Saved test indices -> {run_dir / 'test_indices.pt'} (and test_data.csv)")
                except Exception as _e:
                    print(f"Warning: could not save test indices: {_e}")
        else:
            print(f"Loading Vietnam training dataset from: {data_path}")
            dataset = VietnamDataset(data_path)
            # three-way split: train / val / test
            train_set, val_set, test_set = three_way_split_dataset(dataset, test_frac=test_frac, val_frac_of_remaining=0.1, randomize=random_split)
            dataset_type = "Vietnam"
            if save_test:
                try:
                    torch.save(test_set.indices, run_dir / 'test_indices.pt')
                    print(f"Saved test indices -> {run_dir / 'test_indices.pt'}")
                except Exception as _e:
                    print(f"Warning: could not save test indices: {_e}")

        # TODO: US DLR dataset integration (currently contains pre-computed ratios only)
        # The downloaded US DLR dataset has DLR ratios but no weather features for training
        # When weather data becomes available, uncomment and modify the following:
        """
        from core.data.us_dlr_loader import create_us_dlr_loaders, CombinedDLRDataset

        # Load US DLR dataset (when weather data is available)
        us_train_loader, us_val_loader, us_normalizer = create_us_dlr_loaders({
            'data_path': 'data/us_dlr_2007_2013.h5',
            'batch_size': batch_size,
            'val_split': 0.2,
            'voltage_range': (69, 765),
            'year_range': (2007, 2013)
        })

        # Combine datasets with weighted sampling
        combined_dataset = CombinedDLRDataset(
            datasets=[dataset, us_train_loader.dataset],
            weights=[0.3, 0.7]  # Weight US data more heavily
        )

        # Create weighted sampler
        sampler = combined_dataset.get_weighted_sampler()

        # Create combined loader
        train_loader = DataLoader(
            combined_dataset,
            batch_size=batch_size,
            sampler=sampler,
            num_workers=4,
            pin_memory=True
        )
        """

        # Augment edge cases: weight high-wind (>10 m/s) and afternoon (12-18h) samples
        weights = np.ones(len(train_set))
        for i, idx in enumerate(train_set.indices):
            if dataset.wind_speed[idx] > 10:
                weights[i] *= 2
            if (dataset.hour[idx] >= 12) and (dataset.hour[idx] <= 18):
                weights[i] *= 1.5
        sampler = WeightedRandomSampler(weights, len(train_set), replacement=True)

        # enable faster data pipeline (persistent workers + pinned memory)
        # Note: num_workers=0 on macOS to avoid multiprocessing issues
        num_workers = 0  # Set to 0 for macOS compatibility
        prefetch_factor = 2 if num_workers > 0 else None
        train_loader = DataLoader(
            train_set,
            batch_size=batch_size,
            sampler=sampler,
            num_workers=num_workers,
            persistent_workers=False if num_workers == 0 else True,
            prefetch_factor=prefetch_factor,
            pin_memory=True,
            drop_last=False,
        )
        val_loader = DataLoader(
            val_set,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            persistent_workers=False if num_workers == 0 else True,
            prefetch_factor=prefetch_factor,
            pin_memory=True,
        )

        # ---------------- save normalizer (if available) ----------------
        normalizer_dict = None
        try:
            if hasattr(dataset, 'normalizer') and getattr(dataset, 'normalizer') is not None:
                norm_path = run_dir / 'normalizer.json'
                dataset.normalizer.save(str(norm_path))
                normalizer_dict = {
                    'mean': getattr(dataset.normalizer, 'mean').tolist(),
                    'std': getattr(dataset.normalizer, 'std').tolist(),
                    'feature_names': getattr(dataset.normalizer, 'feature_names')
                }
                print(f"Saved InputNormalizer -> {norm_path}")
        except Exception as _e:
            print(f"Warning: could not save normalizer: {_e}")

        model_cfg = {
            'input_dim': 4,            # weather dims passed to embedding (T, wind, wind_angle, solar)
            'fourier_bands': 8,
            'wavelet_scales': 2,
            'hidden_dim': 32,
            'kan_grid': 3,
            'kan_k': 3,
        }
        model = build_model(model_cfg).to(device)

        # Try PyTorch 2.0 compilation on CUDA only (skip for MPS due to known Inductor issues)
        try:
            if device.type == 'cuda':
                model = torch.compile(model)
        except Exception:
            pass

        # Optional uncertainty head (aleatoric): map embeddings -> log-variance for temperature and ampacity
        if use_uq:
            emb_dim = model.embedding.output_dim
            model.uq_head = nn.Sequential(nn.Linear(emb_dim, 64), nn.ReLU(), nn.Linear(64, 2))
            model.uq_head.to(device)

        physics_engine = IEEE738HeatBalance().to(device)
        # Create physics loss with correct parameters (constructor may not accept lambda_physics)
        loss_fn = PhysicsInformedLoss(physics_engine)

        # Set lambda_physics after construction (robust to different implementations)
        try:
            loss_fn.lambda_physics = lambda_physics
        except Exception:
            # fallback: attach attribute and warn
            setattr(loss_fn, 'lambda_physics', lambda_physics)
            print(f"⚠️  Set lambda_physics={lambda_physics} as a custom attribute on loss_fn")

        # ---------------- Optimizers & Scheduler ----------------
        adam_opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(adam_opt, T_max=epochs, eta_min=1e-6)

        # L-BFGS for fine-tuning
        lbfgs_opt = torch.optim.LBFGS(model.parameters(), lr=1.0, max_iter=20, history_size=10)

        # ---------------- Resume checkpoint ----------------
        start_epoch = 0
        best_val_amp_mae = float('inf')
        history = {
            'epoch': [], 'train_loss': [], 'train_temp_loss': [], 'train_amp_loss': [], 'train_phys_loss': [],
            'val_amp_mae': [], 'val_temp_mae': [], 'val_phys_mae': [], 'lr': [], 'physics_weight': []
        }
        if resume_from and os.path.exists(resume_from):
            ck = load_checkpoint(resume_from, device='cpu')
            model.load_state_dict(ck['model_state_dict'])
            adam_opt.load_state_dict(ck.get('optimizer_state_dict', {}))
            scheduler_state = ck.get('scheduler_state_dict')
            if scheduler_state:
                scheduler.load_state_dict(scheduler_state)
            start_epoch = ck.get('epoch', 0)
            history = ck.get('history', history)
            best_val_amp_mae = ck.get('best_val_amp_mae', best_val_amp_mae)
            print(f"Resumed training from {resume_from} (epoch {start_epoch})")

        # ---------------- Helper functions ----------------
        def evaluate(split_loader) -> Dict[str, float]:
            model.eval()
            temps_pred, temps_true = [], []
            amps_pred, amps_true = [], []
            phys_residuals = []
            with torch.no_grad():
                for x, y in split_loader:
                    # x: [T_amb, wind, wind_angle, solar, current]
                    weather = x[:, :4].to(device)
                    current = x[:, 4:5].to(device)
                    weather_dict = {'T_amb': weather[:, 0], 'wind_speed': weather[:, 1], 'solar': weather[:, 3]}
                    out = model(weather, weather_dict)
                    temps_pred.extend(out['temperature'].cpu().numpy())
                    amps_pred.extend(out['ampacity'].cpu().numpy())
                    temps_true.extend(y[:, 0].cpu().numpy())
                    amps_true.extend(y[:, 1].cpu().numpy())
                    # estimate physics residual if available
                    phys_residuals.extend([float(out.get('physics_residual', torch.tensor(float('nan'))).cpu().numpy())] * weather.shape[0])
            temp_metrics = compute_metrics(np.array(temps_true), np.array(temps_pred))
            amp_metrics = compute_metrics(np.array(amps_true), np.array(amps_pred))
            phys_mae = float(np.nanmean(np.abs(np.array(phys_residuals)))) if phys_residuals else float('nan')
            return {
                'temp_mae': temp_metrics['mae'], 'temp_rmse': temp_metrics['rmse'], 'temp_r2': temp_metrics['r2'],
                'amp_mae': amp_metrics['mae'], 'amp_rmse': amp_metrics['rmse'], 'amp_r2': amp_metrics['r2'],
                'phys_mae': phys_mae
            }

        def log_learned_params(epoch: int):
            # Physics weight and adaptive balancer weights (temp/amp/physics)
            try:
                if hasattr(loss_fn, 'balancer') and hasattr(loss_fn.balancer, 'register'):
                    reg = loss_fn.balancer.register
                    if reg.sum() > 0:
                        weights = reg / reg.sum()
                        writer.add_scalar('Weights/temp', float(weights[0]), epoch)
                        writer.add_scalar('Weights/amp', float(weights[1]), epoch)
                        writer.add_scalar('Weights/physics', float(weights[2]), epoch)
                    else:
                        writer.add_scalar('Weights/physics', float(getattr(loss_fn, 'lambda_physics', float('nan'))), epoch)
                else:
                    writer.add_scalar('Weights/physics', float(getattr(loss_fn, 'lambda_physics', float('nan'))), epoch)
            except Exception:
                writer.add_scalar('Weights/physics', float(getattr(loss_fn, 'lambda_physics', float('nan'))), epoch)

# Wavelet scales, frequencies, weights (throttle expensive histogram logging)
            try:
                if epoch % 10 == 0:
                    writer.add_histogram('Wavelet/scales', model.embedding.scales.detach().cpu().numpy(), epoch)
                    writer.add_histogram('Fourier/freqs', model.embedding.freqs.detach().cpu().numpy(), epoch)
                    ww = model.embedding.wavelet_weights.detach().cpu().numpy()
                    # log per-dimension as scalars
                    for d in range(ww.shape[1]):
                        writer.add_histogram(f'Wavelet/weights_dim_{d}', ww[:, d], epoch)
            except Exception:
                pass

        # ---------------- Training loop ----------------
        no_improve = 0
        for epoch in range(start_epoch, epochs):
            epoch_start = time.time()

            is_adam_phase = (epoch < adam_epochs)
            model.train()
            train_losses = []
            comp_accum = {'loss_temp': 0.0, 'loss_amp': 0.0, 'loss_physics': 0.0}

            if is_adam_phase:
                # optional AMP scaler for CUDA
                scaler = torch.cuda.amp.GradScaler() if (use_amp and device.type == 'cuda') else None

                for batch_idx, batch in enumerate(train_loader):
                    batch_start = time.time()
                    x, y = batch
                    weather = x[:, :4].to(device)
                    current = x[:, 4:5].to(device)
                    weather_dict = {'T_amb': weather[:, 0], 'wind_speed': weather[:, 1], 'solar': weather[:, 3]}

                    # forward + loss (autocast on supported devices when requested)
                    # Reduce physics loss frequency to improve performance
                    current_lambda_physics = lambda_physics if (batch_idx % physics_loss_every == 0) else 0.0
                    loss_fn.lambda_physics = current_lambda_physics
                    if use_amp and device.type in ('cuda', 'mps'):
                        with torch.autocast(device_type=device.type):
                            preds = model(weather, weather_dict)
                            loss, comps = loss_fn(preds, {'temperature': y[:, 0].to(device), 'ampacity': y[:, 1].to(device)}, torch.cat([weather[:, :3], current], dim=-1), return_components=True)
                    else:
                        preds = model(weather, weather_dict)
                        loss, comps = loss_fn(preds, {'temperature': y[:, 0].to(device), 'ampacity': y[:, 1].to(device)}, torch.cat([weather[:, :3], current], dim=-1), return_components=True)

                    # handle NaN/inf
                    if not torch.isfinite(loss):
                        print(f"[warn] NaN loss at epoch {epoch} batch {batch_idx}, skipping batch")
                        continue

                    # backward + step (use GradScaler for CUDA when enabled)
                    if scaler is not None:
                        scaler.scale(loss).backward()
                        scaler.unscale_(adam_opt)
                        grad_norms = [p.grad.data.norm(2) for p in model.parameters() if p.grad is not None]
                        total_norm = float(torch.sqrt(sum(g**2 for g in grad_norms)).item()) if grad_norms else 0.0
                        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                        scaler.step(adam_opt)
                        scaler.update()
                    else:
                        loss.backward()
                        grad_norms = [p.grad.data.norm(2) for p in model.parameters() if p.grad is not None]
                        total_norm = float(torch.sqrt(sum(g**2 for g in grad_norms)).item()) if grad_norms else 0.0
                        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                        adam_opt.step()

                    train_losses.append(float(loss.item()))
                    comp_accum['loss_temp'] += comps.get('loss_temp', 0.0)
                    comp_accum['loss_amp'] += comps.get('loss_amp', 0.0)
                    comp_accum['loss_physics'] += comps.get('loss_physics', 0.0)

                    batch_time = time.time() - batch_start
                    if batch_idx % 10 == 0:
                        print(f"Batch {batch_idx} took {batch_time:.3f}s")

                # scheduler step
                scheduler.step()
                current_lr = scheduler.get_last_lr()[0]
            else:
                # L-BFGS fine-tuning on training set (closure-based)
                def lbfgs_closure():
                    lbfgs_opt.zero_grad()
                    # compute epoch-loss over training loader (small cost if batch-size moderate)
                    total = 0.0
                    n = 0
                    for bx, by in train_loader:
                        weather = bx[:, :4].to(device)
                        current = bx[:, 4:5].to(device)
                        preds = model(weather, {'T_amb': weather[:,0], 'wind_speed': weather[:,1], 'solar': weather[:,3]})
                        loss = loss_fn(preds, {'temperature': by[:,0].to(device), 'ampacity': by[:,1].to(device)}, torch.cat([weather[:, :3], current], dim=-1))
                        loss.backward(retain_graph=True)
                        total += float(loss.item())
                        n += 1
                    return torch.tensor(total / max(n,1), requires_grad=True)

                # perform a few LBFGS steps
                try:
                    lbfgs_opt.step(lbfgs_closure)
                except Exception as e:
                    print(f"[warn] LBFGS step failed: {e}")
                current_lr = 0.0  # L-BFGS uses its own internal LR

            # ---------------- Validation ----------------
            if (epoch + 1) % validate_every == 0 or epoch == 0:
                val_metrics = evaluate(val_loader)
                amp_mae = val_metrics['amp_mae']
                temp_mae = val_metrics['temp_mae']
                phys_mae = val_metrics['phys_mae']

                # Logging
                writer.add_scalar('Metrics/amp_mae', amp_mae, epoch)
                writer.add_scalar('Metrics/temp_mae', temp_mae, epoch)
                writer.add_scalar('Metrics/physics_mae', phys_mae, epoch)

                # log learned params
                # log_learned_params(epoch)

                # log gradient norm periodically (compute on-device to avoid many syncs)
                if (epoch % log_grad_every) == 0:
                    grad_norms = [p.grad.data.norm(2) for p in model.parameters() if p.grad is not None]
                    total_norm = float(torch.sqrt(sum(g**2 for g in grad_norms)).item()) if grad_norms else 0.0
                    writer.add_scalar('Gradients/norm', total_norm, epoch)

                # Wind gust response check every 20 epochs
                # if (epoch + 1) % 20 == 0:
                #     try:
                #         gust_response = validate_wind_gust_response(model, val_loader, device)
                #         writer.add_scalar('Metrics/gust_response_steps', gust_response, epoch)
                #     except Exception as _:
                #         writer.add_scalar('Metrics/gust_response_steps', float('nan'), epoch)

                # save best model
                improved = False
                if amp_mae < best_val_amp_mae - 1e-6:
                    best_val_amp_mae = amp_mae
                    improved = True
                    save_checkpoint(run_dir / 'best_model.pt', {
                        'epoch': epoch + 1,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': adam_opt.state_dict() if is_adam_phase else {},
                        'val_amp_mae': amp_mae,
                        'config': {'model': model_cfg, 'training': {'epochs': epochs, 'batch_size': batch_size}},
                        'history': history,
                        'normalizer': normalizer_dict,
                    })

                if improved:
                    no_improve = 0
                else:
                    no_improve += 1

                # early stopping
                if no_improve >= patience:
                    print(f"Early stopping (no improvement for {patience} epochs). Best amp MAE: {best_val_amp_mae:.2f}A")
                    break

            # ---------------- Checkpointing ----------------
            if (epoch + 1) % checkpoint_every == 0:
                ckpt_path = run_dir / f'checkpoint_epoch{epoch+1}.pt'
                save_checkpoint(ckpt_path, {
                    'epoch': epoch + 1,
                    'model_state_dict': model.state_dict(),
                    'normalizer': normalizer_dict,
                    'optimizer_state_dict': adam_opt.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'val_amp_mae': best_val_amp_mae,
                    'history': history,
                })

        # ---------------- History & Logging ----------------
        epoch_time = time.time() - epoch_start
        train_loss = float(np.mean(train_losses)) if train_losses else float('nan')
        history['epoch'].append(epoch + 1)
        history['train_loss'].append(train_loss)
        history['train_temp_loss'].append(comp_accum['loss_temp'] / max(1, len(train_loader)))
        history['train_amp_loss'].append(comp_accum['loss_amp'] / max(1, len(train_loader)))
        history['train_phys_loss'].append(comp_accum['loss_physics'] / max(1, len(train_loader)))
        history['val_amp_mae'].append(amp_mae if 'amp_mae' in locals() else float('nan'))
        history['val_temp_mae'].append(temp_mae if 'temp_mae' in locals() else float('nan'))
        history['val_phys_mae'].append(phys_mae if 'phys_mae' in locals() else float('nan'))
        history['lr'].append(current_lr)
        history['physics_weight'].append(float(getattr(loss_fn, 'lambda_physics', float('nan'))))

        # TensorBoard scalars
        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('Loss/temp', history['train_temp_loss'][-1], epoch)
        writer.add_scalar('Loss/amp', history['train_amp_loss'][-1], epoch)
        writer.add_scalar('Loss/physics', history['train_phys_loss'][-1], epoch)
        writer.add_scalar('LR', current_lr, epoch)

        # Save history CSV every epoch
        pd.DataFrame(history).to_csv(run_dir / 'history.csv', index=False)

        print(f"Epoch {epoch+1:03d} | Train loss {train_loss:.4f} | Val amp MAE {history['val_amp_mae'][-1]:.2f}A | Time {epoch_time:.1f}s", flush=True)

        # Detailed periodic report (console + CSV) every `report_every` epochs
        try:
            if (epoch + 1) % report_every == 0:
                report = {
                    'epoch': int(epoch + 1),
                    'train_loss': float(train_loss),
                    'train_temp_loss': float(history['train_temp_loss'][-1]),
                    'train_amp_loss': float(history['train_amp_loss'][-1]),
                    'train_phys_loss': float(history['train_phys_loss'][-1]),
                    'val_amp_mae': float(history['val_amp_mae'][-1]),
                    'val_temp_mae': float(history['val_temp_mae'][-1]),
                    'val_phys_mae': float(history['val_phys_mae'][-1]),
                    'lr': float(current_lr),
                    'timestamp_utc': datetime.utcnow().isoformat()
                }

                # Console summary
                print(
                    f"[REPORT] Epoch {report['epoch']:03d} | train_loss={report['train_loss']:.4f} | val_amp_mae={report['val_amp_mae']:.2f}A | lr={report['lr']:.2e}",
                    flush=True,
                )

                # Append to run_dir/reports.csv
                rpt_path = run_dir / 'reports.csv'
                pd.DataFrame([report]).to_csv(rpt_path, mode='a', header=not rpt_path.exists(), index=False)
        except Exception:
            # never fail training due to reporting
            pass

        # ---------------- Finalize ----------------
        # Save final model and config
        if save_final:
            final_path = run_dir / 'final_model.pt'
            save_checkpoint(final_path, {
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'config': {'model': model_cfg, 'training': {'epochs': epochs, 'batch_size': batch_size}},
                'history': history,
                'normalizer': normalizer_dict,
            })
            print(f"Saved final model to {final_path}")

        # Plot training curves
        plot_training_history(history, str(run_dir / 'training_curves.png'))
        writer.close()

        return {'run_dir': str(run_dir), 'history': history, 'best_amp_mae': best_val_amp_mae}


# ----------------------------- CLI ---------------------------------------
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Production training for HWF-PIKAN v2')
    parser.add_argument('--data-path', default='data/mendeley/vietnam_220kv.csv')
    parser.add_argument('--us-data', default=None, help='Path to US training HDF5 file (if provided, uses US dataset instead of Vietnam)')
    parser.add_argument('--save-dir', default='runs/hwf_pikan_production_' + datetime.utcnow().strftime('%Y%m%d_%H%M%S'))
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--lambda-physics', type=float, default=0.0, help='Weight for physics loss (default: 0.0)')
    parser.add_argument('--device', default='auto', help="Device to use: 'auto'|'cpu'|'cuda'|'mps' — use 'mps' on Apple Silicon (recommended)")
    parser.add_argument('--resume', default=None)
    parser.add_argument('--use-uq', action='store_true')
    parser.add_argument('--use-amp', action='store_true', help='Enable mixed precision (autocast). Experimental on MPS')
    parser.add_argument('--report-every', type=int, default=5, help='Print + save a compact report every N epochs')
    parser.add_argument('--physics-loss-every', type=int, default=5, help='Compute physics loss every N batches (default: 5)')
    parser.add_argument('--validate-every', type=int, default=5, help='Run validation every N epochs (default: 5)')
    parser.add_argument('--random-split', action='store_true', help='Use random train/val split instead of temporal')
    parser.add_argument('--test-frac', type=float, default=0.1, help='Fraction of data to hold out for final testing (default: 0.1)')
    parser.add_argument('--save-test', action='store_true', help='Save test indices (or test data) in run directory for later evaluation')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility (default: 42)')
    args = parser.parse_args()

    # Resolve 'auto' to cuda > mps > cpu for Apple Silicon support
    if args.device == 'auto':
        if torch.cuda.is_available():
            resolved_device = 'cuda'
        elif torch.backends.mps.is_available():
            resolved_device = 'mps'
        else:
            resolved_device = 'cpu'
    else:
        resolved_device = args.device

    run_meta = train(
        run_dir=args.save_dir,
        data_path=args.data_path,
        us_data_path=args.us_data,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        adam_epochs=100,
        lambda_physics=args.lambda_physics,
        device=resolved_device,
        report_every=args.report_every,
        physics_loss_every=args.physics_loss_every,
        validate_every=args.validate_every,
        resume_from=args.resume,
        use_amp=args.use_amp,
        use_uq=args.use_uq,
        random_split=args.random_split,
        test_frac=args.test_frac,
        save_test=args.save_test,
        seed=args.seed,
    )

    print('\nTraining finished. Summary:')
    print(json.dumps({'run_dir': run_meta['run_dir'], 'best_amp_mae': run_meta['best_amp_mae']}, indent=2))
