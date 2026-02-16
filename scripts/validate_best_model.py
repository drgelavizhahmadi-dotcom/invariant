"""
Invariant-PIKAN: Adversarially-Robust Physics-Informed Neural Networks for Dynamic Line Rating
Copyright (C) 2025 Gelavizh Ahmadi / Invariant Research

This software is licensed under the Business Source License 1.1 (BSL 1.1).
Commercial production use requires a separate license agreement.
See LICENSE.txt for full terms.

"""

#!/usr/bin/env python3
"""
validate_best_model.py

Comprehensive validation utility for the InvariantPIKAN v2 project.

Behavior:
- Finds the latest training run and its `best_model.pt` (falls back to `models/best_model.pt`).
- Loads a runtime predictor (preferred) or a safe fallback loader.
- Performs a temporal 20% test split on the Vietnam CSV (default path `data/mendeley/vietnam_220kv.csv`).
- Computes overall and per-bin metrics for ampacity and conductor temperature, plus physics residual MAE.
- Produces a Predicted vs Actual ampacity scatter plot and saves results to:
    - `validation_results.txt`
    - `validation_results.json`
    - `validation_per_sample.csv`

Usage:
    python -m scripts.validate_best_model
    python -m scripts.validate_best_model --model-path runs/<ts>/best_model.pt --output-dir results/validation

"""
import argparse
import glob
import json
import math
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


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


def temporal_test_split_df(df: pd.DataFrame, test_fraction: float = 0.2) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if "datetime" in df.columns:
        df = df.sort_values("datetime").reset_index(drop=True)
    n = len(df)
    split_idx = int(n * (1 - test_fraction))
    return df.iloc[:split_idx].reset_index(drop=True), df.iloc[split_idx:].reset_index(drop=True)


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    mae = float(mean_absolute_error(y_true, y_pred))
    rmse = float(math.sqrt(mean_squared_error(y_true, y_pred)))
    # r2_score may return nan for degenerate inputs; coerce to float
    try:
        r2 = float(r2_score(y_true, y_pred))
    except Exception:
        r2 = float("nan")
    return {"mae": mae, "rmse": rmse, "r2": r2}


def breakdown_by_bins(df: pd.DataFrame, pred_amp: np.ndarray, true_amp: np.ndarray) -> Dict[str, Any]:
    results: Dict[str, Any] = {}

    # Wind bins: [0-2, 2-5, 5-10, >10]
    wind_upper = float(df["Wind1"].max()) if not df["Wind1"].isna().all() else 11.0
    wind_max_edge = max(11.0, wind_upper + 1.0)
    wind_bins = [0.0, 2.0, 5.0, 10.0, wind_max_edge]
    wind_labels = ["0-2", "2-5", "5-10", ">10"]
    df["wind_bin"] = pd.cut(df["Wind1"].fillna(-1.0), bins=wind_bins, labels=wind_labels, include_lowest=True)

    # Temperature bins: [<20, 20-30, 30-40, >40]
    temp_upper = float(df["temp"].max()) if not df["temp"].isna().all() else 41.0
    temp_max_edge = max(41.0, temp_upper + 1.0)
    temp_bins = [-1e9, 20.0, 30.0, 40.0, temp_max_edge]
    temp_labels = ["<20", "20-30", "30-40", ">40"]
    df["temp_bin"] = pd.cut(df["temp"].fillna(-9999.0), bins=temp_bins, labels=temp_labels, include_lowest=True)

    # Time of day bins
    if "datetime" in df.columns:
        hours = pd.to_datetime(df["datetime"]).dt.hour
    else:
        hours = pd.Series([i % 24 for i in range(len(df))])

    def tod_label(h: int) -> str:
        if 0 <= h < 6:
            return "night"
        if 6 <= h < 12:
            return "morning"
        if 12 <= h < 18:
            return "afternoon"
        return "evening"

    df["hour"] = hours
    df["tod"] = df["hour"].apply(tod_label)

    def group_metrics(indices: List[int]) -> Optional[Dict[str, float]]:
        if len(indices) == 0:
            return None
        y_t = true_amp[indices]
        y_p = pred_amp[indices]
        return compute_metrics(y_t, y_p)

    wind_results = {lab: group_metrics(df.index[df["wind_bin"] == lab].tolist()) for lab in wind_labels}
    temp_results = {lab: group_metrics(df.index[df["temp_bin"] == lab].tolist()) for lab in temp_labels}
    tod_groups = ["night", "morning", "afternoon", "evening"]
    tod_results = {lab: group_metrics(df.index[df["tod"] == lab].tolist()) for lab in tod_groups}

    results["wind_bins"] = wind_results
    results["temp_bins"] = temp_results
    results["time_of_day"] = tod_results
    return results


def plot_scatter_amp(true_amp: np.ndarray, pred_amp: np.ndarray, out_path: str):
    plt.figure(figsize=(6, 6))
    plt.scatter(true_amp, pred_amp, alpha=0.5, s=8)
    mn = min(float(np.nanmin(true_amp)), float(np.nanmin(pred_amp)))
    mx = max(float(np.nanmax(true_amp)), float(np.nanmax(pred_amp)))
    plt.plot([mn, mx], [mn, mx], "r--")
    plt.xlabel("True Ampacity (A)")
    plt.ylabel("Predicted Ampacity (A)")
    plt.title("Predicted vs Actual Ampacity")
    plt.grid(alpha=0.3)
    plt.savefig(out_path, dpi=150)
    plt.close()


def save_results(output_dir: str, summary: Dict[str, Any], per_sample: pd.DataFrame) -> Tuple[str, str, str]:
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    txt_path = os.path.join(output_dir, "validation_results.txt")
    json_path = os.path.join(output_dir, "validation_results.json")
    csv_path = os.path.join(output_dir, "validation_per_sample.csv")

    with open(txt_path, "w") as f:
        f.write("Validation summary - " + datetime.utcnow().isoformat() + "Z\n")
        f.write(json.dumps(summary, indent=2))

    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)

    per_sample.to_csv(csv_path, index=False)
    return txt_path, json_path, csv_path


def load_predictor_from_checkpoint(ckpt: str):
    """Try to load a runtime predictor (preferred). Mirror fallback order used elsewhere in repo.

    Order:
      1) HWF_PIKAN_V2 model
      2) core.inference.DLRPredictor.from_checkpoint(ckpt)
      3) core.inference.DLRPredictor.from_checkpoint(models/best_model.pt)
      4) scripts.validate_vietnam.load_model(models/best_model.pt) -> SimplePredictor
    """
    # ensure repo root is importable
    repo_root = str(Path(__file__).resolve().parents[1])
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)

    fallback_ckpt = os.path.join("models", "best_model.pt")

    # 0) Try HWF_PIKAN_V2 model
    try:
        from models.invariant_pikan_v2 import create_invariant_pikan_v2
        import torch
        model_cfg = {
            'input_dim': 4,
            'fourier_bands': 8,
            'wavelet_scales': 2,
            'hidden_dim': 32,
            'kan_grid': 3,
            'kan_k': 3,
        }
        model = create_invariant_pikan_v2(config=model_cfg)
        checkpoint = torch.load(ckpt, map_location='cpu')
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()

        class HWFPredictor:
            def __init__(self, model):
                self.model = model

            def predict(self, T_ambient, wind_speed, solar_irradiance, current, wind_angle=45.0):
                import torch
                weather = torch.tensor([[T_ambient, wind_speed, wind_angle, solar_irradiance]], dtype=torch.float32)
                current_t = torch.tensor([[current]], dtype=torch.float32)
                weather_dict = {'T_amb': weather[:, 0], 'wind_speed': weather[:, 1], 'solar': weather[:, 3]}
                with torch.no_grad():
                    out = self.model(weather, weather_dict)
                    temp = out['temperature'].item()
                    amp = out['ampacity'].item()
                    phys_res = out.get('physics_residual', torch.tensor(float('nan'))).item()
                return type("R", (), {"conductor_temperature": temp, "dynamic_rating": amp, "physics_residual": phys_res})

        predictor = HWFPredictor(model)
        print(f"Loaded HWF_PIKAN_V2 predictor from: {ckpt}")
        return predictor
    except Exception as e:
        print(f"[warning] HWF_PIKAN_V2 loader failed: {e}")

    # 1) Try production runtime predictor on requested checkpoint
    try:
        from core.inference import DLRPredictor as RuntimeDLRPredictor
        import torch
        pred = RuntimeDLRPredictor.from_checkpoint(ckpt, device=torch.device("cpu"))
        print(f"Loaded runtime DLRPredictor from: {ckpt}")
        return pred
    except Exception as e:
        print(f"[warning] runtime DLRPredictor.from_checkpoint({ckpt}) failed: {e}")

    # 2) Try runtime predictor on saved models/best_model.pt (if present)
    if os.path.exists(fallback_ckpt):
        try:
            from core.inference import DLRPredictor as RuntimeDLRPredictor
            import torch
            pred = RuntimeDLRPredictor.from_checkpoint(fallback_ckpt, device=torch.device("cpu"))
            print(f"Loaded runtime DLRPredictor from fallback: {fallback_ckpt}")
            return pred
        except Exception as e:
            print(f"[warning] runtime DLRPredictor.from_checkpoint({fallback_ckpt}) failed: {e}")

    # 3) Fallback to validate_vietnam loader (returns model + normalizer)
    try:
        from scripts.validate_vietnam import load_model as load_model_fallback
        # prefer loading the stable fallback file if it exists, else try original ckpt
        load_path = fallback_ckpt if os.path.exists(fallback_ckpt) else ckpt
        model, normalizer = load_model_fallback(load_path, device="cpu")

        class SimplePredictor:
            def __init__(self, model, normalizer):
                self.model = model
                self.normalizer = normalizer

            def predict(self, T_ambient, wind_speed, solar_irradiance, current, wind_angle=45.0):
                import torch
                x = np.array([[T_ambient, wind_speed, wind_angle, solar_irradiance, current, self.model.physics.R_ref.item() if hasattr(self.model, "physics") else 7.283e-5]])
                x_norm = self.normalizer.transform(x)
                xt = torch.tensor(x_norm, dtype=torch.float32)
                with torch.no_grad():
                    temp_t, amp_t = self.model(xt)
                return type("R", (), {"conductor_temperature": float(temp_t.item()), "dynamic_rating": float(amp_t.item()), "physics_residual": float("nan")})

        predictor = SimplePredictor(model, normalizer)
        print(f"Loaded fallback predictor via validate_vietnam loader from: {load_path}")
        return predictor
    except Exception as e:
        print(f"[warning] validate_vietnam fallback loader failed: {e}")

    raise RuntimeError("No compatible model loader available for checkpoint: " + ckpt)


def main():
    parser = argparse.ArgumentParser(description="Validate best InvariantPIKAN v2 model")
    parser.add_argument("--runs-dir", default="runs")
    parser.add_argument("--model-path", default=None)
    parser.add_argument("--data-path", default="data/mendeley/vietnam_220kv.csv")
    parser.add_argument("--test-fraction", type=float, default=0.2)
    parser.add_argument("--assumed-current", type=float, default=1000.0)
    parser.add_argument("--baseline", type=float, default=308.0)
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()

    # locate checkpoint
    if args.model_path:
        ckpt = args.model_path
    else:
        latest_run = find_latest_run(args.runs_dir)
        ckpt = find_best_checkpoint(latest_run)

    if not os.path.exists(ckpt):
        print(f"[error] checkpoint not found: {ckpt}")
        sys.exit(1)

    print(f"Using checkpoint: {ckpt}")

    predictor = load_predictor_from_checkpoint(ckpt)

    # Load CSV
    df = pd.read_csv(args.data_path, parse_dates=["datetime"])
    train_df, test_df = temporal_test_split_df(df, test_fraction=args.test_fraction)
    print(f"Dataset: {len(df)} rows — test set: {len(test_df)} rows (last {args.test_fraction*100:.0f}%)")

    # Prepare conditions for prediction
    conditions = []
    for _, row in test_df.iterrows():
        cond = {
            "T_ambient": float(row["temp"]),
            "wind_speed": float(row["Wind1"]),
            "solar_irradiance": float(row["GHI"]),
            "current": float(args.assumed_current),
            "wind_angle": float(row["WinDir"]) if "WinDir" in row and not pd.isna(row["WinDir"]) else 45.0,
        }
        conditions.append(cond)

    # Batch predict
    preds_temp: List[float] = []
    preds_amp: List[float] = []
    phys_res: List[float] = []
    batch_size = 1024
    for i in range(0, len(conditions), batch_size):
        batch = conditions[i : i + batch_size]
        if hasattr(predictor, "predict_batch"):
            results = predictor.predict_batch(batch)
            for r in results:
                preds_temp.append(r.conductor_temperature)
                preds_amp.append(r.dynamic_rating)
                phys_res.append(getattr(r, "physics_residual", float("nan")))
        else:
            for c in batch:
                r = predictor.predict(**c)
                preds_temp.append(r.conductor_temperature)
                preds_amp.append(r.dynamic_rating)
                phys_res.append(getattr(r, "physics_residual", float("nan")))

    preds_temp_arr = np.array(preds_temp)
    preds_amp_arr = np.array(preds_amp)
    phys_res_arr = np.array(phys_res)

    # True values (note: dataset stores ambient temp; earlier code used ambient+10 as conductor temp proxy — keep consistent with repo)
    true_temp = (test_df["temp"].values + 10.0)
    true_amp = test_df["Ampacity"].values.astype(float)

    amp_metrics = compute_metrics(true_amp, preds_amp_arr)
    temp_metrics = compute_metrics(true_temp, preds_temp_arr)
    phys_res_mae = float(np.nanmean(np.abs(phys_res_arr))) if phys_res_arr.size > 0 else float("nan")

    breakdown = breakdown_by_bins(test_df.copy(), preds_amp_arr, true_amp)

    # Scatter plot
    latest_run_dir = os.path.dirname(ckpt) if "runs" in ckpt else os.path.join("runs", find_latest_run(args.runs_dir))
    output_dir = args.output_dir or latest_run_dir
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    scatter_path = os.path.join(output_dir, "ampacity_scatter.png")
    plot_scatter_amp(true_amp, preds_amp_arr, scatter_path)

    # Per-sample CSV
    per_sample = test_df.copy()
    per_sample["pred_amp"] = preds_amp_arr
    per_sample["pred_temp"] = preds_temp_arr
    per_sample["physics_residual"] = phys_res_arr

    summary = {
        "checkpoint": ckpt,
        "n_test_samples": int(len(test_df)),
        "amp_metrics": amp_metrics,
        "temp_metrics": temp_metrics,
        "physics_residual_mae": phys_res_mae,
        "breakdown": breakdown,
        "scatter_plot": scatter_path,
        "assumed_current": args.assumed_current,
    }

    baseline = args.baseline
    comparison = {
        "baseline_ampacity_mae": baseline,
        "current_ampacity_mae": amp_metrics["mae"],
        "delta_vs_baseline": baseline - amp_metrics["mae"],
        "beats_baseline": amp_metrics["mae"] < baseline,
    }
    summary["comparison_vs_baseline"] = comparison

    txt_path, json_path, csv_path = save_results(output_dir, summary, per_sample)

    # Console summary
    print("\n=== Validation Summary ===")
    print(f"Test samples: {len(test_df)}")
    print(f"Ampacity MAE: {amp_metrics['mae']:.2f} A    RMSE: {amp_metrics['rmse']:.2f} A    R²: {amp_metrics['r2']:.3f}")
    print(f"Temperature MAE: {temp_metrics['mae']:.2f} °C    RMSE: {temp_metrics['rmse']:.2f} °C    R²: {temp_metrics['r2']:.3f}")
    print(f"Physics residual MAE: {phys_res_mae:.3f} W/m")
    print(f"Scatter plot: {scatter_path}")
    print(f"Saved results: {txt_path}, {json_path}, {csv_path}")

    if comparison["beats_baseline"]:
        print(f"✅ Ampacity MAE ({amp_metrics['mae']:.2f}A) is better than baseline {baseline}A by {comparison['delta_vs_baseline']:.2f}A")
    else:
        print(f"⚠️ Ampacity MAE ({amp_metrics['mae']:.2f}A) does NOT beat baseline {baseline}A (needs {comparison['delta_vs_baseline']:.2f}A improvement)")

    print("\nBreakdown by wind-speed bins:")
    for k, v in breakdown["wind_bins"].items():
        print(f"  {k}: {v}")
    print("\nBreakdown by temperature bins:")
    for k, v in breakdown["temp_bins"].items():
        print(f"  {k}: {v}")
    print("\nBreakdown by time of day:")
    for k, v in breakdown["time_of_day"].items():
        print(f"  {k}: {v}")

    print("\nValidation complete.")


if __name__ == "__main__":
    main()
