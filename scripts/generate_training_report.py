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
generate_training_report.py

Create a comprehensive training report for the latest InvariantPIKAN v2 run.

Outputs saved to the latest run directory as `TRAINING_REPORT.md` along with
`training_curves.png` (reused or regenerated) and the validation artifacts.

Usage:
    python -m scripts.generate_training_report
"""
import glob
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# tensorboard EventAccumulator (optional)
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


def read_history_from_run(run_dir: str) -> Optional[Dict[str, Any]]:
    # Prefer explicit history.csv
    hist_csv = os.path.join(run_dir, "history.csv")
    if os.path.exists(hist_csv):
        try:
            df = pd.read_csv(hist_csv)
            # Convert to a dict-of-lists compatible with plotting helper
            history = {c: df[c].tolist() for c in df.columns}
            return history
        except Exception:
            pass

    # Try to extract `history` from best_model.pt
    try:
        import torch
        ckpt = find_best_checkpoint(run_dir)
        data = torch.load(ckpt, map_location="cpu")
        if isinstance(data, dict) and "history" in data:
            return data["history"]
    except Exception:
        pass

    # Last-resort: reconstruct from TensorBoard event file
    if _HAS_EA:
        ev_files = glob.glob(os.path.join(run_dir, "**", "events.out.tfevents.*"), recursive=True)
        if ev_files:
            ev = max(ev_files, key=os.path.getmtime)
            try:
                ea = EventAccumulator(ev, size_guidance={"scalars": 0})
                ea.Reload()
                tags = ea.Tags().get("scalars", [])
                history: Dict[str, Any] = {"epoch": []}
                # collect a few expected tags if present
                tag_map = {
                    "Loss/train": "train_loss",
                    "Loss/temp": "train_temp_loss",
                    "Loss/amp": "train_amp_loss",
                    "Loss/physics": "train_physics_loss",
                    "Metrics/temp_mae": "val_temp_mae",
                    "Metrics/amp_mae": "val_amp_mae",
                    "Metrics/physics_mae": "val_physics_mae",
                    "LR": "lr",
                }
                # build epoch-aligned series by step
                steps = set()
                series = {}
                for tb_tag, out_key in tag_map.items():
                    if tb_tag in tags:
                        vals = ea.Scalars(tb_tag)
                        series[out_key] = {int(v.step): float(v.value) for v in vals}
                        steps.update(series[out_key].keys())
                if not series:
                    return None
                steps_sorted = sorted(steps)
                history = {k: [] for k in ["epoch"] + list(series.keys()) + ["lr"]}
                for step in steps_sorted:
                    history["epoch"].append(step)
                    for k in series.keys():
                        history[k].append(series[k].get(step, float("nan")))
                    # lr may be missing
                    history.setdefault("lr", []).append(float(series.get("lr", {}).get(step, float("nan"))))
                return history
            except Exception:
                return None
    return None


def plot_training_curves(history: Dict[str, Any], out_path: str):
    # try to follow the same layout as training script
    try:
        epochs = history.get("epoch", list(range(len(history.get("train_loss", [])))))
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))

        # Training loss
        axes[0, 0].plot(epochs, history.get("train_loss", []))
        axes[0, 0].set_title("Training Loss")
        axes[0, 0].set_xlabel("Epoch")
        axes[0, 0].grid(True, alpha=0.3)

        # Loss components
        axes[0, 1].plot(epochs, history.get("train_temp_loss", []), label="Temperature")
        axes[0, 1].plot(epochs, history.get("train_amp_loss", []), label="Ampacity")
        axes[0, 1].plot(epochs, history.get("train_physics_loss", []), label="Physics")
        axes[0, 1].set_title("Loss Components")
        axes[0, 1].set_xlabel("Epoch")
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)

        # Validation Temperature MAE
        axes[0, 2].plot(epochs, history.get("val_temp_mae", []))
        axes[0, 2].set_title("Validation Temperature MAE")
        axes[0, 2].set_xlabel("Epoch")
        axes[0, 2].set_ylabel("°C")
        axes[0, 2].grid(True, alpha=0.3)

        # Validation Ampacity MAE
        axes[1, 0].plot(epochs, history.get("val_amp_mae", []))
        axes[1, 0].set_title("Validation Ampacity MAE")
        axes[1, 0].set_xlabel("Epoch")
        axes[1, 0].set_ylabel("A")
        axes[1, 0].grid(True, alpha=0.3)

        # Validation Physics MAE
        axes[1, 1].plot(epochs, history.get("val_physics_mae", []))
        axes[1, 1].set_title("Validation Physics MAE")
        axes[1, 1].set_xlabel("Epoch")
        axes[1, 1].set_ylabel("°C")
        axes[1, 1].grid(True, alpha=0.3)

        # Learning rate
        axes[1, 2].plot(epochs, history.get("lr", []))
        axes[1, 2].set_title("Learning Rate")
        axes[1, 2].set_xlabel("Epoch")
        axes[1, 2].set_yscale("log")
        axes[1, 2].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(out_path, dpi=150)
        plt.close()
        return True
    except Exception:
        return False


def estimate_run_duration(run_dir: str) -> Optional[float]:
    # Prefer TensorBoard wall_time if available
    if _HAS_EA:
        ev_files = glob.glob(os.path.join(run_dir, "**", "events.out.tfevents.*"), recursive=True)
        if ev_files:
            ev = max(ev_files, key=os.path.getmtime)
            try:
                ea = EventAccumulator(ev, size_guidance={"scalars": 0})
                ea.Reload()
                tags = ea.Tags().get("scalars", [])
                # pick any scalar with data
                for t in ["Loss/train", "Metrics/amp_mae", "LR"]:
                    if t in tags:
                        vals = ea.Scalars(t)
                        if len(vals) >= 2:
                            start = float(vals[0].wall_time)
                            end = float(vals[-1].wall_time)
                            return max(0.0, end - start)
            except Exception:
                pass
    # Fallback to filesystem times
    mtimes = [os.path.getmtime(p) for p in glob.glob(os.path.join(run_dir, "**"), recursive=True) if os.path.isfile(p)]
    if not mtimes:
        return None
    return max(mtimes) - min(mtimes)


def load_model_config(ckpt_path: str, run_dir: str) -> Dict[str, Any]:
    # Try checkpoint 'config' first
    try:
        import torch
        data = torch.load(ckpt_path, map_location="cpu")
        if isinstance(data, dict) and "config" in data:
            return data["config"]
    except Exception:
        pass

    # Try a config.json in run_dir
    cfg_json = os.path.join(run_dir, "config.json")
    if os.path.exists(cfg_json):
        try:
            with open(cfg_json, "r") as f:
                return json.load(f)
        except Exception:
            pass

    return {"note": "config not found in checkpoint or run dir"}


def run_validation_for_run(ckpt: str, run_dir: str) -> Dict[str, Any]:
    # Call the existing validation script; it will write validation_results.json into run_dir
    cmd = [sys.executable, "-m", "scripts.validate_best_model", "--model-path", ckpt, "--output-dir", run_dir]
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"Validation script failed: {e}")
    # Read the produced JSON
    json_path = os.path.join(run_dir, "validation_results.json")
    if not os.path.exists(json_path):
        raise FileNotFoundError("validation_results.json not found after running validation")
    with open(json_path, "r") as f:
        return json.load(f)


def make_markdown_report(run_dir: str, cfg: Dict[str, Any], history: Optional[Dict[str, Any]], val_summary: Dict[str, Any], duration_s: Optional[float], baseline: float = 308.0) -> str:
    # metrics
    amp_mae = val_summary.get("amp_metrics", {}).get("mae", float("nan"))
    temp_mae = val_summary.get("temp_metrics", {}).get("mae", float("nan"))
    phys_res = val_summary.get("physics_residual_mae", float("nan"))

    delta = baseline - amp_mae
    pct = (delta / baseline * 100.0) if baseline != 0 else float("nan")

    amp_status = "✅" if amp_mae < 280.0 else "❌"
    temp_status = "✅" if temp_mae < 1.7 else "❌"
    phys_status = "✅" if phys_res < 0.5 else "❌"

    duration_h = (duration_s / 3600.0) if duration_s is not None else None
    timestamp = datetime.utcnow().isoformat() + "Z"

    training_curves_path = os.path.join(run_dir, "training_curves.png")
    if not os.path.exists(training_curves_path):
        # attempt to save a fallback placeholder
        training_curves_path = "training_curves.png"

    md_lines = []
    md_lines.append("# InvariantPIKAN v2 Training Report")
    md_lines.append("")
    md_lines.append("## Summary")
    md_lines.append("")
    md_lines.append(f"- Training completed: {timestamp}")
    md_lines.append(f"- Duration: {duration_h:.2f} hours" if duration_h is not None else "- Duration: unknown")
    md_lines.append(f"- Best validation ampacity MAE: {amp_mae:.2f} A")
    md_lines.append(f"- Improvement vs baseline (308A): {delta:+.2f} A ({pct:+.2f}%)")
    md_lines.append("")
    md_lines.append("## Performance Metrics")
    md_lines.append("")
    md_lines.append("| Metric | Value | Target | Status |")
    md_lines.append("|--------|-------|--------|--------|")
    md_lines.append(f"| Ampacity MAE | {amp_mae:.2f} A | <280A | {amp_status} |")
    md_lines.append(f"| Temperature MAE | {temp_mae:.2f} °C | <1.7°C | {temp_status} |")
    md_lines.append(f"| Physics Residual | {phys_res:.3f} | <0.5 | {phys_status} |")
    md_lines.append("")
    md_lines.append("## Final model configuration")
    md_lines.append("")
    md_lines.append("```json")
    md_lines.append(json.dumps(cfg, indent=2))
    md_lines.append("```")
    md_lines.append("")
    md_lines.append("## Training Curves")
    md_lines.append("")
    md_lines.append(f"![Training curves]({os.path.basename(training_curves_path)})")
    md_lines.append("")
    md_lines.append("## Next Steps")
    md_lines.append("")
    md_lines.append("- [ ] Deploy to production")
    md_lines.append("- [ ] Run pilot")
    md_lines.append("- [ ] Document for investors")
    md_lines.append("")

    return "\n".join(md_lines)


def main(runs_dir: str = "runs", baseline: float = 308.0):
    latest = find_latest_run(runs_dir)
    print(f"Latest run: {latest}")

    ckpt = find_best_checkpoint(latest)
    print(f"Best checkpoint: {ckpt}")

    # History (try to reuse existing training_curves.png if present)
    history = read_history_from_run(latest)
    curves_path = os.path.join(latest, "training_curves.png")
    if not os.path.exists(curves_path) and history is not None:
        ok = plot_training_curves(history, curves_path)
        if ok:
            print(f"Wrote training curves to: {curves_path}")

    # Model config
    cfg = load_model_config(ckpt, latest)

    # Estimate duration
    duration_s = estimate_run_duration(latest)

    # Run validation (reuses existing validation script)
    print("Running validation to collect final metrics...")
    val_summary = run_validation_for_run(ckpt, latest)

    # Generate markdown
    md = make_markdown_report(latest, cfg, history, val_summary, duration_s, baseline=baseline)
    out_md = os.path.join(latest, "TRAINING_REPORT.md")
    with open(out_md, "w") as f:
        f.write(md)

    print(f"Saved training report: {out_md}")
    print("Done.")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate training report for latest run")
    parser.add_argument("--runs-dir", default="runs")
    parser.add_argument("--baseline", type=float, default=308.0)
    args = parser.parse_args()
    main(runs_dir=args.runs_dir, baseline=args.baseline)
