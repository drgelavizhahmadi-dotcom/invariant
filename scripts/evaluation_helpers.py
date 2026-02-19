#!/usr/bin/env python3
"""
Evaluation Helpers for Normalized (Ratio-Based) Models

CRITICAL: Models now predict DLR/SLR ratio (dimensionless), but we report
MAE in amps for interpretability. This module handles the conversion.

Usage:
    from scripts.evaluation_helpers import evaluate_ratio_model

    metrics = evaluate_ratio_model(
        predictions=pred_ratio,  # Model output (dimensionless ratio)
        actuals=true_ratio,      # Ground truth ratio
        region='VN',             # 'VN' or 'US'
    )

    # metrics contains both ratio-space and amp-space metrics
    print(f"MAE (amps): {metrics['mae_amps']:.1f} A")
    print(f"MAE (ratio): {metrics['mae_ratio']:.3f}")
"""

import numpy as np
import pandas as pd
from typing import Dict, Union, Tuple
import json

# Load SLR values from normalization params
def load_slr_values() -> Dict[str, float]:
    """Load documented SLR values from normalization params"""
    try:
        with open('data/processed/normalization_params.json', 'r') as f:
            params = json.load(f)
        return {
            'VN': params['VN_SLR'],
            'US': params['US_SLR'],
        }
    except FileNotFoundError:
        # Fallback to documented values
        print("⚠️  normalization_params.json not found, using documented defaults")
        return {
            'VN': 850.0,  # DynaLiRD Table 2
            'US': 1000.0,  # NREL documentation
        }


def ratio_to_amps(ratio: Union[np.ndarray, pd.Series, float],
                   region: str,
                   slr_values: Dict[str, float] = None) -> Union[np.ndarray, float]:
    """
    Convert DLR/SLR ratio to absolute amperes

    Args:
        ratio: DLR/SLR ratio (dimensionless)
        region: 'VN' or 'US'
        slr_values: Optional dict of SLR values. If None, loads from file.

    Returns:
        Ampacity in amps

    Example:
        >>> ratio_to_amps(1.403, 'VN')
        1192.55  # 1.403 × 850A
    """
    if slr_values is None:
        slr_values = load_slr_values()

    if region not in slr_values:
        raise ValueError(f"Unknown region '{region}'. Must be 'VN' or 'US'")

    slr = slr_values[region]
    return ratio * slr


def amps_to_ratio(amps: Union[np.ndarray, pd.Series, float],
                   region: str,
                   slr_values: Dict[str, float] = None) -> Union[np.ndarray, float]:
    """
    Convert absolute amperes to DLR/SLR ratio

    Args:
        amps: Ampacity in amps
        region: 'VN' or 'US'
        slr_values: Optional dict of SLR values. If None, loads from file.

    Returns:
        DLR/SLR ratio (dimensionless)

    Example:
        >>> amps_to_ratio(1193, 'VN')
        1.403  # 1193A / 850A
    """
    if slr_values is None:
        slr_values = load_slr_values()

    if region not in slr_values:
        raise ValueError(f"Unknown region '{region}'. Must be 'VN' or 'US'")

    slr = slr_values[region]
    return amps / slr


def evaluate_ratio_model(
    predictions: Union[np.ndarray, pd.Series],
    actuals: Union[np.ndarray, pd.Series],
    region: str,
    return_predictions: bool = False,
) -> Dict[str, float]:
    """
    Evaluate model predictions in both ratio and amp space

    Args:
        predictions: Predicted DLR/SLR ratios (dimensionless)
        actuals: Actual DLR/SLR ratios (dimensionless)
        region: 'VN' or 'US'
        return_predictions: If True, include pred_amps and actual_amps in output

    Returns:
        Dictionary with metrics:
        - mae_ratio: MAE in ratio units
        - rmse_ratio: RMSE in ratio units
        - mae_amps: MAE in amps (for reporting)
        - rmse_amps: RMSE in amps (for reporting)
        - bias_amps: Bias in amps
        - mape: Mean absolute percentage error
        - slr_used: SLR value used for conversion

    Example:
        >>> metrics = evaluate_ratio_model(pred_ratio, true_ratio, 'VN')
        >>> print(f"MAE: {metrics['mae_amps']:.1f} A")
        MAE: 81.5 A
    """
    # Convert to numpy arrays
    pred_ratio = np.asarray(predictions)
    true_ratio = np.asarray(actuals)

    # Load SLR values
    slr_values = load_slr_values()
    slr = slr_values[region]

    # Metrics in RATIO space (dimensionless)
    mae_ratio = np.mean(np.abs(pred_ratio - true_ratio))
    rmse_ratio = np.sqrt(np.mean((pred_ratio - true_ratio)**2))

    # Convert to AMPS for interpretability
    pred_amps = ratio_to_amps(pred_ratio, region, slr_values)
    true_amps = ratio_to_amps(true_ratio, region, slr_values)

    # Metrics in AMP space (for reporting in paper)
    mae_amps = np.mean(np.abs(pred_amps - true_amps))
    rmse_amps = np.sqrt(np.mean((pred_amps - true_amps)**2))
    bias_amps = np.mean(pred_amps - true_amps)

    # MAPE (dimensionless percentage)
    mape = np.mean(np.abs((true_amps - pred_amps) / true_amps)) * 100

    metrics = {
        'mae_ratio': float(mae_ratio),
        'rmse_ratio': float(rmse_ratio),
        'mae_amps': float(mae_amps),
        'rmse_amps': float(rmse_amps),
        'bias_amps': float(bias_amps),
        'mape': float(mape),
        'slr_used': float(slr),
        'region': region,
    }

    if return_predictions:
        metrics['pred_amps'] = pred_amps
        metrics['true_amps'] = true_amps
        metrics['pred_ratio'] = pred_ratio
        metrics['true_ratio'] = true_ratio

    return metrics


def compare_cross_region_transfer(
    vn_only_pred: np.ndarray,
    vn_only_true: np.ndarray,
    us_only_pred: np.ndarray,
    us_only_true: np.ndarray,
    vn_to_us_pred: np.ndarray,
    vn_to_us_true: np.ndarray,
    us_to_vn_pred: np.ndarray,
    us_to_vn_true: np.ndarray,
) -> pd.DataFrame:
    """
    Generate cross-region transfer comparison table (in AMPS for paper)

    Args:
        All inputs are DLR/SLR ratios (dimensionless)
        *_pred: Model predictions
        *_true: Ground truth

    Returns:
        DataFrame with MAE in amps for all experiments

    Example:
        >>> results = compare_cross_region_transfer(...)
        >>> print(results.to_markdown())
        | Experiment | Train Region | Test Region | MAE (A) |
        |------------|--------------|-------------|---------|
        | VN_only    | VN           | VN          | 81.5    |
        | ...
    """
    experiments = [
        ('VN_only', vn_only_pred, vn_only_true, 'VN', 'VN'),
        ('US_only', us_only_pred, us_only_true, 'US', 'US'),
        ('VN→US', vn_to_us_pred, vn_to_us_true, 'VN', 'US'),
        ('US→VN', us_to_vn_pred, us_to_vn_true, 'US', 'VN'),
    ]

    results = []
    for name, pred, true, train_region, test_region in experiments:
        metrics = evaluate_ratio_model(pred, true, test_region)

        results.append({
            'Experiment': name,
            'Train Region': train_region,
            'Test Region': test_region,
            'MAE (A)': metrics['mae_amps'],
            'RMSE (A)': metrics['rmse_amps'],
            'Bias (A)': metrics['bias_amps'],
            'MAPE (%)': metrics['mape'],
            'MAE (ratio)': metrics['mae_ratio'],
        })

    df = pd.DataFrame(results)

    # Calculate degradation for cross-region transfers
    vn_baseline = df[df['Experiment'] == 'VN_only']['MAE (A)'].values[0]
    us_baseline = df[df['Experiment'] == 'US_only']['MAE (A)'].values[0]

    df['Degradation (%)'] = 0.0
    df.loc[df['Experiment'] == 'VN→US', 'Degradation (%)'] = (
        (df.loc[df['Experiment'] == 'VN→US', 'MAE (A)'].values[0] - us_baseline) / us_baseline * 100
    )
    df.loc[df['Experiment'] == 'US→VN', 'Degradation (%)'] = (
        (df.loc[df['Experiment'] == 'US→VN', 'MAE (A)'].values[0] - vn_baseline) / vn_baseline * 100
    )

    return df


def print_evaluation_summary(metrics: Dict[str, float], experiment_name: str = "Model"):
    """
    Pretty-print evaluation metrics

    Args:
        metrics: Output from evaluate_ratio_model()
        experiment_name: Name of experiment for display
    """
    print(f"\n{'='*60}")
    print(f"{experiment_name} Evaluation")
    print(f"{'='*60}")
    print(f"\n📊 Metrics in AMPS (for reporting):")
    print(f"   MAE:  {metrics['mae_amps']:7.1f} A")
    print(f"   RMSE: {metrics['rmse_amps']:7.1f} A")
    print(f"   Bias: {metrics['bias_amps']:+7.1f} A")
    print(f"   MAPE: {metrics['mape']:7.1f}%")

    print(f"\n📐 Metrics in RATIO units (dimensionless):")
    print(f"   MAE:  {metrics['mae_ratio']:.3f}")
    print(f"   RMSE: {metrics['rmse_ratio']:.3f}")

    print(f"\n🔧 Conversion:")
    print(f"   Region: {metrics['region']}")
    print(f"   SLR used: {metrics['slr_used']:.0f} A")
    print(f"   Formula: pred_amps = pred_ratio × {metrics['slr_used']:.0f}")


# Example usage
if __name__ == "__main__":
    print("=" * 80)
    print("EVALUATION HELPERS - EXAMPLE USAGE")
    print("=" * 80)

    # Load SLR values
    slr_values = load_slr_values()
    print(f"\n✅ Loaded SLR values:")
    print(f"   VN_SLR: {slr_values['VN']:.0f} A")
    print(f"   US_SLR: {slr_values['US']:.0f} A")

    # Example 1: Convert ratio to amps
    print(f"\n" + "="*60)
    print("Example 1: Convert ratio → amps")
    print("="*60)

    vn_ratio = 1.403
    vn_amps = ratio_to_amps(vn_ratio, 'VN')
    print(f"\nVN ratio {vn_ratio:.3f} → {vn_amps:.0f} A")
    print(f"  (1.403 × {slr_values['VN']:.0f} = {vn_amps:.0f})")

    us_ratio = 1.629
    us_amps = ratio_to_amps(us_ratio, 'US')
    print(f"\nUS ratio {us_ratio:.3f} → {us_amps:.0f} A")
    print(f"  (1.629 × {slr_values['US']:.0f} = {us_amps:.0f})")

    # Example 2: Convert amps to ratio
    print(f"\n" + "="*60)
    print("Example 2: Convert amps → ratio")
    print("="*60)

    vn_amps_in = 1193
    vn_ratio_out = amps_to_ratio(vn_amps_in, 'VN')
    print(f"\nVN {vn_amps_in} A → ratio {vn_ratio_out:.3f}")
    print(f"  ({vn_amps_in} / {slr_values['VN']:.0f} = {vn_ratio_out:.3f})")

    # Example 3: Evaluate model predictions
    print(f"\n" + "="*60)
    print("Example 3: Evaluate model predictions")
    print("="*60)

    # Simulate predictions (ratio units)
    np.random.seed(42)
    n_samples = 1000
    true_ratio = np.random.normal(1.403, 0.25, n_samples)
    pred_ratio = true_ratio + np.random.normal(0, 0.1, n_samples)  # Add noise

    metrics = evaluate_ratio_model(pred_ratio, true_ratio, 'VN')
    print_evaluation_summary(metrics, "VN Model")

    print(f"\n💡 For paper, report: MAE = {metrics['mae_amps']:.1f} A")
    print(f"   (Not MAE = {metrics['mae_ratio']:.3f} ratio)")

    print("\n" + "="*80)
