"""
Invariant-PIKAN: Adversarially-Robust Physics-Informed Neural Networks for Dynamic Line Rating
Copyright (C) 2025 Gelavizh Ahmadi / Invariant Research

This software is licensed under the Business Source License 1.1 (BSL 1.1).
Commercial production use requires a separate license agreement.
See LICENSE.txt for full terms.

DISCLAIMER: This implementation is independent of concurrent academic work on
HWF-PIKAN for plasma physics (Heravifard et al., Sharif University, 2025).
"""

"""
Tests for synthetic data sampling and low‑wind augmentation
"""
import numpy as np
import torch

from core.data import SyntheticDLRDataset, DataConfig


def test_low_wind_oversample_fraction():
    """Ensure synthetic data contains >=30% samples with wind < 1 m/s"""
    cfg = DataConfig()
    cfg.low_wind_fraction = 0.35

    ds = SyntheticDLRDataset(n_samples=10000, config=cfg, seed=123)

    frac_low = np.mean(ds.wind_speed < 1.0)

    assert frac_low >= 0.30, f"Low-wind fraction too small: {frac_low:.2f}"
