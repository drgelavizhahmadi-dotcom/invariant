"""
Invariant-PIKAN: Adversarially-Robust Physics-Informed Neural Networks for Dynamic Line Rating
Copyright (C) 2025 Gelavizh Ahmadi / Invariant Research

This software is licensed under the Business Source License 1.1 (BSL 1.1).
Commercial production use requires a separate license agreement.
See LICENSE.txt for full terms.

"""

"""
Invariant Core Engine
Physics-Informed AI for Dynamic Line Rating
"""

__version__ = "0.1.0"
__author__ = "Dr. Gelavizh Ahmadi"
__email__ = "gelavizh@invariant.energy"

from .physics import IEEE738HeatBalance, physics_loss_fn
from .model import PhysicsDLR, PhysicsInformedLoss
from .data import SyntheticDLRDataset, create_dataloaders
from .inference import DLRPredictor

__all__ = [
    "IEEE738HeatBalance",
    "physics_loss_fn",
    "PhysicsDLR",
    "PhysicsInformedLoss",
    "SyntheticDLRDataset",
    "create_dataloaders",
    "DLRPredictor",
]
