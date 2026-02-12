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
from .train import train, get_device
from .inference import DLRPredictor

__all__ = [
    "IEEE738HeatBalance",
    "physics_loss_fn",
    "PhysicsDLR",
    "PhysicsInformedLoss",
    "SyntheticDLRDataset",
    "create_dataloaders",
    "train",
    "get_device",
    "DLRPredictor",
]
