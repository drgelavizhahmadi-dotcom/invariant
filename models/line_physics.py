"""
Invariant-PIKAN: Adversarially-Robust Physics-Informed Neural Networks for Dynamic Line Rating
Copyright (C) 2025 Gelavizh Ahmadi / Invariant Research

This software is licensed under the Business Source License 1.1 (BSL 1.1).
Commercial production use requires a separate license agreement.
See LICENSE.txt for full terms.

"""

#!/usr/bin/env python3
"""
Learnable Per-Line Physics Parameters for InvariantPIKAN

Implements hierarchical Bayesian shrinkage for line-specific physics parameters:
- Global mean parameters shared across all lines
- Per-line deviations regularized toward zero
- Physical constraints via bounded transformations

Reference: Power-GNN (2021), Inverse PINNs, Differentiable Simulation
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Optional, Tuple
from dataclasses import dataclass


@dataclass
class LinePhysicsConfig:
    """Configuration for LinePhysicsParams"""
    num_lines: int = 10  # Number of unique transmission lines
    reg_weight: float = 0.01  # Regularization weight for deviations
    
    # Initial global means (log space for resistance, raw for others)
    init_resistance_factor_mean: float = 0.0  # log(1.0) = 0
    init_emissivity_mean: float = 0.8
    init_absorptivity_mean: float = 0.8
    
    # Initial global std (in log space for resistance)
    init_log_std: float = -2.0  # Small initial variance
    
    # Physical bounds
    resistance_factor_min: float = 0.3
    resistance_factor_max: float = 3.0
    emissivity_min: float = 0.3
    emissivity_max: float = 0.95
    absorptivity_min: float = 0.3
    absorptivity_max: float = 0.95


class LinePhysicsParams(nn.Module):
    """
    Learnable per-line physics parameters with hierarchical Bayesian shrinkage.
    
    Each line has parameters that deviate from a global mean, with regularization
    encouraging parameters to stay close to the mean unless data strongly suggests
    otherwise.
    
    Parameters:
        - resistance_factor: Multiplier on base resistance (positive)
        - emissivity: Radiative cooling efficiency [0.3, 0.95]
        - absorptivity: Solar heat absorption [0.3, 0.95]
    """
    
    def __init__(self, config: LinePhysicsConfig):
        super().__init__()
        self.config = config
        self.num_lines = config.num_lines
        
        # Global mean parameters (shared across all lines)
        # Stored in "raw" space before transformation to physical space
        self.global_raw_mean = nn.Parameter(torch.tensor([
            config.init_resistance_factor_mean,  # log space
            self._emissivity_to_raw(config.init_emissivity_mean),
            self._absorptivity_to_raw(config.init_absorptivity_mean)
        ], dtype=torch.float32))
        
        # Global log-std (controls how much lines can deviate)
        self.global_log_std = nn.Parameter(torch.full((3,), config.init_log_std))
        
        # Per-line deviations (regularized toward zero)
        # Shape: [num_lines, 3] for [resistance, emissivity, absorptivity]
        self.line_deviations = nn.Parameter(torch.zeros(self.num_lines, 3))
        
        # Initialize deviations with small random noise
        nn.init.normal_(self.line_deviations, mean=0.0, std=0.01)
        
    def _emissivity_to_raw(self, emissivity: float) -> float:
        """Convert emissivity [0.3, 0.95] to raw space for sigmoid."""
        # emissivity = sigmoid(raw) * 0.65 + 0.3
        # Solve for raw
        normalized = (emissivity - self.config.emissivity_min) / \
                     (self.config.emissivity_max - self.config.emissivity_min)
        # Clamp to avoid log(0)
        normalized = np.clip(normalized, 1e-6, 1 - 1e-6)
        return np.log(normalized / (1 - normalized))
    
    def _absorptivity_to_raw(self, absorptivity: float) -> float:
        """Convert absorptivity [0.3, 0.95] to raw space for sigmoid."""
        normalized = (absorptivity - self.config.absorptivity_min) / \
                     (self.config.absorptivity_max - self.config.absorptivity_min)
        normalized = np.clip(normalized, 1e-6, 1 - 1e-6)
        return np.log(normalized / (1 - normalized))
    
    def forward(self, line_ids: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Get physics parameters for given line IDs.
        
        Args:
            line_ids: Tensor of shape [batch] with integer line indices
            
        Returns:
            Dictionary with 'resistance_factor', 'emissivity', 'absorptivity'
            Each tensor has shape [batch]
        """
        # Ensure line_ids are valid indices
        line_ids = torch.clamp(line_ids, 0, self.num_lines - 1)
        
        # Get deviations for this batch
        batch_deviations = self.line_deviations[line_ids]  # [batch, 3]
        
        # Compute raw values: global_mean + deviation * exp(log_std)
        # This is a reparameterization trick similar to VAEs
        std = torch.exp(self.global_log_std)
        raw_values = self.global_raw_mean + batch_deviations * std
        
        # Transform to physical space
        # Resistance factor: exp(raw) -> positive, typically 0.3-3.0
        resistance_factor = torch.exp(raw_values[:, 0])
        resistance_factor = torch.clamp(
            resistance_factor, 
            self.config.resistance_factor_min,
            self.config.resistance_factor_max
        )
        
        # Emissivity: sigmoid(raw) * range + min
        emissivity = torch.sigmoid(raw_values[:, 1])
        emissivity = emissivity * (self.config.emissivity_max - self.config.emissivity_min) \
                     + self.config.emissivity_min
        
        # Absorptivity: sigmoid(raw) * range + min
        absorptivity = torch.sigmoid(raw_values[:, 2])
        absorptivity = absorptivity * (self.config.absorptivity_max - self.config.absorptivity_min) \
                       + self.config.absorptivity_min
        
        return {
            'resistance_factor': resistance_factor,
            'emissivity': emissivity,
            'absorptivity': absorptivity,
            'raw_values': raw_values  # For debugging
        }
    
    def regularization_loss(self) -> torch.Tensor:
        """
        Compute regularization loss to shrink deviations toward zero.
        
        This implements the hierarchical Bayesian prior: we want per-line
        deviations to be small unless the data strongly supports larger values.
        
        Returns:
            Scalar regularization loss
        """
        # L2 penalty on deviations (shrinkage toward zero)
        deviation_penalty = torch.mean(self.line_deviations ** 2)
        
        # Optional: Small L2 penalty on global std to prevent it from growing too large
        std_penalty = torch.mean(torch.exp(self.global_log_std))
        
        return self.config.reg_weight * (deviation_penalty + 0.01 * std_penalty)
    
    def get_global_params(self) -> Dict[str, float]:
        """Get the current global mean parameters in physical space."""
        with torch.no_grad():
            r_factor = np.exp(self.global_raw_mean[0].item())
            
            emiss = torch.sigmoid(self.global_raw_mean[1]).item()
            emiss = emiss * (self.config.emissivity_max - self.config.emissivity_min) \
                    + self.config.emissivity_min
            
            absorpt = torch.sigmoid(self.global_raw_mean[2]).item()
            absorpt = absorpt * (self.config.absorptivity_max - self.config.absorptivity_min) \
                      + self.config.absorptivity_min
            
            return {
                'resistance_factor': float(r_factor),
                'emissivity': float(emiss),
                'absorptivity': float(absorpt)
            }
    
    def get_line_params(self, line_id: int) -> Dict[str, float]:
        """Get parameters for a specific line in physical space."""
        with torch.no_grad():
            line_id_tensor = torch.tensor([line_id])
            params = self.forward(line_id_tensor)
            return {
                'resistance_factor': params['resistance_factor'][0].item(),
                'emissivity': params['emissivity'][0].item(),
                'absorptivity': params['absorptivity'][0].item()
            }
    
    def summary(self):
        """Print summary of learned parameters."""
        global_params = self.get_global_params()
        print("\n" + "="*60)
        print("LINE PHYSICS PARAMETERS SUMMARY")
        print("="*60)
        print(f"\nGlobal Mean Parameters:")
        print(f"  Resistance Factor: {global_params['resistance_factor']:.4f}")
        print(f"  Emissivity:        {global_params['emissivity']:.4f}")
        print(f"  Absorptivity:      {global_params['absorptivity']:.4f}")
        
        print(f"\nGlobal Log-Std: {torch.exp(self.global_log_std).cpu().detach().numpy()}")
        
        print(f"\nPer-Line Deviation Stats:")
        deviations = self.line_deviations.cpu().detach().numpy()
        print(f"  Mean abs deviation: {np.mean(np.abs(deviations)):.4f}")
        print(f"  Max abs deviation:  {np.max(np.abs(deviations)):.4f}")
        print(f"  Std deviation:      {np.std(deviations):.4f}")
        
        # Show first few lines
        print(f"\nFirst 5 Lines Parameters:")
        for i in range(min(5, self.num_lines)):
            params = self.get_line_params(i)
            print(f"  Line {i}: R_factor={params['resistance_factor']:.3f}, "
                  f"emiss={params['emissivity']:.3f}, "
                  f"absorpt={params['absorptivity']:.3f}")
        print("="*60)


def create_line_physics_for_dataset(line_ids: list, config: Optional[LinePhysicsConfig] = None) -> Tuple[LinePhysicsParams, Dict[str, int]]:
    """
    Create LinePhysicsParams and mapping from line identifiers to indices.
    
    Args:
        line_ids: List of line identifiers (strings or ints)
        config: Optional configuration
        
    Returns:
        Tuple of (line_physics_module, line_id_to_idx_mapping)
    """
    # Create unique mapping
    unique_lines = sorted(set(line_ids))
    line_to_idx = {line: idx for idx, line in enumerate(unique_lines)}
    
    if config is None:
        config = LinePhysicsConfig(num_lines=len(unique_lines))
    else:
        config.num_lines = len(unique_lines)
    
    line_physics = LinePhysicsParams(config)
    
    print(f"Created LinePhysicsParams for {len(unique_lines)} unique lines:")
    for line, idx in list(line_to_idx.items())[:10]:
        print(f"  {line} -> {idx}")
    if len(line_to_idx) > 10:
        print(f"  ... and {len(line_to_idx) - 10} more")
    
    return line_physics, line_to_idx


# Test function
if __name__ == "__main__":
    print("Testing LinePhysicsParams...")
    
    # Create module
    config = LinePhysicsConfig(num_lines=5, reg_weight=0.01)
    line_physics = LinePhysicsParams(config)
    
    # Test forward pass
    batch_line_ids = torch.tensor([0, 1, 2, 0, 3])
    params = line_physics(batch_line_ids)
    
    print("\nTest batch output:")
    for key, value in params.items():
        if key != 'raw_values':
            print(f"  {key}: {value.detach().numpy()}")
    
    # Test regularization
    reg_loss = line_physics.regularization_loss()
    print(f"\nRegularization loss: {reg_loss.item():.6f}")
    
    # Test summary
    line_physics.summary()
    
    print("\n✅ LinePhysicsParams test passed!")
