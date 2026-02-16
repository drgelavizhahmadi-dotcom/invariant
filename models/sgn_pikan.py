# models/sgn_pikan.py

import torch
import torch.nn as nn
from .hwf_pikan_v2 import HWFPIKANV2

class SGNPIKAN(HWFPIKANV2):
    """
    SGN‑PIKAN: HWF‑PIKAN with Sketchy Natural Gradient support.

    This model subclasses the canonical HWFPIKANV2 implementation and adds:
    - A compatibility `forward(weather_tensor, timestamp)` wrapper used by the
      SGN training script
    - A `compute_residual_jacobian` method that returns flattened gradients of
      the scalar physics residual w.r.t. model parameters
    - Optional hooks for sketch-mode (optimizer integration)
    """

    def __init__(self, physics_engine=None, config=None):
        # Accept an optional physics_engine (stored for optimizer/hooks) but
        # forward only the model configuration to the HWFPIKANV2 constructor.
        cfg = {} if config is None else dict(config)
        hwf_kwargs = {
            'input_dim': cfg.get('input_dim', 4),
            'hidden_dim': cfg.get('hidden_dim', 32),
            'output_dim': cfg.get('output_dim', 2),
            'fourier_bands': cfg.get('fourier_bands', 16),
            'wavelet_scales': cfg.get('wavelet_scales', 4),
            'kan_grid': cfg.get('kan_grid', 5),
            'kan_k': cfg.get('kan_k', 3),
        }
        super().__init__(**hwf_kwargs)
        # Keep reference to the physics engine (optional, may be None)
        self.physics_engine = physics_engine
        self.sketch_mode = False  # Flag to indicate when sketch is active

    def forward(self, weather_tensor, timestamp=None):
        """
        Compatibility wrapper: accept (weather_tensor, timestamp) as used by the
        SGN training script and convert into the (x, weather_dict) pair expected
        by the upstream HWFPIKANV2.forward implementation.
        """
        if weather_tensor.dim() == 1:
            weather_tensor = weather_tensor.unsqueeze(0)
        if weather_tensor.shape[-1] < 4:
            raise ValueError("weather_tensor must have at least 4 columns: [T_amb, wind_speed, wind_angle, solar]")
        weather_dict = {
            'T_amb': weather_tensor[:, 0],
            'wind_speed': weather_tensor[:, 1],
            'solar': weather_tensor[:, 3]
        }
        return super().forward(weather_tensor, weather_dict)

    def compute_residual_jacobian(self, weather, timestamp=None, targets=None):
        """
        Return flattened gradient (1-D tensor) of scalar physics residual
        w.r.t. all trainable parameters. This is a simple (correct) implementation
        suitable for smoke tests and the sketchy-optimizer placeholder.
        """
        # Prepare inputs / weather_dict
        if isinstance(weather, dict):
            weather_dict = weather
            x = torch.stack([weather_dict['T_amb'], weather_dict['wind_speed'],
                             weather_dict.get('wind_angle', torch.zeros_like(weather_dict['T_amb'])),
                             weather_dict.get('solar', torch.zeros_like(weather_dict['T_amb']))], dim=-1)
        else:
            x = weather
            weather_dict = {'T_amb': x[:, 0], 'wind_speed': x[:, 1], 'solar': x[:, 3] if x.shape[1] > 3 else x[:, 2]}

        # Ensure gradients are tracked for parameters
        for p in self.parameters():
            p.requires_grad_(True)

        # Forward pass and scalar residual
        out = self.forward(x, timestamp)
        T_pred = out['temperature']
        I_pred = out['ampacity']
        residual = self.compute_physics_residual(T_pred, I_pred, weather_dict)
        if residual.dim() > 0:
            residual = residual.sum()

        params = [p for p in self.parameters() if p.requires_grad]
        grads = torch.autograd.grad(residual, params, retain_graph=False, create_graph=False, allow_unused=True)

        flat_grads = []
        for g, p in zip(grads, params):
            if g is None:
                flat_grads.append(torch.zeros(p.numel(), device=p.device))
            else:
                flat_grads.append(g.reshape(-1))
        return torch.cat(flat_grads)

    def set_sketch_mode(self, active=True):
        """Toggle sketch mode for optimizer."""
        self.sketch_mode = active


def create_sgn_pikan(physics_engine, config=None):
    """
    Factory function to create an SGN‑PIKAN model.

    Args:
        physics_engine: Your IEEE738HeatBalance instance
        config: Optional dict with keys:
            - 'input_dim' (default 4)
            - 'fourier_bands' (default 16)
            - 'wavelet_scales' (default 4)
            - 'hidden_dim' (default 32)  # smaller for quick experiments
            - 'kan_grid' (default 5)
            - 'kan_k' (default 3)
    """
    default_config = {
        'input_dim': 4,
        'fourier_bands': 16,
        'wavelet_scales': 4,
        'hidden_dim': 32,   # smaller for faster experiments
        'kan_grid': 5,
        'kan_k': 3,
    }
    if config:
        default_config.update(config)

    return SGNPIKAN(physics_engine, default_config)
