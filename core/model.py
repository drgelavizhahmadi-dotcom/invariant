"""
Physics-Informed Neural Network for Dynamic Line Rating

This module implements the core PINN architecture that predicts:
1. Conductor temperature
2. Dynamic ampacity rating

The model is trained with a combined loss:
- Data loss: MSE between predictions and ground truth
- Physics loss: Violation of IEEE 738 heat balance

Author: Dr. Gelavizh Ahmadi
Copyright (c) 2026 Invariant Energy GmbH
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, Dict
import math


class SinusoidalPositionalEncoding(nn.Module):
    """
    Sinusoidal encoding for continuous input features
    
    Helps the network learn high-frequency patterns in the input space.
    Inspired by NeRF and Transformer positional encodings.
    """
    
    def __init__(self, num_frequencies: int = 4):
        super().__init__()
        self.num_frequencies = num_frequencies
        # Frequency bands: 2^0, 2^1, ..., 2^(L-1)
        self.register_buffer(
            'frequency_bands',
            2.0 ** torch.linspace(0, num_frequencies - 1, num_frequencies)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply sinusoidal encoding to input
        
        Args:
            x: Input tensor [batch, features]
            
        Returns:
            Encoded tensor [batch, features * (1 + 2 * num_frequencies)]
        """
        # Original features
        encoded = [x]
        
        # Add sin and cos at each frequency
        for freq in self.frequency_bands:
            encoded.append(torch.sin(freq * math.pi * x))
            encoded.append(torch.cos(freq * math.pi * x))
        
        return torch.cat(encoded, dim=-1)


class ResidualBlock(nn.Module):
    """
    Residual block with pre-activation (LayerNorm before activation)
    """
    
    def __init__(self, dim: int, dropout: float = 0.1):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.linear1 = nn.Linear(dim, dim)
        self.linear2 = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.GELU()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.norm(x)
        x = self.activation(self.linear1(x))
        x = self.dropout(x)
        x = self.linear2(x)
        x = self.dropout(x)
        return x + residual


class PhysicsDLR(nn.Module):
    """
    Physics-Informed Deep Learning model for Dynamic Line Rating
    
    Architecture:
    - Input encoding (optional sinusoidal)
    - MLP encoder with residual connections
    - Separate prediction heads for temperature and rating
    
    Input features (6):
        [T_ambient, wind_speed, wind_angle, solar_irradiance, current, line_resistance]
    
    Outputs:
        - Conductor temperature (°C)
        - Dynamic ampacity rating (A)
    """
    
    def __init__(
        self,
        input_dim: int = 6,
        hidden_dims: list = [128, 128, 64],
        dropout: float = 0.1,
        use_positional_encoding: bool = False,
        num_frequencies: int = 4,
        use_residual: bool = True,
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.use_positional_encoding = use_positional_encoding
        self.use_residual = use_residual
        
        # Optional positional encoding
        if use_positional_encoding:
            self.pos_encoder = SinusoidalPositionalEncoding(num_frequencies)
            encoder_input_dim = input_dim * (1 + 2 * num_frequencies)
        else:
            self.pos_encoder = None
            encoder_input_dim = input_dim
        
        # Build encoder
        layers = []
        prev_dim = encoder_input_dim
        
        for i, hidden_dim in enumerate(hidden_dims):
            # First layer: project to hidden dim
            if i == 0:
                layers.append(nn.Linear(prev_dim, hidden_dim))
                layers.append(nn.LayerNorm(hidden_dim))
                layers.append(nn.GELU())
                layers.append(nn.Dropout(dropout))
            else:
                if use_residual and prev_dim == hidden_dim:
                    # Use residual block if dimensions match
                    layers.append(ResidualBlock(hidden_dim, dropout))
                else:
                    layers.append(nn.Linear(prev_dim, hidden_dim))
                    layers.append(nn.LayerNorm(hidden_dim))
                    layers.append(nn.GELU())
                    layers.append(nn.Dropout(dropout))
            
            prev_dim = hidden_dim
        
        self.encoder = nn.Sequential(*layers)
        
        # Temperature prediction head
        self.temp_head = nn.Sequential(
            nn.Linear(hidden_dims[-1], 32),
            nn.GELU(),
            nn.Linear(32, 1),
        )
        
        # Ampacity prediction head
        self.rating_head = nn.Sequential(
            nn.Linear(hidden_dims[-1], 32),
            nn.GELU(),
            nn.Linear(32, 1),
            nn.Softplus(),  # Ensure positive output
        )
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights using Xavier/Glorot initialization"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(
        self, 
        x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass
        
        Args:
            x: Input tensor [batch, input_dim]
               Features: [T_ambient, wind_speed, wind_angle, solar, current, resistance]
            
        Returns:
            temperature: Predicted conductor temperature [batch, 1]
            rating: Predicted dynamic ampacity [batch, 1]
        """
        # Optional positional encoding
        if self.pos_encoder is not None:
            x = self.pos_encoder(x)
        
        # Encode
        features = self.encoder(x)
        
        # Predict
        temperature = self.temp_head(features)
        rating = self.rating_head(features)
        
        # Scale rating to reasonable range (100-2000 A)
        rating = rating * 500 + 100  # Softplus output scaled
        
        return temperature, rating
    
    def predict_with_uncertainty(
        self,
        x: torch.Tensor,
        n_samples: int = 20,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Monte Carlo dropout for uncertainty estimation
        
        Runs multiple forward passes with dropout enabled to estimate
        prediction uncertainty.
        
        Args:
            x: Input tensor [batch, input_dim]
            n_samples: Number of MC samples
            
        Returns:
            temp_mean: Mean temperature prediction
            temp_std: Temperature uncertainty (std dev)
            rating_mean: Mean rating prediction
            rating_std: Rating uncertainty (std dev)
        """
        was_training = self.training
        self.train()  # Enable dropout
        
        temps = []
        ratings = []
        
        with torch.no_grad():
            for _ in range(n_samples):
                temp, rating = self.forward(x)
                temps.append(temp)
                ratings.append(rating)
        
        temps = torch.stack(temps)
        ratings = torch.stack(ratings)
        
        temp_mean = temps.mean(dim=0)
        temp_std = temps.std(dim=0)
        rating_mean = ratings.mean(dim=0)
        rating_std = ratings.std(dim=0)
        
        if not was_training:
            self.eval()
        
        return temp_mean, temp_std, rating_mean, rating_std
    
    def count_parameters(self) -> int:
        """Count trainable parameters"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class PhysicsInformedLoss(nn.Module):
    """
    Combined data + physics loss function
    
    L_total = w_temp * L_temp + w_rating * L_rating + w_physics * L_physics
    
    Where:
        L_temp: MSE loss on temperature prediction
        L_rating: MSE loss on ampacity prediction
        L_physics: MSE of heat balance residual (physics violation)
    """
    
    def __init__(
        self,
        physics_weight: float = 0.3,
        temp_weight: float = 1.0,
        rating_weight: float = 0.5,
        physics_residual_scale: float = 0.01,  # Scale residual to similar magnitude
    ):
        super().__init__()
        self.physics_weight = physics_weight
        self.temp_weight = temp_weight
        self.rating_weight = rating_weight
        self.physics_residual_scale = physics_residual_scale
        self.mse = nn.MSELoss()
        self.huber = nn.SmoothL1Loss()
    
    def forward(
        self,
        pred_temp: torch.Tensor,
        pred_rating: torch.Tensor,
        true_temp: torch.Tensor,
        true_rating: torch.Tensor,
        physics_residual: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute combined loss
        
        Args:
            pred_temp: Predicted temperature [batch, 1]
            pred_rating: Predicted ampacity [batch, 1]
            true_temp: Ground truth temperature [batch, 1]
            true_rating: Ground truth ampacity [batch, 1]
            physics_residual: Heat balance residual [batch]
            
        Returns:
            total_loss: Weighted sum of losses
            metrics: Dictionary of individual loss values
        """
        # Data losses
        temp_loss = self.mse(pred_temp, true_temp)
        rating_loss = self.mse(pred_rating / 1000, true_rating / 1000)  # Scale for stability
        
        # Physics loss (scaled)
        physics_loss = torch.mean((physics_residual * self.physics_residual_scale) ** 2)
        
        # Combined loss
        total_loss = (
            self.temp_weight * temp_loss +
            self.rating_weight * rating_loss +
            self.physics_weight * physics_loss
        )
        
        # Metrics for logging
        metrics = {
            'total_loss': total_loss.item(),
            'temp_loss': temp_loss.item(),
            'rating_loss': rating_loss.item(),
            'physics_loss': physics_loss.item(),
            'physics_residual_mean': physics_residual.abs().mean().item(),
            'physics_residual_max': physics_residual.abs().max().item(),
        }
        
        return total_loss, metrics


class TemperatureOnlyLoss(nn.Module):
    """
    Simplified loss for temperature prediction only
    
    Useful for initial training or ablation studies.
    """
    
    def __init__(self, physics_weight: float = 0.5):
        super().__init__()
        self.physics_weight = physics_weight
        self.mse = nn.MSELoss()
    
    def forward(
        self,
        pred_temp: torch.Tensor,
        true_temp: torch.Tensor,
        physics_residual: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        
        data_loss = self.mse(pred_temp, true_temp)
        physics_loss = torch.mean(physics_residual ** 2) * 0.0001
        
        total_loss = data_loss + self.physics_weight * physics_loss
        
        metrics = {
            'total_loss': total_loss.item(),
            'data_loss': data_loss.item(),
            'physics_loss': physics_loss.item(),
        }
        
        return total_loss, metrics


def create_model(
    config: Optional[Dict] = None,
    device: Optional[torch.device] = None,
) -> PhysicsDLR:
    """
    Factory function to create model with optional config
    
    Args:
        config: Model configuration dict
        device: Target device
        
    Returns:
        Initialized PhysicsDLR model
    """
    default_config = {
        'input_dim': 6,
        'hidden_dims': [128, 128, 64],
        'dropout': 0.1,
        'use_positional_encoding': False,
        'use_residual': True,
    }
    
    if config:
        default_config.update(config)
    
    model = PhysicsDLR(**default_config)
    
    if device:
        model = model.to(device)
    
    return model


# Quick test
if __name__ == "__main__":
    # Create model
    model = PhysicsDLR(input_dim=6, hidden_dims=[128, 128, 64])
    print(f"Model parameters: {model.count_parameters():,}")
    
    # Test forward pass
    x = torch.randn(32, 6)
    temp, rating = model(x)
    print(f"Temperature shape: {temp.shape}")
    print(f"Rating shape: {rating.shape}")
    print(f"Temperature range: [{temp.min():.1f}, {temp.max():.1f}]")
    print(f"Rating range: [{rating.min():.0f}, {rating.max():.0f}]")
    
    # Test uncertainty
    temp_mean, temp_std, rating_mean, rating_std = model.predict_with_uncertainty(x)
    print(f"Temperature uncertainty: {temp_std.mean():.2f}°C")
    print(f"Rating uncertainty: {rating_std.mean():.0f}A")
