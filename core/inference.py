"""
Inference API for Invariant DLR

Production-ready inference with:
- Model loading and caching
- Batch prediction
- Uncertainty estimation
- Physics compliance validation

Author: Dr. Gelavizh Ahmadi
Copyright (c) 2026 Invariant Energy GmbH
"""

import torch
import numpy as np
from typing import Dict, Optional, Tuple, Union, List
from pathlib import Path
from dataclasses import dataclass
import json

from .model import PhysicsDLR
from .physics import IEEE738HeatBalance
from .data import InputNormalizer


@dataclass
class DLRPrediction:
    """
    Dynamic Line Rating prediction result
    """
    # Primary outputs
    conductor_temperature: float  # °C
    dynamic_rating: float  # Amperes
    
    # Physics compliance
    physics_residual: float  # W/m (should be near 0)
    is_physics_compliant: bool  # True if residual < threshold
    
    # Uncertainty (if requested)
    temperature_uncertainty: Optional[float] = None
    rating_uncertainty: Optional[float] = None
    
    # Comparison to static rating
    static_rating: Optional[float] = None
    capacity_gain_percent: Optional[float] = None
    
    def to_dict(self) -> Dict:
        return {
            'conductor_temperature_C': round(self.conductor_temperature, 2),
            'dynamic_rating_A': round(self.dynamic_rating, 0),
            'physics_residual_Wm': round(self.physics_residual, 3),
            'is_physics_compliant': self.is_physics_compliant,
            'temperature_uncertainty_C': round(self.temperature_uncertainty, 2) if self.temperature_uncertainty else None,
            'rating_uncertainty_A': round(self.rating_uncertainty, 0) if self.rating_uncertainty else None,
            'static_rating_A': round(self.static_rating, 0) if self.static_rating else None,
            'capacity_gain_percent': round(self.capacity_gain_percent, 1) if self.capacity_gain_percent else None,
        }


class DLRPredictor:
    """
    Production-ready DLR prediction interface
    
    Usage:
        predictor = DLRPredictor.from_checkpoint("models/best_model.pt")
        result = predictor.predict(
            T_ambient=25.0,
            wind_speed=5.0,
            solar_irradiance=800.0,
            current=800.0
        )
        print(f"Dynamic Rating: {result.dynamic_rating:.0f} A")
    """
    
    def __init__(
        self,
        model: PhysicsDLR,
        physics: IEEE738HeatBalance,
        normalizer: InputNormalizer,
        device: torch.device,
        physics_compliance_threshold: float = 15.0,  # W/m
    ):
        self.model = model
        self.physics = physics
        self.normalizer = normalizer
        self.device = device
        self.physics_compliance_threshold = physics_compliance_threshold
        
        # Precompute static rating for comparison
        self._static_rating = physics.static_rating()
    
    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: str,
        device: Optional[torch.device] = None,
    ) -> 'DLRPredictor':
        """
        Load predictor from saved checkpoint
        
        Args:
            checkpoint_path: Path to model checkpoint
            device: Target device (auto-detected if None)
        """
        # Auto-detect device
        if device is None:
            if torch.backends.mps.is_available():
                device = torch.device("mps")
            elif torch.cuda.is_available():
                device = torch.device("cuda")
            else:
                device = torch.device("cpu")
        
        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=device)
        
        # Extract config
        config = checkpoint.get('config', {})
        hidden_dims = config.get('hidden_dims', [128, 128, 64])
        dropout = config.get('dropout', 0.1)
        
        # Create model
        model = PhysicsDLR(
            input_dim=6,
            hidden_dims=hidden_dims,
            dropout=dropout,
        )
        model.load_state_dict(checkpoint['model_state_dict'])
        model = model.to(device)
        model.eval()
        
        # Create physics
        physics = IEEE738HeatBalance().to(device)
        
        # Create normalizer
        normalizer = InputNormalizer()
        if 'normalizer' in checkpoint:
            normalizer.mean = np.array(checkpoint['normalizer']['mean'])
            normalizer.std = np.array(checkpoint['normalizer']['std'])
        
        return cls(model, physics, normalizer, device)
    
    def _prepare_input(
        self,
        T_ambient: float,
        wind_speed: float,
        solar_irradiance: float,
        current: float,
        wind_angle: float = 45.0,
        resistance: Optional[float] = None,
    ) -> torch.Tensor:
        """Prepare normalized input tensor"""
        if resistance is None:
            resistance = self.physics.R_ref.item()
        
        x = np.array([[
            T_ambient,
            wind_speed,
            wind_angle,
            solar_irradiance,
            current,
            resistance,
        ]])
        
        x_norm = self.normalizer.transform(x)
        return torch.tensor(x_norm, dtype=torch.float32, device=self.device)
    
    @torch.no_grad()
    def predict(
        self,
        T_ambient: float,
        wind_speed: float,
        solar_irradiance: float,
        current: float,
        wind_angle: float = 45.0,
        include_uncertainty: bool = False,
        include_static_comparison: bool = True,
    ) -> DLRPrediction:
        """
        Make a single DLR prediction
        
        Args:
            T_ambient: Ambient temperature (°C)
            wind_speed: Wind speed (m/s)
            solar_irradiance: Solar irradiance (W/m²)
            current: Line current (A)
            wind_angle: Wind angle to conductor (degrees, default 45)
            include_uncertainty: Include MC dropout uncertainty
            include_static_comparison: Include comparison to static rating
            
        Returns:
            DLRPrediction with results
        """
        x = self._prepare_input(T_ambient, wind_speed, solar_irradiance, current, wind_angle)
        
        # Predict
        if include_uncertainty:
            temp_mean, temp_std, rating_mean, rating_std = self.model.predict_with_uncertainty(x)
            pred_temp = temp_mean.item()
            pred_rating = rating_mean.item()
            temp_uncertainty = temp_std.item()
            rating_uncertainty = rating_std.item()
        else:
            pred_temp_t, pred_rating_t = self.model(x)
            pred_temp = pred_temp_t.item()
            pred_rating = pred_rating_t.item()
            temp_uncertainty = None
            rating_uncertainty = None
        
        # Calculate physics residual
        residual = self.physics.heat_balance_residual(
            current=torch.tensor([current], device=self.device),
            T_conductor=torch.tensor([pred_temp], device=self.device),
            T_ambient=torch.tensor([T_ambient], device=self.device),
            wind_speed=torch.tensor([wind_speed], device=self.device),
            solar_irradiance=torch.tensor([solar_irradiance], device=self.device),
            wind_angle=torch.tensor([wind_angle], device=self.device),
        ).item()
        
        is_compliant = abs(residual) < self.physics_compliance_threshold
        
        # Static comparison
        if include_static_comparison:
            static_rating = self._static_rating
            capacity_gain = ((pred_rating - static_rating) / static_rating) * 100
        else:
            static_rating = None
            capacity_gain = None
        
        return DLRPrediction(
            conductor_temperature=pred_temp,
            dynamic_rating=pred_rating,
            physics_residual=residual,
            is_physics_compliant=is_compliant,
            temperature_uncertainty=temp_uncertainty,
            rating_uncertainty=rating_uncertainty,
            static_rating=static_rating,
            capacity_gain_percent=capacity_gain,
        )
    
    @torch.no_grad()
    def predict_batch(
        self,
        conditions: List[Dict[str, float]],
    ) -> List[DLRPrediction]:
        """
        Batch prediction for multiple conditions
        
        Args:
            conditions: List of dicts with keys:
                T_ambient, wind_speed, solar_irradiance, current, (wind_angle)
                
        Returns:
            List of DLRPrediction objects
        """
        results = []
        
        # Prepare batch input
        x_list = []
        for cond in conditions:
            x = self._prepare_input(
                T_ambient=cond['T_ambient'],
                wind_speed=cond['wind_speed'],
                solar_irradiance=cond['solar_irradiance'],
                current=cond['current'],
                wind_angle=cond.get('wind_angle', 45.0),
            )
            x_list.append(x)
        
        x_batch = torch.cat(x_list, dim=0)
        
        # Predict
        pred_temp, pred_rating = self.model(x_batch)
        
        # Calculate physics residuals
        for i, cond in enumerate(conditions):
            residual = self.physics.heat_balance_residual(
                current=torch.tensor([cond['current']], device=self.device),
                T_conductor=pred_temp[i],
                T_ambient=torch.tensor([cond['T_ambient']], device=self.device),
                wind_speed=torch.tensor([cond['wind_speed']], device=self.device),
                solar_irradiance=torch.tensor([cond['solar_irradiance']], device=self.device),
                wind_angle=torch.tensor([cond.get('wind_angle', 45.0)], device=self.device),
            ).item()
            
            results.append(DLRPrediction(
                conductor_temperature=pred_temp[i].item(),
                dynamic_rating=pred_rating[i].item(),
                physics_residual=residual,
                is_physics_compliant=abs(residual) < self.physics_compliance_threshold,
                static_rating=self._static_rating,
                capacity_gain_percent=((pred_rating[i].item() - self._static_rating) / self._static_rating) * 100,
            ))
        
        return results
    
    def get_model_info(self) -> Dict:
        """Get model information"""
        return {
            'parameters': self.model.count_parameters(),
            'device': str(self.device),
            'static_rating_A': round(self._static_rating, 0),
            'physics_compliance_threshold_Wm': self.physics_compliance_threshold,
        }


# Convenience function
def load_predictor(
    checkpoint_path: str = "models/best_model.pt"
) -> DLRPredictor:
    """
    Quick load of predictor from checkpoint
    """
    return DLRPredictor.from_checkpoint(checkpoint_path)


# Demo function
def demo_prediction():
    """Run a demo prediction (requires trained model)"""
    try:
        predictor = load_predictor()
        
        # Test conditions
        result = predictor.predict(
            T_ambient=25.0,
            wind_speed=5.0,
            solar_irradiance=800.0,
            current=800.0,
            include_uncertainty=True,
            include_static_comparison=True,
        )
        
        print("\n🔌 Invariant DLR Prediction")
        print("=" * 40)
        print(f"Conductor Temperature: {result.conductor_temperature:.1f}°C")
        print(f"Dynamic Rating: {result.dynamic_rating:.0f} A")
        print(f"Static Rating: {result.static_rating:.0f} A")
        print(f"Capacity Gain: {result.capacity_gain_percent:+.1f}%")
        print(f"Physics Residual: {result.physics_residual:.2f} W/m")
        print(f"Physics Compliant: {'✅ Yes' if result.is_physics_compliant else '❌ No'}")
        
        if result.temperature_uncertainty:
            print(f"Temperature Uncertainty: ±{result.temperature_uncertainty:.2f}°C")
            print(f"Rating Uncertainty: ±{result.rating_uncertainty:.0f} A")
        
    except FileNotFoundError:
        print("❌ No trained model found. Run training first:")
        print("   python -m core.train")


if __name__ == "__main__":
    demo_prediction()
