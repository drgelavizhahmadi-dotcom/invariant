"""
Invariant-PIKAN: Adversarially-Robust Physics-Informed Neural Networks for Dynamic Line Rating
Copyright (C) 2025 Gelavizh Ahmadi / Invariant Research

This software is licensed under the Business Source License 1.1 (BSL 1.1).
Commercial production use requires a separate license agreement.
See LICENSE.txt for full terms.

DISCLAIMER: This implementation is independent of concurrent academic work on
HWF-PIKAN for plasma physics (Heravifard et al., Sharif University, 2025).
"""

#!/usr/bin/env python
"""
Evaluate Hybrid Ensemble with Calibrated US Physics
Combines neural network predictions with US-calibrated physics.
"""

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from pathlib import Path
import sys
import json
import argparse

sys.path.append(str(Path(__file__).parent.parent))

from models.invariant_pikan_v2 import create_invariant_pikan_v2
from core.data import VietnamDataset, USDataset
from torch.utils.data import DataLoader, Subset
from scipy.optimize import minimize_scalar

# Import calibrated US params
sys.path.append(str(Path(__file__).parent.parent / 'calibration_results' / 'us'))
from us_line_params import US_LINE_PARAMS


class USCalibratedHybrid(nn.Module):
    """
    Hybrid model: Neural network + US-calibrated physics
    """
    
    def __init__(self, neural_model, physics_params, blend_weight=0.5):
        super().__init__()
        self.neural_model = neural_model
        self.physics_params = physics_params
        self.blend_weight = blend_weight  # Weight for physics (0=neural only, 1=physics only)
        
        # Cache physics params
        self.diameter = physics_params['diameter']
        self.emissivity = physics_params['emissivity']
        self.absorptivity = physics_params['absorptivity']
        self.R_20 = physics_params['resistance_ac']
        self.alpha = physics_params['temp_coefficient']
        
    def physics_ampacity(self, T_ambient, wind_speed, solar_irradiance, T_max=75.0):
        """
        Compute ampacity using calibrated physics at max allowable temperature.
        This is the standard DLR calculation: what current produces T_max?
        """
        from core.physics import (
            convective_heat_loss,
            radiative_heat_loss,
            solar_heat_gain
        )
        
        batch_size = len(T_ambient)
        ampacities = []
        
        for i in range(batch_size):
            T_amb = T_ambient[i].item() if torch.is_tensor(T_ambient) else T_ambient[i]
            v_wind = wind_speed[i].item() if torch.is_tensor(wind_speed) else wind_speed[i]
            solar = solar_irradiance[i].item() if torch.is_tensor(solar_irradiance) else solar_irradiance[i]
            
            # Calculate heat losses at T_max
            q_conv = convective_heat_loss(T_max, T_amb, v_wind, self.diameter)
            q_rad = radiative_heat_loss(T_max, T_amb, self.diameter, self.emissivity)
            q_solar = solar_heat_gain(solar, self.diameter, self.absorptivity)
            
            # Resistance at T_max
            R_Tmax = self.R_20 * (1 + self.alpha * (T_max - 20))
            
            # Available cooling capacity
            q_available = q_conv + q_rad - q_solar
            q_available = max(q_available, 1.0)  # Prevent negative
            
            # Max current: I = sqrt(q_available / R)
            I_max = (q_available / R_Tmax) ** 0.5
            ampacities.append(I_max)
        
        return torch.tensor(ampacities, dtype=torch.float32, device=T_ambient.device)
    
    def forward(self, weather, weather_dict):
        """
        Forward pass blending neural and physics predictions.
        
        Args:
            weather: [batch, 4] - [T_amb, wind_speed, wind_angle, solar]
            weather_dict: dict with keys T_amb, wind_speed, solar
        """
        # Neural prediction
        with torch.no_grad():
            neural_output = self.neural_model(weather, weather_dict)
            neural_temp = neural_output['temperature']
            neural_amp = neural_output['ampacity']
        
        # Extract conditions
        T_ambient = weather[:, 0]
        wind_speed = weather[:, 1]
        solar_irradiance = weather[:, 3]
        
        # Physics ampacity at max allowable temperature (75°C)
        physics_amp = self.physics_ampacity(
            T_ambient, wind_speed, solar_irradiance, T_max=75.0
        )
        
        # Blend predictions
        # weight * physics + (1-weight) * neural
        blended_amp = self.blend_weight * physics_amp + (1 - self.blend_weight) * neural_amp.squeeze()
        
        return {
            'temperature': neural_temp,
            'ampacity': blended_amp,
            'neural_ampacity': neural_amp,
            'physics_ampacity': physics_amp,
            'blend_weight': self.blend_weight
        }


def compute_metrics(y_true, y_pred):
    errors = y_pred - y_true
    mae = np.mean(np.abs(errors))
    rmse = np.sqrt(np.mean(errors**2))
    bias = np.mean(errors)
    return {'mae': mae, 'rmse': rmse, 'bias': bias}


def main():
    parser = argparse.ArgumentParser(description='Evaluate Hybrid Ensemble with US Physics')
    parser.add_argument('--run-dir', type=str, required=True, help='Training run directory')
    parser.add_argument('--blend-weight', type=float, default=0.5, 
                        help='Weight for physics (0=neural only, 1=physics only)')
    parser.add_argument('--output-dir', type=str, default='hybrid_results')
    parser.add_argument('--device', type=str, default='auto')
    parser.add_argument('--analyze-us-only', action='store_true', 
                        help='Only analyze US region samples')
    
    args = parser.parse_args()
    
    run_dir = Path(args.run_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Device
    if args.device == 'auto':
        device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    print(f"🔧 Device: {device}")
    
    # Load model
    ckpt_path = run_dir / 'best_model.pt'
    print(f"📦 Loading model from {ckpt_path}")
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
    
    model_cfg = checkpoint.get('config', {}).get('model', None) if isinstance(checkpoint, dict) else None
    neural_model = create_invariant_pikan_v2(config=model_cfg)
    neural_model.load_state_dict(checkpoint['model_state_dict'])
    neural_model = neural_model.to(device)
    neural_model.eval()
    
    # Create hybrid model
    print(f"🔀 Creating hybrid ensemble (blend_weight={args.blend_weight})")
    print(f"   Physics params: R_factor={US_LINE_PARAMS['resistance_factor']:.3f}, "
          f"emissivity={US_LINE_PARAMS['emissivity']:.3f}")
    
    hybrid_model = USCalibratedHybrid(neural_model, US_LINE_PARAMS, args.blend_weight)
    hybrid_model = hybrid_model.to(device)
    hybrid_model.eval()
    
    # Load test data
    test_idx = torch.load(run_dir / 'test_indices.pt')
    temp_csv = run_dir / 'temp_unified_data.csv'
    
    if temp_csv.exists():
        dataset = VietnamDataset(str(temp_csv))
        raw_df = pd.read_csv(temp_csv)
    else:
        raise FileNotFoundError(f"No test data found in {run_dir}")
    
    test_dataset = Subset(dataset, test_idx)
    test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False)
    
    print(f"📊 Evaluating on {len(test_dataset)} test samples...")
    
    # Run evaluation
    all_targets = []
    all_neural_preds = []
    all_physics_preds = []
    all_hybrid_preds = []
    all_regions = []
    
    with torch.no_grad():
        for batch_idx, (x, y) in enumerate(test_loader):
            x = x.to(device)
            
            # Prepare inputs
            weather = x[:, :4]
            weather_dict = {
                'T_amb': weather[:, 0],
                'wind_speed': weather[:, 1],
                'solar': weather[:, 3]
            }
            
            # Hybrid prediction
            output = hybrid_model(weather, weather_dict)
            
            # Store
            all_targets.extend(y[:, 1].cpu().numpy())
            all_neural_preds.extend(output['neural_ampacity'].cpu().numpy())
            all_physics_preds.extend(output['physics_ampacity'].cpu().numpy())
            all_hybrid_preds.extend(output['ampacity'].cpu().numpy())
            
            if batch_idx % 10 == 0:
                print(f"  Batch {batch_idx}/{len(test_loader)}")
    
    # Convert to arrays
    targets = np.array(all_targets).flatten()
    neural_preds = np.array(all_neural_preds).flatten()
    physics_preds = np.array(all_physics_preds).flatten()
    hybrid_preds = np.array(all_hybrid_preds).flatten()
    
    # Get regions if available
    if 'region' in raw_df.columns:
        test_regions = raw_df.iloc[test_idx]['region'].values
    else:
        test_regions = np.array(['UNKNOWN'] * len(targets))
    
    # Overall metrics
    print("\n" + "="*60)
    print("OVERALL RESULTS")
    print("="*60)
    
    neural_metrics = compute_metrics(targets, neural_preds)
    physics_metrics = compute_metrics(targets, physics_preds)
    hybrid_metrics = compute_metrics(targets, hybrid_preds)
    
    print(f"\n{'Model':<15} {'MAE':<12} {'RMSE':<12} {'Bias':<12}")
    print("-"*60)
    print(f"{'Neural Only':<15} {neural_metrics['mae']:<12.2f} {neural_metrics['rmse']:<12.2f} {neural_metrics['bias']:<12.2f}")
    print(f"{'Physics Only':<15} {physics_metrics['mae']:<12.2f} {physics_metrics['rmse']:<12.2f} {physics_metrics['bias']:<12.2f}")
    print(f"{'Hybrid':<15} {hybrid_metrics['mae']:<12.2f} {hybrid_metrics['rmse']:<12.2f} {hybrid_metrics['bias']:<12.2f}")
    
    # US-specific analysis
    if args.analyze_us_only or 'US' in test_regions:
        print("\n" + "="*60)
        print("US REGION RESULTS")
        print("="*60)
        
        us_mask = test_regions == 'US'
        us_targets = targets[us_mask]
        us_neural = neural_preds[us_mask]
        us_physics = physics_preds[us_mask]
        us_hybrid = hybrid_preds[us_mask]
        
        us_neural_metrics = compute_metrics(us_targets, us_neural)
        us_physics_metrics = compute_metrics(us_targets, us_physics)
        us_hybrid_metrics = compute_metrics(us_targets, us_hybrid)
        
        print(f"US samples: {len(us_targets)}")
        print(f"\n{'Model':<15} {'MAE':<12} {'RMSE':<12} {'Bias':<12}")
        print("-"*60)
        print(f"{'Neural Only':<15} {us_neural_metrics['mae']:<12.2f} {us_neural_metrics['rmse']:<12.2f} {us_neural_metrics['bias']:<12.2f}")
        print(f"{'Physics Only':<15} {us_physics_metrics['mae']:<12.2f} {us_physics_metrics['rmse']:<12.2f} {us_physics_metrics['bias']:<12.2f}")
        print(f"{'Hybrid':<15} {us_hybrid_metrics['mae']:<12.2f} {us_hybrid_metrics['rmse']:<12.2f} {us_hybrid_metrics['bias']:<12.2f}")
        
        # Calculate improvement
        mae_improvement = us_neural_metrics['mae'] - us_hybrid_metrics['mae']
        bias_improvement = abs(us_neural_metrics['bias']) - abs(us_hybrid_metrics['bias'])
        
        print(f"\n📈 Hybrid vs Neural:")
        print(f"   MAE change: {mae_improvement:+.2f} A ({'better' if mae_improvement > 0 else 'worse'})")
        print(f"   Bias reduction: {bias_improvement:+.2f} A ({'better' if bias_improvement > 0 else 'worse'})")
    
    # Save results
    results = {
        'run_dir': str(run_dir),
        'blend_weight': args.blend_weight,
        'physics_params': US_LINE_PARAMS,
        'overall': {
            'neural': neural_metrics,
            'physics': physics_metrics,
            'hybrid': hybrid_metrics
        },
        'us_region': {
            'neural': us_neural_metrics if 'US' in test_regions else None,
            'physics': us_physics_metrics if 'US' in test_regions else None,
            'hybrid': us_hybrid_metrics if 'US' in test_regions else None
        } if 'US' in test_regions else None
    }
    
    output_file = output_dir / f'hybrid_eval_blend{args.blend_weight:.2f}.json'
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2, default=float)
    print(f"\n💾 Results saved to {output_file}")
    
    # Save predictions for analysis
    pred_df = pd.DataFrame({
        'target': targets,
        'neural_pred': neural_preds,
        'physics_pred': physics_preds,
        'hybrid_pred': hybrid_preds,
        'region': test_regions
    })
    pred_file = output_dir / f'hybrid_predictions_blend{args.blend_weight:.2f}.csv'
    pred_df.to_csv(pred_file, index=False)
    print(f"💾 Predictions saved to {pred_file}")
    
    print("\n✅ Evaluation complete!")


if __name__ == '__main__':
    main()
