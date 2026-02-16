"""
Invariant-PIKAN: Adversarially-Robust Physics-Informed Neural Networks for Dynamic Line Rating
Copyright (C) 2025 Gelavizh Ahmadi / Invariant Research

This software is licensed under the Business Source License 1.1 (BSL 1.1).
Commercial production use requires a separate license agreement.
See LICENSE.txt for full terms.

"""

#!/usr/bin/env python3
"""
Create a bias-corrected model wrapper for US DLR deployment.
Adds a learned or fixed bias correction to reduce systematic under-prediction.
"""

import torch
import torch.nn as nn
import numpy as np
import json
import argparse
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent))

from models.invariant_pikan_v2 import create_invariant_pikan_v2


class BiasCorrectedModel(nn.Module):
    """
    Wrapper that adds bias correction to base model predictions.
    
    correction_mode:
        - 'fixed': Use pre-computed bias (e.g., +326 A for US)
        - 'learned': Learn a per-region bias during calibration
        - 'adaptive': Estimate bias from prediction confidence
    """
    
    def __init__(self, base_model, correction_mode='fixed', 
                 us_bias_correction=326.0, vietnam_bias_correction=0.0,
                 apply_to='ampacity_only'):
        super().__init__()
        self.base_model = base_model
        self.correction_mode = correction_mode
        self.us_bias_correction = us_bias_correction
        self.vietnam_bias_correction = vietnam_bias_correction
        self.apply_to = apply_to  # 'ampacity_only' or 'both'
        
        # Learnable bias (if using learned mode)
        if correction_mode == 'learned':
            self.us_bias = nn.Parameter(torch.tensor(us_bias_correction))
            self.vn_bias = nn.Parameter(torch.tensor(vietnam_bias_correction))
        
    def forward(self, weather, weather_dict, region=None):
        """
        Forward pass with bias correction.
        
        Args:
            weather: [batch, 4] - [T_amb, wind_speed, wind_angle, solar]
            weather_dict: dict with T_amb, wind_speed, solar
            region: 'US' or 'VN' or None (auto-detect from data if possible)
        
        Returns:
            dict with corrected predictions
        """
        # Get base predictions
        with torch.no_grad() if not self.training else torch.enable_grad():
            output = self.base_model(weather, weather_dict)
        
        # Determine bias correction
        if self.correction_mode == 'fixed':
            if region == 'US':
                bias = self.us_bias_correction
            elif region == 'VN':
                bias = self.vietnam_bias_correction
            else:
                # Default to no correction if region unknown
                bias = 0.0
        elif self.correction_mode == 'learned':
            if region == 'US':
                bias = self.us_bias.item()
            elif region == 'VN':
                bias = self.vn_bias.item()
            else:
                bias = 0.0
        else:
            bias = 0.0
        
        # Apply correction
        corrected_ampacity = output['ampacity'] + bias
        
        # Ensure non-negative
        corrected_ampacity = torch.clamp(corrected_ampacity, min=0.0)
        
        result = {
            'temperature': output['temperature'],
            'ampacity': corrected_ampacity,
            'bias_applied': bias,
            'region': region,
            'correction_mode': self.correction_mode
        }
        
        # Keep original predictions for comparison
        if not self.training:
            result['original_ampacity'] = output['ampacity']
        
        return result
    
    def save(self, path, metadata=None):
        """Save the wrapper configuration."""
        save_dict = {
            'base_model_state': self.base_model.state_dict(),
            'correction_mode': self.correction_mode,
            'us_bias_correction': self.us_bias_correction,
            'vietnam_bias_correction': self.vietnam_bias_correction,
            'apply_to': self.apply_to,
            'metadata': metadata or {}
        }
        
        if self.correction_mode == 'learned':
            save_dict['learned_us_bias'] = self.us_bias.item()
            save_dict['learned_vn_bias'] = self.vn_bias.item()
        
        torch.save(save_dict, path)
        print(f"💾 Bias-corrected model saved to {path}")
    
    @classmethod
    def load(cls, path, device='cpu'):
        """Load a saved bias-corrected model."""
        checkpoint = torch.load(path, map_location=device, weights_only=False)
        
        # Recreate base model
        model_cfg = checkpoint.get('metadata', {}).get('model_config', {})
        base_model = create_invariant_pikan_v2(config=model_cfg)
        base_model.load_state_dict(checkpoint['base_model_state'])
        
        # Create wrapper
        wrapper = cls(
            base_model=base_model,
            correction_mode=checkpoint['correction_mode'],
            us_bias_correction=checkpoint['us_bias_correction'],
            vietnam_bias_correction=checkpoint['vietnam_bias_correction'],
            apply_to=checkpoint.get('apply_to', 'ampacity_only')
        )
        
        return wrapper


def calibrate_bias_correction(base_model, test_loader, device='cpu', 
                               region='US', verbose=True):
    """
    Calibrate the optimal bias correction for a region.
    
    Finds the bias that minimizes MAE on validation data.
    """
    from core.data import VietnamDataset
    
    all_preds = []
    all_targets = []
    
    base_model.eval()
    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(device)
            weather = x[:, :4]
            weather_dict = {
                'T_amb': weather[:, 0],
                'wind_speed': weather[:, 1],
                'solar': weather[:, 3]
            }
            
            output = base_model(weather, weather_dict)
            preds = output['ampacity'].cpu().numpy()
            targets = y[:, 1].cpu().numpy()
            
            all_preds.extend(preds.flatten())
            all_targets.extend(targets.flatten())
    
    preds = np.array(all_preds)
    targets = np.array(all_targets)
    
    # Find optimal bias (minimize MAE)
    biases = np.linspace(-500, 500, 1001)
    maes = []
    
    for bias in biases:
        corrected = preds + bias
        mae = np.mean(np.abs(corrected - targets))
        maes.append(mae)
    
    best_idx = np.argmin(maes)
    best_bias = biases[best_idx]
    best_mae = maes[best_idx]
    
    # Original MAE
    original_mae = np.mean(np.abs(preds - targets))
    
    if verbose:
        print(f"\n📊 {region} Bias Calibration Results:")
        print(f"  Original MAE: {original_mae:.2f} A")
        print(f"  Best bias: {best_bias:+.2f} A")
        print(f"  Corrected MAE: {best_mae:.2f} A")
        print(f"  Improvement: {original_mae - best_mae:+.2f} A")
    
    return {
        'optimal_bias': float(best_bias),
        'original_mae': float(original_mae),
        'corrected_mae': float(best_mae),
        'improvement': float(original_mae - best_mae)
    }


def main():
    parser = argparse.ArgumentParser(
        description='Create bias-corrected model for US DLR'
    )
    parser.add_argument('--base-model', type=str, required=True,
                       help='Path to base model checkpoint')
    parser.add_argument('--test-data', type=str, required=True,
                       help='Path to test data CSV (for calibration)')
    parser.add_argument('--test-idx', type=str, required=True,
                       help='Path to test indices')
    parser.add_argument('--us-bias', type=float, default=None,
                       help='US bias correction (auto-calibrate if not provided)')
    parser.add_argument('--vn-bias', type=float, default=0.0,
                       help='Vietnam bias correction')
    parser.add_argument('--output', type=str, 
                       default='models/bias_corrected_us_model.pt',
                       help='Output path')
    parser.add_argument('--device', type=str, default='auto')
    
    args = parser.parse_args()
    
    # Device
    if args.device == 'auto':
        device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    print(f"Device: {device}")
    
    # Load base model
    print(f"\nLoading base model from {args.base_model}")
    checkpoint = torch.load(args.base_model, map_location=device, weights_only=False)
    model_cfg = checkpoint.get('config', {}).get('model', None) if isinstance(checkpoint, dict) and 'config' in checkpoint else {}
    
    base_model = create_invariant_pikan_v2(config=model_cfg)
    base_model.load_state_dict(checkpoint['model_state_dict'])
    base_model = base_model.to(device)
    base_model.eval()
    
    # Load test data
    print(f"Loading test data from {args.test_data}")
    from core.data import VietnamDataset
    from torch.utils.data import DataLoader, Subset
    
    dataset = VietnamDataset(args.test_data)
    test_loader = DataLoader(dataset, batch_size=256, shuffle=False)
    
    # Get region info
    test_df = pd.read_csv(args.test_data)
    
    # Calibrate US bias if not provided
    if args.us_bias is None:
        print("\n🔧 Calibrating US bias correction...")
        
        if 'region' in test_df.columns:
            # Filter to US only for calibration
            us_mask = test_df['region'].values == 'US'
            us_indices = np.where(us_mask)[0]
            us_dataset = Subset(dataset, us_indices)
            us_loader = DataLoader(us_dataset, batch_size=256, shuffle=False)
        else:
            us_loader = test_loader
        
        calibration = calibrate_bias_correction(
            base_model, us_loader, device=device, region='US'
        )
        us_bias = calibration['optimal_bias']
    else:
        us_bias = args.us_bias
        print(f"\nUsing provided US bias: {us_bias:+.2f} A")
    
    # Create bias-corrected wrapper
    print("\n🔧 Creating bias-corrected model...")
    corrected_model = BiasCorrectedModel(
        base_model=base_model,
        correction_mode='fixed',
        us_bias_correction=us_bias,
        vietnam_bias_correction=args.vn_bias
    )
    corrected_model = corrected_model.to(device)
    
    # Save
    metadata = {
        'base_model_path': args.base_model,
        'model_config': model_cfg,
        'calibration_data': args.test_data
    }
    
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    corrected_model.save(args.output, metadata=metadata)
    
    # Test the corrected model
    print("\n🧪 Testing bias-corrected model...")
    corrected_model.eval()
    
    with torch.no_grad():
        # Get one batch
        for x, y in test_loader:
            x = x.to(device)
            weather = x[:, :4]
            weather_dict = {
                'T_amb': weather[:, 0],
                'wind_speed': weather[:, 1],
                'solar': weather[:, 3]
            }
            
            # Test both regions
            for region in ['US', 'VN']:
                output = corrected_model(weather, weather_dict, region=region)
                print(f"\n{region} region:")
                print(f"  Sample predictions: {output['ampacity'][:3].cpu().numpy().flatten()}")
                print(f"  Bias applied: {output['bias_applied']}")
            break
    
    print(f"\n✅ Bias-corrected model saved to {args.output}")
    print(f"\nUsage in production:")
    print(f"  from scripts.create_bias_corrected_model import BiasCorrectedModel")
    print(f"  model = BiasCorrectedModel.load('{args.output}')")
    print(f"  output = model(weather, weather_dict, region='US')")


if __name__ == '__main__':
    import pandas as pd  # Needed for main
    main()
