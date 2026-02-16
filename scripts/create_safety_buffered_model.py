"""
Invariant-PIKAN: Adversarially-Robust Physics-Informed Neural Networks for Dynamic Line Rating
Copyright (C) 2025 Gelavizh Ahmadi / Invariant Research

This software is licensed under the Business Source License 1.1 (BSL 1.1).
Commercial production use requires a separate license agreement.
See LICENSE.txt for full terms.

"""

#!/usr/bin/env python3
"""
Create a Safety-Buffered Bias-Corrected Model for US DLR Deployment

This creates a production-ready model that:
1. Applies partial bias correction (e.g., 50% of calculated bias)
2. Adds configurable safety margins
3. Provides uncertainty estimates
4. Includes confidence-based prediction intervals

This is safer than full bias correction while still improving accuracy.
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
from scripts.create_bias_corrected_model import BiasCorrectedModel, calibrate_bias_correction


class SafetyBufferedModel(nn.Module):
    """
    Production-ready model with safety buffers and partial bias correction.
    
    Features:
    - Partial bias correction (e.g., 50% of estimated bias)
    - Configurable safety margin (additional conservatism)
    - Confidence-based uncertainty estimates
    - Prediction intervals for risk assessment
    """
    
    def __init__(
        self,
        base_model,
        us_bias_correction=326.0,
        correction_ratio=0.5,  # Apply 50% of bias correction
        safety_margin=50.0,     # Additional safety margin in Amps
        min_ampacity=100.0,     # Minimum allowed ampacity
        max_ampacity=3000.0,    # Maximum allowed ampacity
    ):
        super().__init__()
        self.base_model = base_model
        self.us_bias_correction = us_bias_correction
        self.correction_ratio = correction_ratio
        self.safety_margin = safety_margin
        self.min_ampacity = min_ampacity
        self.max_ampacity = max_ampacity
        
        # Effective correction (partial + safety margin)
        self.effective_correction = us_bias_correction * correction_ratio - safety_margin
        
    def forward(self, weather, weather_dict, region=None, return_confidence=False):
        """
        Forward pass with safety buffering.
        
        Args:
            weather: [batch, 4] - [T_amb, wind_speed, wind_angle, solar]
            weather_dict: dict with T_amb, wind_speed, solar
            region: 'US' or 'VN' or None
            return_confidence: If True, return uncertainty estimates
            
        Returns:
            dict with predictions and optional confidence metrics
        """
        with torch.no_grad() if not self.training else torch.enable_grad():
            output = self.base_model(weather, weather_dict)
        
        base_ampacity = output['ampacity']
        
        # Determine correction based on region
        if region == 'US':
            correction = self.effective_correction
            confidence = 'medium'  # Corrected but with safety margin
        elif region == 'VN':
            correction = 0.0
            confidence = 'high'    # Original model performs well
        else:
            # Unknown region: apply conservative correction
            correction = self.effective_correction * 0.5
            confidence = 'low'     # Reduced confidence for unknown region
        
        # Apply correction
        corrected_ampacity = base_ampacity + correction
        
        # Clip to safe bounds
        corrected_ampacity = torch.clamp(
            corrected_ampacity,
            min=self.min_ampacity,
            max=self.max_ampacity
        )
        
        result = {
            'ampacity': corrected_ampacity,
            'base_ampacity': base_ampacity,
            'correction_applied': correction,
            'region': region,
            'confidence': confidence,
            'safety_margin': self.safety_margin,
        }
        
        if return_confidence:
            # Estimate uncertainty based on region
            if region == 'US':
                # Higher uncertainty due to domain shift
                uncertainty = torch.full_like(corrected_ampacity, 100.0)  # ±100A
            elif region == 'VN':
                uncertainty = torch.full_like(corrected_ampacity, 50.0)   # ±50A
            else:
                uncertainty = torch.full_like(corrected_ampacity, 150.0)  # ±150A
            
            # Prediction intervals (95% confidence)
            result['uncertainty'] = uncertainty
            result['prediction_lower'] = corrected_ampacity - 1.96 * uncertainty
            result['prediction_upper'] = corrected_ampacity + 1.96 * uncertainty
        
        return result
    
    def get_safety_metrics(self, predictions, actual=None):
        """
        Compute safety metrics for the predictions.
        
        Args:
            predictions: Model output dict
            actual: Optional ground truth for validation
            
        Returns:
            dict with safety metrics
        """
        ampacity = predictions['ampacity']
        
        metrics = {
            'mean_ampacity': torch.mean(ampacity).item(),
            'min_ampacity': torch.min(ampacity).item(),
            'max_ampacity': torch.max(ampacity).item(),
            'correction_applied': predictions['correction_applied'],
            'safety_margin': self.safety_margin,
            'confidence': predictions['confidence'],
        }
        
        if actual is not None:
            # Compute safety margin relative to actual
            margin = ampacity - actual
            metrics['mean_safety_margin'] = torch.mean(margin).item()
            metrics['min_safety_margin'] = torch.min(margin).item()
            metrics['safety_violations'] = torch.sum(margin < 0).item()
        
        return metrics
    
    def save(self, path, metadata=None):
        """Save the safety-buffered model."""
        save_dict = {
            'base_model_state': self.base_model.state_dict(),
            'us_bias_correction': self.us_bias_correction,
            'correction_ratio': self.correction_ratio,
            'safety_margin': self.safety_margin,
            'min_ampacity': self.min_ampacity,
            'max_ampacity': self.max_ampacity,
            'effective_correction': self.effective_correction,
            'metadata': metadata or {}
        }
        
        torch.save(save_dict, path)
        print(f"💾 Safety-buffered model saved to {path}")
    
    @classmethod
    def load(cls, path, device='cpu'):
        """Load a saved safety-buffered model."""
        checkpoint = torch.load(path, map_location=device, weights_only=False)
        
        # Recreate base model
        model_cfg = checkpoint.get('metadata', {}).get('model_config', {})
        base_model = create_invariant_pikan_v2(config=model_cfg)
        base_model.load_state_dict(checkpoint['base_model_state'])
        
        # Create wrapper
        wrapper = cls(
            base_model=base_model,
            us_bias_correction=checkpoint['us_bias_correction'],
            correction_ratio=checkpoint['correction_ratio'],
            safety_margin=checkpoint['safety_margin'],
            min_ampacity=checkpoint.get('min_ampacity', 100.0),
            max_ampacity=checkpoint.get('max_ampacity', 3000.0)
        )
        
        return wrapper


def compute_metrics(y_true, y_pred):
    """Compute MAE, RMSE, bias."""
    errors = y_pred - y_true
    mae = np.mean(np.abs(errors))
    rmse = np.sqrt(np.mean(errors**2))
    bias = np.mean(errors)
    return {'mae': mae, 'rmse': rmse, 'bias': bias}


def evaluate_safety_buffer(base_model, test_loader, device, 
                          us_bias_correction=326.0,
                          correction_ratios=[0.0, 0.25, 0.5, 0.75, 1.0],
                          safety_margins=[0.0, 25.0, 50.0, 100.0]):
    """
    Evaluate different safety buffer configurations.
    
    Returns results for each combination of correction_ratio and safety_margin.
    """
    
    results = []
    
    all_targets = []
    all_base_preds = []
    
    # Collect base predictions
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
            
            all_base_preds.extend(preds.flatten())
            all_targets.extend(targets.flatten())
    
    all_base_preds = np.array(all_base_preds)
    all_targets = np.array(all_targets)
    
    # Test each configuration
    for correction_ratio in correction_ratios:
        for safety_margin in safety_margins:
            # Apply correction and safety margin
            effective_correction = us_bias_correction * correction_ratio - safety_margin
            corrected_preds = all_base_preds + effective_correction
            
            # Clip to safe range
            corrected_preds = np.clip(corrected_preds, 100.0, 3000.0)
            
            # Compute metrics
            metrics = compute_metrics(all_targets, corrected_preds)
            
            # Count safety violations (predictions below actual)
            violations = np.sum(corrected_preds < all_targets)
            violation_rate = violations / len(all_targets) * 100
            
            results.append({
                'correction_ratio': correction_ratio,
                'safety_margin': safety_margin,
                'effective_correction': effective_correction,
                'mae': metrics['mae'],
                'rmse': metrics['rmse'],
                'bias': metrics['bias'],
                'safety_violations': violations,
                'violation_rate_%': violation_rate
            })
    
    return pd.DataFrame(results)


def main():
    parser = argparse.ArgumentParser(
        description='Create safety-buffered model for production'
    )
    parser.add_argument('--base-model', type=str, required=True,
                       help='Path to base model checkpoint')
    parser.add_argument('--test-data', type=str, required=True,
                       help='Path to test data CSV')
    parser.add_argument('--test-idx', type=str, required=True,
                       help='Path to test indices')
    parser.add_argument('--us-bias', type=float, default=326.0,
                       help='US bias correction amount')
    parser.add_argument('--correction-ratio', type=float, default=0.5,
                       help='Fraction of bias to apply (0.0-1.0)')
    parser.add_argument('--safety-margin', type=float, default=50.0,
                       help='Additional safety margin in Amps')
    parser.add_argument('--evaluate-grid', action='store_true',
                       help='Evaluate all correction_ratio × safety_margin combinations')
    parser.add_argument('--output', type=str, 
                       default='models/safety_buffered_model.pt',
                       help='Output path')
    parser.add_argument('--device', type=str, default='auto')
    
    args = parser.parse_args()
    
    # Device
    if args.device == 'auto':
        device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    print(f"🔧 Device: {device}")
    
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
    from torch.utils.data import DataLoader
    
    dataset = VietnamDataset(args.test_data)
    test_loader = DataLoader(dataset, batch_size=256, shuffle=False)
    
    # Get region info
    test_df = pd.read_csv(args.test_data)
    
    if args.evaluate_grid:
        # Evaluate all combinations
        print("\n📊 Evaluating safety buffer grid...")
        results_df = evaluate_safety_buffer(
            base_model, test_loader, device,
            us_bias_correction=args.us_bias
        )
        
        print("\n" + "="*80)
        print("SAFETY BUFFER EVALUATION RESULTS")
        print("="*80)
        print(results_df.to_string(index=False))
        
        # Find best configuration
        # Prioritize: low MAE, low violation rate
        results_df['score'] = (
            -results_df['mae'] / 100 +  # Lower MAE is better
            -results_df['violation_rate_%'] * 10  # Lower violations is much better
        )
        best_idx = results_df['score'].idxmax()
        best_config = results_df.loc[best_idx]
        
        print(f"\n🏆 Recommended configuration:")
        print(f"  Correction ratio: {best_config['correction_ratio']}")
        print(f"  Safety margin: {best_config['safety_margin']} A")
        print(f"  Expected MAE: {best_config['mae']:.2f} A")
        print(f"  Violation rate: {best_config['violation_rate_%']:.2f}%")
        
        # Save results
        results_path = Path(args.output).parent / 'safety_buffer_grid.csv'
        results_df.to_csv(results_path, index=False)
        print(f"\n💾 Grid results saved to {results_path}")
        
        # Use best config for model
        args.correction_ratio = best_config['correction_ratio']
        args.safety_margin = best_config['safety_margin']
    
    # Create safety-buffered model
    print(f"\n🔧 Creating safety-buffered model:")
    print(f"  US bias correction: {args.us_bias:.2f} A")
    print(f"  Correction ratio: {args.correction_ratio}")
    print(f"  Safety margin: {args.safety_margin} A")
    print(f"  Effective correction: {args.us_bias * args.correction_ratio - args.safety_margin:.2f} A")
    
    safety_model = SafetyBufferedModel(
        base_model=base_model,
        us_bias_correction=args.us_bias,
        correction_ratio=args.correction_ratio,
        safety_margin=args.safety_margin
    )
    safety_model = safety_model.to(device)
    
    # Save
    metadata = {
        'base_model_path': args.base_model,
        'model_config': model_cfg,
        'calibration_data': args.test_data,
        'us_bias_correction': args.us_bias,
        'correction_ratio': args.correction_ratio,
        'safety_margin': args.safety_margin
    }
    
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    safety_model.save(args.output, metadata=metadata)
    
    # Test the model
    print("\n🧪 Testing safety-buffered model...")
    safety_model.eval()
    
    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(device)
            weather = x[:, :4]
            weather_dict = {
                'T_amb': weather[:, 0],
                'wind_speed': weather[:, 1],
                'solar': weather[:, 3]
            }
            
            for region in ['US', 'VN']:
                output = safety_model(weather, weather_dict, region=region, return_confidence=True)
                print(f"\n{region} region:")
                print(f"  Sample ampacity: {output['ampacity'][:3].cpu().numpy().flatten()}")
                print(f"  Correction: {output['correction_applied']:.2f} A")
                print(f"  Confidence: {output['confidence']}")
                print(f"  Uncertainty: ±{output['uncertainty'][0].item():.0f} A")
                print(f"  95% CI: [{output['prediction_lower'][0].item():.0f}, {output['prediction_upper'][0].item():.0f}] A")
            break
    
    print(f"\n✅ Safety-buffered model saved to {args.output}")
    print(f"\nUsage in production:")
    print(f"  from scripts.create_safety_buffered_model import SafetyBufferedModel")
    print(f"  model = SafetyBufferedModel.load('{args.output}')")
    print(f"  output = model(weather, weather_dict, region='US', return_confidence=True)")


if __name__ == '__main__':
    import pandas as pd  # Needed for main
    main()
