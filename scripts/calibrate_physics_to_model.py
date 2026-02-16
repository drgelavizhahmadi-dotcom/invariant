"""
Invariant-PIKAN: Adversarially-Robust Physics-Informed Neural Networks for Dynamic Line Rating
Copyright (C) 2025 Gelavizh Ahmadi / Invariant Research

This software is licensed under the Business Source License 1.1 (BSL 1.1).
Commercial production use requires a separate license agreement.
See LICENSE.txt for full terms.

"""

#!/usr/bin/env python
"""
Calibrate IEEE 738 line parameters to match what the neural network learned.
Uses model predictions to infer actual resistance, emissivity, and absorptivity
for the Vietnam transmission line.
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize, differential_evolution
from pathlib import Path
import sys
import json
import pandas as pd
from datetime import datetime

# Add project root to path
sys.path.append(str(Path(__file__).parent))

# Import your physics engine
from core.physics import (
    convective_heat_loss,
    radiative_heat_loss,
    solar_heat_gain,
    resistive_heat_gain,
    ieee738_temperature
)


class PhysicsCalibrator:
    """
    Calibrates line parameters to match model predictions.
    Uses multiple samples to find consistent parameters.
    """
    
    def __init__(self, model, device='cpu'):
        self.model = model
        self.device = device
        self.model.eval()
        
        # Fixed parameters (known from line specs)
        self.fixed_params = {
            'diameter': 0.0275,        # m (27.5mm)
            'temp_coefficient': 0.0039,  # Typical for aluminum
            'resistance_20c': 1.4085e-4  # Ω/m at 20°C (base)
        }
        
        # Parameter bounds (physically plausible ranges)
        self.param_bounds = {
            'resistance_factor': (0.5, 2.0),     # 0.5x to 2.0x of base
            'emissivity': (0.3, 0.9),             # Typical range
            'absorptivity': (0.3, 0.9)            # Typical range
        }
        
    def collect_predictions(self, loader, max_samples=1000):
        """
        Run model on validation data and collect predictions.
        """
        print("📊 Collecting model predictions...")
        
        all_T_pred = []
        all_I_pred = []
        all_T_amb = []
        all_wind = []
        all_solar = []
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(loader):
                x, y = batch  # VietnamDataset returns (inputs, targets)
                
                x = x.to(self.device)
                
                # Get model predictions
                pred_temp, pred_amp = self.model(x)
                
                # Store predictions
                all_T_pred.extend(pred_temp.cpu().numpy())
                all_I_pred.extend(pred_amp.cpu().numpy())
                
                # Store inputs (denormalized if needed)
                # For Vietnam dataset: [T_amb, wind_speed, wind_angle, solar, current, resistance]
                all_T_amb.extend(x[:, 0].cpu().numpy())  # T_ambient
                all_wind.extend(x[:, 1].cpu().numpy())  # wind_speed
                all_solar.extend(x[:, 3].cpu().numpy()) # solar_irradiance
                
                if len(all_T_pred) >= max_samples:
                    break
        
        # Convert to numpy arrays
        data = {
            'T_pred': np.array(all_T_pred[:max_samples]),
            'I_pred': np.array(all_I_pred[:max_samples]),
            'T_amb': np.array(all_T_amb[:max_samples]),
            'wind_speed': np.array(all_wind[:max_samples]),
            'solar': np.array(all_solar[:max_samples])
        }
        
        print(f"  Collected {len(data['T_pred'])} samples")
        return data
    
    def calculate_physics_residual(self, params, data):
        """
        Calculate how well physics matches model predictions.
        Lower is better.
        """
        R_factor, emissivity, absorptivity = params
        
        # Construct line parameters
        line_params = {
            'diameter': self.fixed_params['diameter'],
            'emissivity': emissivity,
            'absorptivity': absorptivity,
            'resistance_ac': self.fixed_params['resistance_20c'] * R_factor,
            'temp_coefficient': self.fixed_params['temp_coefficient']
        }
        
        residuals = []
        
        for i in range(len(data['T_pred'])):
            T = data['T_pred'][i]
            I = data['I_pred'][i]
            T_amb = data['T_amb'][i]
            wind = data['wind_speed'][i]
            solar = data['solar'][i]
            
            # Calculate each heat term
            q_conv = convective_heat_loss(T, T_amb, wind, line_params['diameter'])
            q_rad = radiative_heat_loss(T, T_amb, line_params['diameter'], line_params['emissivity'])
            q_solar = solar_heat_gain(solar, line_params['diameter'], line_params['absorptivity'])
            q_joule = resistive_heat_gain(I, T, line_params['resistance_ac'], 
                                          line_params['temp_coefficient'])
            
            # Heat balance residual: q_joule + q_solar should equal q_conv + q_rad
            # Positive means more heat generated than dissipated (would increase T)
            residual = (q_joule + q_solar) - (q_conv + q_rad)
            
            # Normalize by heat magnitude
            norm = max(abs(q_joule), abs(q_conv), abs(q_rad), 1.0)
            residuals.append(residual / norm)
        
        return np.array(residuals)
    
    def objective_function(self, params, data):
        """
        Objective for optimization: minimize physics residuals.
        """
        residuals = self.calculate_physics_residual(params, data)
        
        # Mean squared residual
        mse = np.mean(residuals**2)
        
        # Also penalize extreme parameters (regularization)
        R_factor, emissivity, absorptivity = params
        reg = 0.01 * (
            (R_factor - 1.0)**2 + 
            (emissivity - 0.7)**2 + 
            (absorptivity - 0.8)**2
        )
        
        return mse + reg
    
    def calibrate(self, data, method='differential_evolution'):
        """
        Find parameters that minimize physics residuals.
        """
        print("\n🔧 Calibrating line parameters...")
        
        # Initial guess
        x0 = [1.0, 0.7, 0.8]
        bounds = [
            self.param_bounds['resistance_factor'],
            self.param_bounds['emissivity'],
            self.param_bounds['absorptivity']
        ]
        
        if method == 'differential_evolution':
            # Global optimization (slower but more robust)
            result = differential_evolution(
                self.objective_function,
                bounds,
                args=(data,),
                maxiter=100,
                popsize=15,
                tol=1e-6,
                seed=42
            )
        else:
            # Local optimization (faster)
            result = minimize(
                self.objective_function,
                x0,
                args=(data,),
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 100, 'disp': True}
            )
        
        return {
            'resistance_factor': result.x[0],
            'emissivity': result.x[1],
            'absorptivity': result.x[2],
            'final_error': result.fun,
            'success': result.success,
            'message': result.message if hasattr(result, 'message') else 'OK'
        }
    
    def validate_calibration(self, params, data):
        """
        Validate calibration on held-out data.
        """
        residuals = self.calculate_physics_residual(params, data)
        
        stats = {
            'mean_residual': np.mean(residuals),
            'std_residual': np.std(residuals),
            'rms_residual': np.sqrt(np.mean(residuals**2)),
            'max_abs_residual': np.max(np.abs(residuals)),
            'percentiles': {
                '5th': np.percentile(residuals, 5),
                '50th': np.percentile(residuals, 50),
                '95th': np.percentile(residuals, 95)
            }
        }
        
        return stats
    
    def plot_calibration_results(self, params, data, output_path='calibration_results.png'):
        """
        Visualize calibration results.
        """
        # Extract parameter values for residual calculation
        param_values = [params['resistance_factor'], params['emissivity'], params['absorptivity']]
        residuals = self.calculate_physics_residual(param_values, data)
        
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        # 1. Residual distribution
        axes[0, 0].hist(residuals, bins=50, edgecolor='black', alpha=0.7)
        axes[0, 0].axvline(x=0, color='red', linestyle='--')
        axes[0, 0].set_xlabel('Normalized Physics Residual')
        axes[0, 0].set_ylabel('Frequency')
        axes[0, 0].set_title('Physics Residual Distribution')
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. Residual vs Temperature
        axes[0, 1].scatter(data['T_pred'], residuals, alpha=0.3, s=5)
        axes[0, 1].axhline(y=0, color='red', linestyle='--')
        axes[0, 1].set_xlabel('Predicted Temperature (°C)')
        axes[0, 1].set_ylabel('Residual')
        axes[0, 1].set_title('Residual vs Temperature')
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. Residual vs Ampacity
        axes[0, 2].scatter(data['I_pred'], residuals, alpha=0.3, s=5)
        axes[0, 2].axhline(y=0, color='red', linestyle='--')
        axes[0, 2].set_xlabel('Predicted Ampacity (A)')
        axes[0, 2].set_ylabel('Residual')
        axes[0, 2].set_title('Residual vs Ampacity')
        axes[0, 2].grid(True, alpha=0.3)
        
        # 4. Q-Q plot
        from scipy import stats
        try:
            stats.probplot(residuals.flatten(), dist="norm", plot=axes[1, 0])
        except:
            # Fallback if Q-Q plot fails
            axes[1, 0].text(0.5, 0.5, 'Q-Q Plot\n(unavailable)', 
                           ha='center', va='center', transform=axes[1, 0].transAxes)
        axes[1, 0].set_title('Q-Q Plot')
        axes[1, 0].grid(True, alpha=0.3)
        
        # 5. Calibrated parameters
        param_names = ['Resistance\nFactor', 'Emissivity', 'Absorptivity']
        param_values = [params['resistance_factor'], params['emissivity'], params['absorptivity']]
        param_bounds_low = [self.param_bounds['resistance_factor'][0], 
                           self.param_bounds['emissivity'][0],
                           self.param_bounds['absorptivity'][0]]
        param_bounds_high = [self.param_bounds['resistance_factor'][1],
                            self.param_bounds['emissivity'][1],
                            self.param_bounds['absorptivity'][1]]
        
        x_pos = np.arange(len(param_names))
        axes[1, 1].bar(x_pos, param_values, color='skyblue', edgecolor='black')
        axes[1, 1].set_xticks(x_pos)
        axes[1, 1].set_xticklabels(param_names)
        axes[1, 1].set_ylabel('Parameter Value')
        axes[1, 1].set_title('Calibrated Line Parameters')
        axes[1, 1].set_ylim(0, 1.0)
        axes[1, 1].grid(True, alpha=0.3)
        
        # Add bounds as horizontal lines
        for i, (low, high) in enumerate(zip(param_bounds_low, param_bounds_high)):
            axes[1, 1].plot([i-0.4, i+0.4], [low, low], 'r--', alpha=0.5)
            axes[1, 1].plot([i-0.4, i+0.4], [high, high], 'r--', alpha=0.5)
        
        # 6. Summary text
        axes[1, 2].axis('off')
        summary_text = (
            f"Calibration Results:\n"
            f"==================\n\n"
            f"Resistance Factor: {params['resistance_factor']:.3f}\n"
            f"  → Effective R = {self.fixed_params['resistance_20c'] * params['resistance_factor']:.3e} Ω/m\n\n"
            f"Emissivity: {params['emissivity']:.3f}\n"
            f"Absorptivity: {params['absorptivity']:.3f}\n\n"
            f"Final Error: {params['final_error']:.6f}\n"
            f"Success: {params['success']}\n\n"
            f"Validation:\n"
            f"Mean Residual: {self.val_stats['mean_residual']:.4f}\n"
            f"RMS Residual: {self.val_stats['rms_residual']:.4f}\n"
            f"95% Range: [{self.val_stats['percentiles']['5th']:.3f}, "
            f"{self.val_stats['percentiles']['95th']:.3f}]"
        )
        axes[1, 2].text(0.1, 0.9, summary_text, transform=axes[1, 2].transAxes,
                       fontsize=10, verticalalignment='top',
                       bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=150)
        print(f"📈 Plot saved to {output_path}")
        
        return fig
    
    def save_calibrated_params(self, params, output_path='calibrated_params.json'):
        """
        Save calibrated parameters to JSON.
        """
        # Convert numpy types to Python native
        clean_params = {}
        for k, v in params.items():
            if isinstance(v, np.ndarray):
                clean_params[k] = v.tolist()
            elif isinstance(v, np.generic):
                clean_params[k] = v.item()
            else:
                clean_params[k] = v
        
        # Add metadata
        clean_params['metadata'] = {
            'timestamp': datetime.now().isoformat(),
            'fixed_params': self.fixed_params,
            'bounds': self.param_bounds
        }
        
        with open(output_path, 'w') as f:
            json.dump(clean_params, f, indent=2)
        
        print(f"💾 Calibrated parameters saved to {output_path}")
        return clean_params


def main():
    import argparse
    from core.data import VietnamDataset, create_dataloaders
    from core.model import PhysicsDLR  # Adjust to your model class
    
    parser = argparse.ArgumentParser(description='Calibrate physics to model')
    parser.add_argument('--model-path', type=str, required=True,
                        help='Path to trained model')
    parser.add_argument('--data-path', type=str, 
                        default='data/mendeley/vietnam_220kv.csv',
                        help='Path to Vietnam dataset')
    parser.add_argument('--max-samples', type=int, default=1000,
                        help='Maximum samples to use for calibration')
    parser.add_argument('--method', type=str, default='differential_evolution',
                        choices=['differential_evolution', 'L-BFGS-B'],
                        help='Optimization method')
    parser.add_argument('--output-dir', type=str, default='calibration_results',
                        help='Output directory')
    parser.add_argument('--device', type=str, default='auto')
    
    args = parser.parse_args()
    
    # Create output directory
    Path(args.output_dir).mkdir(exist_ok=True)
    
    # Set device
    if args.device == 'auto':
        if torch.cuda.is_available():
            device = 'cuda'
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            device = 'mps'
        else:
            device = 'cpu'
    else:
        device = args.device
    
    print(f"🔧 Using device: {device}")
    
    # Load model
    print(f"📦 Loading model from {args.model_path}")
    checkpoint = torch.load(args.model_path, map_location=device, weights_only=False)
    model = PhysicsDLR()
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model = model.to(device)
    
    # Load data
    print(f"📊 Loading Vietnam dataset...")
    dataset = VietnamDataset(args.data_path)
    
    # Create data loader
    from torch.utils.data import DataLoader, random_split
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    val_loader = DataLoader(val_dataset, batch_size=256, shuffle=False)
    
    # Initialize calibrator
    calibrator = PhysicsCalibrator(model, device)
    
    # Collect predictions
    data = calibrator.collect_predictions(val_loader, max_samples=args.max_samples)
    
    # Split into calibration and validation sets
    n_cal = int(0.7 * len(data['T_pred']))
    cal_data = {k: v[:n_cal] for k, v in data.items()}
    val_data = {k: v[n_cal:] for k, v in data.items()}
    
    print(f"Calibration samples: {n_cal}")
    print(f"Validation samples: {len(val_data['T_pred'])}")
    
    # Calibrate
    params = calibrator.calibrate(cal_data, method=args.method)
    
    print("\n📋 Calibrated Parameters:")
    print(f"  Resistance Factor: {params['resistance_factor']:.3f}")
    print(f"  Emissivity: {params['emissivity']:.3f}")
    print(f"  Absorptivity: {params['absorptivity']:.3f}")
    print(f"  Final Error: {params['final_error']:.6f}")
    print(f"  Success: {params['success']}")
    
    # Validate
    calibrator.val_stats = calibrator.validate_calibration(
        [params['resistance_factor'], params['emissivity'], params['absorptivity']],
        val_data
    )
    
    print("\n✅ Validation Results:")
    print(f"  Mean Residual: {calibrator.val_stats['mean_residual']:.4f}")
    print(f"  RMS Residual: {calibrator.val_stats['rms_residual']:.4f}")
    print(f"  95% Range: [{calibrator.val_stats['percentiles']['5th']:.3f}, "
          f"{calibrator.val_stats['percentiles']['95th']:.3f}]")
    
    # Plot
    calibrator.plot_calibration_results(
        params, 
        val_data,
        output_path=f"{args.output_dir}/calibration_plot.png"
    )
    
    # Save parameters
    calibrator.save_calibrated_params(
        params,
        output_path=f"{args.output_dir}/calibrated_params.json"
    )
    
    # Also save as Python module for easy import
    with open(f"{args.output_dir}/vietnam_params.py", 'w') as f:
        f.write(f"""# Auto-generated Vietnam line parameters
# Generated: {datetime.now().isoformat()}

VIETNAM_LINE_PARAMS = {{
    'diameter': 0.0275,
    'emissivity': {params['emissivity']:.4f},
    'absorptivity': {params['absorptivity']:.4f},
    'resistance_ac': {calibrator.fixed_params['resistance_20c'] * params['resistance_factor']:.4e},
    'temp_coefficient': 0.0039,
    'resistance_factor': {params['resistance_factor']:.4f}
}}

# Validation metrics
CALIBRATION_METRICS = {{
    'mean_residual': {calibrator.val_stats['mean_residual']:.4f},
    'rms_residual': {calibrator.val_stats['rms_residual']:.4f},
    'p95_range': [{calibrator.val_stats['percentiles']['5th']:.3f}, 
                  {calibrator.val_stats['percentiles']['95th']:.3f}]
}}
""")
    
    print(f"\n✅ Calibration complete! Results saved to {args.output_dir}/")


if __name__ == "__main__":
    main()
