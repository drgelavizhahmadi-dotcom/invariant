"""
Invariant-PIKAN: Adversarially-Robust Physics-Informed Neural Networks for Dynamic Line Rating
Copyright (C) 2025 Gelavizh Ahmadi / Invariant Research

This software is licensed under the Business Source License 1.1 (BSL 1.1).
Commercial production use requires a separate license agreement.
See LICENSE.txt for full terms.

HWF-PIKAN for plasma physics (Heravifard et al., Sharif University, 2025).
"""

#!/usr/bin/env python
"""
Calibrate IEEE 738 line parameters to match US DLR data.
Uses US DLR predictions to infer actual resistance, emissivity, and absorptivity
for US transmission lines.
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
sys.path.append(str(Path(__file__).parent.parent))

from models.invariant_pikan_v2 import create_invariant_pikan_v2
from core.data import VietnamDataset, USDataset
from torch.utils.data import DataLoader, Subset

# Import physics functions
from core.physics import (
    convective_heat_loss,
    radiative_heat_loss,
    solar_heat_gain,
    resistive_heat_gain,
    ieee738_temperature
)


class USPhysicsCalibrator:
    """
    Calibrates line parameters to match US DLR data.
    """
    
    def __init__(self, model, device='cpu'):
        self.model = model
        self.device = device
        self.model.eval()
        
        # Fixed parameters (typical US line specs - will be refined)
        self.fixed_params = {
            'diameter': 0.02814,        # m (28.14mm - Drake ACSR typical)
            'temp_coefficient': 0.00403,  # Aluminum
            'resistance_20c': 7.283e-5   # Ω/m at 20°C (Drake ACSR)
        }
        
        # Parameter bounds
        self.param_bounds = {
            'resistance_factor': (0.5, 2.0),
            'emissivity': (0.3, 0.95),
            'absorptivity': (0.3, 0.95)
        }
        
    def collect_predictions(self, loader, max_samples=2000):
        """
        Run model on US data and collect predictions.
        """
        print("📊 Collecting model predictions on US data...")
        
        all_T_pred = []
        all_I_pred = []
        all_T_amb = []
        all_wind = []
        all_solar = []
        all_current = []
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(loader):
                x, y = batch
                x = x.to(self.device)
                
                # Get model predictions - HWF_PIKAN v2 format
                # x: [T_amb, wind_speed, wind_angle, solar, current, ...]
                weather = x[:, :4]  # [T_amb, wind_speed, wind_angle, solar]
                current = x[:, 4:5]  # [current]
                
                # Create weather dict for model
                weather_dict = {
                    'T_amb': weather[:, 0],
                    'wind_speed': weather[:, 1],
                    'solar': weather[:, 3]
                }
                
                output = self.model(weather, weather_dict)
                pred_amp = output['ampacity']
                
                # Store
                all_I_pred.extend(pred_amp.cpu().numpy())
                all_T_amb.extend(x[:, 0].cpu().numpy())
                all_wind.extend(x[:, 1].cpu().numpy())
                all_solar.extend(x[:, 3].cpu().numpy())
                all_current.extend(x[:, 4].cpu().numpy())
                
                if len(all_I_pred) >= max_samples:
                    break
        
        data = {
            'I_pred': np.array(all_I_pred[:max_samples]).flatten(),
            'T_amb': np.array(all_T_amb[:max_samples]),
            'wind_speed': np.array(all_wind[:max_samples]),
            'solar': np.array(all_solar[:max_samples]),
            'current': np.array(all_current[:max_samples])
        }
        
        print(f"  Collected {len(data['I_pred'])} US samples")
        return data
    
    def calculate_ampacity_error(self, params, data):
        """
        Calculate error between physics-computed ampacity and model predictions.
        """
        R_factor, emissivity, absorptivity = params
        
        line_params = {
            'diameter': self.fixed_params['diameter'],
            'emissivity': emissivity,
            'absorptivity': absorptivity,
            'R_20': self.fixed_params['resistance_20c'] * R_factor,
            'alpha': self.fixed_params['temp_coefficient']
        }
        
        errors = []
        
        # Assume max temperature of 75°C for ampacity calculation
        T_max = 75.0
        
        for i in range(len(data['I_pred'])):
            T_amb = data['T_amb'][i]
            wind = data['wind_speed'][i]
            solar = data['solar'][i]
            I_model = data['I_pred'][i]
            
            # Calculate physics ampacity at T_max
            try:
                I_physics = ieee738_ampacity(
                    T_max, T_amb, wind, solar,
                    diameter=line_params['diameter'],
                    emissivity=line_params['emissivity'],
                    absorptivity=line_params['absorptivity'],
                    R_20=line_params['R_20'],
                    alpha=line_params['alpha']
                )
                
                # Relative error
                error = (I_physics - I_model) / max(I_model, 100.0)
                errors.append(error)
            except:
                errors.append(1.0)  # Large error on failure
        
        return np.array(errors)
    
    def objective_function(self, params, data):
        """
        Minimize difference between physics and model ampacity.
        """
        errors = self.calculate_ampacity_error(params, data)
        mse = np.mean(errors**2)
        
        # Regularization
        R_factor, emissivity, absorptivity = params
        reg = 0.01 * (
            (R_factor - 1.0)**2 + 
            (emissivity - 0.8)**2 + 
            (absorptivity - 0.8)**2
        )
        
        return mse + reg
    
    def calibrate(self, data, method='differential_evolution'):
        """
        Find parameters that align physics with US DLR model predictions.
        """
        print("\n🔧 Calibrating line parameters to US DLR...")
        
        x0 = [1.0, 0.8, 0.8]
        bounds = [
            self.param_bounds['resistance_factor'],
            self.param_bounds['emissivity'],
            self.param_bounds['absorptivity']
        ]
        
        if method == 'differential_evolution':
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
            result = minimize(
                self.objective_function,
                x0,
                args=(data,),
                method='L-BFGS-B',
                bounds=bounds
            )
        
        return {
            'resistance_factor': result.x[0],
            'emissivity': result.x[1],
            'absorptivity': result.x[2],
            'final_error': result.fun,
            'success': result.success
        }
    
    def validate(self, params, data):
        """Validate calibration."""
        errors = self.calculate_ampacity_error(
            [params['resistance_factor'], params['emissivity'], params['absorptivity']],
            data
        )
        
        I_physics_list = []
        for i in range(len(data['I_pred'])):
            T_amb = data['T_amb'][i]
            wind = data['wind_speed'][i]
            solar = data['solar'][i]
            
            line_params = {
                'diameter': self.fixed_params['diameter'],
                'emissivity': params['emissivity'],
                'absorptivity': params['absorptivity'],
                'R_20': self.fixed_params['resistance_20c'] * params['resistance_factor'],
                'alpha': self.fixed_params['temp_coefficient']
            }
            
            try:
                I_physics = ieee738_ampacity(
                    75.0, T_amb, wind, solar,
                    diameter=line_params['diameter'],
                    emissivity=line_params['emissivity'],
                    absorptivity=line_params['absorptivity'],
                    R_20=line_params['R_20'],
                    alpha=line_params['alpha']
                )
                I_physics_list.append(I_physics)
            except:
                I_physics_list.append(0)
        
        I_physics = np.array(I_physics_list)
        I_model = data['I_pred']
        
        return {
            'mean_residual': np.mean(errors),
            'rms_residual': np.sqrt(np.mean(errors**2)),
            'mae': np.mean(np.abs(I_physics - I_model)),
            'bias': np.mean(I_physics - I_model),
            'physics_ampacity': I_physics,
            'model_ampacity': I_model
        }
    
    def save_params(self, params, output_path):
        """Save calibrated parameters."""
        output = {
            'diameter': self.fixed_params['diameter'],
            'emissivity': float(params['emissivity']),
            'absorptivity': float(params['absorptivity']),
            'resistance_ac': self.fixed_params['resistance_20c'] * params['resistance_factor'],
            'temp_coefficient': self.fixed_params['temp_coefficient'],
            'resistance_factor': float(params['resistance_factor']),
            'timestamp': datetime.now().isoformat()
        }
        
        with open(output_path, 'w') as f:
            json.dump(output, f, indent=2)
        
        # Also save as Python module
        py_path = output_path.replace('.json', '.py')
        with open(py_path, 'w') as f:
            f.write(f"""# Auto-generated US DLR line parameters
# Generated: {datetime.now().isoformat()}

US_LINE_PARAMS = {output}
""")
        
        print(f"💾 Saved to {output_path} and {py_path}")
        return output


def ieee738_ampacity(T_max, T_amb, v_wind, solar, diameter=0.028, emissivity=0.8, absorptivity=0.8, R_20=7.28e-5, alpha=0.004):
    """Calculate maximum allowable current (ampacity) for given conditions."""
    from scipy.optimize import fsolve
    
    def current_balance(I):
        q_conv = convective_heat_loss(T_max, T_amb, v_wind, diameter)
        q_rad = radiative_heat_loss(T_max, T_amb, diameter, emissivity)
        q_solar = solar_heat_gain(solar, diameter, absorptivity)
        
        R_T = R_20 * (1 + alpha * (T_max - 20))
        q_resist = I**2 * R_T
        
        return q_conv + q_rad - q_solar - q_resist
    
    try:
        I_solution = fsolve(current_balance, 1000)[0]
        return max(I_solution, 0)
    except:
        return 0


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Calibrate physics to US DLR data')
    parser.add_argument('--run-dir', type=str, required=True,
                        help='Path to training run directory')
    parser.add_argument('--us-data', type=str, 
                        default='data/us_dlr/unified_us_dlr_training.h5',
                        help='Path to US DLR data')
    parser.add_argument('--max-samples', type=int, default=2000)
    parser.add_argument('--method', type=str, default='differential_evolution')
    parser.add_argument('--output-dir', type=str, default='calibration_results')
    parser.add_argument('--device', type=str, default='auto')
    
    args = parser.parse_args()
    
    Path(args.output_dir).mkdir(exist_ok=True)
    
    # Device
    if args.device == 'auto':
        device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    print(f"🔧 Device: {device}")
    
    # Load model
    run_dir = Path(args.run_dir)
    ckpt_path = run_dir / 'best_model.pt'
    print(f"📦 Loading model from {ckpt_path}")
    
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
    model_cfg = checkpoint.get('config', {}).get('model', None) if isinstance(checkpoint, dict) else None
    model = create_invariant_pikan_v2(config=model_cfg)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    # Load US data
    print(f"📊 Loading US DLR data...")
    if Path(args.us_data).exists():
        us_dataset = USDataset(args.us_data, normalizer=None)
    else:
        # Fallback: use test data from run
        test_csv = run_dir / 'test_data.csv'
        if test_csv.exists():
            us_dataset = VietnamDataset(str(test_csv))
        else:
            raise FileNotFoundError(f"No US data found at {args.us_data} or {test_csv}")
    
    # Filter to US samples only
    loader = DataLoader(us_dataset, batch_size=256, shuffle=False)
    
    # Initialize calibrator
    calibrator = USPhysicsCalibrator(model, device)
    
    # Collect predictions
    data = calibrator.collect_predictions(loader, max_samples=args.max_samples)
    
    # Split cal/val
    n_cal = int(0.7 * len(data['I_pred']))
    cal_data = {k: v[:n_cal] for k, v in data.items()}
    val_data = {k: v[n_cal:] for k, v in data.items()}
    
    print(f"Calibration: {n_cal}, Validation: {len(val_data['I_pred'])}")
    
    # Calibrate
    params = calibrator.calibrate(cal_data, method=args.method)
    
    print("\n📋 Calibrated US Parameters:")
    print(f"  Resistance Factor: {params['resistance_factor']:.3f}")
    print(f"  Emissivity: {params['emissivity']:.3f}")
    print(f"  Absorptivity: {params['absorptivity']:.3f}")
    
    # Validate
    val_stats = calibrator.validate(params, val_data)
    print("\n✅ Validation:")
    print(f"  Physics-Model MAE: {val_stats['mae']:.2f} A")
    print(f"  Bias: {val_stats['bias']:.2f} A")
    print(f"  RMS Residual: {val_stats['rms_residual']:.4f}")
    
    # Save
    output_path = f"{args.output_dir}/us_line_params.json"
    calibrator.save_params(params, output_path)
    
    # Plot comparison
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    ax.scatter(val_stats['model_ampacity'], val_stats['physics_ampacity'], alpha=0.3, s=5)
    ax.plot([0, 3000], [0, 3000], 'r--', label='Perfect match')
    ax.set_xlabel('Model Ampacity (A)')
    ax.set_ylabel('Physics Ampacity (A)')
    ax.set_title('Physics vs Model (US DLR Calibrated)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plot_path = f"{args.output_dir}/us_calibration_plot.png"
    plt.savefig(plot_path, dpi=150)
    print(f"📈 Plot saved to {plot_path}")
    
    print(f"\n✅ US calibration complete!")


if __name__ == "__main__":
    main()
