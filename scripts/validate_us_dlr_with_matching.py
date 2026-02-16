"""
Invariant-PIKAN: Adversarially-Robust Physics-Informed Neural Networks for Dynamic Line Rating
Copyright (C) 2025 Gelavizh Ahmadi / Invariant Research

This software is licensed under the Business Source License 1.1 (BSL 1.1).
Commercial production use requires a separate license agreement.
See LICENSE.txt for full terms.

HWF-PIKAN for plasma physics (Heravifard et al., Sharif University, 2025).
"""

import torch
import pandas as pd
import numpy as np
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent.parent))

def load_model_with_checkpoint_architecture(checkpoint_path, device):
    """
    Load a model checkpoint by first inspecting its architecture
    and creating a matching model instance.

    This function handles both plain state_dict files and full training
    checkpoints (containing 'model_state_dict'). It also strips common
    prefixes like 'module.' when present.
    """
    raw = torch.load(checkpoint_path, map_location='cpu')

    # extract model state_dict if wrapped in a checkpoint
    if isinstance(raw, dict) and ('model_state_dict' in raw or 'state_dict' in raw):
        state_dict = raw.get('model_state_dict', raw.get('state_dict'))
    else:
        state_dict = raw

    # normalize keys (remove 'module.' prefix if present)
    normalized = {}
    for k, v in state_dict.items():
        nk = k[len('module.'):] if k.startswith('module.') else k
        normalized[nk] = v
    state_dict = normalized

    # Infer architecture from state dict keys (embedding parameter shapes)
    embedding_keys = [k for k in state_dict.keys() if 'embedding' in k]

    # Default config (will be overridden when possible)
    config = {
        'input_dim': 4,
        'fourier_bands': 16,
        'wavelet_scales': 4,
        'hidden_dim': 64,
        'kan_grid': 5,
        'kan_k': 3
    }

    for key in embedding_keys:
        if 'freqs' in key or 'freq' in key:
            # expected shape: [input_dim, fourier_bands]
            s = state_dict[key].shape
            if len(s) == 2:
                config['input_dim'] = int(s[0])
                config['fourier_bands'] = int(s[1])
                print(f"📐 Inferred: input_dim={config['input_dim']}, fourier_bands={config['fourier_bands']}")
        if 'scales' in key:
            s = state_dict[key].shape
            if len(s) == 1:
                config['wavelet_scales'] = int(s[0])
            elif len(s) == 2:
                # possible shape [input_dim, scales]
                config['wavelet_scales'] = int(s[1])
            print(f"📐 Inferred: wavelet_scales={config['wavelet_scales']}")

    # Infer KAN hidden dimension if available
    kan0 = 'kan.layers.0.weight'
    if kan0 in state_dict:
        config['hidden_dim'] = int(state_dict[kan0].shape[0])
        print(f"📐 Inferred: kan hidden_dim={config['hidden_dim']}")

    # Create model with inferred architecture
    from models.invariant_pikan_v2 import create_invariant_pikan_v2
    model = create_invariant_pikan_v2(config=config)

    # Load the normalized state_dict into the model (allow missing keys if shapes differ)
    try:
        model.load_state_dict(state_dict)
    except RuntimeError as e:
        # Provide a clearer error if keys still mismatch
        raise RuntimeError(f"Failed to load checkpoint into inferred model: {e}\n" \
                           f"Checkpoint keys sample: {list(state_dict.keys())[:8]}\n" \
                           f"Model keys sample: {list(model.state_dict().keys())[:8]}")

    model.to(device)
    model.eval()
    print(f"✅ Model loaded successfully with inferred architecture")
    return model, config


def sanitize_weather_data(df):
    """Automatically detect and fix common data issues"""
    
    # Make a copy
    df = df.copy()
    
    # 1. Temperature sanitization
    if 'temperature' in df.columns:
        # If temperatures are extremely low, they might be in Kelvin
        if df['temperature'].min(skipna=True) < -200:
            print("⚠️  Detected extremely low temps - converting from Kelvin to Celsius")
            df['temperature'] = df['temperature'] - 273.15
        
        # If temperatures are extremely high, might be scaled
        if df['temperature'].max(skipna=True) > 200:
            print("⚠️  Detected extremely high temps - possible scaling issue")
            # Could be stored as tenths of degrees
            if df['temperature'].max(skipna=True) > 1000:
                df['temperature'] = df['temperature'] / 10
                print("   Divided by 10")
        
        # Clamp to physically plausible range (conductor temperature)
        df['temperature'] = df['temperature'].clip(lower=-50, upper=150)
    else:
        # no temperature column — nothing to sanitize here
        pass
    
    # 2. Wind speed sanitization
    if 'wind_speed' in df.columns:
        if df['wind_speed'].max(skipna=True) > 100:
            print("⚠️  Wind speed >100 m/s detected - dividing by 10")
            df['wind_speed'] = df['wind_speed'] / 10
        df['wind_speed'] = df['wind_speed'].clip(lower=0, upper=50)

    # 3. Solar irradiance sanitization (support both 'solar' and 'solar_irradiance')
    if 'solar' in df.columns:
        df['solar'] = df['solar'].clip(lower=0, upper=1500)
        if 'solar_irradiance' not in df.columns:
            df['solar_irradiance'] = df['solar']
    elif 'solar_irradiance' in df.columns:
        df['solar_irradiance'] = df['solar_irradiance'].clip(lower=0, upper=1500)
        if 'solar' not in df.columns:
            df['solar'] = df['solar_irradiance']
    
    # 4. DLR ampacity sanitization (be tolerant to column name variants)
    amp_candidates = [c for c in ('actual', 'actual_dlr_ampacity', 'dlr_amps', 'dlr_ratio', 'dlr') if c in df.columns]
    if amp_candidates:
        amp_col = amp_candidates[0]
        df[amp_col] = pd.to_numeric(df[amp_col], errors='coerce')
        if df[amp_col].max(skipna=True) < 100:
            print("⚠️  DLR values seem low - converting from kA to A")
            df[amp_col] = df[amp_col] * 1000
        # ensure there is a canonical 'actual' column for downstream code
        if 'actual' not in df.columns:
            df['actual'] = df[amp_col]

    print(f"✅ Sanitized {len(df)} rows")
    return df


def validate_with_matched_data(model_path, weather_csv, output_dir):
    """
    Validation using ONLY matched real weather-DLR pairs
    No synthetic scenarios
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load model with architecture inference
    print(f"📦 Loading model from {model_path}...")
    model, model_config = load_model_with_checkpoint_architecture(model_path, device)
    
    # Load matched weather-DLR data
    df = pd.read_csv(weather_csv, parse_dates=['timestamp'])
    # perform automatic sanitization/normalization of common issues
    df = sanitize_weather_data(df)

    # detect target column (support common names)
    target_col = None
    for c in ('actual_dlr_ampacity', 'dlr_amps', 'dlr_ratio', 'dlr'):
        if c in df.columns:
            target_col = c
            break
    if target_col is None:
        raise ValueError('No ampacity column found in CSV (expected actual_dlr_ampacity, dlr_amps, dlr_ratio or dlr)')

    # Ensure numeric and detect units: some datasets store ampacity in kA or as small ratios
    df[target_col] = pd.to_numeric(df[target_col], errors='coerce')
    if df[target_col].max(skipna=True) < 100:
        # likely stored in kA or per-unit; assume kA -> convert to A
        print(f"⚠️ Detected small values in '{target_col}' (max < 100). Assuming kA and converting to A by *1000.")
        df[target_col] = df[target_col] * 1000.0

    # Filter to only rows with valid DLR values (now in amps)
    df = df[(df[target_col] > 100) & (df[target_col] < 5000)].reset_index(drop=True)

    print(f"📊 Validating on {len(df)} matched real samples (using '{target_col}' as truth)")

    predictions = []
    actuals = []

    for idx, row in df.iterrows():
        # Prepare input tensor - match model's expected dimensions
        x = torch.tensor([[
            row.get('temperature', 0.0),
            row.get('wind_speed', 0.0),
            row.get('wind_direction', 0.0) if 'wind_direction' in row else 0.0,
            row.get('solar_irradiance', 0.0)
        ]], dtype=torch.float32).to(device)

        weather_dict = {
            'T_amb': torch.tensor([row.get('temperature', 0.0)], dtype=torch.float32).to(device),
            'wind_speed': torch.tensor([row.get('wind_speed', 0.0)], dtype=torch.float32).to(device),
            'solar': torch.tensor([row.get('solar_irradiance', 0.0)], dtype=torch.float32).to(device)
        }

        with torch.no_grad():
            pred = model(x, weather_dict)
            predictions.append(pred['ampacity'].item())
            actuals.append(float(row[target_col]))

        if idx % 1000 == 0 and idx > 0:
            print(f"   Processed {idx}/{len(df)} samples")    
    # Calculate metrics
    predictions = np.array(predictions)
    actuals = np.array(actuals)
    
    mae = np.mean(np.abs(predictions - actuals))
    rmse = np.sqrt(np.mean((predictions - actuals)**2))
    
    # Results by condition
    results = {
        'overall': {'mae': float(mae), 'rmse': float(rmse), 'samples': len(df)},
        'by_wind': {},
        'by_temp': {}
    }
    
    # Wind speed bins
    for low, high in [(0,2), (2,5), (5,10), (10,100)]:
        mask = (df['wind_speed'] >= low) & (df['wind_speed'] < high)
        if mask.sum() > 0:
            results['by_wind'][f'{low}-{high} m/s'] = {
                'mae': float(np.mean(np.abs(predictions[mask] - actuals[mask]))),
                'samples': int(mask.sum())
            }
    
    # Temperature bins
    for low, high in [(-50,0), (0,15), (15,25), (25,50)]:
        mask = (df['temperature'] >= low) & (df['temperature'] < high)
        if mask.sum() > 0:
            results['by_temp'][f'{low}-{high} °C'] = {
                'mae': float(np.mean(np.abs(predictions[mask] - actuals[mask]))),
                'samples': int(mask.sum())
            }
    
    # Save results
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    pd.DataFrame({
        'line_id': df.get('line_id', pd.Series(index=df.index, dtype=object)).values,
        'timestamp': df.get('timestamp', pd.Series(index=df.index, dtype=object)).astype(str).values,
        'actual': actuals,
        'predicted': predictions,
        'error': predictions - actuals,
        'wind_speed': df['wind_speed'].values,
        'temperature': df['temperature'].values,
        'solar': df['solar_irradiance'].values
    }).to_csv(output_dir / 'matched_validation_results.csv', index=False)
    
    # Save summary
    with open(output_dir / 'validation_summary.txt', 'w') as f:
        f.write(f"MATCHED REAL-WEATHER VALIDATION RESULTS\n")
        f.write(f"{'='*50}\n")
        f.write(f"Model architecture: {model_config}\n")
        f.write(f"\nOverall MAE: {mae:.2f} A\n")
        f.write(f"Overall RMSE: {rmse:.2f} A\n")
        f.write(f"Samples: {len(df)}\n\n")
        
        f.write("Performance by Wind Speed:\n")
        for k, v in results['by_wind'].items():
            f.write(f"  {k}: {v['mae']:.2f} A ({v['samples']} samples)\n")
        
        f.write("\nPerformance by Temperature:\n")
        for k, v in results['by_temp'].items():
            f.write(f"  {k}: {v['mae']:.2f} A ({v['samples']} samples)\n")
    
    print(f"\n{'='*50}")
    print(f"MATCHED REAL-WEATHER VALIDATION RESULTS")
    print(f"{'='*50}")
    print(f"Overall MAE: {mae:.2f} A")
    print(f"Overall RMSE: {rmse:.2f} A")
    print(f"Samples: {len(df)}")
    print(f"\n✅ Results saved to {output_dir}")
    
    return results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-path', required=True)
    parser.add_argument('--weather-csv', required=True)
    parser.add_argument('--output-dir', required=True)
    args = parser.parse_args()
    
    validate_with_matched_data(
        model_path=args.model_path,
        weather_csv=args.weather_csv,
        output_dir=args.output_dir
    )
