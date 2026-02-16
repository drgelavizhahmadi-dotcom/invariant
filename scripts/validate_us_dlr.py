import torch
import h5py
import numpy as np
import pandas as pd
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent.parent))

from core.physics import IEEE738HeatBalance
from models.hwf_pikan_v2 import create_hwf_pikan_v2

def validate_on_us_dlr(model_path, us_dlr_path, output_dir='us_validation', weather_csv=None):
    """
    Validate your trained model against US DLR dataset

    Args:
        model_path: Path to trained model
        us_dlr_path: Path to US DLR HDF5 file
        output_dir: Output directory
        weather_csv: Optional CSV with real weather data (columns: timestamp, line_id, temperature, wind_speed, wind_direction, solar_irradiance)
    """

    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)

    print("⚠️  LIMITATION: US DLR dataset contains only pre-computed ratios")
    print("   To validate properly, we need matching weather data from:")
    print("   - WIND Toolkit (wind.speed, direction)")
    print("   - NSRDB (solar irradiance)")
    print("   - NOAA/NCDC (temperature, humidity)")
    print("\n   For now, this script demonstrates the validation approach with synthetic weather.\n")

    # Check if real weather data is provided
    use_real_weather = weather_csv is not None and Path(weather_csv).exists()
    if use_real_weather:
        print(f"✅ Using real weather data from: {weather_csv}")
        weather_df = pd.read_csv(weather_csv)
        weather_df['timestamp'] = pd.to_datetime(weather_df['timestamp'])
    else:
        print("🎭 Using synthetic weather data (provide --weather-csv for real validation)")
        weather_df = None

    # Load US DLR data
    print("📊 Loading US DLR dataset...")
    try:
        with h5py.File(us_dlr_path, 'r') as f:
            print(f"HDF5 keys: {list(f.keys())}")

            # The actual structure may differ - let's inspect
            if 'columns' in f and 'data' in f:
                # pandas HDF5 format
                print("Detected pandas HDF5 format")
                # We can't easily read this without pytables, but let's try h5py inspection
                columns = f['columns'][:].astype(str)
                print(f"Columns (line IDs): {columns[:5]}...")

                # For now, create synthetic validation data
                print("Creating synthetic validation data (replace with real weather when available)...")

    except Exception as e:
        print(f"Error loading HDF5: {e}")
        print("Creating synthetic validation data for demonstration...")

    # Load your model
    print("🔧 Loading your HWF-PIKAN model...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Use the same model configuration as the production training
    model_config = {
        'input_dim': 4,            # weather dims passed to embedding (T, wind, wind_angle, solar)
        'fourier_bands': 8,
        'wavelet_scales': 2,
        'hidden_dim': 32,
        'kan_grid': 3,
        'kan_k': 3,
    }
    model = create_hwf_pikan_v2(config=model_config)
    
    # Load checkpoint (may contain full training state or just state_dict)
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model.to(device)
    model.eval()

    # Create physics engine for validation
    physics = IEEE738HeatBalance()

    # Generate synthetic validation data (REPLACE WITH REAL WEATHER DATA)
    print("🎭 Generating synthetic validation scenarios...")
    n_samples = 1000

    # Create diverse weather scenarios
    weather_scenarios = {
        'summer_hot': {'T_amb': 35.0, 'wind_speed': 2.0, 'solar': 800.0},
        'winter_cold': {'T_amb': -5.0, 'wind_speed': 8.0, 'solar': 200.0},
        'spring_mild': {'T_amb': 15.0, 'wind_speed': 4.0, 'solar': 600.0},
        'fall_moderate': {'T_amb': 20.0, 'wind_speed': 3.0, 'solar': 500.0},
        'high_wind': {'T_amb': 25.0, 'wind_speed': 12.0, 'solar': 700.0},
        'low_wind': {'T_amb': 30.0, 'wind_speed': 0.5, 'solar': 900.0},
    }

    # Validation results storage
    results = {
        'scenario': [],
        'voltage_kv': [],
        'predicted_ampacity': [],
        'physics_ampacity': [],
        'model_vs_physics_error': [],
        'wind_speed': [],
        'temperature': [],
        'solar': []
    }

    print(f"📈 Validating on {n_samples} synthetic scenarios...")

    for i in range(n_samples):
        # Sample random scenario and voltage
        scenario_name = np.random.choice(list(weather_scenarios.keys()))
        weather = weather_scenarios[scenario_name]
        voltage = np.random.choice([115, 230, 345, 500, 765])  # Common US voltages

        # If we have real weather data, use it instead of synthetic
        if use_real_weather and i < len(weather_df):
            real_weather = weather_df.iloc[i]
            weather = {
                'T_amb': real_weather['temperature'],
                'wind_speed': real_weather['wind_speed'],
                'solar': real_weather['solar_irradiance']
            }
            voltage = real_weather.get('voltage_kv', voltage)
            scenario_name = f"real_data_{i}"

        # Run your model
        with torch.no_grad():
            # Prepare input (weather tensor: [T_amb, wind_speed, wind_direction, solar])
            # force float32 to match model parameters and avoid dtype mismatch
            weather = torch.tensor([
                [
                    weather['T_amb'],
                    weather['wind_speed'],
                    0.0,  # wind direction (placeholder)
                    weather['solar']
                ]
            ], dtype=torch.float32, device=device)
            
            weather_dict = {
                'T_amb': weather[:, 0],
                'wind_speed': weather[:, 1],
                'solar': weather[:, 3]
            }
            
            pred = model(weather, weather_dict)
            predicted_ampacity = pred['ampacity'].item()

        # Compare with physics-based calculation
        physics_result = physics.ampacity(
            T_max=75.0,
            T_ambient=weather[0, 0],
            wind_speed=weather[0, 1],
            solar_irradiance=weather[0, 3]
        )
        physics_ampacity = physics_result.item()

        # Store results
        results['scenario'].append(scenario_name)
        results['voltage_kv'].append(voltage)
        results['predicted_ampacity'].append(predicted_ampacity)
        results['physics_ampacity'].append(physics_ampacity)
        results['model_vs_physics_error'].append(abs(predicted_ampacity - physics_ampacity))
        results['wind_speed'].append(weather[0, 1].item())
        results['temperature'].append(weather[0, 0].item())
        results['solar'].append(weather[0, 3].item())

    # Convert to DataFrame and analyze
    df = pd.DataFrame(results)

    # Calculate metrics
    mae = df['model_vs_physics_error'].mean()
    rmse = np.sqrt((df['model_vs_physics_error']**2).mean())

    print(f"\n{'='*60}")
    print(f"SYNTHETIC US DLR VALIDATION RESULTS")
    print(f"{'='*60}")
    print(f"Mean Absolute Error vs Physics: {mae:.2f} A")
    print(f"Root Mean Square Error vs Physics: {rmse:.2f} A")
    print(f"Samples validated: {len(df)}")
    print(f"Voltage range: {df['voltage_kv'].min()}-{df['voltage_kv'].max()} kV")

    # Scenario breakdown
    print(f"\nScenario Performance:")
    scenario_stats = df.groupby('scenario')['model_vs_physics_error'].agg(['mean', 'std', 'count'])
    for scenario, stats in scenario_stats.iterrows():
        print(f"  {scenario}: {stats['mean']:.1f} ± {stats['std']:.1f} A ({int(stats['count'])} samples)")

    # Save results
    df.to_csv(output_dir / 'synthetic_us_validation_results.csv', index=False)
    print(f"\n✅ Results saved to {output_dir / 'synthetic_us_validation_results.csv'}")

    # Create summary report
    with open(output_dir / 'validation_summary.txt', 'w') as f:
        f.write("US DLR Validation Summary\n")
        f.write("========================\n\n")
        f.write("IMPORTANT: This validation uses synthetic weather data.\n")
        f.write("For real validation, obtain matching weather data from:\n")
        f.write("- WIND Toolkit: https://www.nrel.gov/grid/wind-toolkit.html\n")
        f.write("- NSRDB: https://nsrdb.nrel.gov/\n")
        f.write("- NOAA NCDC: https://www.ncdc.noaa.gov/\n\n")

        f.write(f"Model: {Path(model_path).name}\n")
        f.write(f"MAE vs Physics: {mae:.2f} A\n")
        f.write(f"RMSE vs Physics: {rmse:.2f} A\n")
        f.write(f"Samples: {len(df)}\n")

    print(f"📋 Summary saved to {output_dir / 'validation_summary.txt'}")

    return df


def main():
    import argparse

    parser = argparse.ArgumentParser(description='Validate HWF-PIKAN model on US DLR dataset')
    parser.add_argument('--model-path', type=str, required=True,
                       help='Path to trained model checkpoint')
    parser.add_argument('--us-dlr-path', type=str, default='data/us_dlr_2007_2013.h5',
                       help='Path to US DLR HDF5 file')
    parser.add_argument('--output-dir', type=str, default='us_validation',
                       help='Output directory for results')
    parser.add_argument('--weather-csv', type=str, default=None,
                       help='Optional CSV file with real weather data for validation')

    args = parser.parse_args()

    validate_on_us_dlr(args.model_path, args.us_dlr_path, args.output_dir, args.weather_csv)


if __name__ == "__main__":
    main()