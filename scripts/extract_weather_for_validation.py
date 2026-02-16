#!/usr/bin/env python3
"""
extract_weather_for_validation.py

Extract weather data for US transmission lines to enable proper validation
of your HWF-PIKAN model against the US DLR dataset.

This script shows how to:
1. Load US transmission line locations from HIFLD dataset
2. Extract matching weather data from WIND Toolkit and NSRDB
3. Create a dataset that pairs weather data with DLR ratios
4. Use this for proper model validation
"""

import pandas as pd
import numpy as np
import h5py
import h5pyd
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent))

def load_us_dlr_metadata(us_dlr_path):
    """Load US DLR dataset metadata (line IDs, voltages, etc.)"""
    print("📊 Loading US DLR metadata...")

    # Load pandas HDF5 file using h5py since pandas.read_hdf has issues
    with h5py.File(us_dlr_path, 'r') as f:
        # Get column names (line IDs)
        columns = f['columns'][:].astype(str)
        # Get data shape to understand structure
        data_shape = f['data'].shape
        print(f"   Data shape: {data_shape}")

    # Create metadata DataFrame
    metadata = pd.DataFrame({
        'line_id': columns,
        'voltage_kv': np.full(len(columns), 230.0)  # Assume 230kV for now
    })

    print(f"✅ Loaded metadata for {len(metadata)} transmission lines")
    return metadata

def get_sample_line_locations():
    """
    Get sample lat/lon coordinates for US transmission lines.
    In practice, you'd load this from the HIFLD dataset.
    For now, we'll use representative locations across the US.
    """
    # Sample locations representing major transmission corridors
    # In reality, you'd load exact coordinates from HIFLD dataset
    sample_locations = {
        '100001': {'lat': 40.7128, 'lon': -74.0060},  # NYC area
        '100002': {'lat': 34.0522, 'lon': -118.2437}, # LA area
        '100003': {'lat': 41.8781, 'lon': -87.6298},  # Chicago area
        '100004': {'lat': 29.7604, 'lon': -95.3698},  # Houston area
        '100005': {'lat': 33.4484, 'lon': -112.0740}, # Phoenix area
    }

    return sample_locations

def extract_weather_for_lines(line_locations, years=[2010, 2011, 2012]):
    """
    Extract weather data for specific transmission line locations.

    This is a demonstration of the approach. In practice, you'd:
    1. Use the DLR tools to extract data for all line locations
    2. Handle the large data volumes appropriately
    3. Cache results to avoid re-downloading
    """
    print("🌤️  Extracting weather data for transmission lines...")
    print("⚠️  Note: This is a demonstration. Real extraction requires:")
    print("   - NREL API key and HSDS setup")
    print("   - Full HIFLD coordinates (not sample data)")
    print("   - Significant storage space (TB scale)")

    # This would be the actual extraction code using DLR tools
    weather_data = {}

    for line_id, coords in line_locations.items():
        print(f"   Processing {line_id} at {coords['lat']:.2f}, {coords['lon']:.2f}")

        # Placeholder for actual weather extraction
        # In practice, you'd use:
        # from dlr.weather import get_weather_data
        # weather = get_weather_data(lat=coords['lat'], lon=coords['lon'], years=years)

        # Create synthetic weather data for demonstration
        n_hours = len(years) * 8760  # Approximate hours per year
        weather_data[line_id] = {
            'temperature': np.random.normal(20, 10, n_hours),  # °C
            'wind_speed': np.random.exponential(3, n_hours),   # m/s
            'wind_direction': np.random.uniform(0, 360, n_hours),  # degrees
            'solar_irradiance': np.random.beta(2, 5, n_hours) * 1000,  # W/m²
            'timestamp': pd.date_range(f'{years[0]}-01-01', f'{years[-1]}-12-31 23:00:00', freq='H')[:n_hours]
        }

    print(f"✅ Extracted weather data for {len(weather_data)} lines")
    return weather_data

def create_validation_dataset(us_dlr_path, weather_data, output_path):
    """
    Create a validation dataset that pairs weather data with DLR ratios.
    """
    print("🔗 Creating validation dataset...")

    # Load DLR data using h5py
    with h5py.File(us_dlr_path, 'r') as f:
        dlr_ratios = f['data'][:]
        columns = f['columns'][:].astype(str)
        index = f['index'][:]

    # Convert index to timestamps
    timestamps = pd.to_datetime(index.astype(str))
    line_ids = columns

    # Create validation DataFrame
    validation_records = []

    for i, line_id in enumerate(line_ids[:5]):  # Process first 5 lines as example
        if line_id in weather_data:
            weather = weather_data[line_id]

            # Match timestamps (simplified - real implementation would need careful alignment)
            for j, ts in enumerate(timestamps[:100]):  # First 100 timestamps
                if j < len(weather['timestamp']):
                    ratio = dlr_ratios[j, i]

                    record = {
                        'timestamp': ts,
                        'line_id': line_id,
                        'voltage_kv': 230,  # Would load from metadata
                        'temperature': weather['temperature'][j],
                        'wind_speed': weather['wind_speed'][j],
                        'wind_direction': weather['wind_direction'][j],
                        'solar_irradiance': weather['solar_irradiance'][j],
                        'dlr_ratio': ratio,
                        'actual_dlr_ampacity': ratio  # Ratio is already the ampacity value
                    }
                    validation_records.append(record)

    df_out = pd.DataFrame(validation_records)
    df_out.to_csv(output_path, index=False)

    print(f"✅ Created validation dataset with {len(df_out)} records")
    print(f"   Saved to: {output_path}")
    return df_out

def main():
    import argparse

    parser = argparse.ArgumentParser(description='Extract weather data for US DLR validation')
    parser.add_argument('--us-dlr-path', type=str, default='data/us_dlr_2007_2013.h5',
                       help='Path to US DLR HDF5 file')
    parser.add_argument('--output-path', type=str, default='data/us_validation_weather_data.csv',
                       help='Output path for validation dataset')
    parser.add_argument('--years', type=int, nargs='+', default=[2010, 2011, 2012],
                       help='Years to extract weather data for')

    args = parser.parse_args()

    # Check if US DLR file exists
    if not Path(args.us_dlr_path).exists():
        print(f"❌ US DLR file not found: {args.us_dlr_path}")
        print("   Please download it from: https://data.openei.org/files/6231/DLR_SLR_ratio-75C.h5")
        return

    try:
        # Load metadata
        metadata = load_us_dlr_metadata(args.us_dlr_path)

        # Get sample line locations (replace with real HIFLD data)
        line_locations = get_sample_line_locations()

        # Extract weather data
        weather_data = extract_weather_for_lines(line_locations, args.years)

        # Create validation dataset
        validation_df = create_validation_dataset(args.us_dlr_path, weather_data, args.output_path)

        print("\n🎉 Validation dataset created successfully!")
        print(f"   Records: {len(validation_df)}")
        print(f"   Lines: {validation_df['line_id'].nunique()}")
        print(f"   Date range: {validation_df['timestamp'].min()} to {validation_df['timestamp'].max()}")

        print("\n📋 Next steps:")
        print("1. Update your validate_us_dlr.py script to use this real weather data")
        print("2. Replace synthetic weather with actual WIND Toolkit/NSRDB data")
        print("3. Run proper validation: python scripts/validate_us_dlr.py --model-path your_model.pt")

    except Exception as e:
        print(f"❌ Error: {e}")
        print("\n🔧 Troubleshooting:")
        print("1. Make sure you have an NREL API key: export NREL_API_KEY='your_key'")
        print("2. Run the setup script: ./setup_weather_data_access.sh")
        print("3. Check network connectivity to NREL HSDS service")

if __name__ == "__main__":
    main()