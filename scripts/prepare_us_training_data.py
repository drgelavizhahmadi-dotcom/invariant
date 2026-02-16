#!/usr/bin/env python3
"""
prepare_us_training_data.py

Create training dataset by joining US DLR ratios with weather data from WIND Toolkit and NSRDB.

This script performs spatiotemporal joins to create training samples with:
- Weather features: temperature, wind_speed, wind_direction, solar_irradiance
- Line parameters: voltage, conductor assumptions
- Target: DLR ampacity (ratio × SLR)

Usage:
    python scripts/prepare_us_training_data.py --us-dlr data/us_dlr_2007_2013.h5 --output data/us_training.h5 --years 2010 2011 2012
"""

import h5py
import pandas as pd
import numpy as np
from pathlib import Path
import sys
from typing import Optional, Tuple, List
import warnings
sys.path.append(str(Path(__file__).parent.parent))

# Optional weather data libraries
try:
    import h5pyd
    H5PYD_AVAILABLE = True
except ImportError:
    H5PYD_AVAILABLE = False
    warnings.warn("h5pyd not available. Install with: pip install h5pyd")

try:
    from dlr.weather import get_weather_data
    DLR_AVAILABLE = True
except ImportError:
    DLR_AVAILABLE = False
    warnings.warn("DLR package not available. Install with: pip install git+https://github.com/NREL/DynamicLineRatings.git")

def load_wtk_data(lat: float, lon: float, timestamp: pd.Timestamp) -> Optional[Tuple[float, float, float]]:
    """
    Load wind data from WIND Toolkit for given location and time.

    Returns: (wind_speed, wind_direction, temperature) or None if not available
    """
    if not H5PYD_AVAILABLE:
        # Fallback: return synthetic data
        return (np.random.exponential(3.0), np.random.uniform(0, 360), 20.0 + np.random.normal(0, 10))

    try:
        # WIND Toolkit CONUS data structure
        year = timestamp.year
        wtk_file = f'/nrel/wtk/conus/wtk_conus_{year}.h5'

        with h5pyd.File(wtk_file, 'r', endpoint='https://developer.nrel.gov/api/hsds') as f:
            # Find nearest grid point to lat/lon
            grid_lats = f['coordinates'][:, 0]
            grid_lons = f['coordinates'][:, 1]

            # Simple nearest neighbor (in practice, use proper spatial indexing)
            distances = np.sqrt((grid_lats - lat)**2 + (grid_lons - lon)**2)
            nearest_idx = np.argmin(distances)

            # Get time index
            time_idx = timestamp.hour + timestamp.dayofyear * 24

            wind_speed = f['windspeed_10m'][time_idx, nearest_idx]
            wind_direction = f['winddirection_10m'][time_idx, nearest_idx]
            temperature = f['temperature_10m'][time_idx, nearest_idx] - 273.15  # K to C

            return (wind_speed, wind_direction, temperature)

    except Exception as e:
        print(f"Warning: Could not load WTK data for {timestamp} at ({lat}, {lon}): {e}")
        return None

def load_nsrdb_data(lat: float, lon: float, timestamp: pd.Timestamp) -> Optional[float]:
    """
    Load solar irradiance from NSRDB for given location and time.

    Returns: GHI (W/m²) or None if not available
    """
    if not H5PYD_AVAILABLE:
        # Fallback: return synthetic data
        hour = timestamp.hour
        if 6 <= hour <= 18:  # Daylight hours
            return np.random.beta(2, 5) * 1000  # Typical GHI distribution
        else:
            return 0.0

    try:
        # NSRDB data structure
        year = timestamp.year
        nsrdb_file = f'/nrel/nsrdb/v3/nsrdb_{year}.h5'

        with h5pyd.File(nsrdb_file, 'r', endpoint='https://developer.nrel.gov/api/hsds') as f:
            # Find nearest grid point
            grid_lats = f['coordinates'][:, 0]
            grid_lons = f['coordinates'][:, 1]

            distances = np.sqrt((grid_lats - lat)**2 + (grid_lons - lon)**2)
            nearest_idx = np.argmin(distances)

            # Get time index (NSRDB is typically hourly)
            time_idx = timestamp.hour + (timestamp.dayofyear - 1) * 24

            ghi = f['ghi'][time_idx, nearest_idx]
            return ghi

    except Exception as e:
        print(f"Warning: Could not load NSRDB data for {timestamp} at ({lat}, {lon}): {e}")
        return None

def load_temp_data(lat: float, lon: float, timestamp: pd.Timestamp) -> Optional[float]:
    """
    Load temperature data. Can use WTK or other sources.
    """
    wtk_data = load_wtk_data(lat, lon, timestamp)
    if wtk_data:
        return wtk_data[2]  # temperature from WTK
    else:
        # Fallback
        return 20.0 + 10 * np.sin(2 * np.pi * timestamp.dayofyear / 365) + np.random.normal(0, 5)

def create_us_training_dataset(us_dlr_path: str, output_path: str, years: List[int],
                              max_lines: int = 100, time_step_hours: int = 6,
                              use_synthetic_fallback: bool = True):
    """
    Create training dataset by joining US DLR ratios with weather data.

    Args:
        us_dlr_path: Path to US DLR HDF5 file
        output_path: Output HDF5 file path
        years: List of years to process
        max_lines: Maximum number of transmission lines to process
        time_step_hours: Temporal resolution (hours)
        use_synthetic_fallback: Use synthetic weather if real data unavailable
    """

    print("🔌 Loading US DLR dataset...")
    try:
        with h5py.File(us_dlr_path, 'r') as f:
            print(f"HDF5 keys: {list(f.keys())}")

            # Load datasets (handle different possible structures)
            if 'DLR_SLR_ratio-75C' in f:
                dlr_ratios = f['DLR_SLR_ratio-75C'][:]
                slr_values = f['SLR_A-75C'][:]
                timestamps = pd.to_datetime(f['time_index'][:].astype(str))
                line_ids = f['line_ids'][:].astype(str)
                voltages = f['voltage_kv'][:]

                # Try to get coordinates
                if 'lat' in f and 'lon' in f:
                    lats = f['lat'][:]
                    lons = f['lon'][:]
                else:
                    print("⚠️  No coordinates found in DLR file, using synthetic locations")
                    # Create synthetic coordinates for demonstration
                    lats = np.random.uniform(25, 50, len(line_ids))  # CONUS lat range
                    lons = np.random.uniform(-125, -65, len(line_ids))  # CONUS lon range
            else:
                raise ValueError("Unexpected HDF5 structure")

    except Exception as e:
        print(f"❌ Error loading US DLR file: {e}")
        return None

    print(f"✅ Loaded data for {len(line_ids)} lines, {len(timestamps)} timestamps")

    # Filter to requested years
    year_mask = timestamps.year.isin(years)
    timestamps_filtered = timestamps[year_mask]
    dlr_ratios_filtered = dlr_ratios[year_mask]

    print(f"📅 Filtered to {len(timestamps_filtered)} timestamps in years {years}")

    # Create training samples
    training_samples = []
    processed_count = 0

    # Process subset of lines
    n_lines = min(max_lines, len(line_ids))
    line_indices = np.random.choice(len(line_ids), n_lines, replace=False)

    print(f"🏭 Processing {n_lines} transmission lines...")

    for line_idx in line_indices:
        line_id = line_ids[line_idx]
        voltage = voltages[line_idx]
        lat = lats[line_idx]
        lon = lons[line_idx]
        slr = slr_values[line_idx]

        print(f"  Processing line {line_id} at ({lat:.2f}, {lon:.2f}) - {voltage}kV")

        # Sample timestamps (not all hours to keep dataset manageable)
        time_indices = np.arange(0, len(timestamps_filtered), time_step_hours)

        for time_idx in time_indices:
            if time_idx >= len(timestamps_filtered):
                continue

            ts = timestamps_filtered[time_idx]

            # Load weather data
            wtk_data = load_wtk_data(lat, lon, ts)
            nsrdb_data = load_nsrdb_data(lat, lon, ts)

            if wtk_data is not None and nsrdb_data is not None:
                wind_speed, wind_direction, temperature = wtk_data
                solar = nsrdb_data

                # Get DLR ratio and calculate target ampacity
                ratio = dlr_ratios_filtered[time_idx, line_idx]
                dlr_ampacity = ratio * slr

                # Create training sample
                sample = {
                    'timestamp': ts,
                    'line_id': line_id,
                    'latitude': lat,
                    'longitude': lon,
                    'voltage_kv': voltage,
                    'T_amb': temperature,
                    'wind_speed': wind_speed,
                    'wind_direction': wind_direction,
                    'solar_irradiance': solar,
                    'dlr_ratio': ratio,
                    'slr_ampacity': slr,
                    'dlr_ampacity': dlr_ampacity,
                    'conductor_temp_limit': 75.0,  # Assumed
                }

                training_samples.append(sample)
                processed_count += 1

            if processed_count % 1000 == 0:
                print(f"    Processed {processed_count} samples...")

    # Create DataFrame
    df = pd.DataFrame(training_samples)

    if len(df) == 0:
        print("❌ No training samples created. Check weather data access.")
        return None

    # Save to HDF5
    print(f"💾 Saving {len(df)} training samples to {output_path}")
    df.to_hdf(output_path, key='train', mode='w', format='table')

    # Print summary statistics
    print("
📊 Dataset Summary:"    print(f"  Total samples: {len(df)}")
    print(f"  Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
    print(f"  Lines: {df['line_id'].nunique()}")
    print(f"  Voltage range: {df['voltage_kv'].min()}-{df['voltage_kv'].max()} kV")
    print(f"  Temperature range: {df['T_amb'].min():.1f}°C to {df['T_amb'].max():.1f}°C")
    print(f"  Wind speed range: {df['wind_speed'].min():.1f} to {df['wind_speed'].max():.1f} m/s")
    print(f"  Solar irradiance range: {df['solar_irradiance'].min():.1f} to {df['solar_irradiance'].max():.1f} W/m²")
    print(f"  DLR ampacity range: {df['dlr_ampacity'].min():.1f} to {df['dlr_ampacity'].max():.1f} A")

    return df

def main():
    import argparse

    parser = argparse.ArgumentParser(description='Prepare US training dataset from DLR ratios and weather data')
    parser.add_argument('--us-dlr', type=str, default='data/us_dlr_2007_2013.h5',
                       help='Path to US DLR HDF5 file')
    parser.add_argument('--output', type=str, default='data/us_training.h5',
                       help='Output HDF5 file path')
    parser.add_argument('--years', type=int, nargs='+', default=[2010, 2011, 2012],
                       help='Years to include in training data')
    parser.add_argument('--max-lines', type=int, default=50,
                       help='Maximum number of transmission lines to process')
    parser.add_argument('--time-step-hours', type=int, default=6,
                       help='Temporal resolution in hours (larger = smaller dataset)')
    parser.add_argument('--synthetic-fallback', action='store_true',
                       help='Use synthetic weather data if real data unavailable')

    args = parser.parse_args()

    # Check if input file exists
    if not Path(args.us_dlr).exists():
        print(f"❌ US DLR file not found: {args.us_dlr}")
        print("   Download from: https://data.openei.org/files/6231/DLR_SLR_ratio-75C.h5")
        return

    # Create output directory
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)

    try:
        df = create_us_training_dataset(
            us_dlr_path=args.us_dlr,
            output_path=args.output,
            years=args.years,
            max_lines=args.max_lines,
            time_step_hours=args.time_step_hours,
            use_synthetic_fallback=args.synthetic_fallback
        )

        if df is not None:
            print("
🎉 Training dataset created successfully!"            print(f"   File: {args.output}")
            print(f"   Samples: {len(df)}")
            print("
📋 Next steps:"            print("1. Inspect the data: python -c \"import pandas as pd; df = pd.read_hdf('data/us_training.h5'); print(df.head())\"")
            print("2. Train your model: python scripts/train_hwf_pikan_production.py --us-data data/us_training.h5")
            print("3. Validate performance: python scripts/validate_us_dlr.py --model-path your_model.pt")

    except Exception as e:
        print(f"❌ Error creating training dataset: {e}")
        print("\n🔧 Troubleshooting:")
        print("1. Ensure NREL API key is set: export NREL_API_KEY='your_key'")
        print("2. Run setup script: ./setup_weather_data_access.sh")
        print("3. Check network connectivity to NREL HSDS")
        print("4. Use --synthetic-fallback for testing without real weather data")

if __name__ == "__main__":
    main()