#!/usr/bin/env python3
"""
extract_us_dlr_weather.py

Extract weather data from NREL WIND Toolkit and NSRDB
to match US DLR dataset timestamps and locations

Uses HSDS cloud access for efficient subsetting
"""

import h5py
import h5pyd
import pandas as pd
import numpy as np
from pathlib import Path
import sys
import os
import time
from datetime import datetime, timedelta
from tqdm import tqdm

sys.path.append(str(Path(__file__).parent.parent))

class NRELWeatherExtractor:
    """
    Extract weather data from NREL WIND Toolkit and NSRDB
    to match US DLR dataset timestamps and locations

    Uses HSDS cloud access for efficient subsetting
    """

    def __init__(self, api_key=None, cache_dir='data/nrel_weather'):
        self.api_key = api_key or os.environ.get('NREL_API_KEY')
        if not self.api_key:
            raise ValueError("NREL_API_KEY required. Sign up at https://developer.nrel.gov/signup/")

        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # HSDS endpoints - corrected paths
        self.wtk_path = '/nrel/wtk/conus/wtk_conus_2013.h5'  # WIND Toolkit CONUS
        self.nsrdb_path = '/nrel/nsrdb/v3/nsrdb_2013.h5'  # NSRDB solar data

        # Connect to HSDS
        try:
            self.f_wtk = h5pyd.File(self.wtk_path, 'r', endpoint='https://developer.nrel.gov/api/hsds')
            print(f"✅ Connected to WIND Toolkit: {self.wtk_path}")
        except Exception as e:
            print(f"❌ Failed to connect to WIND Toolkit: {e}")
            raise

    def find_nearest_gid(self, target_lat, target_lon, max_distance_km=10):
        """
        Find nearest grid point in WIND Toolkit to target lat/lon

        WIND Toolkit has 2km x 2km resolution across CONUS
        """
        # Load metadata
        meta = self.f_wtk['meta'][:]

        # Calculate distances (simplified - use haversine for production)
        distances = np.sqrt(
            (meta['latitude'] - target_lat)**2 +
            (meta['longitude'] - target_lon)**2
        ) * 111  # Rough km conversion

        min_dist_idx = np.argmin(distances)
        min_dist = distances[min_dist_idx]

        if min_dist > max_distance_km:
            print(f"⚠️  Nearest point {min_dist:.1f}km away > {max_distance_km}km limit")
            return None

        return min_dist_idx, meta[min_dist_idx]

    def extract_weather_for_line(self, line_id, lat, lon, timestamps, output_path=None):
        """
        Extract weather data for a specific transmission line using OPTIMIZED batch extraction

        Args:
            line_id: Line identifier from US DLR dataset
            lat, lon: Line location
            timestamps: List of timestamps to extract
            output_path: Optional path to save extracted data

        Returns:
            DataFrame with weather data
        """
        print(f"\n📍 Processing line {line_id} at ({lat:.4f}, {lon:.4f})")

        # Find nearest grid point
        gid_result = self.find_nearest_gid(lat, lon)
        if gid_result is None:
            return None

        gid, meta = gid_result
        print(f"   Using grid point {gid} at ({meta['latitude']:.4f}, {meta['longitude']:.4f})")

        # OPTIMIZED: Download ALL data for this grid point at once (1 request)
        print(f"   📦 Downloading full dataset for grid point {gid}...")
        try:
            wind_speed_all = self.f_wtk['windspeed_100m'][:, gid]
            wind_dir_all = self.f_wtk['winddirection_100m'][:, gid]
            temp_all = self.f_wtk['temperature_100m'][:, gid]
            print(f"   ✅ Downloaded {len(wind_speed_all)} hourly records")
        except Exception as e:
            print(f"   ❌ Failed to download data: {e}")
            return None

        # Convert timestamps to indices (vectorized)
        wtk_times = pd.to_datetime(self.f_wtk['time_index'][:].astype(str))
        timestamp_values = np.array([ts.value for ts in timestamps])
        wtk_values = np.array([ts.value for ts in wtk_times])

        # Vectorized nearest neighbor search
        indices = np.argmin(np.abs(wtk_values[:, None] - timestamp_values), axis=0)

        # Extract local data (instant)
        weather_data = []

        for i, (ts, idx) in enumerate(zip(timestamps, indices)):
            dt = ts

            # Extract from downloaded data
            wind_speed = float(wind_speed_all[idx])  # m/s
            wind_dir = float(wind_dir_all[idx])  # degrees
            temperature = float(temp_all[idx]) - 273.15  # Convert K to C

            # Solar data - simplified estimation
            solar = self._estimate_solar(lat, lon, dt)

            weather_data.append({
                'timestamp': dt,
                'wind_speed': wind_speed,
                'wind_direction': wind_dir,
                'temperature': temperature,
                'pressure': 101325.0,  # Standard atmospheric pressure (Pa)
                'solar': solar,
                'gid': gid,
                'line_id': line_id
            })

        df = pd.DataFrame(weather_data)

        if output_path:
            df.to_csv(output_path, index=False)
            print(f"✅ Saved weather data to {output_path}")

        return df

    def _estimate_solar(self, lat, lon, timestamp):
        """
        Estimate solar irradiance using simple model
        In production, replace with actual NSRDB query
        """
        hour = timestamp.hour
        day_of_year = timestamp.dayofyear

        # Simple solar model based on hour and latitude
        solar_angle = np.sin(np.pi * (hour - 6) / 12)  # Peak at noon
        seasonal_factor = 1 + 0.3 * np.sin(2 * np.pi * (day_of_year - 80) / 365)  # Summer peak

        irradiance = max(0, 800 * solar_angle * seasonal_factor)  # W/m²
        return irradiance if 6 <= hour <= 18 else 0

    def close(self):
        """Close HSDS connections"""
        self.f_wtk.close()
        print("✅ Closed HSDS connections")


def extract_all_us_dlr_weather(us_dlr_path, output_dir='data/us_dlr_weather'):
    """
    Extract weather for all lines in US DLR dataset

    Args:
        us_dlr_path: Path to US DLR HDF5 file
        output_dir: Directory to save weather data
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load US DLR data using h5py (since pandas HDF5 has issues)
    print("📊 Loading US DLR dataset...")
    with h5py.File(us_dlr_path, 'r') as f:
        dlr_data = f['data'][:]
        columns = f['columns'][:].astype(str)
        index = f['index'][:]

    # Convert to DataFrame
    timestamps = pd.to_datetime(index.astype(str))
    df_dlr = pd.DataFrame(dlr_data, index=timestamps, columns=columns)
    print(f"   Loaded {df_dlr.shape[0]} timestamps x {df_dlr.shape[1]} lines")

    # For now, use sample locations since actual coordinates aren't in the file
    # In production, you'd need to join with HIFLD or other location data
    sample_locations = {
        '100001': {'lat': 40.7128, 'lon': -74.0060},  # NYC area
        '100002': {'lat': 34.0522, 'lon': -118.2437}, # LA area
        '100003': {'lat': 41.8781, 'lon': -87.6298},  # Chicago area
        '100004': {'lat': 29.7604, 'lon': -95.3698},  # Houston area
        '100005': {'lat': 33.4484, 'lon': -112.0740}, # Phoenix area
    }

    # Initialize weather extractor
    extractor = NRELWeatherExtractor()

    # Process sample lines
    all_weather = []

    for line_id, coords in sample_locations.items():
        if line_id in df_dlr.columns:
            print(f"\n📊 Processing line {line_id}")

            # Extract weather for this line
            df_weather = extractor.extract_weather_for_line(
                line_id=line_id,
                lat=coords['lat'],
                lon=coords['lon'],
                timestamps=df_dlr.index,
                output_path=output_dir / f'weather_{line_id}.csv'
            )

            if df_weather is not None:
                all_weather.append(df_weather)

            # Rate limiting
            time.sleep(1)

    # Combine all weather data
    if all_weather:
        combined = pd.concat(all_weather, ignore_index=True)
        combined.to_csv(output_dir / 'all_weather_data.csv', index=False)
        print(f"\n✅ Combined weather data saved to {output_dir / 'all_weather_data.csv'}")
        print(f"   {len(combined)} records from {len(all_weather)} lines")

    extractor.close()
    return output_dir / 'all_weather_data.csv'


def join_weather_with_dlr(us_dlr_path, weather_csv, output_path='data/us_dlr_with_weather.h5'):
    """
    Join extracted weather data with original DLR ratios

    Creates complete training/validation dataset with:
    - Weather features (T_amb, wind_speed, solar)
    - DLR target (ratio values)
    - Line metadata
    """
    print("🔗 Joining weather data with DLR ratios...")

    # Load US DLR data using h5py
    with h5py.File(us_dlr_path, 'r') as f:
        dlr_data = f['data'][:]
        columns = f['columns'][:].astype(str)
        index = f['index'][:]

    timestamps = pd.to_datetime(index.astype(str))
    df_dlr = pd.DataFrame(dlr_data, index=timestamps, columns=columns)
    print(f"   DLR data: {df_dlr.shape}")

    # Load weather data
    df_weather = pd.read_csv(weather_csv, parse_dates=['timestamp'])
    print(f"   Weather data: {df_weather.shape}")

    # Create joined dataset
    joined_data = []

    # For each weather record, find matching DLR ratio
    for _, weather_row in df_weather.iterrows():
        line_id = str(weather_row['line_id'])
        ts = weather_row['timestamp']

        if line_id in df_dlr.columns:
            # Find closest timestamp in DLR data
            time_diffs = np.abs((df_dlr.index - ts).total_seconds())
            closest_idx = np.argmin(time_diffs)

            if time_diffs[closest_idx] < 3600:  # Within 1 hour
                dlr_ratio = df_dlr.loc[df_dlr.index[closest_idx], line_id]

                joined_data.append({
                    'timestamp': ts,
                    'line_id': line_id,
                    'T_amb': weather_row['temperature'],
                    'wind_speed': weather_row['wind_speed'],
                    'wind_direction': weather_row['wind_direction'],
                    'solar': weather_row['solar'],
                    'pressure': weather_row['pressure'],
                    'dlr_ratio': dlr_ratio,
                    'gid': weather_row['gid']
                })

    # Save joined dataset
    joined_df = pd.DataFrame(joined_data)

    # Prefer HDF5 but gracefully fall back to Parquet/CSV if PyTables is unavailable
    try:
        joined_df.to_hdf(output_path, key='data', mode='w')
        print(f"\n✅ Joined dataset saved to {output_path}")
    except Exception as e:
        # Common failures: missing `tables` or binary incompatibility with NumPy
        fallback_path = Path(output_path).with_suffix('.csv')
        joined_df.to_csv(fallback_path, index=False)
        print(f"\n⚠️  Could not write HDF5 ({e}); saved CSV instead: {fallback_path}")
        output_path = str(fallback_path)

    print(f"   {len(joined_df)} samples from {joined_df['line_id'].nunique()} lines")

    return output_path


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Extract weather for US DLR validation')
    parser.add_argument('--us-dlr-path', type=str,
                       default='data/us_dlr_2007_2013.h5',
                       help='Path to US DLR HDF5 file')
    parser.add_argument('--output-dir', '--weather-dir', dest='output_dir', type=str,
                       default='data/us_dlr_weather',
                       help='Directory to save weather data (alias: --weather-dir)')
    parser.add_argument('--weather-csv', type=str, default='',
                       help='Path to pre-extracted combined weather CSV (used with --skip-extraction)')
    parser.add_argument('--joined-output', type=str,
                       default='data/us_dlr_with_weather.h5',
                       help='Output path for joined dataset')
    parser.add_argument('--skip-extraction', action='store_true',
                       help='Skip extracting weather and use existing weather CSV in --output-dir or --weather-csv')

    args = parser.parse_args()

    # Step 1: Extract or reuse weather data
    if args.skip_extraction:
        print("🌤️  Step 1: Skipping extraction (using existing weather CSV)...")
        if args.weather_csv:
            weather_csv = args.weather_csv
        else:
            weather_csv = Path(args.output_dir) / 'all_weather_data.csv'

        if not Path(weather_csv).exists():
            print(f"❌ Weather CSV not found: {weather_csv}")
            print("Run without --skip-extraction or provide --weather-csv <path>")
            sys.exit(1)
    else:
        print("🌤️  Step 1: Extracting weather data...")
        weather_csv = extract_all_us_dlr_weather(
            us_dlr_path=args.us_dlr_path,
            output_dir=args.output_dir
        )

    # Step 2: Join with DLR data
    print("\n🔗 Step 2: Joining with DLR data...")
    joined_path = join_weather_with_dlr(
        us_dlr_path=args.us_dlr_path,
        weather_csv=weather_csv,
        output_path=args.joined_output
    )

    print("\n🎯 Ready for validation!")
    print(f"Run validation with: python -m scripts.validate_us_dlr \\")
    print(f"    --model-path runs/finetuned_189a/best_model.pt \\")
    print(f"    --us-dlr-path {args.us_dlr_path} \\")
    print(f"    --weather-csv {joined_path}")