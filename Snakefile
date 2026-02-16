# Snakemake pipeline to standardise raw datasets and produce a unified training HDF5
# Rules: standardise_vietnam, standardise_us, prepare_europe, unify_all

import pandas as pd
import h5py
import subprocess
from pathlib import Path

# small local sanitiser (kept consistent with scripts/validate_us_dlr_with_matching.py)
def sanitize_weather_data(df):
    df = df.copy()
    if 'temperature' in df.columns:
        if df['temperature'].min(skipna=True) < -200:
            df['temperature'] = df['temperature'] - 273.15
        if df['temperature'].max(skipna=True) > 1000:
            df['temperature'] = df['temperature'] / 10
        df['temperature'] = df['temperature'].clip(lower=-50, upper=150)
    if 'wind_speed' in df.columns:
        if df['wind_speed'].max(skipna=True) > 100:
            df['wind_speed'] = df['wind_speed'] / 10
        df['wind_speed'] = df['wind_speed'].clip(lower=0, upper=50)
    if 'solar' in df.columns:
        df['solar'] = df['solar'].clip(lower=0, upper=1500)
        if 'solar_irradiance' not in df.columns:
            df['solar_irradiance'] = df['solar']
    elif 'solar_irradiance' in df.columns:
        df['solar_irradiance'] = df['solar_irradiance'].clip(lower=0, upper=1500)
        if 'solar' not in df.columns:
            df['solar'] = df['solar_irradiance']
    amp_candidates = [c for c in ('actual', 'actual_dlr_ampacity', 'dlr_amps', 'dlr_ratio', 'dlr') if c in df.columns]
    if amp_candidates:
        amp_col = amp_candidates[0]
        df[amp_col] = pd.to_numeric(df[amp_col], errors='coerce')
        if df[amp_col].max(skipna=True) < 100:
            df[amp_col] = df[amp_col] * 1000
        if 'actual' not in df.columns:
            df['actual'] = df[amp_col]
    return df


rule all:
    input:
        "data/processed/unified_dlr_training.h5"


rule standardise_vietnam:
    input:
        "data/raw/vietnam/vietnam_220kv.csv"
    output:
        "data/processed/vietnam_standard.parquet"
    run:
        df = pd.read_csv(input[0])
        # safe rename (only rename keys that exist)
        rename_map = {
            'temp': 'temperature',
            'temperature_C': 'temperature',
            'Wind1': 'wind_speed',
            'wind_speed_m_s': 'wind_speed',
            'Solar': 'solar_irradiance',
            'GHI': 'solar_irradiance',
            'ampacity': 'actual',
            'dlr': 'actual'
        }
        df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})
        df['conductor_type'] = df.get('conductor_type', 'ACSR')
        df['voltage_kv'] = df.get('voltage_kv', 220)
        df['region'] = 'VN'
        df = sanitize_weather_data(df)
        df.to_parquet(output[0])


rule standardise_us:
    input:
        meta="data/raw/us/us_dlr_2007_2013.h5",
        weather="data/raw/us/us_dlr_weather_corrected.csv"
    output:
        "data/processed/us_standard.parquet"
    run:
        import h5py
        import pandas as pd
        from pathlib import Path
        
        # Load weather data
        weather_df = pd.read_csv(input.weather, parse_dates=['timestamp'])
        
        # Load metadata from HDF5 - extract line_ids from columns
        with h5py.File(input.meta, 'r') as f:
            line_ids = f['columns'][:]
            # Create meta_df with defaults
            meta_df = pd.DataFrame({
                'line_id': line_ids,
                'voltage_kv': 230.0,  # Default US transmission voltage
                'lat': None,
                'lon': None
            })
        
        # Merge weather and metadata
        df = weather_df.merge(meta_df, on='line_id', how='left')
        
        # Add region and conductor type
        df['region'] = 'US'
        df['conductor_type'] = df.get('conductor_type', 'unknown')
        
        # Sanitize and save
        df = sanitize_weather_data(df)
        df.to_parquet(output[0])


rule prepare_europe:
    output:
        "data/processed/europe_standard.parquet"
    run:
        # if pypsa-eur repo not present, write an empty placeholder and warn
        if not Path('pypsa-eur').exists():
            print('⚠️  pypsa-eur not found — creating empty placeholder for Europe')
            pd.DataFrame(columns=['line_id','timestamp','temperature','wind_speed','solar_irradiance','actual','voltage_kv','lat','lon','region']).to_parquet(output[0])
            return
        # run a narrow PyPSA‑Eur workflow and convert
        subprocess.run(["snakemake", "-j1", "resources/networks/base.nc"], cwd="pypsa-eur", check=True)
        subprocess.run(["python", "scripts/pypsaeur_to_standard.py", "--out", output[0]], check=True)


rule unify_all:
    input:
        vn = "data/processed/vietnam_standard.parquet",
        us = "data/processed/us_standard.parquet",
        eu = "data/processed/europe_standard.parquet"
    output:
        "data/processed/unified_dlr_training.h5"
    run:
        dfs = [pd.read_parquet(f) for f in input]
        unified = pd.concat(dfs, ignore_index=True)
        unified.to_hdf(output[0], key='data', mode='w', complevel=5)
