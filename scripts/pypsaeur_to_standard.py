#!/usr/bin/env python3
"""Convert pypsa-eur outputs into the repo's standardized parquet schema.

Strategy:
- Prefer pypsa (if installed) to load the NetCDF network and extract line/bus metadata.
- Fallbacks: try xarray/netCDF4 if pypsa is not available.
- If no usable artifacts are found, emit an empty placeholder (so the Snakefile keeps working).

Output schema (one row per line; time-series fields left empty unless cutouts are provided):
  line_id, timestamp, temperature, wind_speed, solar_irradiance, actual,
  voltage_kv, lat, lon, region, conductor_type

Usage: python scripts/pypsaeur_to_standard.py --out data/processed/europe_standard.parquet [--netcdf pypsa-eur/networks/elec_s_10_ec.nc]
"""
from pathlib import Path
import argparse
import pandas as pd
import numpy as np
import sys

parser = argparse.ArgumentParser()
parser.add_argument('--out', required=True, help='Output parquet path')
parser.add_argument('--netcdf', default='pypsa-eur/resources/networks/base.nc', help='Path to pypsa-eur network NetCDF')
parser.add_argument('--cutout', default=None, help='Path to weather cutout (NetCDF or atlite folder)')
parser.add_argument('--tmax', type=float, default=75.0, help='Conductor max temperature for DLR (°C)')
parser.add_argument('--use-loading', action='store_true', help='Attempt to extract line loading time series from the pypsa network (if available)')
parser.add_argument('--sample-limit', type=int, default=None, help='Limit number of lines processed (for quick tests)')
parser.add_argument('--region', default='EU', help='Region tag to write into rows')
args = parser.parse_args()

out_path = Path(args.out)
out_path.parent.mkdir(parents=True, exist_ok=True)


def write_placeholder(msg=None):
    cols = ['line_id','timestamp','temperature','wind_speed','solar_irradiance','actual','voltage_kv','lat','lon','region','conductor_type']
    df = pd.DataFrame(columns=cols)
    df.to_parquet(out_path)
    if msg:
        print(msg)
    print(f'Wrote placeholder {out_path}')


def try_pypsa(nc_path):
    try:
        import pypsa
    except Exception as e:
        print('pypsa not available:', e)
        return None
    try:
        net = pypsa.Network(nc_path)
    except Exception as e:
        print(f'Failed to load network with pypsa from {nc_path}:', e)
        return None
    return net


def network_to_standard_df(net, region='EU'):
    # lines -> one row per line (no time series)
    lines = net.lines.copy()
    if lines.empty:
        return pd.DataFrame(columns=['line_id','timestamp','temperature','wind_speed','solar_irradiance','actual','voltage_kv','lat','lon','region','conductor_type'])

    # ensure string line ids
    lines = lines.reset_index()
    if 'index' in lines.columns:
        lines['line_id'] = lines['index'].astype(str)
    else:
        lines['line_id'] = lines.index.astype(str)

    # compute approximate midpoint coords from bus coordinates if available
    lat_vals = []
    lon_vals = []
    if {'x', 'y'}.issubset(set(net.buses.columns)):
        buses = net.buses[['x','y']]
        for _, r in lines.iterrows():
            b0 = r.get('bus0')
            b1 = r.get('bus1')
            try:
                c0 = buses.loc[b0]
                c1 = buses.loc[b1]
                # Note: pypsa-eur bus coordinates may be projected (not lat/lon). We still store them.
                lon_vals.append(np.nanmean([c0['x'], c1['x']]))
                lat_vals.append(np.nanmean([c0['y'], c1['y']]))
            except Exception:
                lon_vals.append(np.nan);
                lat_vals.append(np.nan)
    else:
        lon_vals = [np.nan] * len(lines)
        lat_vals = [np.nan] * len(lines)

    # voltage heuristics: prefer explicit voltage columns, else try bus v_nom
    if 'voltage' in lines.columns:
        volt = lines['voltage'].values
    else:
        volt = []
        if 'v_nom' in net.buses.columns:
            v_nom = net.buses['v_nom'].to_dict()
            for _, r in lines.iterrows():
                b0 = r.get('bus0'); b1 = r.get('bus1')
                v0 = v_nom.get(b0, np.nan); v1 = v_nom.get(b1, np.nan)
                try:
                    volt.append(np.nanmean([v0, v1]))
                except Exception:
                    volt.append(np.nan)
            volt = np.array(volt)
        else:
            volt = np.array([np.nan] * len(lines))

    out_df = pd.DataFrame({
        'line_id': lines['line_id'].values,
        'timestamp': pd.NaT,
        'temperature': np.nan,
        'wind_speed': np.nan,
        'solar_irradiance': np.nan,
        'actual': np.nan,
        'voltage_kv': volt,
        'lat': lat_vals,
        'lon': lon_vals,
        'region': region,
        'conductor_type': 'unknown'
    })
    return out_df


# Main
nc = Path(args.netcdf)
if nc.exists():
    net = try_pypsa(str(nc))
    if net is None:
        # xarray/netCDF fallback for network metadata
        try:
            import xarray as xr
            ds = xr.open_dataset(str(nc))
            print('Opened NetCDF with xarray; attempting to find `lines`/`buses` groups')
            if 'lines' in ds:
                lines_df = ds['lines'].to_dataframe().reset_index()
                lines_df['line_id'] = lines_df.index.astype(str)
                df_meta = pd.DataFrame({
                    'line_id': lines_df['line_id'].values,
                    'timestamp': pd.NaT,
                    'temperature': np.nan,
                    'wind_speed': np.nan,
                    'solar_irradiance': np.nan,
                    'actual': np.nan,
                    'voltage_kv': lines_df.get('voltage', np.nan),
                    'lat': np.nan,
                    'lon': np.nan,
                    'region': args.region,
                    'conductor_type': 'unknown'
                })
                print(f'Extracted {len(df_meta)} lines from NetCDF `lines` group')
            else:
                print('No `lines` group found in NetCDF network; trying pypsa (if available)')
                net = try_pypsa(str(nc))
        except Exception as e:
            print('xarray/netCDF fallback failed or not available:', e)

    if net is not None:
        print(f'Loaded pypsa network from {nc}; extracting lines...')
        df_meta = network_to_standard_df(net, region=args.region)

    # If user provided a weather cutout, expand to time-series per line
    if args.cutout:
        cut_path = Path(args.cutout)
        if not cut_path.exists():
            print(f'Warning: cutout {cut_path} not found — writing metadata-only output')
            df_meta.to_parquet(out_path)
            sys.exit(0)

        print(f'Loading weather cutout from {cut_path} (atlite/xarray)')
        # prefer atlite if available and cutout is a directory
        ds = None
        try:
            if cut_path.is_dir():
                import atlite
                cut = atlite.Cutout(str(cut_path))
                # atlite Cutout.get returns xarray DataArrays; we will extract via xarray
                # create an xarray.Dataset from requested variables when needed
                # fallback: try to open a NetCDF inside directory
                ds = None
                print('Using atlite Cutout (sampled per point later)')
        except Exception as e:
            # not fatal — fallback to opening NetCDF directly
            print('atlite not available or failed to open cutout directory:', e)

        if ds is None:
            try:
                import xarray as xr
                ds = xr.open_dataset(str(cut_path))
            except Exception as e:
                print('Failed to open cutout with xarray:', e)
                print('Writing metadata-only output')
                df_meta.to_parquet(out_path)
                sys.exit(0)

        # identify coordinate names
        coord_names = set(ds.coords)
        lat_name = next((n for n in ('latitude','lat','y') if n in coord_names), None)
        lon_name = next((n for n in ('longitude','lon','x') if n in coord_names), None)
        time_name = next((n for n in ('time','datetime','timestamp') if n in coord_names), None)
        if lat_name is None or lon_name is None or time_name is None:
            print('Cutout missing expected coords (lat/lon/time) — aborting time-series extraction')
            df_meta.to_parquet(out_path)
            sys.exit(0)

        # helper to find variable in dataset
        def find_var(ds, candidates):
            for v in candidates:
                if v in ds:
                    return v
            return None

        temp_var = find_var(ds, ['t2m','temperature','air_temperature','tas'])
        u_var = find_var(ds, ['u10','10u','u_component_of_wind_10m'])
        v_var = find_var(ds, ['v10','10v','v_component_of_wind_10m'])
        ws_var = find_var(ds, ['wind_speed','sfcWind','ws'])
        solar_var = find_var(ds, ['ssrd','rsds','surface_solar_radiation_downwards','solar_irradiance'])

        if temp_var is None or (ws_var is None and (u_var is None or v_var is None)):
            print('Cutout does not contain required weather fields (temperature + wind). Proceeding but results may be incomplete.')

        # lines selection (limit optional)
        lines_df = df_meta.copy()
        if args.sample_limit:
            lines_df = lines_df.head(args.sample_limit)
            print(f'Processing {len(lines_df)} lines (sample limit)')
        else:
            print(f'Processing {len(lines_df)} lines')

        # detect whether coords look like lon/lat or projected
        def looks_like_lonlat(xs, ys):
            xs = np.array([v for v in xs if not pd.isna(v)])
            ys = np.array([v for v in ys if not pd.isna(v)])
            if len(xs) == 0 or len(ys) == 0:
                return False
            return (xs.min() >= -180 and xs.max() <= 180 and ys.min() >= -90 and ys.max() <= 90)

        is_lonlat = looks_like_lonlat(lines_df['lon'].values, lines_df['lat'].values)
        if not is_lonlat:
            print('Warning: bus coordinates do not look like lon/lat. Attempting to use them directly; consider reprojection for accurate spatial joins.')

        # prepare physics engine
        try:
            import torch
            from core.physics import IEEE738HeatBalance
            physics = IEEE738HeatBalance()
            use_physics = True
        except Exception:
            physics = None
            use_physics = False
            print('torch/physics not available — falling back to placeholder `actual` values')

        rows = []
        # time index (xarray to pandas DatetimeIndex)
        time_index = pd.to_datetime(ds[time_name].values)

        for i, r in lines_df.reset_index(drop=True).iterrows():
            lid = str(r['line_id'])
            lat = r['lat']
            lon = r['lon']
            if pd.isna(lat) or pd.isna(lon):
                print(f'  Skipping line {lid} — missing coords')
                continue

            # select nearest grid cell
            try:
                sel = {}
                sel[lon_name] = lon
                sel[lat_name] = lat
                # select nearest — returns DataArray with time dim
                tmp = ds[temp_var].sel({lon_name: lon, lat_name: lat}, method='nearest') if temp_var else None
                if ws_var:
                    wsp = ds[ws_var].sel({lon_name: lon, lat_name: lat}, method='nearest')
                else:
                    u = ds[u_var].sel({lon_name: lon, lat_name: lat}, method='nearest') if u_var else None
                    v = ds[v_var].sel({lon_name: lon, lat_name: lat}, method='nearest') if v_var else None
                    if u is not None and v is not None:
                        wsp = (u ** 2 + v ** 2) ** 0.5
                    else:
                        wsp = None
                solar = ds[solar_var].sel({lon_name: lon, lat_name: lat}, method='nearest') if solar_var else None
            except Exception as e:
                print(f'  Point sampling failed for line {lid}:', e)
                continue

            # convert to pandas series and apply unit heuristics
            def xr_to_series(xr_da):
                if xr_da is None:
                    return pd.Series(index=time_index, data=[np.nan] * len(time_index))
                arr = xr_da.values
                s = pd.Series(index=time_index, data=arr)
                return s

            temp_s = xr_to_series(tmp)
            wind_s = xr_to_series(wsp)
            solar_s = xr_to_series(solar)

            # unit heuristics
            if temp_s.dropna().size > 0 and temp_s.dropna().mean() > 200:
                temp_s = temp_s - 273.15
            if solar_s.dropna().size > 0 and solar_s.dropna().max() > 5000:
                # likely accumulated J/m^2 over hour -> convert to W/m^2
                solar_s = solar_s / 3600.0

            # compute DLR (actual) using physics if loading time series not available
            if args.use_loading:
                # attempt to extract line flows from net (snapshots) — best effort
                if hasattr(net, 'lines') and hasattr(net, 'snapshot_weightings'):
                    # not implemented generically — skip to physics fallback
                    pass

            if use_physics:
                # convert to torch tensors (batch)
                t_amb_t = torch.tensor(temp_s.fillna(temp_s.mean() if not temp_s.dropna().empty else 20.0).values, dtype=torch.float32)
                wind_t = torch.tensor(wind_s.fillna(0.5).values, dtype=torch.float32)
                solar_t = torch.tensor(solar_s.fillna(0.0).values, dtype=torch.float32)
                with torch.no_grad():
                    I_max = physics.ampacity(args.tmax, t_amb_t, wind_t, solar_t).numpy()
                actual_s = pd.Series(index=time_index, data=I_max)
            else:
                actual_s = pd.Series(index=time_index, data=[np.nan] * len(time_index))

            # assemble rows
            df_line = pd.DataFrame({
                'line_id': lid,
                'timestamp': time_index,
                'temperature': temp_s.values,
                'wind_speed': wind_s.values,
                'solar_irradiance': solar_s.values,
                'actual': actual_s.values,
                'voltage_kv': r.get('voltage_kv', np.nan),
                'lat': lat,
                'lon': lon,
                'region': args.region,
                'conductor_type': r.get('conductor_type', 'unknown')
            })

            rows.append(df_line)

        if len(rows) == 0:
            print('No lines processed from cutout; writing metadata-only output')
            df_meta.to_parquet(out_path)
            sys.exit(0)

        full = pd.concat(rows, ignore_index=True)
        print(f'Writing {len(full)} rows (line × time) to {out_path}')
        full.to_parquet(out_path)
        sys.exit(0)

    else:
        # no cutout — write metadata-only parquet
        print('No weather cutout provided — writing metadata-only parquet')
        df_meta.to_parquet(out_path)
        sys.exit(0)
else:
    print(f'NetCDF {nc} not found — skipping pypsa-eur extraction')

write_placeholder('Could not extract pypsa-eur data; wrote empty placeholder. If you have pypsa and the network NetCDF available, re-run with --netcdf <path>')
