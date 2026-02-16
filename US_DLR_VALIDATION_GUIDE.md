# US DLR Validation: Weather Data Access Guide

## 🎯 Problem
Your US DLR dataset contains only pre-computed DLR/SLR ratios, but to properly validate your HWF-PIKAN model, you need the underlying weather data (temperature, wind speed/direction, solar irradiance) that was used to calculate those ratios.

## 📊 Available Options

### Option 1: WIND Toolkit + NSRDB (Recommended for Full Validation)
**Pros:** Complete weather dataset used for DLR calculations, high accuracy
**Cons:** Large data volume (TB scale), requires API setup

**Data Sources:**
- **WIND Toolkit**: Wind speed, direction, temperature, pressure
- **NSRDB**: Solar irradiance (GHI, DNI, DHI)

**Setup Steps:**
1. Get NREL API key: https://developer.nrel.gov/signup/
2. Run setup script: `./setup_weather_data_access.sh`
3. Extract weather data: `python scripts/extract_weather_for_validation.py`

### Option 2: NSRDB API Only (Solar Data)
**Pros:** Easier access, focused on solar irradiance
**Cons:** Missing wind data, limited validation scope

```python
# Install NSRDB package
pip install nsrdb

# Example usage
from nsrdb import NSRDB
nsrdb = NSRDB(api_key='YOUR_API_KEY')
data = nsrdb.get_data(lat=40.0, lon=-105.0, years=[2010, 2011, 2012])
```

### Option 3: Pre-processed Weather from OEDI (Simplest)
**Pros:** Ready-to-use, matches DLR calculations exactly
**Cons:** May not be publicly available, check access permissions

The OEDI dataset mentions that weather data validation studies are available. Check if the source weather data is also published.

## 🚀 Quick Start

### Step 1: Get NREL API Key
```bash
# Visit: https://developer.nrel.gov/signup/
# Get your free API key
export NREL_API_KEY='your_api_key_here'
```

### Step 2: Setup Environment
```bash
# Run the setup script
./setup_weather_data_access.sh
```

### Step 3: Extract Weather Data
```bash
# Extract weather for validation
python scripts/extract_weather_for_validation.py \
    --us-dlr-path data/us_dlr_2007_2013.h5 \
    --output-path data/us_validation_weather.csv \
    --years 2010 2011 2012
```

### Step 4: Run Proper Validation
```bash
# Use real weather data instead of synthetic
python scripts/validate_us_dlr.py \
    --model-path runs/hwf_pikan_production_20260214_110318/final_model.pt \
    --us-dlr-path data/us_dlr_2007_2013.h5 \
    --weather-data data/us_validation_weather.csv
```

## 📁 File Structure After Setup

```
data/
├── us_dlr_2007_2013.h5              # DLR ratios (19.4GB)
├── us_validation_weather.csv        # Weather data for validation
└── ...

scripts/
├── validate_us_dlr.py               # Validation script
├── extract_weather_for_validation.py # Weather extraction
└── ...

runs/
└── hwf_pikan_production_.../final_model.pt  # Your trained model
```

## 🔧 Technical Details

### Weather Data Format
Your model expects:
- `temperature`: Ambient air temperature (°C)
- `wind_speed`: Wind speed (m/s)
- `wind_direction`: Wind direction relative to conductor (°)
- `solar_irradiance`: Global horizontal irradiance (W/m²)

### DLR Ratio Relationship
```
DLR_Ampacity = DLR_Ratio × SLR_Ampacity
```

Where:
- `DLR_Ratio`: From your downloaded HDF5 file
- `SLR_Ampacity`: Static rating from the same file
- `DLR_Ampacity`: Dynamic rating you want to predict

### Validation Approach
1. Load weather data for specific locations/times
2. Run your model with that weather data
3. Compare model prediction vs actual DLR ampacity
4. Calculate validation metrics (MAE, RMSE, etc.)

## ⚠️ Important Notes

1. **Data Volume**: WIND Toolkit is ~1.2TB for CONUS, NSRDB is ~500GB
2. **API Limits**: NREL has rate limits, plan data extraction carefully
3. **Network**: HSDS access may require specific network permissions
4. **Coordinates**: Need exact lat/lon for each transmission line (from HIFLD)
5. **Time Alignment**: Weather and DLR data must be properly synchronized

## 🎯 Alternative: Simplified Validation

If full weather data access is challenging, consider:

1. **Regional Validation**: Validate on specific regions with available weather data
2. **Statistical Comparison**: Compare model performance distributions vs DLR distributions
3. **Physics-Based Validation**: Use IEEE 738 equations to validate against DLR ratios
4. **Synthetic Scenarios**: Test model on extreme weather scenarios derived from literature

## 📞 Getting Help

- **NREL HSDS**: https://github.com/NREL/hsds-examples
- **DLR Repository**: https://github.com/NREL/DynamicLineRatings
- **OEDI Support**: Contact the dataset authors via email in the metadata

The validation framework is ready - you just need the weather data to make it fully operational!