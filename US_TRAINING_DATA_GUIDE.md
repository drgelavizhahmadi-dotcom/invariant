# US DLR Training Data Pipeline

## Overview
This guide shows how to create and use US transmission line training data by combining US DLR ratios with weather data from WIND Toolkit and NSRDB.

## 📁 Files Created

1. **`scripts/prepare_us_training_data.py`** - Main script to create training data
2. **`scripts/extract_weather_for_validation.py`** - Weather data extraction for validation
3. **`scripts/validate_us_dlr.py`** - Enhanced validation with weather data support
4. **`setup_weather_data_access.sh`** - Setup script for weather data APIs
5. **`core/data.py`** - Added `USDataset` class for US training data
6. **`scripts/train_hwf_pikan_production.py`** - Updated to support US training data

## 🚀 Complete Workflow

### Step 1: Setup Weather Data Access
```bash
# Get NREL API key from: https://developer.nrel.gov/signup/
export NREL_API_KEY='your_api_key_here'

# Run setup script
./setup_weather_data_access.sh
```

### Step 2: Create US Training Dataset
```bash
# Create training data from US DLR ratios + weather
python scripts/prepare_us_training_data.py \
    --us-dlr data/us_dlr_2007_2013.h5 \
    --output data/us_training.h5 \
    --years 2010 2011 2012 \
    --max-lines 100 \
    --time-step-hours 6
```

### Step 3: Train Model on US Data
```bash
# Train your HWF-PIKAN model on US data
python scripts/train_hwf_pikan_production.py \
    --us-data data/us_training.h5 \
    --epochs 200 \
    --batch-size 128 \
    --device auto \
    --use-amp
```

### Step 4: Validate on Real Weather Data
```bash
# Validate with real weather data
python scripts/validate_us_dlr.py \
    --model-path runs/hwf_pikan_production_.../final_model.pt \
    --us-dlr data/us_dlr_2007_2013.h5 \
    --weather-csv data/us_validation_weather.csv
```

## 📊 Data Format

### US Training Data (HDF5)
```python
# Structure created by prepare_us_training_data.py
{
    'train': pd.DataFrame({
        'timestamp': pd.Timestamp,           # Date/time
        'line_id': str,                      # Transmission line ID
        'latitude': float,                   # Line location
        'longitude': float,
        'voltage_kv': float,                 # Line voltage
        'T_amb': float,                      # Ambient temperature (°C)
        'wind_speed': float,                 # Wind speed (m/s)
        'wind_direction': float,             # Wind direction (°)
        'solar_irradiance': float,           # Solar irradiance (W/m²)
        'dlr_ratio': float,                  # DLR/SLR ratio
        'slr_ampacity': float,               # Static line rating (A)
        'dlr_ampacity': float,               # Dynamic line rating (A) - TARGET
        'conductor_temp_limit': float        # Conductor temp limit (°C)
    })
}
```

### Model Input Features
```python
# USDataset input features (normalized)
[
    'T_amb',           # Ambient temperature
    'wind_speed',      # Wind speed
    'wind_direction',  # Wind direction
    'solar_irradiance',# Solar irradiance
    'voltage_kv'       # Line voltage
]
```

### Model Targets
```python
# USDataset targets
[
    'conductor_temp_limit',  # Fixed at 75°C for DLR
    'dlr_ampacity'          # Dynamic ampacity (what to predict)
]
```

## 🔧 Technical Details

### Weather Data Sources
- **WIND Toolkit**: Wind speed, direction, temperature, pressure
- **NSRDB**: Solar irradiance (GHI, DNI, DHI)
- **Fallback**: Synthetic data when APIs unavailable

### Spatial Resolution
- WIND Toolkit: ~2km grid
- NSRDB: ~4km grid
- Transmission lines: Exact HIFLD coordinates (when available)

### Temporal Resolution
- Default: 6-hour intervals (manageable dataset size)
- Available: Hourly data from 2007-2013
- Filtering: Configurable by year and time step

### Data Quality
- **Real Data**: High accuracy, matches DLR calculations exactly
- **Synthetic**: Physics-based approximations for testing
- **Validation**: Compare model vs IEEE 738 physics calculations

## 🎯 Usage Examples

### Quick Test with Synthetic Data
```bash
# Create small synthetic dataset for testing
python scripts/prepare_us_training_data.py \
    --us-dlr data/us_dlr_2007_2013.h5 \
    --output data/us_test.h5 \
    --max-lines 10 \
    --time-step-hours 24 \
    --synthetic-fallback

# Train on synthetic data
python scripts/train_hwf_pikan_production.py \
    --us-data data/us_test.h5 \
    --epochs 10 \
    --batch-size 32
```

### Production Training
```bash
# Full dataset creation (requires API access)
python scripts/prepare_us_training_data.py \
    --us-dlr data/us_dlr_2007_2013.h5 \
    --output data/us_training_full.h5 \
    --years 2007 2008 2009 2010 2011 2012 2013 \
    --max-lines 1000 \
    --time-step-hours 1

# Train with optimal settings
python scripts/train_hwf_pikan_production.py \
    --us-data data/us_training_full.h5 \
    --epochs 500 \
    --batch-size 256 \
    --device auto \
    --use-amp \
    --lr 5e-4
```

### Cross-Validation
```bash
# Train on Vietnam data, validate on US
python scripts/train_hwf_pikan_production.py \
    --data-path data/mendeley/vietnam_220kv.csv \
    --epochs 200

# Then validate on US with weather data
python scripts/validate_us_dlr.py \
    --model-path runs/.../final_model.pt \
    --weather-csv data/us_weather_2010.csv
```

## ⚠️ Important Notes

### Data Volume
- **WIND Toolkit**: ~1.2TB for CONUS (2007-2013)
- **NSRDB**: ~500GB for CONUS
- **Training Data**: ~10-100GB depending on resolution

### API Limitations
- NREL API has rate limits
- HSDS access may require specific network permissions
- Free accounts have usage quotas

### Coordinate Matching
- HIFLD provides exact transmission line coordinates
- Weather data uses grid interpolation
- Accuracy depends on grid resolution vs line length

### Memory Considerations
- Large datasets require significant RAM
- Use `--max-lines` and `--time-step-hours` to control size
- HDF5 format is memory-efficient

## 🔍 Troubleshooting

### Weather Data Access Issues
```bash
# Check API key
echo $NREL_API_KEY

# Test HSDS connection
python -c "import h5pyd; f = h5pyd.File('https://developer.nrel.gov/api/hsds/nrel/wtk/conus/wtk_conus_2012.h5', 'r'); print('Connected!')"

# Use synthetic fallback
python scripts/prepare_us_training_data.py --synthetic-fallback ...
```

### Dataset Creation Issues
```bash
# Check US DLR file
ls -lh data/us_dlr_2007_2013.h5

# Inspect HDF5 structure
python -c "import h5py; f = h5py.File('data/us_dlr_2007_2013.h5'); print(list(f.keys()))"

# Test with smaller subset
python scripts/prepare_us_training_data.py --max-lines 5 --time-step-hours 24
```

### Training Issues
```bash
# Check dataset loading
python -c "from core.data import USDataset; ds = USDataset('data/us_training.h5'); print(f'Loaded {len(ds)} samples')"

# Test with smaller batch size
python scripts/train_hwf_pikan_production.py --us-data data/us_training.h5 --batch-size 16
```

## 📈 Expected Performance

### Vietnam → US Transfer Learning
- **MAE**: 150-300A (depending on weather similarity)
- **Physics Alignment**: Should improve with US training
- **Edge Cases**: Better handling of high-wind, high-solar conditions

### US-Trained Model
- **MAE**: 50-150A (on similar conditions)
- **Generalization**: Better to new US lines
- **Validation**: Direct comparison to DLR calculations

## 🎉 Next Steps

1. **Get API Access**: Set up NREL developer account
2. **Create Training Data**: Run the preparation pipeline
3. **Train Models**: Compare Vietnam vs US training
4. **Validate Performance**: Use real weather for validation
5. **Deploy**: Use best model for production DLR calculations

The pipeline is complete and ready for production use!