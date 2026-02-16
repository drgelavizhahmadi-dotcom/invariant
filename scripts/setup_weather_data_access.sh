#!/bin/bash
# ============================================================
# NREL WEATHER DATA ACCESS SETUP
# Sets up HSDS API access for WIND Toolkit and NSRDB data
# ============================================================

echo "🌤️  Setting up NREL Weather Data Access..."
echo "=========================================="

# Check if API key is provided
if [ -z "$1" ]; then
    echo "❌ Error: Please provide your NREL API key as an argument"
    echo "Usage: $0 <your_nrel_api_key>"
    echo ""
    echo "Get your free API key at: https://developer.nrel.gov/signup/"
    exit 1
fi

API_KEY="$1"

echo "🔑 Using API key: ${API_KEY:0:10}..."

# Create HSDS configuration file
echo "📝 Creating HSDS configuration..."
mkdir -p ~/.hscfg

cat > ~/.hscfg << EOF
{
    "hsds": {
        "endpoint": "https://developer.nrel.gov/api/hsds",
        "api_key": "$API_KEY",
        "username": null,
        "password": null
    }
}
EOF

echo "✅ HSDS config created at ~/.hscfg"

# Install required packages
echo "📦 Installing weather data packages..."
source venv/bin/activate

# Install h5pyd for HSDS access
pip install h5pyd

# Install additional packages if needed
pip install requests

echo "🔍 Testing HSDS connection..."

# Test connection with a simple query
python3 -c "
import h5pyd
try:
    # Test basic connection
    f = h5pyd.File('/nrel/wtk/conus/wtk_conus_2013.h5', 'r', endpoint='https://developer.nrel.gov/api/hsds', api_key='$API_KEY')
    print('✅ HSDS connection successful!')
    print(f'📊 Available datasets: {list(f.keys())[:5]}...')
    f.close()
except Exception as e:
    print(f'❌ Connection failed: {e}')
    print('💡 Check your API key and internet connection')
    exit(1)
"

echo ""
echo "🎉 Weather data access setup complete!"
echo ""
echo "Next steps:"
echo "1. Run weather extraction: python scripts/extract_weather_for_validation.py"
echo "2. Prepare US training data: python scripts/prepare_us_training_data.py"
echo "3. Validate with real weather: python -m scripts.validate_us_dlr --weather-csv <weather_file.csv>"
echo ""
echo "📚 See US_TRAINING_DATA_GUIDE.md for detailed instructions"