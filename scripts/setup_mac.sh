#!/bin/bash
# ============================================================
# INVARIANT SETUP SCRIPT
# Run this once to set up everything on your M2 MacBook
# ============================================================

echo "⚡ INVARIANT SETUP STARTING..."
echo "================================"

# Create project directory
mkdir -p ~/Projects
cd ~/Projects

# Clone or create invariant folder
if [ -d "invariant" ]; then
    echo "📁 Invariant folder exists, updating..."
    cd invariant
else
    echo "📁 Creating invariant folder..."
    mkdir invariant
    cd invariant
fi

# Create Python virtual environment
echo "🐍 Setting up Python environment..."
python3 -m venv venv
source venv/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install dependencies
echo "📦 Installing packages (this takes 2-3 minutes)..."
pip install torch numpy pandas scikit-learn
pip install matplotlib plotly
pip install gradio
pip install jupyter ipywidgets
pip install pytest black mypy
pip install requests python-dotenv

# Verify PyTorch and MPS (Apple Silicon)
echo ""
echo "🔍 Verifying Apple Silicon (MPS) support..."
python3 -c "
import torch
print(f'PyTorch version: {torch.__version__}')
print(f'MPS available: {torch.backends.mps.is_available()}')
if torch.backends.mps.is_available():
    print('✅ Your M2 chip is ready for AI training!')
else:
    print('⚠️ MPS not available, will use CPU (still works, just slower)')
"

echo ""
echo "✅ SETUP COMPLETE!"
echo ""
echo "Next steps:"
echo "1. Copy the Invariant code files into ~/Projects/invariant/"
echo "2. Run: cd ~/Projects/invariant && source venv/bin/activate"
echo "3. Train: python -m core.train --quick"
echo "4. Demo: python demo/app.py"
echo ""
echo "⚡ Ready to revolutionize grid optimization!"
