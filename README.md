# ⚡ Invariant

**Physics-Informed AI for Dynamic Line Rating**

Unlock 20-40% more transmission capacity with AI that respects the laws of physics.

[![License](https://img.shields.io/badge/license-Proprietary-red.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.11+-blue.svg)](https://python.org)
[![PyTorch](https://img.shields.io/badge/pytorch-2.0+-orange.svg)](https://pytorch.org)

---

## 🎯 What is Invariant?

Invariant predicts **Dynamic Line Ratings (DLR)** for transmission lines using Physics-Informed Neural Networks (PINNs). Unlike black-box AI, our predictions:

- ✅ **Respect physics** — IEEE 738 heat balance embedded in the model
- ✅ **Are explainable** — Full transparency on physics compliance
- ✅ **Extrapolate safely** — Physics constraints bound predictions
- ✅ **Need less data** — 10-100x more efficient than pure ML

## 🚀 Quick Start

### Installation

```bash
# Clone repository
git clone https://github.com/invariant-energy/invariant.git
cd invariant

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Training (M2 MacBook optimized)

```bash
# Quick training (~5 minutes on M2)
python -m core.train --quick --save-path models/quick_model.pt

# Full training (~15 minutes on M2)
python -m core.train --epochs 100 --save-path models/best_model.pt
```

### Run Demo

```bash
# Launch Gradio interface
python demo/app.py

# Opens at http://localhost:7860
```

### Make Predictions

```python
from core.inference import DLRPredictor

# Load trained model
predictor = DLRPredictor.from_checkpoint("models/best_model.pt")

# Predict dynamic rating
result = predictor.predict(
    T_ambient=25.0,       # Ambient temperature (°C)
    wind_speed=5.0,       # Wind speed (m/s)
    solar_irradiance=800, # Solar irradiance (W/m²)
    current=800,          # Line current (A)
)

print(f"Dynamic Rating: {result.dynamic_rating:.0f} A")
print(f"Capacity Gain: {result.capacity_gain_percent:+.1f}%")
print(f"Physics Compliant: {result.is_physics_compliant}")
```

## 📁 Project Structure

```
invariant/
├── core/                # 🧠 PINN Engine
│   ├── physics.py       # IEEE 738 heat balance equations
│   ├── model.py         # Neural network architecture
│   ├── data.py          # Synthetic data generation
│   ├── train.py         # Training script
│   └── inference.py     # Production inference API
│
├── demo/                # 🎮 Interactive Demo
│   └── app.py           # Gradio web interface
│
├── landing/             # 🌐 Website
│   ├── index.html       # Landing page
│   └── style.css        # Styling
│
├── notebooks/           # 📓 Experiments
├── models/              # 💾 Saved models
├── data/                # 📊 Data files
└── tests/               # 🧪 Tests
```

## 🔬 How It Works

### Physics-Informed Loss Function

Our model is trained with a combined loss:

```
L_total = L_data + λ · L_physics
```

Where `L_physics` penalizes violations of the IEEE 738 heat balance:

```
q_convection + q_radiation = q_solar + I²R
```

This ensures predictions are physically consistent, not just statistically accurate.

### Model Architecture

```
Input: [T_ambient, wind_speed, wind_angle, solar, current, resistance]
   │
   ├──► MLP Encoder (3 layers, GELU, LayerNorm)
   │
   ├──► Temperature Head ──► Conductor Temp (°C)
   │
   └──► Rating Head ──► Dynamic Ampacity (A)
```

## 📊 Performance

| Metric | Value |
|--------|-------|
| Temperature MAE | < 2°C |
| Rating MAE | < 25 A |
| Physics Residual | < 10 W/m |
| Training Time (M2) | ~15 min |
| Inference Time | < 1 ms |

## 🛠️ Development

### Run Tests

```bash
pytest tests/ -v
```

### Code Formatting

```bash
black core/ demo/ tests/
```

### Type Checking

```bash
mypy core/
```

## 🚢 Deployment

### HuggingFace Spaces (Demo)

1. Create a new Space at [huggingface.co/spaces](https://huggingface.co/spaces)
2. Upload `demo/app.py` and `core/` directory
3. Add `requirements.txt`
4. Done! Free hosting with GPU option

### GitHub Pages (Landing)

1. Push `landing/` contents to `gh-pages` branch
2. Enable GitHub Pages in repo settings
3. Configure custom domain (invariant.energy)

## 📜 License

Proprietary. Contact gelavizh@invariant.energy for licensing inquiries.

## 📞 Contact

**Dr. Gelavizh Ahmadi**  
Founder & CEO

📧 gelavizh@invariant.energy  
🌐 [invariant.energy](https://invariant.energy)

---

*"Physics doesn't negotiate. Neither does our AI."*
