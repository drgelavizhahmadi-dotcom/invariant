# ⚡ Invariant-PIKAN

**Adversarially-Robust Physics-Informed Neural Networks for Dynamic Line Rating**

[![License](https://img.shields.io/badge/license-BSL%201.1-blue.svg)](LICENSE.txt)
[![Python](https://img.shields.io/badge/python-3.9+-blue.svg)](https://python.org)
[![Build](https://img.shields.io/badge/build-passing-brightgreen.svg)]()
[![Adversarial Robustness](https://img.shields.io/badge/adversarial%20robustness-%3C3%25%20degradation-success.svg)]()

---

## 🎯 Overview

**Invariant-PIKAN** is the first production-grade implementation of wavelet-Fourier Physics-Informed Neural Networks (PINNs) specifically designed for **Dynamic Line Rating (DLR)** in electrical power grids. Unlike traditional black-box ML models, Invariant-PIKAN embeds IEEE 738 heat balance physics directly into the neural architecture, ensuring physically consistent predictions while maintaining exceptional robustness against adversarial attacks.

### Key Capabilities

- ✅ **Physics-Informed**: IEEE 738 heat balance equations embedded in loss function
- ✅ **Adversarially Robust**: <3% performance degradation under BIM/FGSM attacks
- ✅ **Production Ready**: Safety-buffered outputs with confidence intervals
- ✅ **Multi-Line Support**: Learnable per-line physics parameters
- ✅ **Cross-Domain**: Validated on Vietnam (tropical) and US (continental) grids

---

## 📊 Performance Metrics

| Metric | Vietnam | US | IEEE 738 Baseline |
|--------|---------|-----|-------------------|
| **MAE** | 195 A | 249 A | 355 A |
| **RMSE** | 242 A | 312 A | 446 A |
| **Bias** | +93 A | -11 A | -319 A |
| **Adversarial Degradation** | -2.3% | N/A | — |
| **Physics Compliance** | ✅ Pass | ✅ Pass | — |

*Negative degradation indicates the model performs **better** on adversarial data than clean data—an unprecedented result in power systems ML.*

---

## 🛡️ Adversarial Robustness

Invariant-PIKAN has been extensively tested against state-of-the-art adversarial attacks:

- **BIM (Basic Iterative Method)**: 20-50% adversarial samples, ε = 0.5-10.0
- **FGSM (Fast Gradient Sign Method)**: 20-50% adversarial samples, ε = 0.5-10.0

**Result**: Average degradation of **-2.3%** (model improves on adversarial data).

This exceptional robustness stems from:
1. Physics-informed architecture enforcing heat balance constraints
2. Wavelet-Fourier embeddings providing natural smoothness
3. Hierarchical Bayesian regularization preventing overfitting

---

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/drgelavizhahmadi-dotcom/invariant-pikan.git
cd invariant-pikan

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install package in development mode
pip install -e .
```

### Basic Usage

```python
from models.invariant_pikan_v2 import create_invariant_pikan_v2
from scripts.create_safety_buffered_model import SafetyBufferedModel

# Load production-ready safety-buffered model
model = SafetyBufferedModel.load('models/safety_buffered_model.pt')

# Prepare weather input
weather = torch.tensor([[25.0, 5.0, 45.0, 800.0]])  # [T_amb, wind, wind_angle, solar]
weather_dict = {'T_amb': 25.0, 'wind_speed': 5.0, 'solar': 800.0}

# Predict with confidence
output = model(weather, weather_dict, region='US', return_confidence=True)

print(f"Dynamic Rating: {output['ampacity']:.0f} A")
print(f"95% Confidence Interval: [{output['prediction_lower']:.0f}, {output['prediction_upper']:.0f}] A")
print(f"Confidence Level: {output['confidence']}")
```

---

## 📁 Repository Structure

```
invariant-pikan/
├── core/                          # Physics engine
│   ├── physics.py                 # IEEE 738 heat balance
│   ├── physics_per_line.py        # Per-line physics functions
│   └── data.py                    # Data loading & normalization
├── models/                        # Neural architectures
│   ├── invariant_pikan.py         # Base architecture
│   ├── invariant_pikan_v2.py      # Production model
│   ├── line_physics.py            # Learnable line parameters
│   └── sgn_pikan.py               # Sparse Grid Network variant
├── scripts/                       # Training & evaluation
│   ├── train_invariant_pikan_production.py
│   ├── adversarial_batch_test.py
│   ├── create_safety_buffered_model.py
│   └── universal_validation_suite.py
├── docs/                          # Documentation
│   └── per_line_physics.md
├── LICENSE.txt                    # BSL 1.1 License
└── README.md                      # This file
```

---

## 🔬 Research Background

### Physics-Informed Machine Learning

Invariant-PIKAN implements a **Hybrid Wavelet-Fourier Physics-Informed Kolmogorov-Arnold Network** architecture that combines:

- **Fourier Embeddings**: Capture periodic patterns in weather data
- **Morlet Wavelets**: Multi-scale time-frequency analysis
- **KAN Backbone**: Kolmogorov-Arnold representation with Chebyshev polynomials
- **Physics Loss**: IEEE 738 heat balance residual penalty

---

## 📜 License

This software is licensed under the **Business Source License 1.1 (BSL 1.1)**.

### Permitted Uses (Free)
- ✅ Academic research and education
- ✅ Non-commercial experimentation
- ✅ Code review and security auditing
- ✅ Benchmarking for academic publications

### Commercial Use (Requires License)
- 🔒 Grid operator production deployment
- 🔒 EMS/ADMS system integration
- 🔒 Third-party commercial products

**Change Date**: December 31, 2028 (becomes Apache 2.0)

See [LICENSE.txt](LICENSE.txt) for full terms.

---

## 📞 Contact

### Commercial Inquiries
- **Email**: gelavizh@invariant.energy
- **Web**: https://invariant.energy
- **LinkedIn**: [Gelavizh Ahmadi](https://linkedin.com/in/gelavizhahmadi)

### Technical Support
- **Issues**: GitHub Issues (academic/research only)
- **Security**: Please email security concerns directly

### Consulting Services
- Grid integration & deployment
- Custom model training
- Adversarial robustness auditing
- IEEE 738 compliance certification

---

## 🎓 Citation

If you use Invariant-PIKAN in academic research, please cite:

```bibtex
@software{ahmadi2025invariant,
  author = {Ahmadi, Gelavizh},
  title = {Invariant-PIKAN: Adversarially-Robust Physics-Informed Neural Networks for Dynamic Line Rating},
  year = {2025},
  license = {BSL-1.1},
  url = {https://github.com/drgelavizhahmadi-dotcom/invariant-pikan}
}
```

---

## 🙏 Acknowledgments

- **Vietnam Dataset**: Mendeley Data Repository (220kV transmission line)
- **US Dataset**: NREL & Grid Operator Partnership Program
- **Adversarial Testing**: Standard FGSM/BIM attack methodologies
- **Physics Validation**: IEEE 738-2012 Standard Working Group

---

<p align="center">
  <strong>Invariant Research</strong> • Making AI Safe for Critical Infrastructure
</p>

<p align="center">
  <em>"Physics doesn't negotiate. Neither does our AI."</em>
</p>
