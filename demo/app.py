"""
Invariant DLR Demo - Fixed Colors
"""

import gradio as gr
import torch
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from core.model import PhysicsDLR
from core.physics import IEEE738HeatBalance
from core.data import InputNormalizer

DEVICE = torch.device("cpu")

def load_model():
    model = PhysicsDLR(input_dim=6, hidden_dims=[128, 128, 64]).to(DEVICE)
    physics = IEEE738HeatBalance().to(DEVICE)
    normalizer = InputNormalizer()
    
    model_path = Path(__file__).parent.parent / "models" / "best_model.pt"
    
    if model_path.exists():
        try:
            checkpoint = torch.load(model_path, map_location=DEVICE)
            model.load_state_dict(checkpoint['model_state_dict'])
            if 'normalizer' in checkpoint:
                normalizer.mean = np.array(checkpoint['normalizer']['mean'])
                normalizer.std = np.array(checkpoint['normalizer']['std'])
            print(f"✅ Loaded trained model")
        except Exception as e:
            print(f"⚠️ Could not load model: {e}")
    
    model.eval()
    return model, physics, normalizer

MODEL, PHYSICS, NORMALIZER = load_model()

def predict_dlr(T_ambient, wind_speed, wind_angle, solar_irradiance, current, max_temp=75.0):
    x = np.array([[T_ambient, wind_speed, wind_angle, solar_irradiance, current, PHYSICS.R_ref.item()]])
    x_norm = NORMALIZER.transform(x)
    x_tensor = torch.tensor(x_norm, dtype=torch.float32, device=DEVICE)
    
    with torch.no_grad():
        pred_temp, pred_rating = MODEL(x_tensor)
    
    T_conductor = pred_temp.item()
    
    T_amb_t = torch.tensor([T_ambient], dtype=torch.float32, device=DEVICE)
    wind_t = torch.tensor([wind_speed], dtype=torch.float32, device=DEVICE)
    angle_t = torch.tensor([wind_angle], dtype=torch.float32, device=DEVICE)
    solar_t = torch.tensor([solar_irradiance], dtype=torch.float32, device=DEVICE)
    current_t = torch.tensor([current], dtype=torch.float32, device=DEVICE)
    
    physics_temp = PHYSICS.steady_state_temperature(current_t, T_amb_t, wind_t, solar_t, angle_t).item()
    physics_rating = PHYSICS.ampacity(max_temp, T_amb_t, wind_t, solar_t, angle_t).item()
    residual = PHYSICS.heat_balance_residual(current_t, torch.tensor([T_conductor], device=DEVICE), T_amb_t, wind_t, solar_t, angle_t).item()
    
    static_rating = PHYSICS.ampacity(max_temp, torch.tensor([35.0], device=DEVICE), torch.tensor([0.61], device=DEVICE), torch.tensor([1000.0], device=DEVICE)).item()
    capacity_gain = ((physics_rating - static_rating) / static_rating) * 100
    
    # FIXED: Light background charts with visible colors
    fig = make_subplots(rows=1, cols=2, subplot_titles=("⚡ Ampacity Comparison", "🎯 Physics Compliance"),
        specs=[[{"type": "bar"}, {"type": "indicator"}]], horizontal_spacing=0.15)
    
    # Bar chart - light theme
    fig.add_trace(go.Bar(
        x=["Static Rating", "Dynamic Rating"], 
        y=[static_rating, physics_rating],
        marker_color=["#6b7280", "#06b6d4"],
        text=[f"{static_rating:.0f} A", f"{physics_rating:.0f} A"],
        textposition="outside",
        textfont=dict(size=18, color="#1f2937", family="Arial Black")
    ), row=1, col=1)
    
    # Physics gauge
    compliance_color = "#10b981" if abs(residual) < 10 else "#f59e0b" if abs(residual) < 30 else "#ef4444"
    fig.add_trace(go.Indicator(
        mode="gauge+number",
        value=min(abs(residual), 50),
        number={"suffix": " W/m", "font": {"size": 32, "color": "#1f2937"}},
        title={"text": "Heat Balance Residual", "font": {"size": 16, "color": "#374151"}},
        gauge={
            "axis": {"range": [0, 50], "tickcolor": "#374151", "tickfont": {"color": "#374151", "size": 12}},
            "bar": {"color": compliance_color, "thickness": 0.8},
            "bgcolor": "#e5e7eb",
            "bordercolor": "#9ca3af",
            "borderwidth": 2,
            "steps": [
                {"range": [0, 10], "color": "#d1fae5"},
                {"range": [10, 30], "color": "#fef3c7"},
                {"range": [30, 50], "color": "#fee2e2"}
            ],
            "threshold": {"line": {"color": "#1f2937", "width": 3}, "thickness": 0.8, "value": 10}
        }
    ), row=1, col=2)
    
    fig.update_layout(
        height=400,
        showlegend=False,
        paper_bgcolor="#ffffff",
        plot_bgcolor="#f9fafb",
        font=dict(color="#1f2937", family="Arial"),
        margin=dict(t=80, b=40, l=60, r=60),
        title=dict(text="", font=dict(size=1))
    )
    
    fig.update_xaxes(tickfont=dict(color="#374151", size=14), gridcolor="#e5e7eb", linecolor="#d1d5db")
    fig.update_yaxes(tickfont=dict(color="#374151", size=14), gridcolor="#e5e7eb", linecolor="#d1d5db", title_font=dict(color="#374151"))
    
    # Status indicators
    physics_status = "✅ Compliant" if abs(residual) < 10 else "⚠️ Review" if abs(residual) < 30 else "❌ Violation"
    temp_status = "🟢 Safe" if T_conductor < max_temp * 0.9 else "🟡 Caution" if T_conductor < max_temp else "🔴 Critical"
    gain_status = "🚀" if capacity_gain > 30 else "📈" if capacity_gain > 10 else "➡️"
    
    summary = f"""
## 📊 Dynamic Line Rating Results

| Metric | Value | Status |
|:-------|------:|:-------|
| **Dynamic Rating** | **{physics_rating:.0f} A** | {gain_status} {'+' if capacity_gain > 0 else ''}{capacity_gain:.1f}% vs static |
| **Static Rating** | {static_rating:.0f} A | Baseline |
| **Conductor Temp** | {physics_temp:.1f}°C | {temp_status} |
| **Physics Residual** | {abs(residual):.2f} W/m | {physics_status} |

---

### 💡 Interpretation

{'🎉 **Excellent conditions!** High wind provides significant cooling — capacity increased by ' + f'{capacity_gain:.0f}%!' if capacity_gain > 30 else '👍 **Good conditions.** Favorable weather enables capacity increase.' if capacity_gain > 10 else '⚖️ **Near baseline.** Conditions similar to conservative assumptions.'}

{'✅ **IEEE 738 Compliant** — Heat balance equation satisfied (residual < 10 W/m)' if abs(residual) < 10 else '⚠️ **Review recommended** — Elevated physics residual'}
"""
    return summary, fig

def sensitivity_plot(T_ambient, wind_speed, solar_irradiance, current):
    wind_speeds = np.linspace(0.5, 20, 40)
    T_amb_t = torch.tensor([T_ambient], dtype=torch.float32, device=DEVICE)
    solar_t = torch.tensor([solar_irradiance], dtype=torch.float32, device=DEVICE)
    
    ratings = [PHYSICS.ampacity(75.0, T_amb_t, torch.tensor([ws], device=DEVICE), solar_t).item() for ws in wind_speeds]
    static_rating = PHYSICS.ampacity(75.0, torch.tensor([35.0], device=DEVICE), torch.tensor([0.61], device=DEVICE), torch.tensor([1000.0], device=DEVICE)).item()
    current_rating = PHYSICS.ampacity(75.0, T_amb_t, torch.tensor([wind_speed], device=DEVICE), solar_t).item()
    
    fig = go.Figure()
    
    # Area fill
    fig.add_trace(go.Scatter(
        x=wind_speeds, y=ratings, mode='lines', name='Dynamic Rating',
        line=dict(color='#0891b2', width=3),
        fill='tozeroy', fillcolor='rgba(8, 145, 178, 0.2)'
    ))
    
    # Static baseline
    fig.add_hline(y=static_rating, line_dash="dash", line_color="#dc2626", line_width=2,
        annotation_text=f"Static: {static_rating:.0f} A", annotation_position="right",
        annotation_font=dict(color="#dc2626", size=14))
    
    # Current point
    fig.add_trace(go.Scatter(
        x=[wind_speed], y=[current_rating], mode='markers+text', name='Current Conditions',
        marker=dict(color='#10b981', size=18, symbol='diamond', line=dict(color='#065f46', width=2)),
        text=[f"{current_rating:.0f} A"], textposition="top center", textfont=dict(color="#065f46", size=14, family="Arial Black")
    ))
    
    fig.update_layout(
        title=dict(text="📈 Wind Speed vs Capacity", font=dict(size=20, color="#1f2937")),
        xaxis_title="Wind Speed (m/s)",
        yaxis_title="Ampacity (A)",
        paper_bgcolor="#ffffff",
        plot_bgcolor="#f9fafb",
        font=dict(color="#374151", family="Arial", size=14),
        showlegend=True,
        legend=dict(x=0.02, y=0.98, bgcolor="rgba(255,255,255,0.8)", bordercolor="#d1d5db", borderwidth=1),
        height=350,
        margin=dict(t=60, b=50, l=60, r=40)
    )
    
    fig.update_xaxes(gridcolor="#e5e7eb", linecolor="#d1d5db", tickfont=dict(size=12))
    fig.update_yaxes(gridcolor="#e5e7eb", linecolor="#d1d5db", tickfont=dict(size=12))
    
    return fig

# Custom CSS for better visibility
custom_css = """
.gradio-container { background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%) !important; }
.gr-button-primary { background: linear-gradient(90deg, #06b6d4, #10b981) !important; border: none !important; font-weight: bold !important; }
.gr-button-primary:hover { background: linear-gradient(90deg, #0891b2, #059669) !important; transform: scale(1.02); }
.gr-form { background: rgba(30, 41, 59, 0.8) !important; border: 1px solid #334155 !important; border-radius: 12px !important; }
.gr-box { border-radius: 8px !important; }
.gr-padded { padding: 16px !important; }
.prose { color: #e2e8f0 !important; }
.prose h2 { color: #f8fafc !important; }
.prose table { background: rgba(51, 65, 85, 0.5) !important; border-radius: 8px !important; }
.prose th, .prose td { border-color: #475569 !important; padding: 12px !important; }
.prose strong { color: #38bdf8 !important; }
"""

with gr.Blocks(title="Invariant - Dynamic Line Rating", theme=gr.themes.Soft(primary_hue="cyan", secondary_hue="emerald"), css=custom_css) as demo:
    
    gr.Markdown("""
# ⚡ Invariant — Dynamic Line Rating
### Physics-Informed AI for Grid Optimization

Enter real-time weather and load conditions. Our AI predicts capacity while respecting **IEEE 738 heat balance physics**.
    """)
    
    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### 🌡️ Weather Conditions")
            T_ambient = gr.Slider(-20, 45, 25, step=1, label="Ambient Temperature (°C)", info="Air temperature near the line")
            wind_speed = gr.Slider(0.3, 20, 5.0, step=0.5, label="Wind Speed (m/s)", info="Higher wind = more cooling = more capacity")
            wind_angle = gr.Slider(0, 90, 45, step=5, label="Wind Angle (°)", info="90° = perpendicular (best cooling)")
            solar_irradiance = gr.Slider(0, 1100, 800, step=50, label="Solar Irradiance (W/m²)", info="0 at night, ~1000 peak sun")
            
            gr.Markdown("### ⚡ Line Conditions")
            current = gr.Slider(100, 1500, 800, step=50, label="Current Load (A)")
            max_temp = gr.Slider(50, 100, 75, step=5, label="Max Conductor Temp (°C)", info="Safety limit")
            
            predict_btn = gr.Button("🔮 Calculate Dynamic Rating", variant="primary", size="lg")
        
        with gr.Column(scale=2):
            output_text = gr.Markdown()
            output_plot = gr.Plot()
    
    gr.Markdown("### 📈 Sensitivity Analysis")
    sensitivity = gr.Plot()
    
    predict_btn.click(predict_dlr, [T_ambient, wind_speed, wind_angle, solar_irradiance, current, max_temp], [output_text, output_plot])
    
    for inp in [T_ambient, wind_speed, solar_irradiance, current]:
        inp.change(sensitivity_plot, [T_ambient, wind_speed, solar_irradiance, current], sensitivity)
    
    demo.load(predict_dlr, [T_ambient, wind_speed, wind_angle, solar_irradiance, current, max_temp], [output_text, output_plot])
    demo.load(sensitivity_plot, [T_ambient, wind_speed, solar_irradiance, current], sensitivity)
    
    gr.Markdown("""
---
**Invariant Energy** | 📧 gelavizh@invariant.energy | 🌐 [invariant.energy](https://invariant.energy)

*Physics doesn't negotiate. Neither does our AI.*
    """)

if __name__ == "__main__":
    demo.launch(share=True, server_name="0.0.0.0", server_port=7860)
