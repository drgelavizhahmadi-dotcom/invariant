import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import matplotlib.ticker as ticker


def analyze_validation_results(csv_path='tests/ml_validation_results.csv'):
    df = pd.read_csv(csv_path)
    output_dir = Path("reports/plots")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # matplotlib-only visual style (seaborn not required in venv)
    plt.style.use('ggplot')
    
    # --- 1. THE SAFETY MOAT (Conservative Bias) ---
    plt.figure(figsize=(10, 6))
    plt.hist(df['I_rating_error_pct'].dropna(), bins=50, color='teal', alpha=0.7)
    try:
        df['I_rating_error_pct'].plot.kde(color='black')
    except Exception:
        pass
    plt.axvline(0, color='red', linestyle='--', label='Regulatory Limit (True Ampacity)')
    plt.title("Distribution of Ampacity Prediction Error\n(100% Safety Bias Verification)")
    plt.xlabel("Prediction Error (%)")
    plt.ylabel("Case Count")
    plt.legend()
    plt.savefig(output_dir / "safety_bias_dist.png")
    plt.close()

    # --- 2. THE PHYSICS RESIDUAL HEATMAP ---
    plt.figure(figsize=(10, 8))
    pivot = df.copy()
    pivot['Wind_Bin'] = pd.qcut(df['V_wind'], 10)
    pivot['Temp_Bin'] = pd.qcut(df['T_ambient'], 10)
    heatmap_data = pivot.pivot_table(
        values='physics_residual_Wm', 
        index='Wind_Bin', 
        columns='Temp_Bin', 
        aggfunc='mean'
    )
    im = plt.imshow(heatmap_data.values, cmap='RdYlGn_r', aspect='auto', origin='lower')
    plt.colorbar(im, label='Mean residual (W/m)')
    plt.title("Mean Physics Residual (W/m) across Weather Envelope")
    plt.yticks(range(len(heatmap_data.index)), [str(i) for i in heatmap_data.index])
    plt.xticks(range(len(heatmap_data.columns)), [str(c) for c in heatmap_data.columns], rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(output_dir / "physics_heatmap.png")
    plt.close()

    # --- 3. CORNER CASE AUDIT ---
    # Identify the 1% worst physics violations
    worst_cases = df.nlargest(max(1, int(len(df) * 0.01)), 'physics_residual_Wm')
    print("\n--- CRITICAL AUDIT: TOP 1% PHYSICS RESIDUE ---")
    print(worst_cases[['T_ambient', 'V_wind', 'physics_residual_Wm', 'T_error']].head(10))

    # --- 4. EXPORTING THE REGULATORY SUMMARY ---
    summary = {
        "Total_Scenarios": len(df),
        "Safety_Pass_Rate": f"{(df['I_rating_error_pct'] <= 0).mean() * 100:.2f}%",
        "Mean_Residual_Wm": df['physics_residual_Wm'].mean(),
        "P95_Residual_Wm": df['physics_residual_Wm'].quantile(0.95),
        "Max_Residual_Wm": df['physics_residual_Wm'].max(),
        "Conductor_Temp_RMSE": np.sqrt((df['T_error']**2).mean())
    }
    
    pd.DataFrame([summary]).to_csv("reports/compliance_summary.csv", index=False)
    print("\n✅ Compliance Summary generated in reports/")
    return summary


if __name__ == "__main__":
    s = analyze_validation_results()
    print(s)
