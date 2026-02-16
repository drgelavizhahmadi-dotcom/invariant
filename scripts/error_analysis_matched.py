"""
Invariant-PIKAN: Adversarially-Robust Physics-Informed Neural Networks for Dynamic Line Rating
Copyright (C) 2025 Gelavizh Ahmadi / Invariant Research

This software is licensed under the Business Source License 1.1 (BSL 1.1).
Commercial production use requires a separate license agreement.
See LICENSE.txt for full terms.

HWF-PIKAN for plasma physics (Heravifard et al., Sharif University, 2025).
"""

import pandas as pd
import numpy as np
from pathlib import Path

OUT_DIR = Path('us_validation_matched')
OUT_DIR.mkdir(exist_ok=True)

df = pd.read_csv(OUT_DIR / 'matched_validation_results.csv')

lines = []
lines.append("📊 ERROR ANALYSIS BY CONDITION")
lines.append("="*50)

# Wind speed bins
lines.append("\n🌪️  BY WIND SPEED:")
for low, high in [(0,2), (2,5), (5,10), (10,20), (20,100)]:
    mask = (df['wind_speed'] >= low) & (df['wind_speed'] < high)
    if mask.sum() > 0:
        mae = np.mean(np.abs(df.loc[mask, 'error']))
        lines.append(f"  {low}-{high} m/s: {mae:6.1f} A ({mask.sum():5d} samples)")

# Temperature bins
lines.append("\n🌡️  BY TEMPERATURE:")
for low, high in [(-20,0), (0,10), (10,20), (20,30), (30,50)]:
    mask = (df['temperature'] >= low) & (df['temperature'] < high)
    if mask.sum() > 0:
        mae = np.mean(np.abs(df.loc[mask, 'error']))
        lines.append(f"  {low:2.0f}-{high:2.0f}°C: {mae:6.1f} A ({mask.sum():5d} samples)")

# By line - top 10 only
lines.append('\n📍 TOP 10 LINES BY MAE:')
line_stats = df.groupby('line_id').agg(samples=('error','size'), mae=('error', lambda x: np.mean(np.abs(x))))
line_stats = line_stats.sort_values('mae', ascending=False)
for line_id, row in line_stats.head(10).iterrows():
    lines.append(f"  Line {int(line_id)}: {row['mae']:6.1f} A ({int(row['samples']):5d} samples)")

# Write to file and print
out_file = OUT_DIR / 'error_analysis_by_condition.txt'
with open(out_file, 'w') as f:
    f.write('\n'.join(lines))

print('\n'.join(lines))
print('\n✅ Wrote:', out_file)
