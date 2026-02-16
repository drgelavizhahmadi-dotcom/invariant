"""
Invariant-PIKAN: Adversarially-Robust Physics-Informed Neural Networks for Dynamic Line Rating
Copyright (C) 2025 Gelavizh Ahmadi / Invariant Research

This software is licensed under the Business Source License 1.1 (BSL 1.1).
Commercial production use requires a separate license agreement.
See LICENSE.txt for full terms.

DISCLAIMER: This implementation is independent of concurrent academic work on
HWF-PIKAN for plasma physics (Heravifard et al., Sharif University, 2025).
"""

#!/usr/bin/env python3
import glob
import os
import torch

run_dir = 'runs/20260213_234337'
print(f'🔍 Checking: {run_dir}\n')

best_model = os.path.join(run_dir, 'best_model.pt')
if os.path.exists(best_model):
    print('✅ best_model.pt exists')
    data = torch.load(best_model, map_location='cpu')
    if 'history' in data:
        h = data['history']
        if 'val_amp_mae' in h and len(h['val_amp_mae']) > 0:
            epochs = len(h.get('epoch', []))
            current_mae = h['val_amp_mae'][-1]
            best_mae = min(h['val_amp_mae'])
            print(f'📊 Epochs completed: {epochs}')
            print(f'📊 Current amp MAE: {current_mae:.1f}A')
            print(f'🏆 Best amp MAE: {best_mae:.1f}A')
        else:
            print('No validation metrics in best_model.pt')
else:
    print('⏳ best_model.pt not found')

history_csv = os.path.join(run_dir, 'history.csv')
if os.path.exists(history_csv):
    print('\n✅ history.csv exists')
    with open(history_csv, 'r') as f:
        lines = f.readlines()
        print(f'📊 Total entries: {max(0, len(lines)-1)}')
        if len(lines) > 1:
            print(f'Last entry: {lines[-1].strip()}')
else:
    print('\n⏳ history.csv not yet created')

print('\n📁 Recent files in run dir:')
files = sorted(glob.glob(f'{run_dir}/*'))
for f in files[-20:]:
    print('  ', os.path.basename(f))
