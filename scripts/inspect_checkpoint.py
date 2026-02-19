#!/usr/bin/env python3
"""Quick script to inspect checkpoint contents."""

import torch

checkpoint_path = '/Users/gelavizhahmadi/Projects/invariant/cross_region_results/model_VN_only.pt'

checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)

print("Checkpoint keys:")
for key in checkpoint.keys():
    print(f"  - {key}")

if 'config' in checkpoint:
    print("\nConfig:")
    print(checkpoint['config'])

if 'model_state_dict' in checkpoint:
    print("\nModel state dict keys (first 20):")
    keys = list(checkpoint['model_state_dict'].keys())
    for key in keys[:20]:
        print(f"  - {key}")
    print(f"  ... {len(keys)} keys total")
