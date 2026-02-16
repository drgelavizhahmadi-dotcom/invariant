#!/bin/bash

# Activate environment
cd ~/Projects/invariant
source venv/bin/activate

# Set configuration for 100-epoch run
export CUDA_VISIBLE_DEVICES=0  # If using GPU

python -m scripts.train_hwf_pikan_v2 \
    --config-override \
    epochs=100 \
    batch_size=64 \
    lr=1e-3 \
    lambda_physics=0.1 \
    fourier_bands=16 \
    wavelet_scales=4 \
    hidden_dim=64 \
    save_dir=runs/hwf_pikan_100ep_$(date +%Y%m%d_%H%M%S) \
    validate_every=5 \
    log_every=10 \
    --device auto
