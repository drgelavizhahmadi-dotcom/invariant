# Learnable Per-Line Physics Parameters

## Implementation Complete

### Created Files

1. **`models/line_physics.py`** - LinePhysicsParams module
   - Hierarchical Bayesian model with global mean + per-line deviations
   - Physical constraints via bounded transformations
   - Regularization loss for shrinkage toward global mean

2. **`core/physics_per_line.py`** - Physics functions
   - `heat_balance_residual_per_line()`: Computes residual with line-specific parameters
   - Support for per-line resistance, emissivity, absorptivity

3. **`scripts/train_per_line_physics.py`** - Training script
   - Integrates LinePhysicsParams with HWF-PIKAN
   - Separate learning rates for model and physics params
   - Line ID mapping from dataset

### Usage

```bash
python -m scripts.train_per_line_physics \
    --data-path data/processed/unified_dlr_training.h5 \
    --line-id-col line_id \
    --epochs 100 \
    --reg-weight 0.01
```

### Key Features

- **Hierarchical Prior**: Global mean + regularized per-line deviations
- **Physical Bounds**: Parameters constrained to realistic ranges
- **Interpretable**: Learned parameters can be inspected per line
- **Safe**: Strong regularization prevents overfitting

### Expected Outcomes

- Lower MAE on diverse datasets (target <300 A on US data)
- Reduced bias through line-specific physics adaptation
- Better generalization to lines with different characteristics
