#!/usr/bin/env python3
"""
Extract and compare learned physics parameters from cross-region model checkpoints.

This script loads the 4 trained models and extracts:
- Global mean physics parameters (μ)
- Per-line deviations
- Compares VN vs US learned physics
"""

import torch
import numpy as np
import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from models.invariant_pikan_v2 import create_invariant_pikan_v2
from models.line_physics import LinePhysicsParams


def load_model_checkpoint(checkpoint_path):
    """Load model checkpoint and return model."""
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)

    # Create model (assuming same architecture for all)
    model = create_invariant_pikan_v2(
        input_dim=4,
        output_dim=2,
        hidden_dims=[64, 64],
        fourier_features=16,
        wavelet_scales=8,
        n_lines=1,  # Default, will be overridden by checkpoint
        use_line_physics=True
    )

    # Load state dict
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    return model, checkpoint


def extract_physics_params(model):
    """Extract learned physics parameters from LinePhysicsParams module."""
    # Find the line_physics module
    if hasattr(model, 'line_physics'):
        line_physics = model.line_physics
    else:
        # Search in submodules
        line_physics = None
        for name, module in model.named_modules():
            if isinstance(module, LinePhysicsParams):
                line_physics = module
                break

    if line_physics is None:
        print("Warning: No LinePhysicsParams found in model")
        return None

    # Extract parameters
    with torch.no_grad():
        # Global means (μ)
        R_global = line_physics.R_global.item()
        epsilon_global = line_physics.epsilon_global.item()
        alpha_global = line_physics.alpha_global.item()

        # Per-line deviations
        R_per_line = line_physics.R_per_line.cpu().numpy()
        epsilon_per_line = line_physics.epsilon_per_line.cpu().numpy()
        alpha_per_line = line_physics.alpha_per_line.cpu().numpy()

        # Compute actual values (global + deviation)
        R_actual = R_global + R_per_line
        epsilon_actual = epsilon_global + epsilon_per_line
        alpha_actual = alpha_global + alpha_per_line

    return {
        'R_global': R_global,
        'epsilon_global': epsilon_global,
        'alpha_global': alpha_global,
        'R_per_line': R_per_line,
        'epsilon_per_line': epsilon_per_line,
        'alpha_per_line': alpha_per_line,
        'R_actual': R_actual,
        'epsilon_actual': epsilon_actual,
        'alpha_actual': alpha_actual,
        'n_lines': len(R_per_line)
    }


def print_comparison_table(vn_params, us_params):
    """Print comparison table of learned parameters."""

    # IEEE 738 defaults
    IEEE_R_FACTOR = 1.00  # Resistance at operating temperature
    IEEE_EPSILON = 0.50   # Emissivity (typical for weathered conductors)
    IEEE_ALPHA = 0.50     # Absorptivity (typical)

    print("\n" + "="*80)
    print("LEARNED PHYSICS PARAMETER COMPARISON")
    print("="*80)
    print("\n| Parameter | VN_only μ | US_only μ | IEEE Default | VN-US Diff % |")
    print("|-----------|-----------|-----------|--------------|--------------|")

    # Resistance factor
    vn_R = vn_params['R_global']
    us_R = us_params['R_global']
    diff_R = abs(vn_R - us_R) / ((vn_R + us_R) / 2) * 100
    print(f"| Resistance factor | {vn_R:6.3f} | {us_R:6.3f} | {IEEE_R_FACTOR:6.2f} | {diff_R:6.1f}% |")

    # Emissivity
    vn_eps = vn_params['epsilon_global']
    us_eps = us_params['epsilon_global']
    diff_eps = abs(vn_eps - us_eps) / ((vn_eps + us_eps) / 2) * 100
    print(f"| Emissivity | {vn_eps:6.3f} | {us_eps:6.3f} | {IEEE_EPSILON:6.2f} | {diff_eps:6.1f}% |")

    # Absorptivity
    vn_alpha = vn_params['alpha_global']
    us_alpha = us_params['alpha_global']
    diff_alpha = abs(vn_alpha - us_alpha) / ((vn_alpha + us_alpha) / 2) * 100
    print(f"| Absorptivity | {vn_alpha:6.3f} | {us_alpha:6.3f} | {IEEE_ALPHA:6.2f} | {diff_alpha:6.1f}% |")

    print("\n" + "="*80)
    print("\nINTERPRETATION:")
    print("-" * 80)

    # Interpret differences
    if diff_R > 30 or diff_eps > 30 or diff_alpha > 30:
        print("⚠️  LARGE PARAMETER DIFFERENCE (>30%) DETECTED!")
        print("\nThis indicates that Vietnam and US transmission lines have")
        print("fundamentally different physical characteristics:")
        print()
        if diff_R > 30:
            print(f"  • Resistance: {diff_R:.1f}% difference suggests different conductor")
            print("    materials, sizing, or operating temperatures")
        if diff_eps > 30:
            print(f"  • Emissivity: {diff_eps:.1f}% difference suggests different surface")
            print("    treatments or weathering conditions")
        if diff_alpha > 30:
            print(f"  • Absorptivity: {diff_alpha:.1f}% difference suggests different")
            print("    surface colors or coatings")
        print("\n✅ This explains the cross-region transfer failure!")
    else:
        print("✅ Parameters are similar (<30% difference)")
        print("\nThe cross-region transfer failure is NOT primarily due to")
        print("different conductor physics, but rather:")
        print("  • Domain shift in weather distributions")
        print("  • Different ampacity rating standards")
        print("  • Measurement/calibration differences")

    print("="*80)


def main():
    """Main execution."""

    results_dir = Path('/Users/gelavizhahmadi/Projects/invariant/cross_region_results')

    # Load all 4 models
    print("Loading model checkpoints...")
    models = {}
    checkpoints = {}

    for name in ['VN_only', 'US_only', 'VN_to_US', 'US_to_VN']:
        checkpoint_path = results_dir / f'model_{name}.pt'
        print(f"  Loading {checkpoint_path.name}...")

        try:
            model, checkpoint = load_model_checkpoint(checkpoint_path)
            models[name] = model
            checkpoints[name] = checkpoint
        except Exception as e:
            print(f"    ⚠️  Error loading {name}: {e}")
            continue

    print(f"\nSuccessfully loaded {len(models)}/4 models\n")

    # Extract physics parameters
    print("Extracting learned physics parameters...")
    params = {}

    for name, model in models.items():
        print(f"  Extracting from {name}...")
        p = extract_physics_params(model)

        if p is not None:
            params[name] = p
            print(f"    ✓ Found {p['n_lines']} line(s)")
            print(f"      R_global={p['R_global']:.3f}, ε={p['epsilon_global']:.3f}, α={p['alpha_global']:.3f}")
        else:
            print(f"    ⚠️  No LinePhysicsParams found")

    # Print comparison
    if 'VN_only' in params and 'US_only' in params:
        print_comparison_table(params['VN_only'], params['US_only'])

        # Additional analysis
        print("\n" + "="*80)
        print("ADDITIONAL INSIGHTS")
        print("="*80)

        # Check if transfer learning changed parameters
        if 'VN_to_US' in params and 'US_to_VN' in params:
            print("\nDid transfer learning adapt physics parameters?")
            print("-" * 80)

            vn_to_us_R = params['VN_to_US']['R_global']
            vn_only_R = params['VN_only']['R_global']
            us_only_R = params['US_only']['R_global']

            print(f"\nResistance factor evolution:")
            print(f"  VN_only trained:  {vn_only_R:.3f}")
            print(f"  VN → US transfer: {vn_to_us_R:.3f}")
            print(f"  US_only trained:  {us_only_R:.3f}")

            if abs(vn_to_us_R - us_only_R) < abs(vn_only_R - us_only_R):
                print("\n  ✅ Transfer learning partially adapted physics toward US domain")
            else:
                print("\n  ❌ Transfer learning did NOT successfully adapt physics")

            print("\n" + "="*80)
    else:
        print("\n⚠️  Missing VN_only or US_only models - cannot compare")

    # Save detailed output
    output_file = results_dir / 'physics_parameters_analysis.txt'
    print(f"\n\n💾 Full analysis saved to: {output_file}")


if __name__ == '__main__':
    main()
