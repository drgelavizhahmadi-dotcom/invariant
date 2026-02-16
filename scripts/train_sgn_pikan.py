"""
Invariant-PIKAN: Adversarially-Robust Physics-Informed Neural Networks for Dynamic Line Rating
Copyright (C) 2025 Gelavizh Ahmadi / Invariant Research

This software is licensed under the Business Source License 1.1 (BSL 1.1).
Commercial production use requires a separate license agreement.
See LICENSE.txt for full terms.

"""

#!/usr/bin/env python
# scripts/train_sgn_pikan.py

"""
Training script for SGN‑PIKAN with Sketchy Natural Gradients.
Completely separate from the main HWF‑PIKAN training pipeline.
"""

import argparse
import torch
import numpy as np
from pathlib import Path
import sys
import json
import yaml
from datetime import datetime

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from models.sgn_pikan import create_sgn_pikan
from core.physics import IEEE738HeatBalance
from core.data import VietnamDataset
from experiments.sketchy_natural_gradient.sketchy_optimizer import SketchyNaturalGradient


def parse_args():
    parser = argparse.ArgumentParser(description='Train SGN‑PIKAN with Sketchy Natural Gradients')
    parser.add_argument('--data-path', type=str, required=True,
                        help='Path to unified HDF5 or CSV data')
    parser.add_argument('--dataset', type=str, default='vietnam',
                        choices=['vietnam', 'us', 'unified'],
                        help='Dataset to use')
    parser.add_argument('--epochs', type=int, default=50,
                        help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=0.01,
                        help='Learning rate for sketchy optimizer')
    parser.add_argument('--sketch-size', type=int, default=None,
                        help='Sketch dimension (default: auto)')
    parser.add_argument('--damping', type=float, default=1e-4,
                        help='Damping for numerical stability')
    parser.add_argument('--ema-decay', type=float, default=0.9,
                        help='EMA decay for sketched covariance (0-1)')
    parser.add_argument('--use-adaptive-rank', action='store_true', default=True,
                        help='Automatically tune sketch rank based on eigenvalue decay')
    parser.add_argument('--device', type=str, default='auto',
                        help='Device: auto, mps, cuda, cpu')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--run-dir', type=str, default=None,
                        help='Custom run directory (default: runs/sgn_pikan_experiments/timestamp)')
    parser.add_argument('--validate-every', type=int, default=5)
    parser.add_argument('--save-every', type=int, default=10)
    return parser.parse_args()


def setup_device(device_str):
    if device_str == 'auto':
        if torch.cuda.is_available():
            return 'cuda'
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            return 'mps'
        else:
            return 'cpu'
    return device_str


def load_dataset(args):
    """Load dataset based on args."""
    if args.dataset == 'vietnam':
        from core.data import VietnamDataset
        dataset = VietnamDataset(args.data_path)
    elif args.dataset == 'unified':
        # Placeholder: you'll need to implement a unified loader similar to your HDF5 logic
        raise NotImplementedError("Unified dataset loading not yet implemented. Please use 'vietnam' for now.")
    else:
        raise ValueError(f"Unknown dataset: {args.dataset}")
    return dataset


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = setup_device(args.device)
    print(f"Using device: {device}")

    # Create run directory
    if args.run_dir is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        run_dir = Path('runs/sgn_pikan_experiments') / timestamp
    else:
        run_dir = Path(args.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    # Save arguments
    with open(run_dir / 'args.json', 'w') as f:
        json.dump(vars(args), f, indent=2)

    # Load dataset
    print("Loading dataset...")
    dataset = load_dataset(args)

    # Simple temporal split (80/20)
    n = len(dataset)
    split = int(n * 0.8)
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [split, n - split]
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=0, pin_memory=False
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=0, pin_memory=False
    )

    # Create model
    physics_engine = IEEE738HeatBalance().to(device)
    model = create_sgn_pikan(physics_engine).to(device)

    # Setup sketchy optimizer
    optimizer = SketchyNaturalGradient(
        model.parameters(),
        lr=args.lr,
        sketch_size=args.sketch_size,
        damping=args.damping,
        ema_decay=args.ema_decay,
        use_adaptive_rank=args.use_adaptive_rank
    )

    # Loss function (simple MSE for now; natural gradient will handle geometry)
    criterion = torch.nn.MSELoss()

    print(f"Model has {sum(p.numel() for p in model.parameters()):,} parameters")

    # Training loop
    best_val_loss = float('inf')
    history = []

    for epoch in range(args.epochs):
        model.train()
        train_loss = 0.0
        for batch in train_loader:
            # Assume batch returns (x, y) where x is input features, y is target ampacity
            # Adapt this to your actual data loader output
            x, y = batch
            x = x.to(device)
            y = y.to(device)

            def closure():
                optimizer.zero_grad()
                # Forward pass – need to extract weather and timestamp from x
                # This depends on your dataset; for VietnamDataset we need weather_dict.
                # We'll pass a dummy timestamp (None) for now.
                # In practice, you'll need to adapt to your data format.
                # For simplicity, we assume the model can handle x directly.
                # Since our SGN model expects (weather, timestamp), we need to split x.
                # If x is already [weather, current, voltage], we can separate.
                # This is a placeholder; you must adjust based on your data loader.
                # For example, if x is [T_amb, wind_speed, wind_angle, solar, current, voltage],
                # we can take first 4 as weather.
                weather = x[:, :4]  # first 4 columns
                # Create dummy timestamp
                timestamp = torch.zeros(x.shape[0], device=device)
                output = model(weather, timestamp)
                # dataset.target = [T_conductor, ampacity] -> take ampacity (col 1)
                loss = criterion(output['ampacity'], y[:, 1])
                loss.backward()
                return loss

            loss = optimizer.step(closure)
            train_loss += loss.item()

        train_loss /= len(train_loader)

        # Validation
        if (epoch + 1) % args.validate_every == 0:
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for batch in val_loader:
                    x, y = batch
                    x = x.to(device)
                    y = y.to(device)
                    weather = x[:, :4]
                    timestamp = torch.zeros(x.shape[0], device=device)
                    output = model(weather, timestamp)
                    # Take ampacity target (second column)
                    loss = criterion(output['ampacity'], y[:, 1])
                    val_loss += loss.item()
            val_loss /= len(val_loader)

            print(f"Epoch {epoch+1}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}")

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(model.state_dict(), run_dir / 'best_model.pt')
        else:
            print(f"Epoch {epoch+1}: train_loss={train_loss:.4f}")

        # Save checkpoint
        if (epoch + 1) % args.save_every == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss if 'val_loss' in locals() else None,
            }, run_dir / f'checkpoint_epoch{epoch+1}.pt')

        history.append({'epoch': epoch+1, 'train_loss': train_loss, 'val_loss': val_loss if 'val_loss' in locals() else None})

    # Save final model
    torch.save(model.state_dict(), run_dir / 'final_model.pt')

    # Save history
    import pandas as pd
    pd.DataFrame(history).to_csv(run_dir / 'history.csv', index=False)

    print(f"Training completed. Best validation loss: {best_val_loss:.4f}")
    print(f"Results saved to {run_dir}")


if __name__ == '__main__':
    main()
