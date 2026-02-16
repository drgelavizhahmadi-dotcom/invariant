"""
Invariant-PIKAN: Adversarially-Robust Physics-Informed Neural Networks for Dynamic Line Rating
Copyright (C) 2025 Gelavizh Ahmadi / Invariant Research

This software is licensed under the Business Source License 1.1 (BSL 1.1).
Commercial production use requires a separate license agreement.
See LICENSE.txt for full terms.

HWF-PIKAN for plasma physics (Heravifard et al., Sharif University, 2025).
"""

#!/usr/bin/env python3
"""
Invariant Quick Start Script

Run this script to:
1. Train the model (if not already trained)
2. Launch the demo

Usage:
    python run.py          # Train + launch demo
    python run.py --train  # Train only
    python run.py --demo   # Demo only (requires trained model)
"""

import argparse
import sys
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description="Invariant Quick Start")
    parser.add_argument("--train", action="store_true", help="Train model only")
    parser.add_argument("--demo", action="store_true", help="Launch demo only")
    parser.add_argument("--epochs", type=int, default=50, help="Training epochs")
    parser.add_argument("--quick", action="store_true", help="Quick training (30 epochs)")
    args = parser.parse_args()
    
    model_path = Path("models/best_model.pt")
    
    # Default: both train and demo
    if not args.train and not args.demo:
        args.train = True
        args.demo = True
    
    # Training
    if args.train:
        print("\n" + "="*60)
        print("⚡ INVARIANT - Physics-Informed AI for Grid Optimization")
        print("="*60)
        
        from core.train import train, TrainConfig
        
        epochs = 30 if args.quick else args.epochs
        
        config = TrainConfig(
            n_epochs=epochs,
            n_train=10000,
            n_val=2000,
            batch_size=256,
            save_path=str(model_path),
            log_every=10,
        )
        
        model, history = train(config)
        print(f"\n✅ Model saved to {model_path}")
    
    # Demo
    if args.demo:
        if not model_path.exists() and not args.train:
            print(f"\n❌ No trained model found at {model_path}")
            print("   Run with --train first, or use --quick for fast training")
            sys.exit(1)
        
        print("\n" + "="*60)
        print("🎮 Launching Gradio Demo...")
        print("="*60)
        
        # Import and launch demo
        import demo.app
        demo.app.demo.launch(share=True)


if __name__ == "__main__":
    main()
