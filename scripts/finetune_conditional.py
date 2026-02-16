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
finetune_conditional.py

Fine-tune a pre-trained InvariantPIKAN v2 model on specific data subsets
defined by conditional filters (e.g., high-wind or afternoon conditions).

This script loads a base model, filters the training dataset based on
the provided condition, and performs targeted fine-tuning to improve
performance in high-error regions.

Usage:
    python -m scripts.finetune_conditional \
        --base-model runs/invariant_pikan_production_20260214_110318/final_model.pt \
        --condition "wind_speed > 10 or (hour >= 12 and hour <= 18)" \
        --epochs 50 \
        --lr 1e-4
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torch.utils.tensorboard import SummaryWriter

# Project imports
from core.data import VietnamDataset, InputNormalizer
from models.invariant_pikan_v2 import create_invariant_pikan_v2
from core.physics import IEEE738HeatBalance

# Use the same PhysicsInformedLoss as production training
try:
    from scripts.train_invariant_pikan_v2 import PhysicsInformedLoss
except Exception:
    from core.model import PhysicsInformedLoss


def parse_condition(condition_str: str) -> callable:
    """
    Parse a condition string into a filter function.

    Supports simple conditions like:
    - "wind_speed > 10"
    - "hour >= 12 and hour <= 18"
    - "wind_speed > 10 or (hour >= 12 and hour <= 18)"

    Args:
        condition_str: String condition to parse

    Returns:
        Filter function that takes a dataset index and returns bool
    """
    # Simple parser for basic conditions
    # This is a basic implementation - could be extended with a proper expression parser

    def create_filter(dataset):
        # Extract variables from condition
        vars_needed = []
        if 'wind_speed' in condition_str:
            vars_needed.append('wind_speed')
        if 'hour' in condition_str:
            vars_needed.append('hour')

        def filter_func(idx):
            # Get values
            values = {}
            if 'wind_speed' in vars_needed:
                values['wind_speed'] = dataset.wind_speed[idx]
            if 'hour' in vars_needed:
                values['hour'] = dataset._hour[idx]

            # Evaluate condition (use eval with restricted globals)
            try:
                return eval(condition_str, {"__builtins__": {}}, values)
            except Exception as e:
                raise ValueError(f"Failed to evaluate condition '{condition_str}': {e}")

        return filter_func

    return create_filter


def filter_dataset_by_condition(dataset, condition_str: str) -> Subset:
    """
    Filter dataset indices based on condition string.

    Args:
        dataset: The dataset to filter
        condition_str: Condition string

    Returns:
        Subset of dataset matching the condition
    """
    create_filter = parse_condition(condition_str)
    filter_func = create_filter(dataset)

    # Find matching indices
    matching_indices = [idx for idx in range(len(dataset)) if filter_func(idx)]

    print(f"Condition '{condition_str}' matches {len(matching_indices)}/{len(dataset)} samples")

    return Subset(dataset, matching_indices)


def main():
    parser = argparse.ArgumentParser(description="Fine-tune InvariantPIKAN v2 on conditional subsets")
    parser.add_argument("--base-model", type=str, required=True,
                       help="Path to base model checkpoint")
    parser.add_argument("--condition", type=str, required=True,
                       help="Condition string to filter training data")
    parser.add_argument("--loss-weights", type=str, default='{"temp": 2.0, "amp": 1.0, "physics": 0.05}',
                       help="JSON string with loss weights for temp, amp, physics")
    parser.add_argument("--epochs", type=int, default=50,
                       help="Number of fine-tuning epochs")
    parser.add_argument("--lr", type=float, default=1e-4,
                       help="Learning rate for fine-tuning")
    parser.add_argument("--batch-size", type=int, default=32,
                       help="Batch size for fine-tuning")
    parser.add_argument("--device", type=str, default="auto",
                       help="Device: 'cpu', 'mps', 'cuda', or 'auto'")
    parser.add_argument("--output-dir", type=str, default=None,
                       help="Output directory (default: auto-generated)")

    args = parser.parse_args()

    # Setup device
    if args.device == "auto":
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device(args.device)

    print(f"Using device: {device}")

    # Load base model
    print(f"Loading base model from {args.base_model}")
    checkpoint = torch.load(args.base_model, map_location=device)

    # Get model config from checkpoint
    model_config = checkpoint.get('config', {})
    if not model_config:
        # Use the known config from the fine-tuned model
        model_config = {
            'input_dim': 4,
            'fourier_bands': 8,
            'wavelet_scales': 2,
            'hidden_dim': 32,
            'kan_grid': 3,
            'kan_k': 3
        }

    print(f"Using model config: {model_config}")

    # Create model with config
    model = create_invariant_pikan_v2(config=model_config)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.train()

    # Load dataset
    print("Loading Vietnam dataset...")
    dataset = VietnamDataset(csv_path="data/mendeley/vietnam_220kv.csv")

    # Filter dataset by condition
    print(f"Filtering dataset with condition: {args.condition}")
    filtered_dataset = filter_dataset_by_condition(dataset, args.condition)

    if len(filtered_dataset) == 0:
        print("No samples match the condition. Exiting.")
        sys.exit(1)

    # Create data loader
    train_loader = DataLoader(
        filtered_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        persistent_workers=True,
        prefetch_factor=2,
        pin_memory=(device.type == 'cuda'),
    )

    # Setup output directory
    if args.output_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        condition_safe = args.condition.replace(" ", "_").replace(">", "gt").replace("<", "lt").replace("=", "eq").replace("or", "OR").replace("and", "AND").replace("(", "").replace(")", "")
        args.output_dir = f"runs/finetune_{condition_safe}_{timestamp}"

    os.makedirs(args.output_dir, exist_ok=True)
    print(f"Output directory: {args.output_dir}")

    # Setup TensorBoard
    writer = SummaryWriter(args.output_dir)

    # Parse loss weights
    loss_weights = json.loads(args.loss_weights)
    temp_weight = loss_weights.get('temp', 2.0)
    amp_weight = loss_weights.get('amp', 1.0)
    physics_weight = loss_weights.get('physics', 0.05)

    # Setup loss and optimizer
    physics_engine = IEEE738HeatBalance()
    loss_fn = PhysicsInformedLoss(physics_engine=physics_engine, lambda_physics=physics_weight)
    # Set the weights if the loss function supports it
    if hasattr(loss_fn, 'temp_weight'):
        loss_fn.temp_weight = temp_weight
    if hasattr(loss_fn, 'rating_weight'):
        loss_fn.rating_weight = amp_weight

    # Fine-tuning: use smaller learning rate, only train certain layers if desired
    # For now, fine-tune all parameters
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    # Training loop
    print(f"Starting fine-tuning for {args.epochs} epochs...")
    best_loss = float('inf')

    for epoch in range(args.epochs):
        epoch_start = time.time()
        total_loss = 0.0
        n_batches = 0

        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            
            # Split inputs: weather (4) + current (1)
            weather = batch_x[:, :4]
            current = batch_x[:, 4:5]
            weather_dict = {
                'T_amb': weather[:, 0],
                'wind_speed': weather[:, 1], 
                'solar': weather[:, 3]
            }

            # Forward pass
            predictions = model(weather, weather_dict)
            targets = {'temperature': batch_y[:, 0], 'ampacity': batch_y[:, 1]}

            # Get raw weather for physics loss (approximate with normalized)
            weather_for_loss = torch.cat([weather[:, :3], current], dim=-1)

            loss, loss_components = loss_fn(predictions, targets, weather_for_loss, return_components=True)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        avg_loss = total_loss / n_batches
        epoch_time = time.time() - epoch_start

        print(f"Epoch {epoch+1}/{args.epochs} - Loss: {avg_loss:.4f} - Time: {epoch_time:.2f}s")

        # Log to TensorBoard
        writer.add_scalar('Loss/train', avg_loss, epoch)
        for comp_name, comp_value in loss_components.items():
            if isinstance(comp_value, (int, float)):
                writer.add_scalar(f'Loss/{comp_name}', comp_value, epoch)
            elif isinstance(comp_value, torch.Tensor):
                writer.add_scalar(f'Loss/{comp_name}', comp_value.item(), epoch)
            else:
                # Skip if it's a list or other type
                pass

    # Save best model
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': best_loss,
                'condition': args.condition,
                'config': model_config,
            }, os.path.join(args.output_dir, 'best_model.pt'))

    # Save final model
    torch.save({
        'epoch': args.epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': avg_loss,
        'condition': args.condition,
        'config': model_config,
    }, os.path.join(args.output_dir, 'final_model.pt'))

    print(f"Fine-tuning complete. Best loss: {best_loss:.4f}")
    print(f"Models saved to {args.output_dir}")

    writer.close()


if __name__ == "__main__":
    main()
