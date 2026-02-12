"""
Training Script for Invariant PINN

Optimized for:
- Apple Silicon (M1/M2) with MPS backend
- CPU fallback for Linux/Windows
- CUDA for cloud GPUs

Author: Dr. Gelavizh Ahmadi
Copyright (c) 2026 Invariant Energy GmbH
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from typing import Optional, Tuple, Dict, Callable
import time
from pathlib import Path
from dataclasses import dataclass
import json

from .model import PhysicsDLR, PhysicsInformedLoss
from .physics import IEEE738HeatBalance
from .data import (
    SyntheticDLRDataset, 
    create_dataloaders, 
    denormalize_batch,
    InputNormalizer,
)


@dataclass
class TrainConfig:
    """Training configuration"""
    # Data
    n_train: int = 15000
    n_val: int = 3000
    batch_size: int = 256
    
    # Model
    hidden_dims: list = None  # Will default to [128, 128, 64]
    dropout: float = 0.1
    
    # Training
    n_epochs: int = 100
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    
    # Loss weights
    physics_weight: float = 0.3
    temp_weight: float = 1.0
    rating_weight: float = 0.5
    
    # Scheduling
    scheduler: str = 'cosine'  # 'cosine', 'step', 'none'
    warmup_epochs: int = 5
    
    # Checkpointing
    save_path: str = 'models/best_model.pt'
    save_every: int = 20
    
    # Logging
    log_every: int = 10
    verbose: bool = True
    
    def __post_init__(self):
        if self.hidden_dims is None:
            self.hidden_dims = [128, 128, 64]


def get_device() -> torch.device:
    """
    Get the best available compute device
    
    Priority: MPS (Apple Silicon) > CUDA > CPU
    """
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print(f"Using Apple Silicon (MPS)")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Using CUDA: {torch.cuda.get_device_name()}")
    else:
        device = torch.device("cpu")
        print("Using CPU")
    
    return device


def train_epoch(
    model: PhysicsDLR,
    train_loader: DataLoader,
    optimizer: optim.Optimizer,
    loss_fn: PhysicsInformedLoss,
    physics: IEEE738HeatBalance,
    device: torch.device,
    normalizer: InputNormalizer,
) -> Dict[str, float]:
    """
    Train for one epoch
    
    Returns:
        Dictionary of average metrics for the epoch
    """
    model.train()
    
    total_metrics = {
        'total_loss': 0,
        'temp_loss': 0,
        'rating_loss': 0,
        'physics_loss': 0,
        'physics_residual_mean': 0,
    }
    n_batches = 0
    
    for x, y in train_loader:
        x, y = x.to(device), y.to(device)
        
        # Forward pass
        pred_temp, pred_rating = model(x)
        
        # Denormalize inputs for physics calculation
        raw_inputs = denormalize_batch(x, normalizer)
        
        # Calculate physics residual
        physics_residual = physics.heat_balance_residual(
            current=raw_inputs['current'],
            T_conductor=pred_temp.squeeze(),
            T_ambient=raw_inputs['T_ambient'],
            wind_speed=raw_inputs['wind_speed'],
            solar_irradiance=raw_inputs['solar_irradiance'],
            wind_angle=raw_inputs['wind_angle'],
        )
        
        # Compute loss
        loss, metrics = loss_fn(
            pred_temp=pred_temp,
            pred_rating=pred_rating,
            true_temp=y[:, 0:1],
            true_rating=y[:, 1:2],
            physics_residual=physics_residual,
        )
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        # Accumulate metrics
        for k in total_metrics:
            if k in metrics:
                total_metrics[k] += metrics[k]
        n_batches += 1
    
    # Average metrics
    for k in total_metrics:
        total_metrics[k] /= n_batches
    
    return total_metrics


@torch.no_grad()
def validate(
    model: PhysicsDLR,
    val_loader: DataLoader,
    loss_fn: PhysicsInformedLoss,
    physics: IEEE738HeatBalance,
    device: torch.device,
    normalizer: InputNormalizer,
) -> Dict[str, float]:
    """
    Validation pass
    
    Returns:
        Dictionary of validation metrics
    """
    model.eval()
    
    total_metrics = {
        'total_loss': 0,
        'temp_loss': 0,
        'rating_loss': 0,
        'physics_loss': 0,
        'physics_residual_mean': 0,
        'temp_mae': 0,
        'rating_mae': 0,
        'temp_rmse': 0,
        'rating_rmse': 0,
    }
    n_batches = 0
    
    all_temp_errors = []
    all_rating_errors = []
    
    for x, y in val_loader:
        x, y = x.to(device), y.to(device)
        
        pred_temp, pred_rating = model(x)
        
        raw_inputs = denormalize_batch(x, normalizer)
        
        physics_residual = physics.heat_balance_residual(
            current=raw_inputs['current'],
            T_conductor=pred_temp.squeeze(),
            T_ambient=raw_inputs['T_ambient'],
            wind_speed=raw_inputs['wind_speed'],
            solar_irradiance=raw_inputs['solar_irradiance'],
            wind_angle=raw_inputs['wind_angle'],
        )
        
        loss, metrics = loss_fn(
            pred_temp=pred_temp,
            pred_rating=pred_rating,
            true_temp=y[:, 0:1],
            true_rating=y[:, 1:2],
            physics_residual=physics_residual,
        )
        
        # Calculate additional metrics
        temp_error = pred_temp.squeeze() - y[:, 0]
        rating_error = pred_rating.squeeze() - y[:, 1]
        
        all_temp_errors.append(temp_error.cpu())
        all_rating_errors.append(rating_error.cpu())
        
        metrics['temp_mae'] = temp_error.abs().mean().item()
        metrics['rating_mae'] = rating_error.abs().mean().item()
        metrics['temp_rmse'] = torch.sqrt((temp_error ** 2).mean()).item()
        metrics['rating_rmse'] = torch.sqrt((rating_error ** 2).mean()).item()
        
        for k in total_metrics:
            if k in metrics:
                total_metrics[k] += metrics[k]
        n_batches += 1
    
    # Average metrics
    for k in total_metrics:
        total_metrics[k] /= n_batches
    
    # Overall RMSE
    all_temp = torch.cat(all_temp_errors)
    all_rating = torch.cat(all_rating_errors)
    total_metrics['temp_rmse_overall'] = torch.sqrt((all_temp ** 2).mean()).item()
    total_metrics['rating_rmse_overall'] = torch.sqrt((all_rating ** 2).mean()).item()
    
    return total_metrics


def create_scheduler(
    optimizer: optim.Optimizer,
    config: TrainConfig,
) -> Optional[optim.lr_scheduler._LRScheduler]:
    """Create learning rate scheduler"""
    
    if config.scheduler == 'cosine':
        return optim.lr_scheduler.CosineAnnealingLR(
            optimizer, 
            T_max=config.n_epochs - config.warmup_epochs,
            eta_min=config.learning_rate * 0.01
        )
    elif config.scheduler == 'step':
        return optim.lr_scheduler.StepLR(
            optimizer,
            step_size=30,
            gamma=0.5
        )
    else:
        return None


def train(
    config: Optional[TrainConfig] = None,
    model: Optional[PhysicsDLR] = None,
    callback: Optional[Callable] = None,
) -> Tuple[PhysicsDLR, Dict]:
    """
    Main training function
    
    Args:
        config: TrainConfig instance (uses defaults if None)
        model: Pre-initialized model (creates new if None)
        callback: Optional callback function called each epoch
        
    Returns:
        model: Trained model
        history: Training history dictionary
    """
    config = config or TrainConfig()
    device = get_device()
    
    # Create data
    print("\n📊 Creating datasets...")
    physics = IEEE738HeatBalance().to(device)
    
    train_loader, val_loader, train_dataset, val_dataset = create_dataloaders(
        n_train=config.n_train,
        n_val=config.n_val,
        batch_size=config.batch_size,
        num_workers=0,  # MPS doesn't support multiprocessing well
    )
    
    normalizer = train_dataset.normalizer
    
    print(f"   Train samples: {config.n_train:,}")
    print(f"   Val samples: {config.n_val:,}")
    print(f"   Batch size: {config.batch_size}")
    
    # Create model
    if model is None:
        print("\n🧠 Creating model...")
        model = PhysicsDLR(
            input_dim=6,
            hidden_dims=config.hidden_dims,
            dropout=config.dropout,
        ).to(device)
    else:
        model = model.to(device)
    
    print(f"   Parameters: {model.count_parameters():,}")
    print(f"   Hidden dims: {config.hidden_dims}")
    
    # Optimizer and loss
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay
    )
    
    scheduler = create_scheduler(optimizer, config)
    
    loss_fn = PhysicsInformedLoss(
        physics_weight=config.physics_weight,
        temp_weight=config.temp_weight,
        rating_weight=config.rating_weight,
    )
    
    # Training loop
    history = {
        'train': [],
        'val': [],
        'lr': [],
    }
    best_val_loss = float('inf')
    best_epoch = 0
    
    # Create save directory
    save_dir = Path(config.save_path).parent
    save_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n🚀 Starting training for {config.n_epochs} epochs...")
    print("-" * 70)
    
    start_time = time.time()
    
    for epoch in range(config.n_epochs):
        epoch_start = time.time()
        
        # Warmup learning rate
        if epoch < config.warmup_epochs:
            warmup_factor = (epoch + 1) / config.warmup_epochs
            for pg in optimizer.param_groups:
                pg['lr'] = config.learning_rate * warmup_factor
        
        # Train
        train_metrics = train_epoch(
            model, train_loader, optimizer, loss_fn, 
            physics, device, normalizer
        )
        
        # Validate
        val_metrics = validate(
            model, val_loader, loss_fn, 
            physics, device, normalizer
        )
        
        # Update scheduler (after warmup)
        current_lr = optimizer.param_groups[0]['lr']
        if scheduler and epoch >= config.warmup_epochs:
            scheduler.step()
        
        # Record history
        history['train'].append(train_metrics)
        history['val'].append(val_metrics)
        history['lr'].append(current_lr)
        
        # Save best model
        if val_metrics['total_loss'] < best_val_loss:
            best_val_loss = val_metrics['total_loss']
            best_epoch = epoch
            
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': best_val_loss,
                'config': config.__dict__,
                'normalizer': {
                    'mean': normalizer.mean.tolist(),
                    'std': normalizer.std.tolist(),
                }
            }
            torch.save(checkpoint, config.save_path)
        
        # Periodic checkpoint
        if (epoch + 1) % config.save_every == 0:
            checkpoint_path = save_dir / f"checkpoint_epoch{epoch+1}.pt"
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'val_loss': val_metrics['total_loss'],
            }, checkpoint_path)
        
        # Logging
        epoch_time = time.time() - epoch_start
        
        if config.verbose and (epoch + 1) % config.log_every == 0:
            print(
                f"Epoch {epoch+1:3d}/{config.n_epochs} │ "
                f"Train: {train_metrics['total_loss']:.4f} │ "
                f"Val: {val_metrics['total_loss']:.4f} │ "
                f"T_MAE: {val_metrics['temp_mae']:.2f}°C │ "
                f"I_MAE: {val_metrics['rating_mae']:.0f}A │ "
                f"φ: {val_metrics['physics_residual_mean']:.1f} │ "
                f"LR: {current_lr:.1e} │ "
                f"{epoch_time:.1f}s"
            )
        
        # Callback
        if callback:
            callback(epoch, train_metrics, val_metrics)
    
    # Training complete
    total_time = time.time() - start_time
    print("-" * 70)
    print(f"\n✅ Training complete in {total_time/60:.1f} minutes")
    print(f"   Best validation loss: {best_val_loss:.4f} (epoch {best_epoch + 1})")
    print(f"   Model saved to: {config.save_path}")
    
    # Final validation metrics
    print(f"\n📈 Final Validation Metrics:")
    print(f"   Temperature MAE: {val_metrics['temp_mae']:.2f}°C")
    print(f"   Temperature RMSE: {val_metrics['temp_rmse_overall']:.2f}°C")
    print(f"   Ampacity MAE: {val_metrics['rating_mae']:.0f} A")
    print(f"   Ampacity RMSE: {val_metrics['rating_rmse_overall']:.0f} A")
    print(f"   Physics Residual: {val_metrics['physics_residual_mean']:.2f} W/m")
    
    # Load best model
    checkpoint = torch.load(config.save_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    return model, history


def quick_train(
    n_epochs: int = 50,
    batch_size: int = 256,
    save_path: str = "models/quick_model.pt",
) -> PhysicsDLR:
    """
    Quick training with default settings
    
    Good for testing and development.
    """
    config = TrainConfig(
        n_epochs=n_epochs,
        batch_size=batch_size,
        save_path=save_path,
        n_train=10000,
        n_val=2000,
        log_every=5,
    )
    
    model, _ = train(config)
    return model


# Main entry point
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Train Invariant DLR Model")
    parser.add_argument("--epochs", type=int, default=100, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=256, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--physics-weight", type=float, default=0.3, help="Physics loss weight")
    parser.add_argument("--save-path", type=str, default="models/best_model.pt", help="Save path")
    parser.add_argument("--quick", action="store_true", help="Quick training (fewer samples)")
    
    args = parser.parse_args()
    
    if args.quick:
        model = quick_train(n_epochs=30, save_path=args.save_path)
    else:
        config = TrainConfig(
            n_epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.lr,
            physics_weight=args.physics_weight,
            save_path=args.save_path,
        )
        model, history = train(config)
