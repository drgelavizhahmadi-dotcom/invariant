"""
Training Script for Invariant PINN

Optimized for:
- Apple Silicon (M1/M2) with MPS backend
- CPU fallback for Linux/Windows
- CUDA for cloud GPUs

Author: Dr. Gelavizh Ahmadi
Copyright (c) 2026 Invariant Energy GmbH
"""

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader
except Exception as e:
    raise ImportError(
        "PyTorch is required to run training.\n"
        "Install with: `pip install -r requirements.txt` or follow platform-specific instructions:\n"
        "  - macOS (MPS): `pip install torch --index-url https://download.pytorch.org/whl/nightly/cpu`\n"
        "  - CPU (Linux/macOS): `pip install torch --index-url https://download.pytorch.org/whl/cpu`\n"
        "  - CUDA: see https://pytorch.org for the correct wheel for your CUDA version"
    ) from e
from typing import Optional, Tuple, Dict, Callable
import time
from pathlib import Path
from dataclasses import dataclass
import json
import argparse

from .model import PhysicsDLR, PhysicsInformedLoss
from .physics import IEEE738HeatBalance, physics_loss_fn
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
    real_data_path: Optional[str] = None
    synthetic_ratio: float = 0.7
    
    # Model
    hidden_dims: list = None  # Will default to [128, 128, 64]
    dropout: float = 0.1
    
    # Training
    n_epochs: int = 100
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    
    # Loss weights
    physics_weight: float = 0.05  # Lowered for fine-tuning with real data
    temp_weight: float = 1.0
    rating_weight: float = 0.5
    consistency_weight: float = 0.1
    
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
    log_var_data: Optional[torch.Tensor] = None,
    log_var_phys: Optional[torch.Tensor] = None,
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
        'consistency_loss': 0,
        'physics_residual_mean': 0,
    }
    n_batches = 0
    
    for x, y in train_loader:
        x, y = x.to(device), y.to(device)
        
        # Forward pass with physics consistency
        pred_temp, pred_rating, consistency_loss = model.forward_with_physics_consistency(
            x, physics, consistency_weight=0.1
        )
        
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
        

        # Compute data loss (existing)
        data_loss, metrics = loss_fn(
            pred_temp=pred_temp,
            pred_rating=pred_rating,
            true_temp=y[:, 0:1],
            true_rating=y[:, 1:2],
            physics_residual=physics_residual,
            consistency_loss=consistency_loss,
        )

        # Compute new physics loss using physics_loss_fn and physics
        residual = physics.heat_balance_residual(
            current=raw_inputs['current'],
            T_conductor=pred_temp.squeeze(),
            T_ambient=raw_inputs['T_ambient'],
            wind_speed=raw_inputs['wind_speed'],
            solar_irradiance=raw_inputs['solar_irradiance'],
        )
        # --- Physics loss (per-sample L2 + small L1), weighted in low-wind bins ---
        # Per-user request: increase physics penalty when wind_speed < 5 m/s
        sample_phys_l2 = residual ** 2
        sample_phys_l1 = torch.abs(residual)
        sample_phys = sample_phys_l2 + 0.05 * sample_phys_l1

        # Low-wind mask (denormalized wind speed)
        low_wind_mask = (raw_inputs['wind_speed'] < 5.0).float()

        # Increase physics weight by 1.5x for low-wind samples (conservative)
        phys_sample_weight = torch.where(low_wind_mask.bool(), 1.5, 1.0)

        # Batch-level physics loss (weighted mean)
        phys_loss = torch.mean(sample_phys * phys_sample_weight)

        # --- Governed / self-adaptive weighting (trainable uncertainty) ---
        # Use the tensors passed from `train()` (ensures optimizer updates them)
        if log_var_data is None or log_var_phys is None:
            # fallback to fixed-weight behavior
            mse_temp = torch.mean((pred_temp - y[:, 0:1]) ** 2)
            mse_rating = torch.mean((pred_rating - y[:, 1:2]) ** 2)
            data_loss_simple = mse_temp + 0.5 * mse_rating
            weighted_loss = data_loss_simple + 0.1 * phys_loss
        else:
            mse_temp = torch.mean((pred_temp - y[:, 0:1]) ** 2)
            mse_rating = torch.mean((pred_rating - y[:, 1:2]) ** 2)
            data_loss_simple = mse_temp + 0.5 * mse_rating

            weighted_loss = (
                data_loss_simple * torch.exp(-log_var_data) + log_var_data +
                10 * phys_loss * torch.exp(-log_var_phys) + log_var_phys
            )

        # --- Asymmetric safety barrier (soft constraint) ---
        # percentage error (pred / true - 1). penalize over-prediction exponentially
        amp_error_pct = (pred_rating.squeeze() / (y[:, 1] + 1e-9)) - 1.0
        safety_penalty = torch.mean(torch.exp(10.0 * amp_error_pct))

        # Final loss: governed weighted loss + small safety penalty
        loss = weighted_loss + 0.1 * safety_penalty

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


def calibrated_ampacity(pred_temp: torch.Tensor, ambient_temp: torch.Tensor, T_max: float = 100.0, base_ampacity: torch.Tensor = None):
    """
    Post-processing calibration for ampacity prediction
    
    Uses power-law correction based on temperature difference
    """
    # Design temperature difference (T_max - typical ambient)
    delta_T_design = T_max - 50.0  # Assume 50°C typical ambient
    
    # Predicted temperature difference
    delta_T_pred = T_max - pred_temp.squeeze()
    
    # Scale factor based on ratio of temperature differences
    scale_factor = torch.clamp(delta_T_pred / delta_T_design, min=0.1, max=3.0)
    
    # If base_ampacity provided, scale it; otherwise return scale_factor
    if base_ampacity is not None:
        return base_ampacity * scale_factor
    else:
        return scale_factor


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
        'consistency_loss': 0,
        'physics_residual_mean': 0,
        'temp_mae': 0,
        'rating_mae': 0,
        'temp_rmse': 0,
        'rating_rmse': 0,
    }
    n_batches = 0
    
    total_temp_abs = 0.0
    total_temp_sq = 0.0
    total_rating_abs = 0.0
    total_rating_sq = 0.0
    total_count = 0
    
    for x, y in val_loader:
        x, y = x.to(device), y.to(device)
        
        pred_temp, pred_rating, consistency_loss = model.forward_with_physics_consistency(
            x, physics, consistency_weight=0.1
        )
        
        raw_inputs = denormalize_batch(x, normalizer)
        
        # Apply ampacity calibration
        calibrated_rating = calibrated_ampacity(pred_temp, raw_inputs['T_ambient'], base_ampacity=pred_rating)
        
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
            consistency_loss=consistency_loss,
        )
        
        # Calculate additional metrics
        temp_error = pred_temp.squeeze() - y[:, 0]
        rating_error = calibrated_rating.squeeze() - y[:, 1]
        
        # Accumulate for overall metrics
        batch_count = temp_error.numel()
        total_temp_abs += temp_error.abs().sum()
        total_temp_sq += (temp_error ** 2).sum()
        total_rating_abs += rating_error.abs().sum()
        total_rating_sq += (rating_error ** 2).sum()
        total_count += batch_count
        
        for k in total_metrics:
            if k in metrics:
                total_metrics[k] += metrics[k]
        n_batches += 1
    
    # Average metrics
    for k in total_metrics:
        total_metrics[k] /= n_batches
    
    # Overall MAE/RMSE
    total_metrics['temp_mae_overall'] = (total_temp_abs / total_count).item()
    total_metrics['temp_rmse_overall'] = torch.sqrt(total_temp_sq / total_count).item()
    total_metrics['rating_mae_overall'] = (total_rating_abs / total_count).item()
    total_metrics['rating_rmse_overall'] = torch.sqrt(total_rating_sq / total_count).item()
    
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
        real_data_path=config.real_data_path,
        synthetic_ratio=config.synthetic_ratio,
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

        # NOTE: adaptive log-variance tensors are created in `train()` and
        # added to the optimizer param groups so they are updated during training.
    else:
        model = model.to(device)
    
    print(f"   Parameters: {model.count_parameters():,}")
    print(f"   Hidden dims: {config.hidden_dims}")

    # Instantiate physics engine (Drake ACSR typical values)
    physics_engine = IEEE738HeatBalance(
        conductor_diameter=0.02814,
        conductor_emissivity=0.8,
        conductor_absorptivity=0.8,
        resistance_per_meter_25C=7.283e-5,
        temp_coeff_resistance=0.00403,
        max_conductor_temp=100.0
    ).to(device)
    
    # Optimizer and loss
    # Initialize adaptive weights (trainable tensors) and include them in optimizer
    log_var_data = torch.zeros(1, requires_grad=True, device=device)
    log_var_phys = torch.tensor(-0.5, requires_grad=True, device=device)  # Start with higher physics weight

    optimizer = optim.AdamW(
        [
            {'params': model.parameters()},
            {'params': [log_var_data, log_var_phys], 'lr': config.learning_rate}
        ],
        lr=config.learning_rate,
        weight_decay=config.weight_decay
    )
    
    # Use ReduceLROnPlateau for learning rate scheduling
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.3, patience=20, min_lr=3e-4
    )
    
    loss_fn = PhysicsInformedLoss(
        physics_weight=config.physics_weight,
        temp_weight=config.temp_weight,
        rating_weight=config.rating_weight,
        consistency_weight=config.consistency_weight,
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
            physics, device, normalizer, log_var_data=log_var_data, log_var_phys=log_var_phys
        )
        
        # Validate
        val_metrics = validate(
            model, val_loader, loss_fn, 
            physics, device, normalizer
        )
        
        # Update scheduler (after warmup)
        current_lr = optimizer.param_groups[0]['lr']
        if scheduler and epoch >= config.warmup_epochs:
            scheduler.step(val_metrics['total_loss'])
        
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
            lv_data = log_var_data.item() if 'log_var_data' in locals() else float('nan')
            lv_phys = log_var_phys.item() if 'log_var_phys' in locals() else float('nan')
            print(
                f"Epoch {epoch+1:3d}/{config.n_epochs} │ "
                f"Train: {train_metrics['total_loss']:.4f} │ "
                f"Val: {val_metrics['total_loss']:.4f} │ "
                f"T_MAE: {val_metrics['temp_mae_overall']:.2f}°C │ "
                f"I_MAE: {val_metrics['rating_mae_overall']:.0f}A │ "
                f"φ: {val_metrics['physics_residual_mean']:.1f} │ "
                f"log_var_data: {lv_data:.3f} │ log_var_phys: {lv_phys:.3f} │ "
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
    print(f"   Temperature MAE: {val_metrics['temp_mae_overall']:.2f}°C")
    print(f"   Temperature RMSE: {val_metrics['temp_rmse_overall']:.2f}°C")
    print(f"   Ampacity MAE: {val_metrics['rating_mae_overall']:.0f} A")
    print(f"   Ampacity RMSE: {val_metrics['rating_rmse_overall']:.0f} A")
    print(f"   Physics Residual: {val_metrics['physics_residual_mean']:.2f} W/m")
    
    # Load best model
    checkpoint = torch.load(config.save_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    return model, history


def quick_train(
    n_epochs: int = 200,
    batch_size: int = 256,
    save_path: str = "models/fine_tuned_model.pt",
) -> PhysicsDLR:
    """
    Quick training with real data fine-tuning
    
    Uses 70% synthetic + 30% real data, lower physics weight.
    """
    config = TrainConfig(
        n_epochs=n_epochs,
        batch_size=batch_size,
        save_path=save_path,
        physics_weight=0.05,  # Lower for fine-tuning
        real_data_path="data/real_dlr_data_berlin_2023_2025.csv",
        synthetic_ratio=0.7,
    )
    # Call main training loop
    return train(config=config)


def main():
    """Main entry point for training script"""
    parser = argparse.ArgumentParser(description="Train Invariant Physics-Informed AI Model")
    parser.add_argument("--quick", action="store_true", help="Quick training with default settings")
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=256, help="Batch size")
    parser.add_argument("--save-path", type=str, default="models/best_model.pt", help="Path to save trained model")
    parser.add_argument("--n-train", type=int, default=15000, help="Number of training samples")
    parser.add_argument("--n-val", type=int, default=3000, help="Number of validation samples")
    
    args = parser.parse_args()
    
    if args.quick:
        print("Starting quick training...")
        quick_train(
            n_epochs=args.epochs,
            batch_size=args.batch_size,
            save_path=args.save_path
        )
    else:
        print(f"Starting full training for {args.epochs} epochs...")
        config = TrainConfig(
            n_epochs=args.epochs,
            batch_size=args.batch_size,
            save_path=args.save_path,
            n_train=args.n_train,
            n_val=args.n_val
        )
        train(config=config)
    
    print("Training completed!")


if __name__ == "__main__":
    main()
