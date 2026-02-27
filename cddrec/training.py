"""Training loop with validation and checkpointing"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import time
import json
import random
from pathlib import Path
from datetime import datetime
from tqdm.auto import tqdm

from cddrec import models, data
from .losses import compute_total_loss, compute_single_timestep_loss
from .utils import evaluate_metrics
from .data.augmentation import augment_sequence


def train_epoch(
    model: models.CDDRec,
    train_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    lambda_contrast: float = 0.1,
    temperature: float = 0.1,
    mask_ratio: float = 0.3,
    shuffle_ratio: float = 0.6,
    crop_keep_ratio: float = 0.6,
    verbose: bool = True,
    epoch: int | None = None,
    separate_timestep_updates: bool = False,
) -> dict[str, float]:
    """
    Train for one epoch.

    Args:
        model: CDDRec model
        train_loader: Training data loader
        optimizer: Optimizer
        device: Device to train on
        lambda_contrast: Weight for contrastive losses
        temperature: Temperature for contrastive losses
        mask_ratio: Fraction of items to mask (default: 0.3, authors' gamma)
        shuffle_ratio: Fraction of items to shuffle (default: 0.6, authors' beta)
        crop_keep_ratio: Fraction of items to KEEP in crop (default: 0.6, authors' eta)
        verbose: Whether to show progress bar
        epoch: Current epoch number (for progress bar description)
        separate_timestep_updates: If True, do separate gradient updates per timestep
                                    instead of summing all timesteps into one loss.
                                    This avoids gradient conflicts between timesteps.

    Returns:
        Dictionary with average losses
    """
    model.train()

    total_loss = 0.0
    total_cd_loss = 0.0
    total_in_loss = 0.0
    total_cross_loss = 0.0
    num_batches = 0

    # Create progress bar
    pbar = tqdm(train_loader, desc=f"Epoch {epoch}" if epoch is not None else "Training",
                disable=not verbose, leave=False)

    for batch in pbar:
        # Move to device
        sequence = batch["sequence"].to(device)
        negatives = batch["negatives"].to(device)

        # Augment sequence (randomly chooses one of mask/shuffle/crop)
        sequence_aug = augment_sequence(
            sequence,
            mask_token=model.mask_idx,
            pad_token=model.padding_idx,
            mask_ratio=mask_ratio,
            shuffle_ratio=shuffle_ratio,
            crop_keep_ratio=crop_keep_ratio,
        )

        if separate_timestep_updates:
            # Alternative approach: Separate gradient step per timestep
            # This avoids mixing gradients from different timesteps in the same backward pass
            batch_total_loss = 0.0
            batch_cd_loss = 0.0
            batch_in_loss = 0.0
            batch_cross_loss = 0.0

            for t in range(model.num_diffusion_steps):
                # Compute loss for this timestep only
                loss_t, loss_dict_t = compute_single_timestep_loss(
                    model=model,
                    sequence=sequence,
                    sequence_aug=sequence_aug,
                    negatives=negatives,
                    timestep=t,
                    lambda_contrast=lambda_contrast,
                    temperature=temperature,
                )

                # Apply rescaling factor
                rescale_factor = 1.0 / (t + 1.0)
                scaled_loss = rescale_factor * loss_t

                # Backward pass for this timestep
                optimizer.zero_grad()
                scaled_loss.backward()

                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

                optimizer.step()

                # Accumulate unscaled losses for logging
                batch_total_loss += scaled_loss.item()
                batch_cd_loss += loss_dict_t["cross_divergence"]
                batch_in_loss += loss_dict_t["in_view_contrastive"]
                batch_cross_loss += loss_dict_t["cross_view_contrastive"]

            # Average for display
            total_loss += batch_total_loss
            total_cd_loss += batch_cd_loss / model.num_diffusion_steps
            total_in_loss += batch_in_loss / model.num_diffusion_steps
            total_cross_loss += batch_cross_loss / model.num_diffusion_steps

            # Update progress bar
            if verbose:
                pbar.set_postfix({
                    "loss": f"{batch_total_loss:.4f}",
                    "cd": f"{batch_cd_loss / model.num_diffusion_steps:.4f}",
                })

        else:
            # Original approach: Sum all timesteps, single gradient step
            loss, loss_dict = compute_total_loss(
                model=model,
                sequence=sequence,
                sequence_aug=sequence_aug,
                negatives=negatives,
                lambda_contrast=lambda_contrast,
                temperature=temperature,
            )

            # Backward pass
            optimizer.zero_grad()
            loss.backward()

            # Gradient clipping for stability (all timesteps contribute gradients)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()

            # Accumulate losses
            total_loss += loss_dict["total_loss"]
            total_cd_loss += loss_dict["cross_divergence"]
            total_in_loss += loss_dict["in_view_contrastive"]
            total_cross_loss += loss_dict["cross_view_contrastive"]

            # Update progress bar with current loss
            if verbose:
                pbar.set_postfix({
                    "loss": f"{loss_dict['total_loss']:.4f}",
                    "cd": f"{loss_dict['cross_divergence']:.4f}",
                })

        num_batches += 1

    # Average losses
    avg_losses = {
        "train_loss": total_loss / num_batches,
        "train_cd_loss": total_cd_loss / num_batches,
        "train_in_loss": total_in_loss / num_batches,
        "train_cross_loss": total_cross_loss / num_batches,
    }

    return avg_losses


@torch.no_grad()
def validate(
    model: models.CDDRec,
    val_loader: DataLoader,
    device: torch.device,
    ks: list = [1, 5, 10],
    sample_ratio: float = 1.0,
    sample_seed: int = 42,
    verbose: bool = True,
) -> dict[str, float]:
    """
    Validate model on validation set.

    Args:
        model: CDDRec model
        val_loader: Validation data loader
        device: Device
        ks: K values for Recall@K and NDCG@K
        sample_ratio: Fraction of validation data to use (default: 1.0 for full validation).
                     E.g., 0.2 = validate on 20% of data. Sampling is deterministic for
                     reproducibility (same batches every time with same seed).
        sample_seed: Random seed for deterministic batch sampling (default: 42)
        verbose: Whether to show progress bar

    Returns:
        Dictionary with validation metrics
    """
    model.eval()

    all_recall = {k: [] for k in ks}
    all_ndcg = {k: [] for k in ks}
    all_mrr = []

    # Deterministic sampling: select which batches to process
    sampled_indices = None
    if sample_ratio < 1.0:
        num_batches = len(val_loader)
        num_sample = max(1, int(num_batches * sample_ratio))
        rng = random.Random(sample_seed)
        sampled_indices = set(rng.sample(range(num_batches), num_sample))

    # Create progress bar
    desc = f"Validating ({sample_ratio*100:.0f}%)" if sample_ratio < 1.0 else "Validating"
    pbar = tqdm(val_loader, desc=desc, disable=not verbose, leave=False)

    for batch_idx, batch in enumerate(pbar):
        # Skip batch if not in sampled set
        if sampled_indices is not None and batch_idx not in sampled_indices:
            continue

        # Move to device
        sequence = batch["sequence"].to(device)
        seq_len = batch["seq_len"].to(device)

        # Extract targets using advanced indexing (last real item from each sequence)
        from cddrec.data import extract_targets
        target_item = extract_targets(sequence, seq_len)

        # Prepare input: exclude target from each sequence
        # For seq [1,2,3,4,5,0,0,0] with len=5, we want input [1,2,3,4,0,0,0,0]
        item_seq = sequence.clone()

        # Create mask: True for positions >= target_pos for each sequence
        # target_pos = seq_len - 1 (position of target item)
        max_seq_len = sequence.shape[1]
        positions = torch.arange(max_seq_len, device=device).unsqueeze(0)  # (1, max_seq_len)
        target_positions = (seq_len - 1).unsqueeze(1)  # (batch_size, 1)
        mask_out = positions >= target_positions  # (batch_size, max_seq_len)

        # Zero out target and beyond
        item_seq[mask_out] = 0

        # Generate predictions
        scores = model.forward_inference(item_seq)

        # Compute metrics
        metrics = evaluate_metrics(scores, target_item, ks=ks)

        for k in ks:
            all_recall[k].append(metrics[f"recall@{k}"])
            all_ndcg[k].append(metrics[f"ndcg@{k}"])
        all_mrr.append(metrics["mrr"])

    # Average metrics
    val_metrics = {}
    for k in ks:
        val_metrics[f"val_recall@{k}"] = sum(all_recall[k]) / len(all_recall[k])
        val_metrics[f"val_ndcg@{k}"] = sum(all_ndcg[k]) / len(all_ndcg[k])
    val_metrics["val_mrr"] = sum(all_mrr) / len(all_mrr)

    return val_metrics


def train(
    model: models.CDDRec,
    data: data.DataBundle,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    num_epochs: int = 100,
    early_stopping_patience: int | None = 50,
    checkpoint_dir: str = "checkpoints",
    lambda_contrast: float = 0.1,
    temperature: float = 0.1,
    mask_ratio: float = 0.3,
    shuffle_ratio: float = 0.6,
    crop_keep_ratio: float = 0.6,
    val_metric: str = "val_recall@10",
    val_sample_ratio: float = 1.0,
    val_sample_seed: int = 42,
    verbose: bool = True,
    separate_timestep_updates: bool = False,
) -> tuple[dict[str, list], str]:
    """
    Main training loop with early stopping and checkpointing.

    Args:
        model: CDDRec model
        data: DataBundle containing train/val loaders and metadata
        optimizer: Optimizer
        device: Device
        num_epochs: Maximum number of epochs
        early_stopping_patience: Patience for early stopping (None to disable)
        checkpoint_dir: Directory to save checkpoints
        lambda_contrast: Weight for contrastive losses
        temperature: Temperature for contrastive losses
        mask_ratio: Fraction of items to mask (default: 0.3, authors' gamma)
        shuffle_ratio: Fraction of items to shuffle (default: 0.6, authors' beta)
        crop_keep_ratio: Fraction of items to KEEP in crop (default: 0.6, authors' eta)
        val_metric: Metric to use for early stopping (e.g., 'val_recall@10')
        val_sample_ratio: Fraction of validation data to use per epoch (default: 1.0).
                         E.g., 0.2 = validate on 20% of data. Sampling is deterministic.
                         Use < 1.0 to speed up validation during training.
        val_sample_seed: Random seed for deterministic validation sampling (default: 42)
        verbose: Whether to show progress bars and detailed output
        separate_timestep_updates: If True, do separate gradient updates per timestep.
                                    This avoids mixing gradients from different timesteps.
                                    Default False (sum all timesteps, as in Equation 16).

    Returns:
        Tuple of (history, experiment_id) where:
        - history: Dictionary with training history (train_loss, val_metrics)
        - experiment_id: Timestamped experiment identifier
    """
    # Extract loaders from DataBundle
    train_loader = data.train_loader
    val_loader = data.val_loader

    # Create checkpoint directory
    checkpoint_path = Path(checkpoint_dir)
    checkpoint_path.mkdir(parents=True, exist_ok=True)

    # Generate experiment ID with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_id = f"cddrec_{timestamp}"

    # Build experiment configuration for logging
    experiment_config = {
        'experiment_id': experiment_id,
        'timestamp': datetime.now().isoformat(),
        'model': model.to_config(),
        'data': {
            'num_users': data.num_users,
            'num_items': data.num_items,
            'train_samples': len(data.train_dataset),
            'val_samples': len(data.val_dataset),
            'test_samples': len(data.test_dataset),
        },
        'training': {
            'num_epochs': num_epochs,
            'early_stopping_patience': early_stopping_patience,
            'lambda_contrast': lambda_contrast,
            'temperature': temperature,
            'mask_ratio': mask_ratio,
            'shuffle_ratio': shuffle_ratio,
            'crop_keep_ratio': crop_keep_ratio,
            'val_metric': val_metric,
            'val_sample_ratio': val_sample_ratio,
            'val_sample_seed': val_sample_seed,
            'separate_timestep_updates': separate_timestep_updates,
        },
        'optimizer': {
            'type': type(optimizer).__name__,
            'lr': optimizer.param_groups[0]['lr'],
        },
        'device': str(device),
    }

    # Save experiment config
    config_file = checkpoint_path / f"{experiment_id}_config.json"
    with open(config_file, 'w') as f:
        json.dump(experiment_config, f, indent=2)
    print(f"Experiment config saved: {config_file}")

    # Training history
    history = {
        "train_loss": [],
        "val_metrics": [],
    }

    # Early stopping
    best_val_metric = 0.0
    patience_counter = 0
    best_epoch = 0

    print(f"\nStarting training for {num_epochs} epochs...")
    print(f"Experiment ID: {experiment_id}")
    print(f"Device: {device}")

    for epoch in range(num_epochs):
        start_time = time.time()

        # Train
        train_losses = train_epoch(
            model=model,
            train_loader=train_loader,
            optimizer=optimizer,
            device=device,
            lambda_contrast=lambda_contrast,
            temperature=temperature,
            mask_ratio=mask_ratio,
            shuffle_ratio=shuffle_ratio,
            crop_keep_ratio=crop_keep_ratio,
            verbose=verbose,
            epoch=epoch + 1,
            separate_timestep_updates=separate_timestep_updates,
        )

        # Validate
        val_metrics = validate(
            model, val_loader, device,
            sample_ratio=val_sample_ratio,
            sample_seed=val_sample_seed,
            verbose=verbose
        )

        # Record history
        history["train_loss"].append(train_losses["train_loss"])
        history["val_metrics"].append(val_metrics)

        epoch_time = time.time() - start_time

        # Print progress
        print(f"Epoch {epoch+1}/{num_epochs} ({epoch_time:.2f}s)")
        print(f"  Train Loss: {train_losses['train_loss']:.4f}")
        print(f"  Val Recall@10: {val_metrics['val_recall@10']:.4f}")
        print(f"  Val NDCG@10: {val_metrics['val_ndcg@10']:.4f}")
        print(f"  Val MRR: {val_metrics['val_mrr']:.4f}")

        # Check for improvement
        current_val_metric = val_metrics[val_metric]

        if current_val_metric > best_val_metric:
            best_val_metric = current_val_metric
            patience_counter = 0
            best_epoch = epoch + 1

            # Save best model with timestamped name
            checkpoint_file = checkpoint_path / f"{experiment_id}_best.pth"
            torch.save({
                "epoch": epoch + 1,
                "experiment_id": experiment_id,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_metrics": val_metrics,
                "model_config": model.to_config(),
            }, checkpoint_file)

            print(f"  → Best model saved: {checkpoint_file.name} (epoch {best_epoch})")

        else:
            patience_counter += 1
            if early_stopping_patience is not None:
                print(f"  No improvement ({patience_counter}/{early_stopping_patience})")

                if patience_counter >= early_stopping_patience:
                    print(f"\nEarly stopping at epoch {epoch+1}")
                    print(f"Best epoch: {best_epoch} with {val_metric}={best_val_metric:.4f}")
                    break
            else:
                print(f"  No improvement (early stopping disabled)")

    print("\nTraining completed!")
    print(f"Experiment ID: {experiment_id}")
    return history, experiment_id


def load_checkpoint(
    model: models.CDDRec,
    checkpoint_path: str,
    optimizer: torch.optim.Optimizer | None = None,
    device: torch.device = torch.device("cpu"),
):
    """
    Load model from checkpoint.

    Args:
        model: CDDRec model
        checkpoint_path: Path to checkpoint file
        optimizer: Optional optimizer to load state
        device: Device to load to

    Returns:
        Loaded epoch and validation metrics
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)

    model.load_state_dict(checkpoint["model_state_dict"])

    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    epoch = checkpoint.get("epoch", 0)
    val_metrics = checkpoint.get("val_metrics", {})

    print(f"Loaded checkpoint from epoch {epoch}")
    print(f"Validation metrics: {val_metrics}")

    return epoch, val_metrics
