"""Training loop with validation and checkpointing"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import time
import json
from pathlib import Path
from tqdm.auto import tqdm

from cddrec import models
from .losses import compute_total_loss
from .utils import evaluate_metrics
from .data.augmentation import augment_sequence


def train_epoch(
    model: models.CDDRec,
    train_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    lambda_contrast: float = 0.1,
    temperature: float = 0.1,
    margin: float = 1.0,
    augmentation_type: str = "random",
    augmentation_ratio: float = 0.2,
    verbose: bool = True,
    epoch: int | None = None,
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
        margin: Margin for cross-divergence loss
        augmentation_type: Type of augmentation
        augmentation_ratio: Augmentation intensity
        verbose: Whether to show progress bar
        epoch: Current epoch number (for progress bar description)

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

        # Augment sequence
        sequence_aug = augment_sequence(
            sequence,
            augmentation_type=augmentation_type,
            augmentation_ratio=augmentation_ratio,
        )

        # Forward pass and compute loss
        loss, loss_dict = compute_total_loss(
            model=model,
            sequence=sequence,
            sequence_aug=sequence_aug,
            negatives=negatives,
            lambda_contrast=lambda_contrast,
            temperature=temperature,
            margin=margin,
        )

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Accumulate losses
        total_loss += loss_dict["total_loss"]
        total_cd_loss += loss_dict["cross_divergence"]
        total_in_loss += loss_dict["in_view_contrastive"]
        total_cross_loss += loss_dict["cross_view_contrastive"]
        num_batches += 1

        # Update progress bar with current loss
        if verbose:
            pbar.set_postfix({
                "loss": f"{loss_dict['total_loss']:.4f}",
                "cd": f"{loss_dict['cross_divergence']:.4f}",
            })

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
    verbose: bool = True,
) -> dict[str, float]:
    """
    Validate model on validation set.

    Args:
        model: CDDRec model
        val_loader: Validation data loader
        device: Device
        ks: K values for Recall@K and NDCG@K
        verbose: Whether to show progress bar

    Returns:
        Dictionary with validation metrics
    """
    model.eval()

    all_recall = {k: [] for k in ks}
    all_ndcg = {k: [] for k in ks}
    all_mrr = []

    # Create progress bar
    pbar = tqdm(val_loader, desc="Validating", disable=not verbose, leave=False)

    for batch in pbar:
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
    train_loader: DataLoader,
    val_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    num_epochs: int = 100,
    early_stopping_patience: int | None = 50,
    checkpoint_dir: str = "checkpoints",
    lambda_contrast: float = 0.1,
    temperature: float = 0.1,
    margin: float = 1.0,
    augmentation_type: str = "random",
    augmentation_ratio: float = 0.2,
    val_metric: str = "val_recall@10",
    verbose: bool = True,
) -> dict[str, list]:
    """
    Main training loop with early stopping and checkpointing.

    Args:
        model: CDDRec model
        train_loader: Training data loader
        val_loader: Validation data loader
        optimizer: Optimizer
        device: Device
        num_epochs: Maximum number of epochs
        early_stopping_patience: Patience for early stopping (None to disable)
        checkpoint_dir: Directory to save checkpoints
        lambda_contrast: Weight for contrastive losses
        temperature: Temperature
        margin: Margin
        augmentation_type: Augmentation type
        augmentation_ratio: Augmentation ratio
        val_metric: Metric to use for early stopping
        verbose: Whether to show progress bars and detailed output

    Returns:
        Dictionary with training history
    """
    # Create checkpoint directory
    checkpoint_path = Path(checkpoint_dir)
    checkpoint_path.mkdir(parents=True, exist_ok=True)

    # Training history
    history = {
        "train_loss": [],
        "val_metrics": [],
    }

    # Early stopping
    best_val_metric = 0.0
    patience_counter = 0
    best_epoch = 0

    print(f"Starting training for {num_epochs} epochs...")
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
            margin=margin,
            augmentation_type=augmentation_type,
            augmentation_ratio=augmentation_ratio,
            verbose=verbose,
            epoch=epoch + 1,
        )

        # Validate
        val_metrics = validate(model, val_loader, device, verbose=verbose)

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

            # Save best model
            checkpoint_file = checkpoint_path / "best_model.pth"
            torch.save({
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_metrics": val_metrics,
            }, checkpoint_file)

            print(f"  → Best model saved (epoch {best_epoch})")

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
    return history


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
