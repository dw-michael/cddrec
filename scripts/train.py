"""Training script for CDDRec"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import argparse
import json

from cddrec.models import CDDRec
from cddrec.data import SeqRecDataset, create_dataloader
from cddrec.training import train, load_checkpoint
from cddrec.utils import set_seed, count_parameters


def parse_args():
    """Parse command-line arguments"""
    parser = argparse.ArgumentParser(description="Train CDDRec model")

    # Data arguments
    parser.add_argument("--data_path", type=str, required=True,
                        help="Path to processed data file (.json)")
    parser.add_argument("--num_items", type=int, required=True,
                        help="Total number of items")

    # Model arguments
    parser.add_argument("--embedding_dim", type=int, default=128,
                        help="Embedding dimension")
    parser.add_argument("--encoder_layers", type=int, default=2,
                        help="Number of encoder layers")
    parser.add_argument("--decoder_layers", type=int, default=2,
                        help="Number of decoder layers")
    parser.add_argument("--num_heads", type=int, default=2,
                        help="Number of attention heads")
    parser.add_argument("--dropout", type=float, default=0.2,
                        help="Dropout probability")
    parser.add_argument("--max_seq_len", type=int, default=20,
                        help="Maximum sequence length")

    # Diffusion arguments
    parser.add_argument("--diffusion_steps", type=int, default=30,
                        help="Number of diffusion steps")
    parser.add_argument("--noise_schedule", type=str, default="linear",
                        choices=["linear", "cosine"],
                        help="Noise schedule type")
    parser.add_argument("--max_beta", type=float, default=0.1,
                        help="Maximum beta for linear schedule")

    # Training arguments
    parser.add_argument("--batch_size", type=int, default=128,
                        help="Batch size")
    parser.add_argument("--lr", type=float, default=0.001,
                        help="Learning rate")
    parser.add_argument("--num_epochs", type=int, default=100,
                        help="Number of epochs")
    parser.add_argument("--early_stopping", type=int, default=None,
                        help="Early stopping patience (default: None/disabled for sanity checks)")

    # Loss arguments
    parser.add_argument("--lambda_contrast", type=float, default=0.1,
                        help="Weight for contrastive losses")
    parser.add_argument("--temperature", type=float, default=0.1,
                        help="Temperature for contrastive losses")
    parser.add_argument("--margin", type=float, default=1.0,
                        help="Margin for cross-divergence loss")

    # Augmentation arguments
    parser.add_argument("--augmentation", type=str, default="random",
                        choices=["mask", "shuffle", "crop", "random"],
                        help="Augmentation type")
    parser.add_argument("--augmentation_ratio", type=float, default=0.2,
                        help="Augmentation intensity")

    # Other arguments
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    parser.add_argument("--device", type=str, default="cuda",
                        choices=["cuda", "cpu"],
                        help="Device to train on")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints",
                        help="Directory to save checkpoints")
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to checkpoint to resume from")

    return parser.parse_args()


def load_data(data_path: str):
    """Load processed data from JSON file"""
    with open(data_path, "r") as f:
        data = json.load(f)

    train_sequences = data["train"]["sequences"]
    train_targets = data["train"]["targets"]

    val_sequences = data["val"]["sequences"]
    val_targets = data["val"]["targets"]

    return (train_sequences, train_targets), (val_sequences, val_targets)


def main():
    args = parse_args()

    # Set random seed
    set_seed(args.seed)

    # Set device
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load data
    print("Loading data...")
    (train_sequences, train_targets), (val_sequences, val_targets) = load_data(args.data_path)

    print(f"Train samples: {len(train_sequences)}")
    print(f"Val samples: {len(val_sequences)}")

    # Create datasets
    train_dataset = SeqRecDataset(
        sequences=train_sequences,
        targets=train_targets,
        num_items=args.num_items,
        max_seq_len=args.max_seq_len,
    )

    val_dataset = SeqRecDataset(
        sequences=val_sequences,
        targets=val_targets,
        num_items=args.num_items,
        max_seq_len=args.max_seq_len,
    )

    # Create dataloaders
    train_loader = create_dataloader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
    )

    val_loader = create_dataloader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
    )

    # Create model
    print("\nInitializing model...")
    model = CDDRec(
        num_items=args.num_items,
        embedding_dim=args.embedding_dim,
        encoder_layers=args.encoder_layers,
        decoder_layers=args.decoder_layers,
        num_heads=args.num_heads,
        dropout=args.dropout,
        max_seq_len=args.max_seq_len,
        num_diffusion_steps=args.diffusion_steps,
        noise_schedule=args.noise_schedule,
        max_beta=args.max_beta,
    ).to(device)

    print(f"Model parameters: {count_parameters(model):,}")

    # Create optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # Resume from checkpoint if specified
    start_epoch = 0
    if args.resume:
        print(f"\nResuming from checkpoint: {args.resume}")
        start_epoch, _ = load_checkpoint(model, args.resume, optimizer, device)

    # Train model
    print("\nStarting training...")
    history = train(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        device=device,
        num_epochs=args.num_epochs,
        early_stopping_patience=args.early_stopping,
        checkpoint_dir=args.checkpoint_dir,
        lambda_contrast=args.lambda_contrast,
        temperature=args.temperature,
        margin=args.margin,
        augmentation_type=args.augmentation,
        augmentation_ratio=args.augmentation_ratio,
    )

    # Save training history
    history_path = Path(args.checkpoint_dir) / "training_history.json"
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)

    print(f"\nTraining history saved to {history_path}")


if __name__ == "__main__":
    main()
