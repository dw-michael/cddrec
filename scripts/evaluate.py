"""Evaluation script for CDDRec"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import argparse
import json

from cddrec.models import CDDRec
from cddrec.data import setup_data_from_file
from cddrec.training import load_checkpoint, validate
from cddrec.utils import set_seed


def parse_args():
    """Parse command-line arguments"""
    parser = argparse.ArgumentParser(description="Evaluate CDDRec model")

    # Data arguments
    parser.add_argument("--data_path", type=str, required=True,
                        help="Path to processed data file (.json)")
    parser.add_argument("--split", type=str, default="test",
                        choices=["val", "test"],
                        help="Dataset split to evaluate on")

    # Model arguments
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to model checkpoint")
    parser.add_argument("--embedding_dim", type=int, default=128)
    parser.add_argument("--encoder_layers", type=int, default=2)
    parser.add_argument("--decoder_layers", type=int, default=2)
    parser.add_argument("--num_heads", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--max_seq_len", type=int, default=20)
    parser.add_argument("--diffusion_steps", type=int, default=30)
    parser.add_argument("--noise_schedule", type=str, default="linear")
    parser.add_argument("--max_beta", type=float, default=0.1)

    # Evaluation arguments
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--metrics", type=str, nargs="+",
                        default=["recall@1", "recall@5", "recall@10", "ndcg@5", "ndcg@10", "mrr"],
                        help="Metrics to compute")

    # Other arguments
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda",
                        choices=["cuda", "cpu"])

    return parser.parse_args()


def main():
    args = parse_args()

    # Set random seed
    set_seed(args.seed)

    # Set device
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load data and create dataloaders
    print("Loading data...")
    data = setup_data_from_file(
        json_path=args.data_path,
        batch_size=args.batch_size,
        max_seq_len=args.max_seq_len,
    )

    # Select the appropriate split
    if args.split == "val":
        dataloader = data.val_loader
        dataset = data.val_dataset
    else:  # test
        dataloader = data.test_loader
        dataset = data.test_dataset

    print(f"Items: {data.num_items}")
    print(f"{args.split.capitalize()} samples: {len(dataset)}")

    # Create model
    print("\nInitializing model...")
    model = CDDRec(
        num_items=data.num_items,
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

    # Load checkpoint
    print(f"Loading checkpoint: {args.checkpoint}")
    load_checkpoint(model, args.checkpoint, device=device)

    # Extract K values from metric names
    ks = []
    for metric in args.metrics:
        if "@" in metric:
            k = int(metric.split("@")[1])
            if k not in ks:
                ks.append(k)

    if not ks:
        ks = [1, 5, 10]

    # Evaluate
    print(f"\nEvaluating on {args.split} set...")
    metrics = validate(model, dataloader, device, ks=ks)

    # Print results
    print("\n" + "="*50)
    print("EVALUATION RESULTS")
    print("="*50)

    for metric_name, value in sorted(metrics.items()):
        clean_name = metric_name.replace("val_", "")
        print(f"{clean_name:15s}: {value:.4f}")

    print("="*50)

    # Save results
    results_path = Path(args.checkpoint).parent / f"{args.split}_results.json"
    with open(results_path, "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"\nResults saved to {results_path}")


if __name__ == "__main__":
    main()
