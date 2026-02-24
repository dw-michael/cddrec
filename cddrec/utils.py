"""Utility functions for evaluation metrics and misc helpers"""

import torch
import numpy as np


# =============================================================================
# Evaluation Metrics
# =============================================================================

def recall_at_k(predictions: torch.Tensor, targets: torch.Tensor, k: int) -> float:
    """
    Calculate Recall@K for batch of predictions.

    Args:
        predictions: (batch_size, num_items) logits or scores
        targets: (batch_size,) ground truth item indices
        k: Top-K to consider

    Returns:
        Recall@K score (fraction of correct predictions in top-K)
    """
    _, top_k_indices = torch.topk(predictions, k, dim=1)

    targets = targets.view(-1, 1).expand_as(top_k_indices)
    hits = (top_k_indices == targets).any(dim=1).float()

    return hits.mean().item()


def ndcg_at_k(predictions: torch.Tensor, targets: torch.Tensor, k: int) -> float:
    """
    Calculate NDCG@K (Normalized Discounted Cumulative Gain).

    Args:
        predictions: (batch_size, num_items) logits or scores
        targets: (batch_size,) ground truth item indices
        k: Top-K to consider

    Returns:
        NDCG@K score
    """
    _, top_k_indices = torch.topk(predictions, k, dim=1)

    # Calculate DCG
    batch_size = predictions.size(0)
    dcg = torch.zeros(batch_size, device=predictions.device)

    for i in range(batch_size):
        for j in range(k):
            if top_k_indices[i, j] == targets[i]:
                dcg[i] = 1.0 / torch.log2(torch.tensor(j + 2.0))
                break

    # IDCG is 1.0 (only one relevant item)
    idcg = 1.0

    return (dcg / idcg).mean().item()


def mrr(predictions: torch.Tensor, targets: torch.Tensor) -> float:
    """
    Calculate Mean Reciprocal Rank.

    Args:
        predictions: (batch_size, num_items) logits or scores
        targets: (batch_size,) ground truth item indices

    Returns:
        MRR score
    """
    _, sorted_indices = torch.sort(predictions, dim=1, descending=True)

    batch_size = predictions.size(0)
    reciprocal_ranks = torch.zeros(batch_size, device=predictions.device)

    for i in range(batch_size):
        rank = (sorted_indices[i] == targets[i]).nonzero(as_tuple=True)[0]
        if len(rank) > 0:
            reciprocal_ranks[i] = 1.0 / (rank[0].item() + 1)

    return reciprocal_ranks.mean().item()


def evaluate_metrics(predictions: torch.Tensor, targets: torch.Tensor,
                      ks: list[int] = [1, 5, 10]) -> dict:
    """
    Evaluate all metrics at once.

    Args:
        predictions: (batch_size, num_items) logits or scores
        targets: (batch_size,) ground truth item indices
        ks: List of K values for Recall@K and NDCG@K

    Returns:
        Dictionary with all metric values
    """
    metrics = {}

    for k in ks:
        metrics[f"recall@{k}"] = recall_at_k(predictions, targets, k)
        metrics[f"ndcg@{k}"] = ndcg_at_k(predictions, targets, k)

    metrics["mrr"] = mrr(predictions, targets)

    return metrics


# =============================================================================
# Misc Utilities
# =============================================================================

def set_seed(seed: int):
    """Set random seed for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


def count_parameters(model: torch.nn.Module) -> int:
    """Count trainable parameters in a model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
