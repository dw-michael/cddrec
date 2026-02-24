"""Loss functions for CDDRec training"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from cddrec import models


def cross_divergence_loss(
    x_0_pred: torch.Tensor,
    x_0_target: torch.Tensor,
    x_neg: torch.Tensor,
    mask: torch.Tensor | None = None,
    margin: float = 1.0,
) -> torch.Tensor:
    """
    Cross-divergence loss: ensures predicted embeddings are close to targets
    and far from negative samples.

    From paper Equation 12-13:
        L_cd^t = (1/N) ∑_n [log σ(-D_KL[q(x_t^n|x_0^n) || p_θ(x̂_t^n|es,t)])
                          + log(1 - σ(-D_KL[q(x_t'^n|x_0'^n) || p_θ(x̂_t^n|es,t)]))]

    Simplified using L2 distance as proxy for KL divergence:
        D_KL ∝ ||x_t - x̂_t||^2 ∝ -x_t^T x̂_t

    Implemented as ranking loss (hinge loss):
        L_cd = max(0, ||x̂_0 - x_0||^2 - ||x̂_0 - x_neg||^2 + margin)

    This prevents representation collapse when learning with randomly initialized
    item embeddings by ensuring predictions are closer to positive targets than
    negative samples.

    Args:
        x_0_pred: (batch_size, seq_len, embedding_dim) predicted target embeddings
        x_0_target: (batch_size, seq_len, embedding_dim) true target embeddings
        x_neg: (batch_size, seq_len, embedding_dim) negative sample embeddings
        mask: (batch_size, seq_len) boolean mask for valid positions
        margin: Margin for ranking loss

    Returns:
        Scalar loss value
    """
    # Distance to positive target: (batch_size, seq_len)
    pos_dist = torch.sum((x_0_pred - x_0_target) ** 2, dim=-1)

    # Distance to negative sample: (batch_size, seq_len)
    neg_dist = torch.sum((x_0_pred - x_neg) ** 2, dim=-1)

    # Ranking loss: want pos_dist < neg_dist
    loss = F.relu(pos_dist - neg_dist + margin)

    # Apply mask if provided
    if mask is not None:
        loss = loss * mask.float()
        return loss.sum() / mask.sum().clamp(min=1)
    else:
        return loss.mean()


def in_view_contrastive_loss(
    x_0_pred: torch.Tensor,
    x_t: torch.Tensor,
    mask: torch.Tensor | None = None,
    temperature: float = 0.1,
) -> torch.Tensor:
    """
    In-view contrastive loss: aligns predicted embeddings with corrupted embeddings
    at the same diffusion step.

    From paper Equation 14:
        L_in^t = -(1/N) ∑_{i=1}^N log [exp(x̂_i^T x_i / τ) /
                                        (∑_j exp(x̂_i^T x_j / τ) + ∑_{j≠i} exp(x̂_i^T x̂_j / τ))]

    Where:
        x̂_i: predicted clean embedding for sample i (x_0_pred)
        x_i: noised embedding at step t for sample i (x_t)
        τ: temperature parameter

    Uses InfoNCE loss treating other examples in batch as negatives. Ensures
    the model predicts consistent representations given noise interpolation, i.e.,
    the predicted clean x̂_0 should be similar to the actual noised x_t.

    Args:
        x_0_pred: (batch_size, seq_len, embedding_dim) predicted target embeddings
        x_t: (batch_size, seq_len, embedding_dim) noised target embeddings
        mask: (batch_size, seq_len) boolean mask for valid positions
        temperature: Temperature parameter for softmax

    Returns:
        Scalar loss value
    """
    batch_size, seq_len, embedding_dim = x_0_pred.shape

    # Flatten batch and sequence dimensions
    x_0_pred_flat = x_0_pred.view(batch_size * seq_len, embedding_dim)
    x_t_flat = x_t.view(batch_size * seq_len, embedding_dim)

    # Normalize embeddings
    x_0_pred_flat = F.normalize(x_0_pred_flat, dim=1)
    x_t_flat = F.normalize(x_t_flat, dim=1)

    # Compute similarity matrix: (batch_size * seq_len, batch_size * seq_len)
    logits = torch.matmul(x_0_pred_flat, x_t_flat.T) / temperature

    # Labels: diagonal elements are positive pairs
    labels = torch.arange(batch_size * seq_len, device=x_0_pred.device)

    # Cross-entropy loss with reduction='none' to apply masking
    loss = F.cross_entropy(logits, labels, reduction='none')

    # Apply mask if provided
    if mask is not None:
        mask_flat = mask.reshape(-1).float()
        loss = loss * mask_flat
        return loss.sum() / mask_flat.sum().clamp(min=1)
    else:
        return loss.mean()


def cross_view_contrastive_loss(
    x_0_pred: torch.Tensor,
    x_0_pred_aug: torch.Tensor,
    mask: torch.Tensor | None = None,
    temperature: float = 0.1,
) -> torch.Tensor:
    """
    Cross-view contrastive loss: ensures robustness to data augmentation.

    From paper Equation 15:
        L_cross^t = -(1/N) ∑_{i=1}^N log [exp(x̂_i^T x̃_i / τ) /
                                           (∑_j exp(x̂_i^T x̃_j / τ) + ∑_{j≠i} exp(x̂_i^T x̂_j / τ))]

    Where:
        x̂_i: predicted embedding from original sequence
        x̃_i: predicted embedding from augmented sequence (random crop/shuffle/mask)
        τ: temperature parameter

    Aligns predictions from original and augmented sequences, promoting robustness
    to input perturbations. The augmentation simulates noise in user behavior
    (e.g., missing interactions, reordering).

    Args:
        x_0_pred: (batch_size, seq_len, embedding_dim) predictions from original sequence
        x_0_pred_aug: (batch_size, seq_len, embedding_dim) predictions from augmented sequence
        mask: (batch_size, seq_len) boolean mask for valid positions
        temperature: Temperature parameter

    Returns:
        Scalar loss value
    """
    batch_size, seq_len, embedding_dim = x_0_pred.shape

    # Flatten batch and sequence dimensions
    x_0_pred_flat = x_0_pred.view(batch_size * seq_len, embedding_dim)
    x_0_pred_aug_flat = x_0_pred_aug.view(batch_size * seq_len, embedding_dim)

    # Normalize embeddings
    x_0_pred_flat = F.normalize(x_0_pred_flat, dim=1)
    x_0_pred_aug_flat = F.normalize(x_0_pred_aug_flat, dim=1)

    # Compute similarity matrix
    logits = torch.matmul(x_0_pred_flat, x_0_pred_aug_flat.T) / temperature

    # Labels: diagonal elements are positive pairs
    labels = torch.arange(batch_size * seq_len, device=x_0_pred.device)

    # Cross-entropy loss with reduction='none' to apply masking
    loss = F.cross_entropy(logits, labels, reduction='none')

    # Apply mask if provided
    if mask is not None:
        mask_flat = mask.reshape(-1).float()
        loss = loss * mask_flat
        return loss.sum() / mask_flat.sum().clamp(min=1)
    else:
        return loss.mean()


def compute_total_loss(
    model: models.CDDRec,
    sequence: torch.Tensor,
    sequence_aug: torch.Tensor,
    negatives: torch.Tensor,
    padding_mask: torch.Tensor | None = None,
    lambda_contrast: float = 0.1,
    temperature: float = 0.1,
    margin: float = 1.0,
) -> tuple[torch.Tensor, dict]:
    """
    Compute total CDDRec loss over all diffusion timesteps.

    From paper Equation 16:
        L_re = ∑_{t=0}^{T-1} (1/(t+1)) * [L_cd^t + λ(L_in^t + L_cross^t)]

    Unlike traditional DDPM which samples random timesteps, CDDRec explicitly
    calculates loss for every diffusion step. The 1/(t+1) rescaling prevents
    over-focusing on non-informative noise at higher steps.

    Args:
        model: CDDRec model
        sequence: (batch_size, seq_len) full sequence
        sequence_aug: (batch_size, seq_len) augmented sequence
        negatives: (batch_size, seq_len) negative items for each position
        padding_mask: (batch_size, seq_len) padding mask
        lambda_contrast: Weight for contrastive losses (λ)
        temperature: Temperature for contrastive losses (τ)
        margin: Margin for cross-divergence loss

    Returns:
        total_loss: Scalar loss
        loss_dict: Dictionary with loss components
    """
    # Precompute target and negative embeddings (shared across timesteps)
    target_seq = sequence[:, 1:]  # (batch_size, seq_len-1)
    x_0_target = model.get_target_embedding(target_seq)

    neg_seq = negatives[:, 1:]  # (batch_size, seq_len-1)
    x_neg = model.get_target_embedding(neg_seq)

    # Accumulate losses across all timesteps
    total_loss = torch.scalar_tensor(0.0, device=sequence.device)
    total_cd = torch.scalar_tensor(0.0, device=sequence.device)
    total_in = torch.scalar_tensor(0.0, device=sequence.device)
    total_cross = torch.scalar_tensor(0.0, device=sequence.device)

    # Iterate over all diffusion timesteps
    for t in range(model.num_diffusion_steps):
        # Forward pass at timestep t
        x_0_pred, x_t, input_mask, target_mask = model.forward_train(sequence, t, padding_mask)
        x_0_pred_aug, _, _, _ = model.forward_train(sequence_aug, t, padding_mask)

        # Compute losses at this timestep
        l_cd = cross_divergence_loss(x_0_pred, x_0_target, x_neg, target_mask, margin)
        l_in = in_view_contrastive_loss(x_0_pred, x_t, target_mask, temperature)
        l_cross = cross_view_contrastive_loss(x_0_pred, x_0_pred_aug, target_mask, temperature)

        # Apply step rescaling: 1/(t+1)
        rescale_factor = 1.0 / (t + 1.0)

        # Accumulate rescaled loss
        step_loss = rescale_factor * (l_cd + lambda_contrast * (l_in + l_cross))
        total_loss = total_loss + step_loss

        # Accumulate unscaled losses for logging
        total_cd = total_cd + l_cd.item()
        total_in = total_in + l_in.item()
        total_cross = total_cross + l_cross.item()

    # Loss dictionary for logging (averaged over timesteps)
    loss_dict = {
        "total_loss": total_loss.item(),
        "cross_divergence": total_cd / model.num_diffusion_steps,
        "in_view_contrastive": total_in / model.num_diffusion_steps,
        "cross_view_contrastive": total_cross / model.num_diffusion_steps,
    }

    return total_loss, loss_dict
