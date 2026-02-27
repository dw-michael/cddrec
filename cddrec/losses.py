"""Loss functions for CDDRec training"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from cddrec import models


def cross_divergence_loss(
    x_pred_t: torch.Tensor,
    x_t: torch.Tensor,
    x_t_neg: torch.Tensor,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    Cross-divergence loss: ensures predicted embeddings are close to targets
    and far from negative samples.

    FIXED: Now uses dot product similarity (matching authors' implementation)
    instead of squared distances. The authors' implementation in their GitHub repo
    uses binary cross-entropy on dot products, not squared distances.

    Authors' implementation (models.py:262-274):
        pos_logits = torch.sum(pos * seq_emb, -1)  # Dot product
        neg_logits = torch.sum(neg * seq_emb, -1)  # Dot product
        loss = -log(sigmoid(pos_logits)) - log(1 - sigmoid(neg_logits))

    This simplifies to:
        L_cd = -(1/N) ∑_n [log(σ(dot(x̂_t, x_t))) + log(1 - σ(dot(x̂_t, x_t')))]
             = -(1/N) ∑_n [log(σ(dot(x̂_t, x_t))) + log(σ(-dot(x̂_t, x_t')))]

    Where:
        - dot(a, b) = sum(a * b) = dot product
        - σ is sigmoid function
        - Maximizes similarity (dot product) with positive targets
        - Minimizes similarity (dot product) with negative targets

    CRITICAL: Both positive and negative targets must be noised at the SAME
    timestep t. The model learns to predict noised embeddings at each diffusion
    step, not clean embeddings.

    Args:
        x_pred_t: (batch_size, seq_len, embedding_dim) sampled predicted embeddings
                  (x̂_t = μ_θ + sqrt(β_t) * ε from Equation 10)
        x_t: (batch_size, seq_len, embedding_dim) noised target embeddings at timestep t
             (from Equation 11: x_t = √(ᾱ_t) x_0 + √(1 - ᾱ_t) ε)
        x_t_neg: (batch_size, seq_len, embedding_dim) noised negative embeddings at timestep t
        mask: (batch_size, seq_len) boolean mask for valid positions

    Returns:
        Scalar loss value
    """
    # Compute dot product similarity: (batch_size, seq_len)
    # Higher dot product = more similar
    pos_logits = torch.sum(x_pred_t * x_t, dim=-1)
    neg_logits = torch.sum(x_pred_t * x_t_neg, dim=-1)

    # Binary cross-entropy on similarity scores
    # Maximize positive similarity, minimize negative similarity
    # Using log(1 - sigmoid(x)) = log(sigmoid(-x)) for numerical stability
    loss_pos = -torch.log(torch.sigmoid(pos_logits) + 1e-24)  # Want this high
    loss_neg = -torch.log(1 - torch.sigmoid(neg_logits) + 1e-24)  # Want this low

    loss = loss_pos + loss_neg

    # Apply mask if provided
    if mask is not None:
        loss = loss * mask.float()
        return loss.sum() / mask.sum().clamp(min=1)
    else:
        return loss.mean()


def in_view_contrastive_loss(
    x_pred_t: torch.Tensor,
    x_t: torch.Tensor,
    mask: torch.Tensor | None = None,
    temperature: float = 0.1,
) -> torch.Tensor:
    """
    In-view contrastive loss: aligns predicted embeddings with corrupted embeddings
    at the same diffusion step.

    UPDATED: Now uses BATCH-LEVEL contrastive (matches authors' XNetLoss):
    - Flattens to (B*S, D) treating all positions as independent samples
    - Each position contrasts against ALL other positions across the batch
    - Positive pair: x_pred[i] with x_t[i] (same position)
    - Negative samples: ALL other B*S - 1 positions

    From paper Equation 14:
        L_in^t = -(1/N) ∑_{i=1}^N log [exp(x̂_i^T x_i / τ) / ∑_j exp(x̂_i^T x_j / τ)]

    Where i, j range over ALL positions in the batch (batch_size * seq_len).

    Args:
        x_pred_t: (batch_size, seq_len, embedding_dim) sampled predicted embeddings
        x_t: (batch_size, seq_len, embedding_dim) noised target embeddings
        mask: (batch_size, seq_len) boolean mask. True = valid data, False = padding
        temperature: Temperature parameter for softmax

    Returns:
        Scalar loss value
    """
    B, S, D = x_pred_t.shape

    # Flatten to treat all positions as independent samples
    view1 = x_pred_t.view(-1, D)  # (B*S, D)
    view2 = x_t.view(-1, D)  # (B*S, D)

    # Apply mask to zero out padding (authors don't do this, but we should for correctness)
    if mask is not None:
        mask_flat = mask.view(-1, 1).float()  # (B*S, 1)
        view1 = view1 * mask_flat
        view2 = view2 * mask_flat

    # Normalize embeddings
    view1 = F.normalize(view1, p=2, dim=1)  # (B*S, D)
    view2 = F.normalize(view2, p=2, dim=1)  # (B*S, D)

    # Concatenate views: [view1; view2]
    features = torch.cat([view1, view2], dim=0)  # (2*B*S, D)

    # Compute global similarity matrix
    similarity = torch.matmul(features, features.T) / temperature  # (2*B*S, 2*B*S)

    # Extract positive pairs: view1[i] with view2[i]
    batch_size = view1.shape[0]  # B*S
    sim_ij = torch.diag(similarity, batch_size)  # view1[i] · view2[i]
    sim_ji = torch.diag(similarity, -batch_size)  # view2[i] · view1[i]
    positives = torch.cat([sim_ij, sim_ji], dim=0)  # (2*B*S,)

    # Mask out self-similarity (diagonal)
    mask_eye = (~torch.eye(batch_size * 2, batch_size * 2, dtype=bool, device=features.device)).float()

    # Compute InfoNCE loss
    nominator = torch.exp(positives)
    denominator = (mask_eye * torch.exp(similarity)).sum(dim=1)

    losses = -torch.log(nominator / (denominator + 1e-8))

    # Apply mask to losses (only compute on valid positions)
    if mask is not None:
        mask_double = torch.cat([mask_flat, mask_flat], dim=0).squeeze()  # (2*B*S,)
        losses = losses * mask_double
        return losses.sum() / (mask_double.sum() + 1e-8)

    return losses.mean()


def cross_view_contrastive_loss(
    x_pred_t: torch.Tensor,
    x_pred_t_aug: torch.Tensor,
    mask: torch.Tensor | None = None,
    temperature: float = 0.1,
) -> torch.Tensor:
    """
    Cross-view contrastive loss: ensures robustness to data augmentation.

    UPDATED: Now uses BATCH-LEVEL contrastive (matches authors' XNetLossCrossView):
    - Flattens to (B*S, D) treating all positions as independent samples
    - Each position contrasts against ALL other positions across the batch
    - Positive pair: x_pred[i] with x_pred_aug[i] (same position in both views)
    - Negative samples: ALL other B*S - 1 positions

    From paper Equation 15:
        L_cross^t = -(1/N) ∑_{i=1}^N log [exp(x̂_i^T x̃_i / τ) / ∑_j exp(x̂_i^T x̃_j / τ)]

    Where i, j range over ALL positions in the batch (batch_size * seq_len).

    Args:
        x_pred_t: (batch_size, seq_len, embedding_dim) predictions from original sequence
        x_pred_t_aug: (batch_size, seq_len, embedding_dim) predictions from augmented sequence
        mask: (batch_size, seq_len) boolean mask. True = valid data, False = padding
        temperature: Temperature parameter

    Returns:
        Scalar loss value
    """
    B, S, D = x_pred_t.shape

    # Flatten to treat all positions as independent samples
    view1 = x_pred_t.view(-1, D)  # (B*S, D)
    view2 = x_pred_t_aug.view(-1, D)  # (B*S, D)

    # Apply mask to zero out padding (authors don't do this, but we should for correctness)
    if mask is not None:
        mask_flat = mask.view(-1, 1).float()  # (B*S, 1)
        view1 = view1 * mask_flat
        view2 = view2 * mask_flat

    # Normalize embeddings
    view1 = F.normalize(view1, p=2, dim=1)  # (B*S, D)
    view2 = F.normalize(view2, p=2, dim=1)  # (B*S, D)

    # Concatenate views: [view1; view2]
    features = torch.cat([view1, view2], dim=0)  # (2*B*S, D)

    # Compute global similarity matrix
    similarity = torch.matmul(features, features.T) / temperature  # (2*B*S, 2*B*S)

    # Extract positive pairs: view1[i] with view2[i]
    batch_size = view1.shape[0]  # B*S
    sim_ij = torch.diag(similarity, batch_size)  # view1[i] · view2[i]
    sim_ji = torch.diag(similarity, -batch_size)  # view2[i] · view1[i]
    positives = torch.cat([sim_ij, sim_ji], dim=0)  # (2*B*S,)

    # Mask out self-similarity (diagonal)
    mask_eye = (~torch.eye(batch_size * 2, batch_size * 2, dtype=bool, device=features.device)).float()

    # Compute InfoNCE loss
    nominator = torch.exp(positives)
    denominator = (mask_eye * torch.exp(similarity)).sum(dim=1)

    losses = -torch.log(nominator / (denominator + 1e-8))

    # Apply mask to losses (only compute on valid positions)
    if mask is not None:
        mask_double = torch.cat([mask_flat, mask_flat], dim=0).squeeze()  # (2*B*S,)
        losses = losses * mask_double
        return losses.sum() / (mask_double.sum() + 1e-8)

    return losses.mean()


def compute_single_timestep_loss(
    model: models.CDDRec,
    sequence: torch.Tensor,
    sequence_aug: torch.Tensor,
    negatives: torch.Tensor,
    timestep: int,
    lambda_contrast: float = 0.1,
    temperature: float = 0.1,
) -> tuple[torch.Tensor, dict]:
    """
    Compute CDDRec loss for a single diffusion timestep.

    This is used when doing separate gradient updates per timestep to avoid
    gradient conflicts between different timesteps.

    Args:
        model: CDDRec model
        sequence: (batch_size, seq_len) full sequence
        sequence_aug: (batch_size, seq_len) augmented sequence
        negatives: (batch_size, seq_len) negative items for each position
        timestep: Specific diffusion timestep (0 to T-1)
        lambda_contrast: Weight for contrastive losses (λ)
        temperature: Temperature for contrastive losses (τ)

    Returns:
        loss: Scalar loss for this timestep (before rescaling)
        loss_dict: Dictionary with loss components
    """
    # Forward pass at this timestep for both original and augmented views
    x_pred_t, x_t, input_mask, target_mask = model.forward_train(sequence, timestep)
    x_pred_t_aug, _, input_mask_aug, target_mask_aug = model.forward_train(sequence_aug, timestep)

    # Get clean negative embeddings and noise them at this timestep
    neg_seq = negatives[:, 1:]  # (batch_size, seq_len-1)
    x_0_neg = model.get_target_embedding(neg_seq)

    # Convert timestep to tensor for add_noise (expects (batch_size,) tensor)
    batch_size = x_0_neg.size(0)
    device = x_0_neg.device
    timestep_tensor = torch.full((batch_size,), timestep, device=device, dtype=torch.long)
    x_t_neg = model.diffuser.add_noise(x_0_neg, timestep_tensor)

    # Compute losses at this timestep
    # Cross-divergence on BOTH original and augmented views (matches authors)
    l_cd = cross_divergence_loss(x_pred_t, x_t, x_t_neg, target_mask)
    l_cd_aug = cross_divergence_loss(x_pred_t_aug, x_t, x_t_neg, target_mask_aug)

    # Contrastive losses
    l_in = in_view_contrastive_loss(x_pred_t, x_t, target_mask, temperature)
    l_cross = cross_view_contrastive_loss(x_pred_t, x_pred_t_aug, target_mask, temperature)

    # Combine losses (without rescaling - that's done in the training loop)
    # Matches authors: loss_t + loss_t_aug + 0.3*loss_clr + 0.3*loss_crossview
    loss = (l_cd + l_cd_aug) + lambda_contrast * (l_in + l_cross)

    # Loss dictionary for logging
    loss_dict = {
        "cross_divergence": (l_cd.item() + l_cd_aug.item()) / 2,  # Average of both views
        "in_view_contrastive": l_in.item(),
        "cross_view_contrastive": l_cross.item(),
    }

    return loss, loss_dict


def compute_total_loss(
    model: models.CDDRec,
    sequence: torch.Tensor,
    sequence_aug: torch.Tensor,
    negatives: torch.Tensor,
    lambda_contrast: float = 0.1,
    temperature: float = 0.1,
) -> tuple[torch.Tensor, dict]:
    """
    Compute total CDDRec loss over all diffusion timesteps.

    From paper Equation 16:
        L_re = ∑_{t=0}^{T-1} (1/(t+1)) * [L_cd^t + λ(L_in^t + L_cross^t)]

    Unlike traditional DDPM which samples random timesteps, CDDRec explicitly
    calculates loss for every diffusion step. The 1/(t+1) rescaling prevents
    over-focusing on non-informative noise at higher steps.

    NOTE: This sums all timestep losses into a single loss for one gradient update.
    An alternative approach is to do separate gradient updates per timestep using
    compute_single_timestep_loss(). Both are valid interpretations of Equation 16,
    but have different gradient dynamics.

    Args:
        model: CDDRec model
        sequence: (batch_size, seq_len) full sequence
        sequence_aug: (batch_size, seq_len) augmented sequence
        negatives: (batch_size, seq_len) negative items for each position
        lambda_contrast: Weight for contrastive losses (λ)
        temperature: Temperature for contrastive losses (τ)

    Returns:
        total_loss: Scalar loss
        loss_dict: Dictionary with loss components
    """
    # Accumulate losses across all timesteps
    total_loss = torch.scalar_tensor(0.0, device=sequence.device)
    total_cd = 0.0
    total_in = 0.0
    total_cross = 0.0

    # Iterate over all diffusion timesteps
    for t in range(model.num_diffusion_steps):
        # Compute loss for this timestep
        # NOTE: Cannot precompute embeddings because they must be noised
        # at each specific timestep t
        loss_t, loss_dict_t = compute_single_timestep_loss(
            model=model,
            sequence=sequence,
            sequence_aug=sequence_aug,
            negatives=negatives,
            timestep=t,
            lambda_contrast=lambda_contrast,
            temperature=temperature,
        )

        # Apply step rescaling: 1/(t+1)
        rescale_factor = 1.0 / (t + 1.0)

        # Accumulate rescaled loss
        total_loss = total_loss + rescale_factor * loss_t

        # Accumulate unscaled losses for logging
        total_cd += loss_dict_t["cross_divergence"]
        total_in += loss_dict_t["in_view_contrastive"]
        total_cross += loss_dict_t["cross_view_contrastive"]

    # Loss dictionary for logging (averaged over timesteps)
    loss_dict = {
        "total_loss": total_loss.item(),
        "cross_divergence": total_cd / model.num_diffusion_steps,
        "in_view_contrastive": total_in / model.num_diffusion_steps,
        "cross_view_contrastive": total_cross / model.num_diffusion_steps,
    }

    return total_loss, loss_dict
