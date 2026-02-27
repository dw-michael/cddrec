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

    From paper Equation 13 (exact formulation):
        L_cd^t = (1/N) ∑_n [log σ(-D_KL[q(x_t^n|x_0^n) || p_θ(x̂_t^n|es,t)])
                          + log(1 - σ(-D_KL[q(x_t'^n|x_0'^n) || p_θ(x̂_t^n|es,t)]))]

    Where:
        - D_KL is KL divergence between distributions
        - For Gaussians with same variance: D_KL ∝ ||μ1 - μ2||²
        - σ is sigmoid function
        - Using identity: log(1 - σ(x)) = log(σ(-x))
        - x_t: noised target at timestep t (from Equation 11)
        - x_t': noised negative at timestep t

    This becomes:
        L_cd = -(1/N) ∑_n [log σ(-||x̂_t - x_t||²) + log σ(||x̂_t - x_t'||²)]

    The negative sign converts the paper's formulation (which maximizes)
    to a minimization objective. This smooth formulation provides better
    gradients than hard-margin hinge loss and prevents representation collapse.

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
    # Compute KL divergence using squared distance: (batch_size, seq_len)
    # D_KL ∝ ||x̂_t - x_t||² for Gaussians with same variance
    kl_pos = torch.sum((x_pred_t - x_t) ** 2, dim=-1)
    kl_neg = torch.sum((x_pred_t - x_t_neg) ** 2, dim=-1)

    # Apply paper's formulation: -[log(σ(-D_KL_pos)) + log(σ(D_KL_neg))]
    # Interpretation:
    #   - log(σ(-kl_pos)): reward small distance to positive (kl_pos → 0)
    #   - log(σ(kl_neg)): reward large distance to negative (kl_neg → ∞)
    loss_pos = F.logsigmoid(-kl_pos)  # log(sigmoid(-kl_pos))
    loss_neg = F.logsigmoid(kl_neg)    # log(sigmoid(kl_neg))

    # Negate to convert maximization to minimization
    loss = -(loss_pos + loss_neg)

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
    at the same diffusion step, within each sequence.

    From paper Equation 14:
        L_in^t = -(1/N) ∑_{i=1}^N log [exp(x̂_i^T x_i / τ) /
                                        (∑_j exp(x̂_i^T x_j / τ) + ∑_{j≠i} exp(x̂_i^T x̂_j / τ))]

    Where:
        x̂_i: sampled predicted embedding for position i (Eq. 10: x̂_t = μ_θ + sqrt(β_t) * ε)
        x_i: noised embedding at step t for position i (from diffuser via Eq. 11)
        i, j: positions within the same sequence
        τ: temperature parameter

    Key architectural note: x_pred_t and x_t are computed independently:
        - μ_θ = decoder(t, encoded_seq) - mean prediction from (timestep, history)
        - x_pred_t = diffuser.sample_prediction(μ_θ, t) - sampled via Eq. 10
        - x_t = diffuser.add_noise(target_embeddings, t) - noised target via Eq. 11

    Applies InfoNCE within each sequence separately (not across the batch).
    Position i in a sequence is contrasted against other positions j in the
    same sequence, not against positions from other sequences in the batch.

    Args:
        x_pred_t: (batch_size, seq_len, embedding_dim) sampled predicted embeddings
                  (x̂_t = μ_θ + sqrt(β_t) * ε from Equation 10)
        x_t: (batch_size, seq_len, embedding_dim) noised target embeddings
             (from Equation 11: x_t = √(ᾱ_t) x_0 + √(1 - ᾱ_t) ε)
        mask: (batch_size, seq_len) boolean mask. True = valid data, False = padding
        temperature: Temperature parameter for softmax

    Returns:
        Scalar loss value
    """
    batch_size, seq_len, embedding_dim = x_pred_t.shape

    # Apply mask before normalization to avoid issues with padding
    if mask is not None:
        mask_expanded = mask.unsqueeze(-1).float()  # (batch, seq, 1)
        x_pred_t = x_pred_t * mask_expanded
        x_t = x_t * mask_expanded

    # Normalize embeddings (eps prevents division by zero)
    x_pred_t = F.normalize(x_pred_t, dim=-1, eps=1e-8)  # (batch, seq, emb)
    x_t = F.normalize(x_t, dim=-1, eps=1e-8)  # (batch, seq, emb)

    # Compute similarity matrix within each sequence
    # For each batch b: logits[b] = x_pred_t[b] @ x_t[b].T
    logits = torch.bmm(x_pred_t, x_t.transpose(1, 2)) / temperature
    # Shape: (batch, seq, seq)

    # Labels: diagonal matching (position i should match position i)
    labels = torch.arange(seq_len, device=x_pred_t.device)
    labels = labels.unsqueeze(0).expand(batch_size, -1)  # (batch, seq)

    # Reshape for cross_entropy: (batch * seq, seq) and (batch * seq,)
    logits_flat = logits.reshape(batch_size * seq_len, seq_len)
    labels_flat = labels.reshape(-1)

    # Compute cross-entropy
    loss = F.cross_entropy(logits_flat, labels_flat, reduction='none')
    loss = loss.reshape(batch_size, seq_len)

    # Apply mask to loss: zero out padding positions
    if mask is not None:
        loss = loss * mask.float()  # Zero out padding
        return loss.sum() / mask.sum().clamp(min=1)  # Average over valid positions only
    else:
        return loss.mean()


def cross_view_contrastive_loss(
    x_pred_t: torch.Tensor,
    x_pred_t_aug: torch.Tensor,
    mask: torch.Tensor | None = None,
    temperature: float = 0.1,
) -> torch.Tensor:
    """
    Cross-view contrastive loss: ensures robustness to data augmentation.

    From paper Equation 15:
        L_cross^t = -(1/N) ∑_{i=1}^N log [exp(x̂_i^T x̃_i / τ) /
                                           (∑_j exp(x̂_i^T x̃_j / τ) + ∑_{j≠i} exp(x̂_i^T x̂_j / τ))]

    Where:
        x̂_i: sampled predicted embedding from original sequence at position i
        x̃_i: sampled predicted embedding from augmented sequence at position i
        i, j: positions within the same sequence
        τ: temperature parameter

    Applies InfoNCE within each sequence separately (not across the batch).
    Aligns predictions from original and augmented sequences, promoting robustness
    to input perturbations. The augmentation simulates noise in user behavior
    (e.g., missing interactions, reordering).

    Args:
        x_pred_t: (batch_size, seq_len, embedding_dim) sampled predictions from original sequence
                  (x̂_t = μ_θ + sqrt(β_t) * ε from Equation 10)
        x_pred_t_aug: (batch_size, seq_len, embedding_dim) sampled predictions from augmented sequence
        mask: (batch_size, seq_len) boolean mask. True = valid data, False = padding
        temperature: Temperature parameter

    Returns:
        Scalar loss value
    """
    batch_size, seq_len, embedding_dim = x_pred_t.shape

    # Apply mask before normalization
    if mask is not None:
        mask_expanded = mask.unsqueeze(-1).float()  # (batch, seq, 1)
        x_pred_t = x_pred_t * mask_expanded
        x_pred_t_aug = x_pred_t_aug * mask_expanded

    # Normalize embeddings
    x_pred_t = F.normalize(x_pred_t, dim=-1, eps=1e-8)
    x_pred_t_aug = F.normalize(x_pred_t_aug, dim=-1, eps=1e-8)

    # Compute similarity matrix within each sequence
    # For each batch b: logits[b] = x_pred_t[b] @ x_pred_t_aug[b].T
    logits = torch.bmm(x_pred_t, x_pred_t_aug.transpose(1, 2)) / temperature
    # Shape: (batch, seq, seq)

    # Labels: diagonal matching (position i should match position i)
    labels = torch.arange(seq_len, device=x_pred_t.device)
    labels = labels.unsqueeze(0).expand(batch_size, -1)  # (batch, seq)

    # Reshape for cross_entropy: (batch * seq, seq) and (batch * seq,)
    logits_flat = logits.reshape(batch_size * seq_len, seq_len)
    labels_flat = labels.reshape(-1)

    # Compute cross-entropy
    loss = F.cross_entropy(logits_flat, labels_flat, reduction='none')
    loss = loss.reshape(batch_size, seq_len)

    # Apply mask to loss: zero out padding positions
    if mask is not None:
        loss = loss * mask.float()  # Zero out padding
        return loss.sum() / mask.sum().clamp(min=1)  # Average over valid positions only
    else:
        return loss.mean()


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
    # Forward pass at this timestep
    x_pred_t, x_t, input_mask, target_mask = model.forward_train(sequence, timestep)
    x_pred_t_aug, _, _, _ = model.forward_train(sequence_aug, timestep)

    # Get clean negative embeddings and noise them at this timestep
    neg_seq = negatives[:, 1:]  # (batch_size, seq_len-1)
    x_0_neg = model.get_target_embedding(neg_seq)

    # Convert timestep to tensor for add_noise (expects (batch_size,) tensor)
    batch_size = x_0_neg.size(0)
    device = x_0_neg.device
    timestep_tensor = torch.full((batch_size,), timestep, device=device, dtype=torch.long)
    x_t_neg = model.diffuser.add_noise(x_0_neg, timestep_tensor)

    # Compute losses at this timestep
    l_cd = cross_divergence_loss(x_pred_t, x_t, x_t_neg, target_mask)
    l_in = in_view_contrastive_loss(x_pred_t, x_t, target_mask, temperature)
    l_cross = cross_view_contrastive_loss(x_pred_t, x_pred_t_aug, target_mask, temperature)

    # Combine losses (without rescaling - that's done in the training loop)
    loss = l_cd + lambda_contrast * (l_in + l_cross)

    # Loss dictionary for logging
    loss_dict = {
        "cross_divergence": l_cd.item(),
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
