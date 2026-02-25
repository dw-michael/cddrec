"""Sequence augmentation for contrastive learning"""

import torch
import random


def mask_sequence(
    item_seq: torch.Tensor,
    mask_ratio: float = 0.2,
    mask_token: int = 0,
) -> torch.Tensor:
    """
    Randomly mask items in sequence.

    IMPORTANT: Never masks position 0 to maintain causal attention structure.
    If position 0 is masked, it creates all-masked attention rows → NaN.

    Args:
        item_seq: (batch_size, seq_len) item IDs
        mask_ratio: Fraction of items to mask
        mask_token: Token ID to use for masking (default: 0 for padding)

    Returns:
        Masked sequence of same shape
    """
    batch_size, seq_len = item_seq.size()
    masked_seq = item_seq.clone()

    for i in range(batch_size):
        # Identify non-padding positions
        non_padding = (item_seq[i] != mask_token).nonzero(as_tuple=True)[0]

        if len(non_padding) > 1:  # Need at least 2 items to mask some
            # CRITICAL: Exclude position 0 from masking to prevent all-masked attention rows
            # In causal attention, position 0 can only attend to itself.
            # If position 0 is masked, it has no valid attention targets → NaN
            maskable_positions = non_padding[non_padding > 0].tolist()

            if maskable_positions:
                num_mask = max(1, int(len(maskable_positions) * mask_ratio))
                num_mask = min(num_mask, len(maskable_positions))

                # Randomly select positions to mask (excluding position 0)
                mask_positions = random.sample(maskable_positions, num_mask)
                masked_seq[i, mask_positions] = mask_token

    return masked_seq


def shuffle_sequence(
    item_seq: torch.Tensor,
    shuffle_ratio: float = 0.2,
) -> torch.Tensor:
    """
    Randomly shuffle items locally within sequence.

    IMPORTANT: Never shuffles position 0 to maintain causal attention structure.
    If position 0 contains padding after shuffle, it creates all-masked attention rows → NaN.

    Local shuffle maintains some sequential structure while adding noise.

    Args:
        item_seq: (batch_size, seq_len) item IDs
        shuffle_ratio: Fraction of positions to shuffle

    Returns:
        Shuffled sequence of same shape
    """
    batch_size, seq_len = item_seq.size()
    shuffled_seq = item_seq.clone()

    for i in range(batch_size):
        num_shuffle = int(seq_len * shuffle_ratio)
        if num_shuffle > 1:
            # CRITICAL: Never start shuffle at position 0 to prevent padding at position 0
            # Position 0 with causal attention can only attend to itself.
            # If position 0 becomes padding after shuffle, it has no valid attention targets → NaN
            if seq_len - num_shuffle >= 1:
                start_pos = random.randint(1, seq_len - num_shuffle)
                end_pos = start_pos + num_shuffle

                # Shuffle the segment (excluding position 0)
                segment = shuffled_seq[i, start_pos:end_pos].clone()
                perm = torch.randperm(num_shuffle)
                shuffled_seq[i, start_pos:end_pos] = segment[perm]

    return shuffled_seq


def crop_sequence(
    item_seq: torch.Tensor,
    crop_ratio: float = 0.2,
    pad_token: int = 0,
) -> torch.Tensor:
    """
    Randomly crop sequence by removing a portion.

    IMPORTANT: Always keeps position 0 to maintain causal attention structure.
    If position 0 is cropped out, it creates all-masked attention rows → NaN.

    Args:
        item_seq: (batch_size, seq_len) item IDs
        crop_ratio: Fraction of sequence to crop
        pad_token: Token to pad with after cropping

    Returns:
        Cropped and padded sequence of same shape
    """
    batch_size, seq_len = item_seq.size()
    cropped_seq = item_seq.clone()

    for i in range(batch_size):
        num_crop = int(seq_len * crop_ratio)
        if num_crop > 0 and num_crop < seq_len:
            # Keep (seq_len - num_crop) items
            keep_len = seq_len - num_crop

            # CRITICAL: Always start crop at position 0 to prevent padding at position 0
            # Position 0 with causal attention can only attend to itself.
            # If position 0 is cropped out, it becomes padding → no valid attention targets → NaN
            start_pos = 0
            end_pos = start_pos + keep_len

            # Create cropped sequence with padding
            # This simplifies to just keeping the first keep_len items
            cropped_seq[i, :keep_len] = item_seq[i, :keep_len]
            cropped_seq[i, keep_len:] = pad_token

    return cropped_seq


def augment_sequence(
    item_seq: torch.Tensor,
    augmentation_type: str = "mask",
    augmentation_ratio: float = 0.2,
    mask_token: int = 0,
) -> torch.Tensor:
    """
    Apply data augmentation to sequence.

    Args:
        item_seq: (batch_size, seq_len) item IDs
        augmentation_type: 'mask', 'shuffle', or 'crop'
        augmentation_ratio: Intensity of augmentation
        mask_token: Token for masking/padding

    Returns:
        Augmented sequence of same shape
    """
    if augmentation_type == "mask":
        return mask_sequence(item_seq, augmentation_ratio, mask_token)
    elif augmentation_type == "shuffle":
        return shuffle_sequence(item_seq, augmentation_ratio)
    elif augmentation_type == "crop":
        return crop_sequence(item_seq, augmentation_ratio, mask_token)
    elif augmentation_type == "random":
        # Randomly choose augmentation
        aug_type = random.choice(["mask", "shuffle", "crop"])
        return augment_sequence(item_seq, aug_type, augmentation_ratio, mask_token)
    else:
        raise ValueError(f"Unknown augmentation type: {augmentation_type}")
