"""Sequence augmentation for contrastive learning

Matches the augmentation logic from the authors' implementation exactly.
"""

import torch
import random
import math


def mask_sequence(
    item_seq: torch.Tensor,
    mask_token: int,
    pad_token: int = 0,
    mask_ratio: float = 0.3,
) -> torch.Tensor:
    """
    Randomly mask items in sequence (authors' item_mask with gamma=0.3).

    Matches authors' implementation:
    - Masks mask_ratio (default 30%) of NON-PADDING positions
    - Uses special mask_token (not padding token)
    - Random (non-contiguous) positions
    - Can mask any position including position 0

    Args:
        item_seq: (batch_size, seq_len) item IDs
        mask_token: Token ID to use for masking (typically num_items + 1)
        pad_token: Padding token ID (default: 0)
        mask_ratio: Fraction of non-padding items to mask (default: 0.3)

    Returns:
        Masked sequence of same shape
    """

    batch_size, seq_len = item_seq.size()
    masked_seq = item_seq.clone()

    for i in range(batch_size):
        # Identify non-padding positions only
        non_padding = (item_seq[i] != pad_token).nonzero(as_tuple=True)[0].tolist()

        if len(non_padding) > 0:
            # Mask mask_ratio of non-padding positions
            num_mask = math.floor(len(non_padding) * mask_ratio)
            if num_mask > 0:
                # Randomly select positions to mask
                mask_positions = random.sample(non_padding, num_mask)
                masked_seq[i, mask_positions] = mask_token

    return masked_seq


def shuffle_sequence(
    item_seq: torch.Tensor,
    shuffle_ratio: float = 0.6,
    pad_token: int = 0,
) -> torch.Tensor:
    """
    Randomly shuffle items locally within sequence (authors' item_reorder with beta=0.6).

    Matches authors' implementation:
    - Shuffles shuffle_ratio (default 60%) of NON-PADDING items
    - Selects a contiguous segment and shuffles within it (local shuffle)
    - Can include position 0 in the shuffled segment

    Args:
        item_seq: (batch_size, seq_len) item IDs
        shuffle_ratio: Fraction of non-padding items to shuffle (default: 0.6)
        pad_token: Padding token ID (default: 0)

    Returns:
        Shuffled sequence of same shape
    """
    batch_size, seq_len = item_seq.size()
    shuffled_seq = item_seq.clone()

    for i in range(batch_size):
        # Count non-padding items
        non_padding = (item_seq[i] != pad_token).nonzero(as_tuple=True)[0]
        item_seq_len = len(non_padding)

        if item_seq_len > 1:
            # Shuffle shuffle_ratio of non-padding items
            num_reorder = math.floor(item_seq_len * shuffle_ratio)

            if num_reorder > 1 and num_reorder <= item_seq_len:
                # Random starting position for the shuffle segment
                reorder_begin = random.randint(0, item_seq_len - num_reorder)
                reorder_end = reorder_begin + num_reorder

                # Get actual positions in sequence (mapping from non-padding index to seq position)
                actual_positions = non_padding[reorder_begin:reorder_end].tolist()

                # Extract the segment and shuffle it
                segment = shuffled_seq[i, actual_positions].clone()
                shuffle_indices = torch.randperm(len(actual_positions))
                shuffled_seq[i, actual_positions] = segment[shuffle_indices]

    return shuffled_seq


def crop_sequence(
    item_seq: torch.Tensor,
    keep_ratio: float = 0.6,
    pad_token: int = 0,
) -> torch.Tensor:
    """
    Randomly crop sequence by keeping a contiguous segment (authors' item_crop with eta=0.6).

    Matches authors' implementation:
    - Keeps keep_ratio (default 60%) of NON-PADDING items, removes the rest
    - Selects random contiguous segment from non-padding items
    - Zero-pads removed positions
    - Can start crop anywhere (position 0 not protected)

    Args:
        item_seq: (batch_size, seq_len) item IDs
        keep_ratio: Fraction of non-padding items to KEEP (default: 0.6, removes 40%)
        pad_token: Token to pad with after cropping (default: 0)

    Returns:
        Cropped and padded sequence of same shape
    """
    batch_size, seq_len = item_seq.size()
    cropped_seq = torch.full_like(item_seq, pad_token)

    for i in range(batch_size):
        # Count non-padding items
        non_padding_mask = item_seq[i] != pad_token
        non_padding_positions = non_padding_mask.nonzero(as_tuple=True)[0]
        item_seq_len = len(non_padding_positions)

        if item_seq_len > 0:
            # Keep keep_ratio of non-padding items
            num_left = math.floor(item_seq_len * keep_ratio)
            num_left = max(1, num_left)  # Keep at least 1 item

            if num_left < item_seq_len:
                # Random starting position for the kept segment
                crop_begin = random.randint(0, item_seq_len - num_left)
                crop_end = crop_begin + num_left

                # Extract segment from non-padding items
                kept_positions = non_padding_positions[crop_begin:crop_end]
                kept_items = item_seq[i, kept_positions]

                # Place kept items at the beginning, pad the rest
                cropped_seq[i, :len(kept_items)] = kept_items
            else:
                # Keep all items (no cropping needed)
                cropped_seq[i, :item_seq_len] = item_seq[i, non_padding_positions]

    return cropped_seq


def augment_sequence(
    item_seq: torch.Tensor,
    mask_token: int,
    pad_token: int = 0,
    mask_ratio: float = 0.3,
    shuffle_ratio: float = 0.6,
    crop_keep_ratio: float = 0.6,
) -> torch.Tensor:
    """
    Apply data augmentation to sequence.

    Matches authors' implementation:
    - Randomly selects ONE augmentation type per batch
    - Uses authors' default intensities (configurable)
    - Special mask_token for masking (not padding)

    Args:
        item_seq: (batch_size, seq_len) item IDs
        mask_token: Special token for masking augmentation (typically num_items + 1)
        pad_token: Padding token ID (default: 0)
        mask_ratio: Fraction of items to mask (default: 0.3, authors' gamma)
        shuffle_ratio: Fraction of items to shuffle (default: 0.6, authors' beta)
        crop_keep_ratio: Fraction of items to KEEP in crop (default: 0.6, authors' eta)

    Returns:
        Augmented sequence of same shape
    """
    # Randomly choose ONE augmentation type (authors' approach)
    aug_type = random.choice(["mask", "shuffle", "crop"])

    if aug_type == "mask":
        return mask_sequence(item_seq, mask_token, pad_token, mask_ratio)
    elif aug_type == "shuffle":
        return shuffle_sequence(item_seq, shuffle_ratio, pad_token)
    else:  # crop
        return crop_sequence(item_seq, crop_keep_ratio, pad_token)
