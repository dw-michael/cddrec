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
        num_mask = int(seq_len * mask_ratio)
        if num_mask > 0:
            # Randomly select positions to mask
            mask_positions = random.sample(range(seq_len), num_mask)
            masked_seq[i, mask_positions] = mask_token

    return masked_seq


def shuffle_sequence(
    item_seq: torch.Tensor,
    shuffle_ratio: float = 0.2,
) -> torch.Tensor:
    """
    Randomly shuffle items locally within sequence.

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
            # Randomly select a contiguous segment to shuffle
            start_pos = random.randint(0, seq_len - num_shuffle)
            end_pos = start_pos + num_shuffle

            # Shuffle the segment
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

            # Randomly select start position for crop
            start_pos = random.randint(0, seq_len - keep_len)
            end_pos = start_pos + keep_len

            # Create cropped sequence with padding
            cropped_seq[i, :keep_len] = item_seq[i, start_pos:end_pos]
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


def create_padding_mask(item_seq: torch.Tensor, pad_token: int = 0) -> torch.Tensor:
    """
    Create boolean padding mask from sequence.

    Args:
        item_seq: (batch_size, seq_len) item IDs
        pad_token: Padding token ID

    Returns:
        Padding mask: (batch_size, seq_len) True for valid, False for padding
    """
    return item_seq != pad_token
