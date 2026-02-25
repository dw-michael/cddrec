"""Type definitions for data structures used throughout CDDRec."""

from typing import TypedDict
import torch

# ============================================================================
# Type Aliases (local to data module, not exported)
# ============================================================================

# ID mappings: original ID (string or int) → model ID (int)
IDMapping = dict[str | int, int]

# Item sequences (lists of item IDs)
ItemSequence = list[int]

# ============================================================================
# Preprocessing Types
# ============================================================================


class SplitData(TypedDict):
    """Data for a single split (train/val/test).

    All splits use the same format. The last item in each sequence is what's being predicted.
    """
    sequences: list[ItemSequence]  # Full sequences (history + target as last item)
    user_ids: list[int]            # Which user owns each sequence


class ProcessedData(TypedDict):
    """Processed dataset structure (saved to/loaded from JSON).

    Contains train/val/test splits with integer item IDs.
    """
    train: SplitData
    val: SplitData
    test: SplitData
    num_users: int
    num_items: int


class PreprocessingResult(TypedDict):
    """Result from preprocessing pipeline.

    Includes both the processed data and ID mappings for production use.
    """
    # Data splits
    train: SplitData
    val: SplitData
    test: SplitData

    # Metadata
    num_users: int
    num_items: int

    # ID mappings (original → model IDs)
    user_mapping: IDMapping
    item_mapping: IDMapping


# ============================================================================
# Dataset Types
# ============================================================================


class DatasetSample(TypedDict):
    """Single sample from SeqRecDataset.

    Returned by SeqRecDataset.__getitem__().
    """
    sequence: torch.Tensor   # (max_seq_len,) padded sequence of item IDs
    negatives: torch.Tensor  # (max_seq_len,) negative items for each position
    seq_len: torch.Tensor    # Scalar, length before padding


class DatasetBatch(TypedDict):
    """Batched samples from DataLoader.

    Returned by collate_fn() after batching multiple DatasetSamples.
    """
    sequence: torch.Tensor   # (batch_size, max_seq_len) padded sequences
    negatives: torch.Tensor  # (batch_size, max_seq_len) negative items
    seq_len: torch.Tensor    # (batch_size,) sequence lengths before padding
