"""Type definitions for data structures used throughout CDDRec."""

from typing import TypedDict


class SplitData(TypedDict):
    """Data for a single split (train/val/test).

    All splits use the same format. The last item in each sequence is what's being predicted.
    """
    sequences: list[list[int]]  # Full sequences (history + target as last item)
    user_ids: list[int]         # Which user owns each sequence


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
    user_mapping: dict[str | int, int]
    item_mapping: dict[str | int, int]
