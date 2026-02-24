"""Data handling utilities for CDDRec"""

from .dataset import (
    SeqRecDataset,
    create_dataloader,
    collate_fn,
    extract_targets,
    DataBundle,
    setup_data_from_file,
)
from .augmentation import mask_sequence, shuffle_sequence, crop_sequence, augment_sequence, create_padding_mask
from .preprocessing import (
    preprocess_interactions,
    load_processed_data,
    load_id_mappings,
    create_reverse_mappings,
)
from .types import SplitData, ProcessedData, PreprocessingResult

__all__ = [
    # Dataset
    "SeqRecDataset",
    "create_dataloader",
    "collate_fn",
    "extract_targets",
    "DataBundle",
    "setup_data_from_file",
    # Augmentation
    "mask_sequence",
    "shuffle_sequence",
    "crop_sequence",
    "augment_sequence",
    "create_padding_mask",
    # Preprocessing
    "preprocess_interactions",
    "load_processed_data",
    "load_id_mappings",
    "create_reverse_mappings",
    # Types
    "SplitData",
    "ProcessedData",
    "PreprocessingResult",
]
