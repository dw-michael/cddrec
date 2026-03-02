"""Data handling utilities for CDDRec"""

from .dataset import (
    SeqRecDataset,
    create_dataloader,
    collate_fn,
    extract_targets,
    DataBundle,
    load_data,
)
from .augmentation import mask_sequence, shuffle_sequence, crop_sequence, augment_sequence
from .preprocessing import (
    preprocess_interactions,
    load_processed_data,
    load_id_mappings,
    create_reverse_mappings,
    convert_to_author_format,
)
from .types import (
    SplitData,
    ProcessedData,
    PreprocessingResult,
    DatasetSample,
    DatasetBatch,
)

__all__ = [
    # Dataset
    "SeqRecDataset",
    "create_dataloader",
    "collate_fn",
    "extract_targets",
    "DataBundle",
    "load_data",
    # Augmentation
    "mask_sequence",
    "shuffle_sequence",
    "crop_sequence",
    "augment_sequence",
    # Preprocessing
    "preprocess_interactions",
    "load_processed_data",
    "load_id_mappings",
    "create_reverse_mappings",
    "convert_to_author_format",
    # Types
    "SplitData",
    "ProcessedData",
    "PreprocessingResult",
    "DatasetSample",
    "DatasetBatch",
]
