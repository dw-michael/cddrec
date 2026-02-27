"""PyTorch Dataset for sequential recommendation"""

import torch
from torch.utils.data import Dataset, DataLoader
from typing import NamedTuple
import random

from .types import ItemSequence, DatasetSample, DatasetBatch


class SeqRecDataset(Dataset):
    """
    Dataset for sequential recommendation with teacher forcing.

    Each sample contains:
    - Full interaction sequence (input and target will be sliced in model)
    - Negative items for each position (for cross-divergence loss)

    In training mode, randomly samples contiguous subsequences of length >= 2
    to provide diverse training samples and avoid bias toward longer sequences.
    """

    def __init__(
        self,
        sequences: list[ItemSequence],
        num_items: int,
        max_seq_len: int = 20,
        pad_token: int = 0,
        mask_token: int | None = None,
        train_mode: bool = True,
        seed: int | None = None,
        min_subseq_len: int = 2,
    ):
        """
        Args:
            sequences: List of full sequences (history + target as last item)
            num_items: Total number of items (for negative sampling)
            max_seq_len: Maximum sequence length (pad/truncate to this)
            pad_token: Padding token ID (default: 0)
            mask_token: Special mask token ID for augmentation (default: num_items + 1).
                       If None, defaults to num_items + 1.
            train_mode: If True, randomly sample subsequences during training.
                       If False, use full sequences (for val/test).
            seed: Random seed for subsequence sampling (for reproducibility).
                 If None, uses default random state.
            min_subseq_len: Minimum length for sampled subsequences (default: 2).
                           Must be >= 2 (need at least 1 context + 1 target).
                           Higher values may filter out noisy short sequences.
        """
        # Validate minimum subsequence length
        if min_subseq_len < 2:
            raise ValueError(
                f"min_subseq_len must be >= 2 (need at least 1 context + 1 target), "
                f"got {min_subseq_len}"
            )

        self.num_items = num_items
        self.max_seq_len = max_seq_len
        self.pad_token = pad_token
        self.mask_token = mask_token if mask_token is not None else num_items + 1
        self.train_mode = train_mode
        self.min_subseq_len = min_subseq_len

        # Sequences are already complete (no concatenation needed)
        self.full_sequences = sequences

        # Per-dataset RNG for reproducible subsequence sampling
        self.rng = random.Random(seed) if seed is not None else random.Random()

    def __len__(self) -> int:
        return len(self.full_sequences)

    def _sample_subsequence(self, sequence: ItemSequence) -> ItemSequence:
        """
        Sample a random contiguous subsequence uniformly over all valid subsequences.

        Samples start and end positions such that each possible contiguous
        subsequence of length >= min_subseq_len has equal probability.

        Args:
            sequence: Full sequence to sample from

        Returns:
            Sampled contiguous subsequence of length >= min_subseq_len
        """
        seq_len = len(sequence)
        min_len = self.min_subseq_len

        if seq_len <= min_len:
            return sequence

        # Sample start position: [0, seq_len - min_len]
        start = self.rng.randint(0, seq_len - min_len)

        # Sample end position: [start + min_len, seq_len]
        end = self.rng.randint(start + min_len, seq_len)

        return sequence[start:end]

    def _normalize_sequence_length(self, seq: ItemSequence) -> ItemSequence:
        """
        Normalize sequence to exactly max_seq_len items.

        - If too long: truncate to keep most recent items
        - If too short: pad at the end with pad_token

        Returns:
            Sequence of exactly max_seq_len items
        """
        if len(seq) > self.max_seq_len:
            # Keep most recent items (truncate from beginning)
            return seq[-self.max_seq_len:]
        elif len(seq) < self.max_seq_len:
            # Pad at the end (back-padding)
            return seq + [self.pad_token] * (self.max_seq_len - len(seq))
        else:
            # Already correct length
            return seq

    def _sample_negatives(self, user_items: ItemSequence) -> ItemSequence:
        """
        Sample negative items for each position in the sequence.

        Args:
            user_items: All items in user's history

        Returns:
            List of negative item IDs (length = max_seq_len)
        """
        user_item_set = set(user_items)
        negatives = []

        for _ in range(self.max_seq_len):
            # Sample a negative item not in user's history
            while True:
                neg_item = random.randint(1, self.num_items)  # 1-indexed (0 is padding)
                if neg_item not in user_item_set:
                    negatives.append(neg_item)
                    break

        return negatives

    def __getitem__(self, idx: int) -> DatasetSample:
        """
        Get a single sample.

        In training mode, samples a random contiguous subsequence.
        In val/test mode, uses the full sequence.

        Returns:
            DatasetSample with:
            - sequence: (max_seq_len,) padded sequence
            - negatives: (max_seq_len,) negative items for each position
            - seq_len: Scalar, length of sequence before padding
        """
        full_seq = self.full_sequences[idx]

        # Sample subsequence during training, use full sequence for val/test
        if self.train_mode:
            seq = self._sample_subsequence(full_seq)
        else:
            seq = full_seq

        # Normalize to exactly max_seq_len (truncate if too long, pad if too short)
        normalized_seq = self._normalize_sequence_length(seq)

        # IMPORTANT: Record actual length (clamped to max_seq_len)
        # This ensures seq_len never exceeds the actual tensor size
        seq_len = min(len(seq), self.max_seq_len)

        # Sample negatives for all positions (use full history for negative sampling)
        negatives = self._sample_negatives(full_seq)

        sample = {
            "sequence": torch.tensor(normalized_seq, dtype=torch.long),
            "negatives": torch.tensor(negatives, dtype=torch.long),
            "seq_len": torch.tensor(seq_len, dtype=torch.long),
        }

        return sample


def collate_fn(batch: list[DatasetSample]) -> DatasetBatch:
    """
    Collate function for DataLoader.

    Args:
        batch: List of DatasetSamples from dataset

    Returns:
        DatasetBatch with:
            - sequence: (batch_size, max_seq_len) padded sequences
            - negatives: (batch_size, max_seq_len) negative items
            - seq_len: (batch_size,) sequence lengths (before padding)
    """
    sequences = torch.stack([sample["sequence"] for sample in batch])
    negatives = torch.stack([sample["negatives"] for sample in batch])
    seq_lens = torch.stack([sample["seq_len"] for sample in batch])

    return {
        "sequence": sequences,
        "negatives": negatives,
        "seq_len": seq_lens,
    }


def extract_targets(sequences: torch.Tensor, seq_lens: torch.Tensor) -> torch.Tensor:
    """
    Extract target items from padded sequences using sequence lengths.

    Uses advanced indexing to extract the last real item from each sequence
    (the item at position seq_len - 1 for each sequence in the batch).

    Args:
        sequences: (batch_size, max_seq_len) padded sequences
        seq_lens: (batch_size,) sequence lengths before padding

    Returns:
        targets: (batch_size,) target items (last real item from each sequence)

    Example:
        >>> sequences = torch.tensor([[1, 2, 3, 0, 0], [6, 7, 8, 9, 0]])
        >>> seq_lens = torch.tensor([3, 4])
        >>> extract_targets(sequences, seq_lens)
        tensor([3, 9])
    """
    batch_size = sequences.shape[0]
    batch_indices = torch.arange(batch_size, device=sequences.device)
    target_positions = seq_lens - 1  # Last item is at seq_len - 1
    targets = sequences[batch_indices, target_positions]
    return targets


def create_dataloader(
    dataset: SeqRecDataset,
    batch_size: int = 128,
    shuffle: bool = True,
    num_workers: int = 0,
) -> DataLoader:
    """
    Create DataLoader for dataset.

    Args:
        dataset: SeqRecDataset instance
        batch_size: Batch size
        shuffle: Whether to shuffle data
        num_workers: Number of worker processes

    Returns:
        DataLoader instance
    """
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_fn,
    )


class DataBundle(NamedTuple):
    """Container for datasets, dataloaders, and metadata.

    This is returned by load_data() for convenience.
    """
    # Datasets
    train_dataset: SeqRecDataset
    val_dataset: SeqRecDataset
    test_dataset: SeqRecDataset

    # DataLoaders
    train_loader: DataLoader
    val_loader: DataLoader
    test_loader: DataLoader

    # Metadata
    num_users: int
    num_items: int


def load_data(
    json_path: str,
    batch_size: int = 128,
    val_batch_size: int | None = None,
    max_seq_len: int = 20,
    num_workers: int = 0,
    min_subseq_len: int = 2,
) -> DataBundle:
    """
    Load preprocessed data and create everything needed for training.

    This is the main entry point for loading data - it handles everything:
    1. Loads processed data from JSON
    2. Creates train/val/test datasets
    3. Creates train/val/test dataloaders
    4. Returns everything in a convenient bundle

    Args:
        json_path: Path to preprocessed JSON file (from preprocess_interactions)
        batch_size: Batch size for training dataloader
        val_batch_size: Batch size for validation/test dataloaders (default: same as batch_size).
                       Can be larger than training batch size since validation doesn't compute
                       gradients or store optimizer states. Typically 2-4x larger is safe.
        max_seq_len: Maximum sequence length (pad/truncate)
        num_workers: Number of worker processes for dataloaders
        min_subseq_len: Minimum length for training subsequences (default: 2).
                       Must be >= 2. Higher values filter out short sequences.
                       Only affects training; val/test always use full sequences.

    Returns:
        DataBundle with all datasets, dataloaders, and metadata

    Example:
        >>> from cddrec.data import load_data
        >>> # Use larger batch size for validation (no gradients needed)
        >>> data = load_data("data/processed/beauty.json",
        ...                  batch_size=128,
        ...                  val_batch_size=512)
        >>> print(f"Training on {data.num_items} items")
        >>> model = CDDRec(num_items=data.num_items, ...)
        >>> for batch in data.train_loader:
        ...     # train model
    """
    # Default to same batch size for validation if not specified
    if val_batch_size is None:
        val_batch_size = batch_size
    # Import here to avoid circular dependency
    from .preprocessing import load_processed_data

    # Load preprocessed data
    processed_data = load_processed_data(json_path)

    # Create datasets
    train_dataset = SeqRecDataset(
        sequences=processed_data["train"]["sequences"],
        num_items=processed_data["num_items"],
        max_seq_len=max_seq_len,
        train_mode=True,   # Enable random subsequence sampling
        seed=None,         # Random sampling each time
        min_subseq_len=min_subseq_len,
    )

    val_dataset = SeqRecDataset(
        sequences=processed_data["val"]["sequences"],
        num_items=processed_data["num_items"],
        max_seq_len=max_seq_len,
        train_mode=False,  # Use full sequences (deterministic)
        seed=42,           # Seed not used in eval mode, but set for consistency
        min_subseq_len=min_subseq_len,  # Not used in eval mode but kept for consistency
    )

    test_dataset = SeqRecDataset(
        sequences=processed_data["test"]["sequences"],
        num_items=processed_data["num_items"],
        max_seq_len=max_seq_len,
        train_mode=False,  # Use full sequences (deterministic)
        seed=42,           # Seed not used in eval mode, but set for consistency
        min_subseq_len=min_subseq_len,  # Not used in eval mode but kept for consistency
    )

    # Create dataloaders
    train_loader = create_dataloader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
    )

    val_loader = create_dataloader(
        val_dataset,
        batch_size=val_batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    test_loader = create_dataloader(
        test_dataset,
        batch_size=val_batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    return DataBundle(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        test_dataset=test_dataset,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        num_users=processed_data["num_users"],
        num_items=processed_data["num_items"],
    )
