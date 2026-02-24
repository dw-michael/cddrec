"""PyTorch Dataset for sequential recommendation"""

import torch
from torch.utils.data import Dataset
import random


class SeqRecDataset(Dataset):
    """
    Dataset for sequential recommendation with teacher forcing.

    Each sample contains:
    - Full interaction sequence (input and target will be sliced in model)
    - Negative items for each position (for cross-divergence loss)
    """

    def __init__(
        self,
        sequences: list[list[int]],
        targets: list[int],
        num_items: int,
        max_seq_len: int = 20,
        pad_token: int = 0,
    ):
        """
        Args:
            sequences: List of interaction sequences (each is list of item IDs)
            targets: List of target items (next item for each sequence)
            num_items: Total number of items (for negative sampling)
            max_seq_len: Maximum sequence length (pad/truncate to this)
            pad_token: Padding token ID
        """
        self.num_items = num_items
        self.max_seq_len = max_seq_len
        self.pad_token = pad_token

        # Concatenate sequences with targets to form full sequences
        self.full_sequences = []
        for seq, target in zip(sequences, targets):
            full_seq = seq + [target]
            self.full_sequences.append(full_seq)

    def __len__(self) -> int:
        return len(self.full_sequences)

    def _pad_sequence(self, seq: list[int]) -> list[int]:
        """Pad or truncate sequence to max_seq_len"""
        if len(seq) > self.max_seq_len:
            # Keep most recent items
            return seq[-self.max_seq_len:]
        else:
            # Pad at the end (back-padding)
            return seq + [self.pad_token] * (self.max_seq_len - len(seq))

    def _sample_negatives(self, user_items: list[int]) -> list[int]:
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

    def __getitem__(self, idx: int) -> dict:
        """
        Get a single sample.

        Returns:
            Dictionary with:
            - sequence: (max_seq_len,) padded full sequence
            - negatives: (max_seq_len,) negative items for each position
        """
        full_seq = self.full_sequences[idx]

        # Pad sequence
        padded_seq = self._pad_sequence(full_seq)

        # Sample negatives for all positions
        negatives = self._sample_negatives(full_seq)

        sample = {
            "sequence": torch.tensor(padded_seq, dtype=torch.long),
            "negatives": torch.tensor(negatives, dtype=torch.long),
        }

        return sample


def collate_fn(batch: list[dict]) -> dict:
    """
    Collate function for DataLoader.

    Args:
        batch: List of samples from dataset

    Returns:
        Batched dictionary
    """
    sequences = torch.stack([sample["sequence"] for sample in batch])
    negatives = torch.stack([sample["negatives"] for sample in batch])

    return {
        "sequence": sequences,
        "negatives": negatives,
    }


def create_dataloader(
    dataset: SeqRecDataset,
    batch_size: int = 128,
    shuffle: bool = True,
    num_workers: int = 0,
) -> torch.utils.data.DataLoader:
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
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_fn,
    )
