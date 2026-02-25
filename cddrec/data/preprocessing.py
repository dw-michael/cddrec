"""Data preprocessing utilities for sequential recommendation datasets"""

import json
from pathlib import Path
from collections import defaultdict, Counter
from typing import Iterable
from .types import ProcessedData, PreprocessingResult


def preprocess_interactions(
    interactions: Iterable[tuple[str | int, str | int, float]],
    min_user_interactions: int = 5,
    min_item_interactions: int = 5,
    max_items: int | None = None,
    output_path: str | None = None,
) -> PreprocessingResult:
    """
    Complete preprocessing pipeline from raw interactions to train/val/test splits.

    Memory-efficient single-pass processing:
    1. Read iterator once, build ID mappings, convert to integers immediately
    2. Apply k-core filtering (iterative removal until stable)
    3. Create sequences and split into train/val/test
    4. Optionally save to JSON

    Args:
        interactions: Iterable of (user_id, item_id, timestamp) tuples
            - user_id: any hashable (int, str, etc.) - converted to 0-indexed int
            - item_id: any hashable (int, str, etc.) - converted to 1-indexed int (0=padding)
            - timestamp: any sortable value
            Can be list, generator, database cursor, etc.
        min_user_interactions: Minimum interactions per user (hard constraint, default: 5)
            Note: Need ≥4 for train/val/test split (2 train + 1 val + 1 test minimum)
        min_item_interactions: Minimum interactions per item (soft constraint, default: 5)
        max_items: If set, keep only top-K most popular items
        output_path: If provided, saves data to JSON and mappings to separate file
            Example: "data/beauty.json" → data: "data/beauty.json",
                     mappings: "data/beauty_mappings.json"

    Returns:
        PreprocessingResult with:
            - train/val/test: splits with integer IDs
            - num_users, num_items: counts
            - user_mapping, item_mapping: original_id → model_id

    Examples:
        >>> # With string IDs from database
        >>> interactions = [
        ...     ("user_abc", "item_xyz", 1000.0),
        ...     ("user_abc", "item_123", 1001.0),
        ...     # ...
        ... ]
        >>> data = preprocess_interactions(interactions)

        >>> # Streaming from database (memory-efficient!)
        >>> def stream_from_db():
        ...     for row in db.execute("SELECT user_id, item_id, timestamp FROM interactions"):
        ...         yield (row[0], row[1], row[2])
        >>> data = preprocess_interactions(stream_from_db())

        >>> # Mixed types are fine
        >>> interactions = [
        ...     (123, "SKU-456", 1.0),  # int user, string item
        ...     (456, "SKU-789", 2.0),
        ... ]
        >>> data = preprocess_interactions(interactions)
    """
    print("=" * 60)
    print("Preprocessing interactions")
    print("=" * 60)

    # Step 1: Single pass - build mappings and convert to integers
    print("\n[1/4] Reading interactions and building ID mappings...")
    integer_interactions, user_mapping, item_mapping = _read_and_index(interactions)
    print(f"  Original: {len(integer_interactions)} interactions")
    print(f"  Users: {len(user_mapping)}, Items: {len(item_mapping)}")

    # Step 2: K-core filtering (iterative removal until stable)
    print("\n[2/4] Filtering with k-core algorithm...")
    filtered_interactions, valid_users, valid_items = _kcore_filter(
        integer_interactions,
        min_user_count=min_user_interactions,
        min_item_count=min_item_interactions,
        max_items=max_items,
    )
    print(f"  After filtering: {len(filtered_interactions)} interactions")
    print(f"  Users: {len(valid_users)}, Items: {len(valid_items)}")

    # Remap to continuous IDs (0-indexed users, 1-indexed items)
    user_mapping, item_mapping = _remap_ids(
        user_mapping, item_mapping, valid_users, valid_items
    )

    # Step 3: Create sequences
    print("\n[3/4] Creating sequences...")
    user_sequences = _create_sequences(filtered_interactions, user_mapping, item_mapping)
    print(f"  Created sequences for {len(user_sequences)} users")

    # Step 4: Split train/val/test
    print("\n[4/4] Splitting train/val/test...")
    train_seqs, train_users, val_seqs, val_users, test_seqs, test_users = (
        _train_val_test_split(user_sequences)
    )
    print(f"  Train: {len(train_seqs)}, Val: {len(val_seqs)}, Test: {len(test_seqs)}")

    # Save if requested
    if output_path is not None:
        print("\nSaving to files...")
        _save_data(
            output_path,
            train_seqs, train_users,
            val_seqs, val_users,
            test_seqs, test_users,
            len(user_mapping),
            len(item_mapping),
        )
        _save_mappings(output_path, user_mapping, item_mapping)

    print("\n" + "=" * 60)
    print("Preprocessing complete!")
    print("=" * 60)

    return PreprocessingResult(
        train={"sequences": train_seqs, "user_ids": train_users},
        val={"sequences": val_seqs, "user_ids": val_users},
        test={"sequences": test_seqs, "user_ids": test_users},
        num_users=len(user_mapping),
        num_items=len(item_mapping),
        user_mapping=user_mapping,
        item_mapping=item_mapping,
    )


def _read_and_index(
    interactions: Iterable[tuple[str | int, str | int, float]]
) -> tuple[list[tuple[int, int, float]], dict[str | int, int], dict[str | int, int]]:
    """
    Single pass: read iterator, build mappings, convert to integers.

    Returns integer interactions for memory efficiency (strings discarded immediately).
    """
    user_mapping: dict[str | int, int] = {}
    item_mapping: dict[str | int, int] = {}
    integer_interactions: list[tuple[int, int, float]] = []

    user_counter = 0
    item_counter = 0

    for user_id, item_id, timestamp in interactions:
        # Map user to integer (0-indexed)
        if user_id not in user_mapping:
            user_mapping[user_id] = user_counter
            user_counter += 1

        # Map item to integer (0-indexed temporarily, will shift to 1-indexed later)
        if item_id not in item_mapping:
            item_mapping[item_id] = item_counter
            item_counter += 1

        # Store as integers immediately (discard strings)
        integer_interactions.append(
            (user_mapping[user_id], item_mapping[item_id], timestamp)
        )

    return integer_interactions, user_mapping, item_mapping


def _kcore_filter(
    interactions: list[tuple[int, int, float]],
    min_user_count: int,
    min_item_count: int,
    max_items: int | None = None,
) -> tuple[list[tuple[int, int, float]], set[int], set[int]]:
    """
    K-core filtering: iteratively remove users/items until stable.

    Hard constraint on users (must have min_user_count).
    Soft constraint on items (apply before user filtering, then keep whatever remains).
    """
    # Count occurrences
    user_counts = Counter(u for u, i, t in interactions)
    item_counts = Counter(i for u, i, t in interactions)

    # Apply item constraints FIRST (before user filtering)
    # 1. Max items constraint (keep top-K most popular)
    if max_items is not None and len(item_counts) > max_items:
        top_items = [i for i, _ in item_counts.most_common(max_items)]
        valid_items = set(top_items)
    else:
        valid_items = set(item_counts.keys())

    # 2. Min item interactions (soft constraint - apply initially)
    if min_item_count > 0:
        valid_items = {i for i, c in item_counts.items() if i in valid_items and c >= min_item_count}

    # Now apply hard user constraint with k-core iteration
    valid_users = {u for u, c in user_counts.items() if c >= min_user_count}

    # K-core iteration on USERS only (items are fixed after initial filter)
    prev_size = -1
    iteration = 0
    while len(valid_users) != prev_size:
        prev_size = len(valid_users)
        iteration += 1

        # Filter interactions
        filtered = [
            (u, i, t)
            for u, i, t in interactions
            if u in valid_users and i in valid_items
        ]

        # Recount users (hard constraint - must meet minimum)
        user_counts = Counter(u for u, i, t in filtered)
        valid_users = {u for u, c in user_counts.items() if c >= min_user_count}

    print(f"  K-core converged in {iteration} iteration(s)")

    # Final filter
    filtered = [(u, i, t) for u, i, t in interactions if u in valid_users and i in valid_items]

    return filtered, valid_users, valid_items


def _remap_ids(
    user_mapping: dict[str | int, int],
    item_mapping: dict[str | int, int],
    valid_users: set[int],
    valid_items: set[int],
) -> tuple[dict[str | int, int], dict[str | int, int]]:
    """
    Remap to continuous IDs after filtering.

    Users: 0-indexed
    Items: 1-indexed (0 reserved for padding)
    """
    # Find original IDs for valid indices
    reverse_user = {v: k for k, v in user_mapping.items()}
    reverse_item = {v: k for k, v in item_mapping.items()}

    # Create new continuous mappings
    new_user_mapping = {
        reverse_user[idx]: new_idx
        for new_idx, idx in enumerate(sorted(valid_users))
    }

    new_item_mapping = {
        reverse_item[idx]: new_idx + 1  # 1-indexed
        for new_idx, idx in enumerate(sorted(valid_items))
    }

    return new_user_mapping, new_item_mapping


def _create_sequences(
    interactions: list[tuple[int, int, float]],
    user_mapping: dict[str | int, int],
    item_mapping: dict[str | int, int],
) -> dict[int, list[int]]:
    """Create user sequences sorted by timestamp."""
    # Build reverse mapping to get back to current indices
    reverse_user = {v: k for k, v in user_mapping.items()}
    reverse_item = {v: k for k, v in item_mapping.items()}

    user_sequences = defaultdict(list)

    for user_idx, item_idx, timestamp in interactions:
        # Map old indices to new continuous indices
        try:
            new_user_id = user_mapping[reverse_user[user_idx]]
            new_item_id = item_mapping[reverse_item[item_idx]]
            user_sequences[new_user_id].append((timestamp, new_item_id))
        except KeyError:
            # Filtered out
            continue

    # Sort by timestamp
    for user_id in user_sequences:
        user_sequences[user_id].sort(key=lambda x: x[0])
        user_sequences[user_id] = [item_id for _, item_id in user_sequences[user_id]]

    return dict(user_sequences)


def _train_val_test_split(
    user_sequences: dict[int, list[int]]
) -> tuple[list[list[int]], list[int], list[list[int]], list[int], list[list[int]], list[int]]:
    """
    Split sequences into train/val/test.

    For each user with sequence [1, 2, 3, 4, 5]:
    - Test: [1, 2, 3, 4, 5] (predict 5 from [1,2,3,4])
    - Val:  [1, 2, 3, 4] (predict 4 from [1,2,3])
    - Train: [1, 2], [1, 2, 3] (multiple samples with increasing history)

    Returns:
        (train_sequences, train_user_ids,
         val_sequences, val_user_ids,
         test_sequences, test_user_ids)
    """
    train_sequences = []
    train_user_ids = []
    val_sequences = []
    val_user_ids = []
    test_sequences = []
    test_user_ids = []

    for user_id, sequence in user_sequences.items():
        if len(sequence) < 3:
            # Need at least 3 items for train/val/test split
            continue

        # Test: full sequence up to last item (predict last)
        test_sequences.append(sequence)
        test_user_ids.append(user_id)

        # Validation: full sequence up to second-to-last (predict second-to-last)
        val_sequences.append(sequence[:-1])
        val_user_ids.append(user_id)

        # Training: multiple samples with increasing history
        # For sequence [1,2,3,4,5], create:
        #   [1, 2] (predict 2 from 1)
        #   [1, 2, 3] (predict 3 from 1,2)
        for i in range(1, len(sequence) - 2):
            train_sequences.append(sequence[:i+1])  # Include target in sequence
            train_user_ids.append(user_id)

    return (
        train_sequences, train_user_ids,
        val_sequences, val_user_ids,
        test_sequences, test_user_ids,
    )


def _save_data(
    output_path: str,
    train_sequences: list[list[int]],
    train_user_ids: list[int],
    val_sequences: list[list[int]],
    val_user_ids: list[int],
    test_sequences: list[list[int]],
    test_user_ids: list[int],
    num_users: int,
    num_items: int,
):
    """Save processed data to JSON."""
    data = {
        "train": {"sequences": train_sequences, "user_ids": train_user_ids},
        "val": {"sequences": val_sequences, "user_ids": val_user_ids},
        "test": {"sequences": test_sequences, "user_ids": test_user_ids},
        "num_users": num_users,
        "num_items": num_items,
    }

    with open(output_path, "w") as f:
        json.dump(data, f, indent=2)

    print(f"  Data saved: {output_path}")
    print(f"    Users: {num_users}, Items: {num_items}")
    print(f"    Train: {len(train_sequences)}, Val: {len(val_sequences)}, Test: {len(test_sequences)}")


def _save_mappings(
    output_path: str,
    user_mapping: dict[str | int, int],
    item_mapping: dict[str | int, int],
):
    """Save ID mappings to separate JSON file."""
    path_obj = Path(output_path)
    mappings_path = path_obj.parent / f"{path_obj.stem}_mappings{path_obj.suffix}"

    mappings = {
        "user_mapping": {str(k): v for k, v in user_mapping.items()},
        "item_mapping": {str(k): v for k, v in item_mapping.items()},
        "num_users": len(user_mapping),
        "num_items": len(item_mapping),
    }

    with open(mappings_path, "w") as f:
        json.dump(mappings, f, indent=2)

    print(f"  Mappings saved: {mappings_path}")
    print(f"    User IDs: {len(user_mapping)}, Item IDs: {len(item_mapping)}")


# ============================================================================
# Loading and utility functions
# ============================================================================


def load_processed_data(json_path: str) -> ProcessedData:
    """
    Load preprocessed data from JSON file.

    Args:
        json_path: Path to data JSON file

    Returns:
        ProcessedData with train/val/test splits and metadata
    """
    with open(json_path, "r") as f:
        data: ProcessedData = json.load(f)

    # Validate structure
    required_keys = {"train", "val", "test", "num_users", "num_items"}
    if not required_keys.issubset(data.keys()):
        missing = required_keys - set(data.keys())
        raise ValueError(f"Invalid data file. Missing keys: {missing}")

    for split in ["train", "val", "test"]:
        if "sequences" not in data[split] or "user_ids" not in data[split]:
            raise ValueError(f"Invalid data. '{split}' missing 'sequences' or 'user_ids'")

    return data


def load_id_mappings(json_path: str) -> tuple[dict[str, int], dict[str, int]]:
    """
    Load ID mappings from JSON file.

    Args:
        json_path: Path to mappings JSON file (e.g., "data/beauty_mappings.json")

    Returns:
        user_mapping: original_user_id (as string) → model_user_id
        item_mapping: original_item_id (as string) → model_item_id

    Note:
        Keys are strings in JSON. Convert if needed:
        >>> user_map, item_map = load_id_mappings("mappings.json")
        >>> # If original IDs were integers:
        >>> user_map_int = {int(k): v for k, v in user_map.items()}

    Example:
        >>> user_map, item_map = load_id_mappings("data/beauty_mappings.json")
        >>> model_item_id = item_map["SKU-12345"]
    """
    with open(json_path, "r") as f:
        data = json.load(f)

    return data["user_mapping"], data["item_mapping"]


def create_reverse_mappings(
    user_mapping: dict[str | int, int],
    item_mapping: dict[str | int, int],
) -> tuple[dict[int, str | int], dict[int, str | int]]:
    """
    Create reverse mappings from model IDs → original IDs.

    Useful for translating predictions back to original IDs.

    Args:
        user_mapping: original_user_id → model_user_id
        item_mapping: original_item_id → model_item_id

    Returns:
        reverse_user_mapping: model_user_id → original_user_id
        reverse_item_mapping: model_item_id → original_item_id

    Example:
        >>> rev_user, rev_item = create_reverse_mappings(user_map, item_map)
        >>> # Get top-10 predictions from model
        >>> top_10_model_ids = model.predict(...)
        >>> # Convert back to original IDs
        >>> top_10_original_ids = [rev_item[i] for i in top_10_model_ids]
    """
    reverse_user_mapping = {v: k for k, v in user_mapping.items()}
    reverse_item_mapping = {v: k for k, v in item_mapping.items()}
    return reverse_user_mapping, reverse_item_mapping
