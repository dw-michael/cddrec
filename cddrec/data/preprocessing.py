"""Data preprocessing utilities for sequential recommendation datasets"""

import json
from collections import defaultdict, Counter


def filter_interactions(
    interactions: list[tuple[int, int, float]],
    min_user_interactions: int = 5,
    min_item_interactions: int = 5,
    max_items: int | None = None,
) -> tuple[list[tuple[int, int, float]], dict[int, int], dict[int, int]]:
    """
    Filter out users and items with too few interactions.

    Args:
        interactions: List of (user_id, item_id, timestamp) tuples
        min_user_interactions: Minimum interactions per user
        min_item_interactions: Minimum interactions per item
        max_items: If set, keep only top-K most popular items

    Returns:
        filtered_interactions: Filtered interactions with (user_id, item_id, timestamp)
        user_mapping: Original user_id -> new user_id (0-indexed)
        item_mapping: Original item_id -> new item_id (1-indexed, 0 reserved for padding)
    """
    print(f"Original interactions: {len(interactions)}")

    # Count interactions
    user_counts = defaultdict(int)
    item_counts = defaultdict(int)

    for user_id, item_id, _ in interactions:
        user_counts[user_id] += 1
        item_counts[item_id] += 1

    print(f"Original users: {len(user_counts)}, items: {len(item_counts)}")

    # Filter by popularity
    valid_items = set(item_counts.keys())
    if max_items is not None and len(valid_items) > max_items:
        # Keep only top-K most popular items
        top_items = [item for item, _ in Counter(item_counts).most_common(max_items)]
        valid_items = set(top_items)
        print(f"Keeping top {max_items} most popular items")

    # Filter by minimum interactions
    valid_users = {u for u, c in user_counts.items() if c >= min_user_interactions}
    valid_items = {i for i in valid_items if item_counts[i] >= min_item_interactions}

    # Iteratively filter until stable (users may drop below threshold after item filtering)
    prev_size = -1
    while len(valid_users) != prev_size:
        prev_size = len(valid_users)

        # Filter interactions
        filtered = [
            (u, i, t) for u, i, t in interactions
            if u in valid_users and i in valid_items
        ]

        # Recount
        user_counts_new = defaultdict(int)
        for u, _, _ in filtered:
            user_counts_new[u] += 1

        valid_users = {u for u, c in user_counts_new.items() if c >= min_user_interactions}

    filtered_interactions = [
        (u, i, t) for u, i, t in interactions
        if u in valid_users and i in valid_items
    ]

    print(f"After filtering: {len(filtered_interactions)} interactions")
    print(f"Remaining users: {len(valid_users)}, items: {len(valid_items)}")

    # Create mappings (continuous IDs)
    user_mapping = {u: idx for idx, u in enumerate(sorted(valid_users))}
    item_mapping = {i: idx + 1 for idx, i in enumerate(sorted(valid_items))}  # 1-indexed

    return filtered_interactions, user_mapping, item_mapping


def create_sequences(
    interactions: list[tuple[int, int, float]],
    user_mapping: dict[int, int],
    item_mapping: dict[int, int],
) -> dict[int, list[int]]:
    """
    Create user interaction sequences sorted by timestamp.

    Args:
        interactions: List of (user_id, item_id, timestamp) tuples
        user_mapping: User ID mapping
        item_mapping: Item ID mapping

    Returns:
        Dictionary mapping new_user_id -> sequence of new_item_ids (chronologically sorted)
    """
    user_sequences = defaultdict(list)

    for user_id, item_id, timestamp in interactions:
        if user_id in user_mapping and item_id in item_mapping:
            new_user_id = user_mapping[user_id]
            new_item_id = item_mapping[item_id]
            user_sequences[new_user_id].append((timestamp, new_item_id))

    # Sort each user's sequence by timestamp
    for user_id in user_sequences:
        user_sequences[user_id].sort(key=lambda x: x[0])
        # Keep only item IDs
        user_sequences[user_id] = [item_id for _, item_id in user_sequences[user_id]]

    return dict(user_sequences)


def train_val_test_split(
    user_sequences: dict[int, list[int]],
) -> tuple[list[list[int]], list[int], list[list[int]], list[int], list[list[int]], list[int]]:
    """
    Split sequences into train/val/test sets.

    For each user:
    - Last item: test target
    - Second-to-last item: validation target
    - All previous items: training sequence

    Args:
        user_sequences: Dictionary mapping user_id -> sequence

    Returns:
        train_sequences, train_targets, val_sequences, val_targets, test_sequences, test_targets
    """
    train_sequences = []
    train_targets = []
    val_sequences = []
    val_targets = []
    test_sequences = []
    test_targets = []

    for user_id, sequence in user_sequences.items():
        if len(sequence) < 3:
            # Need at least 3 items for train/val/test split
            continue

        # Test: last item
        test_sequences.append(sequence[:-1])
        test_targets.append(sequence[-1])

        # Validation: second-to-last item
        val_sequences.append(sequence[:-2])
        val_targets.append(sequence[-2])

        # Training: all items except last two as targets
        for i in range(1, len(sequence) - 2):
            train_sequences.append(sequence[:i])
            train_targets.append(sequence[i])

    return (
        train_sequences, train_targets,
        val_sequences, val_targets,
        test_sequences, test_targets
    )


def save_processed_data(
    train_sequences: list[list[int]],
    train_targets: list[int],
    val_sequences: list[list[int]],
    val_targets: list[int],
    test_sequences: list[list[int]],
    test_targets: list[int],
    num_users: int,
    num_items: int,
    output_path: str,
):
    """Save processed data to JSON file"""
    data = {
        "train": {
            "sequences": train_sequences,
            "targets": train_targets,
        },
        "val": {
            "sequences": val_sequences,
            "targets": val_targets,
        },
        "test": {
            "sequences": test_sequences,
            "targets": test_targets,
        },
        "num_users": num_users,
        "num_items": num_items,
    }

    with open(output_path, "w") as f:
        json.dump(data, f)

    print(f"Processed data saved to {output_path}")
    print(f"Users: {num_users}, Items: {num_items}")
    print(f"Train: {len(train_sequences)}, Val: {len(val_sequences)}, Test: {len(test_sequences)}")


def preprocess_interactions(
    interactions: list[tuple[int, int, float]],
    min_user_interactions: int = 5,
    min_item_interactions: int = 5,
    max_items: int | None = None,
    output_path: str | None = None,
) -> dict:
    """
    Complete preprocessing pipeline from raw interactions to train/val/test splits.

    This is the main entry point for data preprocessing. It handles:
    1. Filtering users/items by interaction count
    2. Optionally limiting to top-K most popular items
    3. Creating chronologically-sorted sequences
    4. Splitting into train/val/test sets
    5. Optionally saving to JSON

    Args:
        interactions: List of (user_id, item_id, timestamp) tuples
            - user_id: any hashable ID (will be remapped to 0-indexed)
            - item_id: any hashable ID (will be remapped to 1-indexed, 0 reserved for padding)
            - timestamp: any sortable value (unix timestamp, datetime, int, etc.)
        min_user_interactions: Minimum interactions per user (default: 5)
        min_item_interactions: Minimum interactions per item (default: 5)
        max_items: If set, keep only top-K most popular items (useful for large datasets)
        output_path: If provided, save processed data to this JSON file

    Returns:
        Dictionary with keys:
            - "train": {"sequences": [...], "targets": [...]}
            - "val": {"sequences": [...], "targets": [...]}
            - "test": {"sequences": [...], "targets": [...]}
            - "num_users": int
            - "num_items": int
            - "user_mapping": dict (original_user_id -> new_user_id)
            - "item_mapping": dict (original_item_id -> new_item_id)

    Example:
        >>> interactions = [
        ...     (0, 101, 1000.0),  # user 0 interacted with item 101 at time 1000
        ...     (0, 102, 1001.0),
        ...     (0, 103, 1002.0),
        ...     (1, 102, 2000.0),
        ...     # ... more interactions
        ... ]
        >>> data = preprocess_interactions(
        ...     interactions,
        ...     min_user_interactions=5,
        ...     max_items=50000,  # Keep only top 50K items
        ... )
        >>> print(f"Train samples: {len(data['train']['sequences'])}")
    """
    print("="*60)
    print("Preprocessing interactions")
    print("="*60)

    # Step 1: Filter interactions
    print("\n[1/4] Filtering interactions...")
    filtered_interactions, user_mapping, item_mapping = filter_interactions(
        interactions,
        min_user_interactions=min_user_interactions,
        min_item_interactions=min_item_interactions,
        max_items=max_items,
    )

    # Step 2: Create sequences
    print("\n[2/4] Creating sequences...")
    user_sequences = create_sequences(
        filtered_interactions,
        user_mapping,
        item_mapping,
    )
    print(f"Created sequences for {len(user_sequences)} users")

    # Step 3: Split train/val/test
    print("\n[3/4] Splitting train/val/test...")
    (train_seqs, train_tgts,
     val_seqs, val_tgts,
     test_seqs, test_tgts) = train_val_test_split(user_sequences)

    print(f"Train samples: {len(train_seqs)}")
    print(f"Val samples: {len(val_seqs)}")
    print(f"Test samples: {len(test_seqs)}")

    # Step 4: Save if requested
    if output_path is not None:
        print("\n[4/4] Saving to file...")
        save_processed_data(
            train_seqs, train_tgts,
            val_seqs, val_tgts,
            test_seqs, test_tgts,
            num_users=len(user_mapping),
            num_items=len(item_mapping),
            output_path=output_path,
        )

    print("\n" + "="*60)
    print("Preprocessing complete!")
    print("="*60)

    return {
        "train": {"sequences": train_seqs, "targets": train_tgts},
        "val": {"sequences": val_seqs, "targets": val_tgts},
        "test": {"sequences": test_seqs, "targets": test_tgts},
        "num_users": len(user_mapping),
        "num_items": len(item_mapping),
        "user_mapping": user_mapping,
        "item_mapping": item_mapping,
    }
