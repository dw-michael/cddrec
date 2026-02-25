"""Recommender wrapper for CDDRec inference."""

import torch
from typing import Any
from cddrec.models import CDDRec
from cddrec.data import load_processed_data, load_id_mappings, create_reverse_mappings


class Recommender:
    """
    Inference wrapper for CDDRec model.

    Handles ID mapping, history lookup, and recommendation generation.
    Accepts original IDs (strings/ints) and returns original IDs.

    Examples:
        >>> # From preprocessing result (has mappings + histories)
        >>> data = preprocess_interactions(interactions)
        >>> model = CDDRec(num_items=data["num_items"], ...)
        >>> recommender = Recommender(model, data)
        >>> recs = recommender.recommend("alice", k=10)

        >>> # From saved files
        >>> model = load_model(...)
        >>> recommender = Recommender.from_files(
        ...     model=model,
        ...     mappings_path="data/beauty_mappings.json",
        ...     data_path="data/beauty.json",
        ... )
        >>> recs = recommender.recommend("user_12345", k=10)

        >>> # New user (cold start)
        >>> recs = recommender.recommend_with_history(
        ...     user_history=["item_a", "item_b", "item_c"],
        ...     k=10,
        ... )
    """

    def __init__(
        self,
        model: CDDRec,
        data: dict[str, Any] | None = None,
        user_mapping: dict[str | int, int] | None = None,
        item_mapping: dict[str | int, int] | None = None,
        device: str = "cpu",
    ):
        """
        Initialize recommender.

        Args:
            model: Trained CDDRec model
            data: PreprocessingResult from preprocessing (includes mappings + data)
            user_mapping: Original user_id → model user_id (if not using data)
            item_mapping: Original item_id → model item_id (if not using data)
            device: Device for inference

        Note: Either provide `data` OR both `user_mapping` and `item_mapping`.
        """
        self.model = model
        self.device = torch.device(device)
        self.model.to(self.device)
        self.model.eval()

        # Extract mappings
        if data is not None:
            self.user_mapping = data.get("user_mapping", {})
            self.item_mapping = data.get("item_mapping", {})
            # Build user history lookup from test set
            self.user_histories = self._build_user_histories(data)
        else:
            self.user_mapping = user_mapping or {}
            self.item_mapping = item_mapping or {}
            self.user_histories = {}

        # Create reverse mappings (model ID → original ID)
        self.reverse_user = {v: k for k, v in self.user_mapping.items()}
        self.reverse_item = {v: k for k, v in self.item_mapping.items()}

        # Model config
        self.max_seq_len = model.encoder.max_seq_len
        self.num_items = model.num_items

    @classmethod
    def from_files(
        cls,
        model: CDDRec,
        mappings_path: str,
        data_path: str | None = None,
        device: str = "cpu",
    ) -> "Recommender":
        """
        Load recommender from saved files.

        Args:
            model: Trained CDDRec model
            mappings_path: Path to ID mappings JSON (e.g., "data/beauty_mappings.json")
            data_path: Optional path to data JSON for user history lookup
            device: Device for inference

        Returns:
            Recommender instance

        Example:
            >>> model = CDDRec(...)
            >>> load_checkpoint(model, "checkpoints/best.pth")
            >>> recommender = Recommender.from_files(
            ...     model=model,
            ...     mappings_path="data/beauty_mappings.json",
            ...     data_path="data/beauty.json",
            ... )
        """
        # Load ID mappings
        user_mapping, item_mapping = load_id_mappings(mappings_path)

        # Optionally load data for history lookup
        data_dict = None
        if data_path is not None:
            processed_data = load_processed_data(data_path)
            # Combine with mappings for history building
            data_dict = {
                "user_mapping": user_mapping,
                "item_mapping": item_mapping,
                "test": processed_data["test"],
                "num_users": processed_data["num_users"],
                "num_items": processed_data["num_items"],
            }

        return cls(
            model=model,
            data=data_dict,
            user_mapping=user_mapping if data_dict is None else None,
            item_mapping=item_mapping if data_dict is None else None,
            device=device,
        )

    def recommend(
        self,
        user_id: str | int,
        k: int = 10,
        exclude_seen: bool = True,
        return_original_ids: bool = True,
    ) -> list[str | int]:
        """
        Generate top-K recommendations for an existing user.

        User history is looked up automatically from the data. For new users,
        use `recommend_with_history()` instead.

        Args:
            user_id: User ID (original ID)
            k: Number of recommendations
            exclude_seen: Don't recommend items user has already seen
            return_original_ids: Return original IDs (vs model IDs)

        Returns:
            List of recommended item IDs

        Example:
            >>> recs = recommender.recommend("alice", k=10)
            >>> # ['item_xyz', 'item_abc', ...]
        """
        # Convert user ID to model ID
        if user_id not in self.user_mapping:
            raise ValueError(
                f"User '{user_id}' not found in data. "
                f"Use recommend_with_history() for new users."
            )

        model_user_id = self.user_mapping[user_id]

        # Look up user history
        if model_user_id not in self.user_histories:
            raise ValueError(
                f"User '{user_id}' has no history. "
                f"Use recommend_with_history() for new users."
            )

        user_history = self.user_histories[model_user_id]

        # Generate recommendations
        return self._recommend_from_history(
            item_history=user_history,
            k=k,
            exclude_seen=user_history if exclude_seen else None,
            return_original_ids=return_original_ids,
        )

    def recommend_with_history(
        self,
        user_history: list[str | int],
        k: int = 10,
        exclude_seen: bool = True,
        return_original_ids: bool = True,
        are_model_ids: bool = False,
    ) -> list[str | int]:
        """
        Generate recommendations given explicit user history.

        Use this for new users or custom histories.

        Args:
            user_history: List of item IDs (chronologically ordered)
            k: Number of recommendations
            exclude_seen: Don't recommend items in history
            return_original_ids: Return original IDs (vs model IDs)
            are_model_ids: If True, history contains model IDs (skip conversion)

        Returns:
            List of recommended item IDs

        Example:
            >>> # New user with browsing history
            >>> recs = recommender.recommend_with_history(
            ...     user_history=["item_a", "item_b", "item_c"],
            ...     k=10,
            ... )
        """
        # Convert history to model IDs if needed
        if not are_model_ids:
            model_history = []
            for item_id in user_history:
                if item_id not in self.item_mapping:
                    # Skip unknown items
                    continue
                model_history.append(self.item_mapping[item_id])
        else:
            model_history = user_history

        if len(model_history) == 0:
            raise ValueError("User history is empty after filtering unknown items.")

        # Generate recommendations
        return self._recommend_from_history(
            item_history=model_history,
            k=k,
            exclude_seen=model_history if exclude_seen else None,
            return_original_ids=return_original_ids,
        )

    def predict_scores(
        self,
        user_id: str | int | None = None,
        user_history: list[str | int] | None = None,
        return_original_ids: bool = True,
    ) -> dict[str | int, float]:
        """
        Get prediction scores for all items.

        Either `user_id` or `user_history` must be provided.

        Args:
            user_id: User ID (for existing users)
            user_history: Item history (for new users)
            return_original_ids: Return original IDs in dict keys

        Returns:
            Dictionary mapping item_id → score

        Example:
            >>> scores = recommender.predict_scores(user_id="alice")
            >>> # {'item_a': 0.95, 'item_b': 0.87, ...}
        """
        if user_id is None and user_history is None:
            raise ValueError("Must provide either user_id or user_history")

        # Get history
        if user_id is not None:
            model_user_id = self.user_mapping[user_id]
            item_history = self.user_histories[model_user_id]
        else:
            # Convert history to model IDs
            item_history = [self.item_mapping[iid] for iid in user_history if iid in self.item_mapping]

        # Prepare sequence
        item_seq, padding_mask = self._prepare_sequence(item_history)

        # Get scores
        with torch.no_grad():
            scores = self.model.forward_inference(item_seq, padding_mask)
            scores = scores[0].cpu()  # (num_items,)

        # Convert to dictionary
        score_dict = {}
        for model_item_id in range(1, self.num_items + 1):  # 1-indexed
            score = scores[model_item_id - 1].item()
            if return_original_ids:
                original_id = self.reverse_item.get(model_item_id)
                if original_id is not None:
                    score_dict[original_id] = score
            else:
                score_dict[model_item_id] = score

        return score_dict

    def _recommend_from_history(
        self,
        item_history: list[int],  # Model IDs
        k: int,
        exclude_seen: list[int] | None,  # Model IDs
        return_original_ids: bool,
    ) -> list[str | int]:
        """Core recommendation logic."""

        # Prepare sequence
        item_seq, padding_mask = self._prepare_sequence(item_history)

        # Get scores from model
        with torch.no_grad():
            scores = self.model.forward_inference(item_seq, padding_mask)
            scores = scores[0]  # (num_items,)

        # Filter out seen items if requested
        if exclude_seen is not None:
            for item_id in exclude_seen:
                if 1 <= item_id <= self.num_items:  # Valid model item ID (1-indexed)
                    scores[item_id - 1] = float('-inf')

        # Get top-K
        k_clamped = min(k, self.num_items)
        top_k_scores, top_k_indices = torch.topk(scores, k=k_clamped)
        top_k_model_ids = (top_k_indices + 1).cpu().tolist()  # Convert to 1-indexed

        # Convert to original IDs if requested
        if return_original_ids:
            return [self.reverse_item[model_id] for model_id in top_k_model_ids]
        else:
            return top_k_model_ids

    def _prepare_sequence(self, item_history: list[int]) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Pad or truncate sequence to max_seq_len and create padding mask.

        Args:
            item_history: List of model item IDs

        Returns:
            item_seq: (1, max_seq_len) padded sequence tensor
            padding_mask: (1, max_seq_len) boolean mask (True for valid positions)
        """
        # Truncate if too long
        if len(item_history) > self.max_seq_len:
            sequence = item_history[-self.max_seq_len:]
        else:
            sequence = item_history

        # Pad to max_seq_len
        seq_len = len(sequence)
        padded = sequence + [0] * (self.max_seq_len - seq_len)

        # Create tensors
        item_seq = torch.tensor([padded], dtype=torch.long, device=self.device)
        padding_mask = item_seq != 0  # True for valid positions, False for padding

        return item_seq, padding_mask

    def _build_user_histories(self, data: dict) -> dict[int, list[int]]:
        """
        Build user history lookup from test set.

        Uses test set because it has the most complete history for each user.

        Args:
            data: PreprocessingResult or loaded data with test split

        Returns:
            Dictionary mapping model_user_id → list of model_item_ids
        """
        user_histories = {}

        if "test" not in data:
            return user_histories

        test_sequences = data["test"]["sequences"]
        test_user_ids = data["test"]["user_ids"]

        for user_id, sequence in zip(test_user_ids, test_sequences):
            # Test sequences are full (include target), so we can use them as-is
            # But for recommendations, we typically want to exclude the target
            # So store the full history and let recommend() decide whether to exclude
            user_histories[user_id] = sequence

        return user_histories
