"""Sequence Encoder using self-attention to encode historical user interactions"""

import torch
import torch.nn as nn


class SequenceEncoder(nn.Module):
    """
    Encodes user interaction sequences using self-attention (Transformer).

    From paper Equation 8:
        es = SA(E) = Softmax((EW_Q)(EW_K)^T / √d)(EW_V)

    Where:
        E = [e1+p1, e2+p2, ..., e_{n-1}+p_{n-1}] ∈ R^{(n-1)×d}
        ei: item embedding for item i
        pi: positional embedding for position i
        es: encoded sequence representations (used as conditioning)
        SA: self-attention with autoregressive masking

    Architecture:
    - Item embeddings + positional embeddings
    - Multi-layer self-attention (Transformer encoder)
    - Autoregressive masking to prevent attending to future items

    Input: sequence of item IDs
    Output: encoded sequence representations for conditioning the decoder
    """

    def __init__(
        self,
        num_items: int,
        embedding_dim: int = 128,
        num_layers: int = 1,
        num_heads: int = 4,
        attention_dropout: float = 0.2,
        hidden_dropout: float = 0.0,
        max_seq_len: int = 20,
        padding_idx: int = 0,
    ):
        """
        Args:
            num_items: Total number of items in the dataset
            embedding_dim: Dimension of item embeddings
            num_layers: Number of transformer encoder layers
            num_heads: Number of attention heads
            attention_dropout: Dropout probability for attention weights (authors: 0.2)
            hidden_dropout: Dropout probability for hidden states (authors: 0.0)
            max_seq_len: Maximum sequence length
            padding_idx: Index used for padding tokens
        """
        super().__init__()

        self.num_items = num_items
        self.embedding_dim = embedding_dim
        self.max_seq_len = max_seq_len
        self.padding_idx = padding_idx

        # Item embeddings (randomly initialized, learned during training)
        # num_items already includes special tokens (padding + mask)
        self.item_embedding = nn.Embedding(
            num_items,
            embedding_dim,
            padding_idx=padding_idx
        )

        # Positional embeddings
        self.position_embedding = nn.Embedding(max_seq_len, embedding_dim)

        # Transformer encoder layers
        # Authors use dim_feedforward=hidden_size (1x, not 4x), we keep 4x for now
        # but set hidden_dropout separately to match authors' behavior
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=num_heads,
            dim_feedforward=embedding_dim * 4,  # We use 4x, authors use 1x
            dropout=attention_dropout,  # This sets the default for all dropouts initially
            activation="gelu",
            batch_first=True,
        )

        # Configure separate dropout rates to match authors' implementation
        # Authors: attention_probs_dropout_prob=0.2, hidden_dropout_prob=0.0
        encoder_layer.dropout1.p = attention_dropout  # Dropout after self-attention
        encoder_layer.dropout2.p = hidden_dropout     # Dropout after FFN
        encoder_layer.dropout.p = hidden_dropout      # Dropout in FFN activation

        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
        )

        self.dropout = nn.Dropout(hidden_dropout)
        self.layer_norm = nn.LayerNorm(embedding_dim)

        self._init_weights()

    def _init_weights(self):
        """Initialize embeddings with normal distribution"""
        nn.init.normal_(self.item_embedding.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.position_embedding.weight, mean=0.0, std=0.02)

    def forward(self, item_seq: torch.Tensor, padding_mask: torch.Tensor) -> torch.Tensor:
        """
        Encode item sequence into representations.

        Args:
            item_seq: (batch_size, seq_len) item IDs
            padding_mask: (batch_size, seq_len) True = data, False = padding

        Returns:
            encoded_seq: (batch_size, seq_len, embedding_dim) encoded representations
        """
        batch_size, seq_len = item_seq.size()

        # Get item embeddings
        item_emb = self.item_embedding(item_seq)  # (batch_size, seq_len, embedding_dim)

        # Add positional embeddings
        positions = torch.arange(seq_len, device=item_seq.device).unsqueeze(0).expand(batch_size, -1)
        pos_emb = self.position_embedding(positions)

        # Combine embeddings
        x = self.dropout(item_emb + pos_emb)
        x = self.layer_norm(x)

        # Create causal mask (autoregressive)
        # Boolean mask: True = positions to ignore (future), False = allowed (past/present)
        causal_mask = nn.Transformer.generate_square_subsequent_mask(
            seq_len, device=item_seq.device, dtype=torch.bool
        )

        # Create padding mask for transformer: True = padding (ignore), False = data (attend)
        src_key_padding_mask = ~padding_mask

        # Apply transformer encoder
        # Use is_causal=True as a hint for better performance
        encoded_seq = self.transformer_encoder(
            x,
            mask=causal_mask,
            src_key_padding_mask=src_key_padding_mask,
            is_causal=True,
        )

        return encoded_seq
