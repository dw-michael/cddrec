"""Conditional Denoising Decoder with cross-attention mechanism"""

import torch
import torch.nn as nn


class ConditionalDenoisingDecoder(nn.Module):
    """
    Generates target item embeddings conditioned ONLY on encoded sequences and timestep.

    CRITICAL: This decoder does NOT take noised embeddings (x_t) as input!

    From paper Equation 9 (Section 4.2):
        μ_θ(es, t) = CA(es, e_t) = Softmax((e_t W_Q)(es W_K)^T / √d)(es W_V)

    The decoder predicts x̂_t directly from:
        - es: encoded sequence from encoder
        - e_t: timestep embedding (learnable embedding table)

    The noised embeddings x_t from the diffuser are used ONLY as targets for
    loss computation, NOT as inputs to this decoder.

    Architecture:
    - Query: timestep embedding e_t (expanded to target sequence length)
    - Key/Value: encoded sequence representations es
    - Causal masking for autoregressive generation
    - Output: predicted target embeddings x̂_t
    """

    def __init__(
        self,
        embedding_dim: int = 128,
        num_layers: int = 2,
        num_heads: int = 2,
        dropout: float = 0.2,
        num_diffusion_steps: int = 30,
    ):
        """
        Args:
            embedding_dim: Dimension of embeddings
            num_layers: Number of cross-attention layers
            num_heads: Number of attention heads
            dropout: Dropout probability
            num_diffusion_steps: Number of diffusion steps (for timestep embedding)
        """
        super().__init__()

        self.embedding_dim = embedding_dim
        self.num_diffusion_steps = num_diffusion_steps

        # Timestep embedding: learnable embedding table (as per paper)
        # "Initially, we acquire a learnable embedding et for the indicator t
        # from a time lookup embedding table" (paper lines 610-611)
        # This creates the query for cross-attention
        self.time_embed = nn.Embedding(num_diffusion_steps, embedding_dim)

        # Cross-attention layers
        # Query: timestep embedding, Key/Value: encoded sequence
        self.cross_attention_layers = nn.ModuleList([
            CrossAttentionLayer(
                embedding_dim=embedding_dim,
                num_heads=num_heads,
                dropout=dropout,
            )
            for _ in range(num_layers)
        ])

        # Output projection to predict target embeddings
        self.output_proj = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embedding_dim, embedding_dim),
        )

        self.layer_norm = nn.LayerNorm(embedding_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        t: torch.Tensor,
        encoded_seq: torch.Tensor,
        padding_mask: torch.Tensor | None = None,
        target_seq_len: int | None = None,
    ) -> torch.Tensor:
        """
        Predict target embeddings conditioned on encoded sequence and timestep.

        Args:
            t: (batch_size,) timestep indices
            encoded_seq: (batch_size, seq_len, embedding_dim) encoded sequence from encoder
            padding_mask: (batch_size, seq_len) boolean mask (True = valid, False = padding)
            target_seq_len: Length of output sequence (defaults to encoded_seq length)

        Returns:
            x_pred: (batch_size, target_seq_len, embedding_dim) predicted target embeddings
        """
        batch_size = encoded_seq.size(0)
        if target_seq_len is None:
            target_seq_len = encoded_seq.size(1)

        # Embed timestep: (batch_size, embedding_dim)
        t_emb = self.time_embed(t)

        # Expand timestep embedding to target sequence length
        # This serves as the query for cross-attention
        query = t_emb.unsqueeze(1).expand(-1, target_seq_len, -1)
        query = self.layer_norm(query)

        # Apply cross-attention layers
        # Query: timestep embedding, Key/Value: encoded sequence
        for layer in self.cross_attention_layers:
            query = layer(query, encoded_seq, padding_mask)

        # Predict target embeddings: (batch_size, target_seq_len, embedding_dim)
        x_pred = self.output_proj(query)

        return x_pred


class CrossAttentionLayer(nn.Module):
    """Single cross-attention layer with residual connection"""

    def __init__(self, embedding_dim: int, num_heads: int, dropout: float):
        super().__init__()

        self.cross_attention = nn.MultiheadAttention(
            embed_dim=embedding_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

        self.feed_forward = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embedding_dim * 4, embedding_dim),
        )

        self.layer_norm1 = nn.LayerNorm(embedding_dim)
        self.layer_norm2 = nn.LayerNorm(embedding_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        query: torch.Tensor,
        key_value: torch.Tensor,
        key_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Args:
            query: (batch_size, query_len, embedding_dim) queries from timestep embedding
            key_value: (batch_size, seq_len, embedding_dim) encoded sequence
            key_padding_mask: (batch_size, seq_len) mask for padding

        Returns:
            output: (batch_size, query_len, embedding_dim)
        """
        query_len = query.size(1)
        key_len = key_value.size(1)

        # Create causal mask for autoregressive cross-attention only when
        # query and key have the same sequence length (training case).
        # During inference, query_len=1 and key_len=seq_len, so we skip the causal mask
        # and allow the single query position to attend to the full history.
        causal_mask = None
        use_is_causal = False
        if query_len == key_len:
            # Position i in query can only attend to positions 0...i in key_value
            # Use boolean mask: True for positions to ignore (future), False for allowed (past/present)
            causal_mask = nn.Transformer.generate_square_subsequent_mask(
                query_len, device=query.device, dtype=torch.bool
            )
            use_is_causal = True

        # Convert key_padding_mask for PyTorch MultiheadAttention convention
        # Input key_padding_mask: True = valid, False = padding
        # PyTorch expects: True = masked, False = valid
        # Keep as boolean to match causal_mask dtype and avoid warnings
        padding_mask = None
        if key_padding_mask is not None:
            padding_mask = ~key_padding_mask  # Invert: True = ignore (padding), False = valid

        attn_output, _ = self.cross_attention(
            query=query,
            key=key_value,
            value=key_value,
            attn_mask=causal_mask,  # Causal mask for autoregressive property (training only)
            key_padding_mask=padding_mask,  # Padding mask
            is_causal=use_is_causal,  # Hint for better performance when using causal mask
        )

        query = self.layer_norm1(query + self.dropout(attn_output))

        # Feed-forward with residual
        ff_output = self.feed_forward(query)
        query = self.layer_norm2(query + self.dropout(ff_output))

        return query
