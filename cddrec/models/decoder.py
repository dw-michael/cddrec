"""Conditional Denoising Decoder with cross-attention mechanism"""

import torch
import torch.nn as nn
import math


class ConditionalDenoisingDecoder(nn.Module):
    """
    Generates target item embeddings by denoising conditioned on encoded sequences.

    From paper Equations 9-10:
        μ_θ(es, t) = CA(es, e_t) = Softmax((e_t W_Q)(es W_K)^T / √d)(es W_V)
        x̂_t = μ_θ(es, t) + √β̂_t ε, where ε ~ N(0, I)

    Where:
        es: encoded sequence from encoder
        e_t: timestep embedding (sinusoidal + MLP)
        CA: cross-attention mechanism
        μ_θ: predicted mean of target embedding
        β̂_t: noise variance at step t

    Key mechanism: Direct conditioning on encoder output at every denoising step
    (not on previous denoising predictions). Cross-attention where:
    - Query: timestep embedding e_t
    - Key/Value: encoded sequence representations es
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

        # Timestep embedding (sinusoidal + learned projection)
        self.time_embed = nn.Sequential(
            SinusoidalPositionEmbedding(embedding_dim),
            nn.Linear(embedding_dim, embedding_dim),
            nn.GELU(),
            nn.Linear(embedding_dim, embedding_dim),
        )

        # Project noised input to query space
        self.input_proj = nn.Linear(embedding_dim, embedding_dim)

        # Cross-attention layers
        self.cross_attention_layers = nn.ModuleList([
            CrossAttentionLayer(
                embedding_dim=embedding_dim,
                num_heads=num_heads,
                dropout=dropout,
            )
            for _ in range(num_layers)
        ])

        # Output projection to predict x_0 (mean of target embedding)
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
        x_t: torch.Tensor,
        t: torch.Tensor,
        encoded_seq: torch.Tensor,
        padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Denoise x_t conditioned on encoded sequence to predict x_0.

        Args:
            x_t: (batch_size, seq_len, embedding_dim) noised embeddings at step t
            t: (batch_size,) timestep indices
            encoded_seq: (batch_size, seq_len, embedding_dim) encoded sequence from encoder
            padding_mask: (batch_size, seq_len) boolean mask for valid positions

        Returns:
            x_0_pred: (batch_size, seq_len, embedding_dim) predicted mean of target embeddings
        """
        batch_size, seq_len, _ = x_t.shape

        # Embed timestep: (batch_size, embedding_dim)
        t_emb = self.time_embed(t)

        # Broadcast timestep embedding across sequence: (batch_size, seq_len, embedding_dim)
        t_emb = t_emb.unsqueeze(1).expand(-1, seq_len, -1)

        # Project noised input: (batch_size, seq_len, embedding_dim)
        x = self.input_proj(x_t)

        # Combine with timestep embedding
        query = x + t_emb  # (batch_size, seq_len, embedding_dim)
        query = self.layer_norm(query)

        # Apply cross-attention layers
        for layer in self.cross_attention_layers:
            query = layer(query, encoded_seq, padding_mask)

        # Predict x_0: (batch_size, seq_len, embedding_dim)
        x_0_pred = self.output_proj(query)

        return x_0_pred


class SinusoidalPositionEmbedding(nn.Module):
    """Sinusoidal timestep embedding as used in Transformer and DDPM"""

    def __init__(self, embedding_dim: int):
        super().__init__()
        self.embedding_dim = embedding_dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        Args:
            t: (batch_size,) timestep indices

        Returns:
            embeddings: (batch_size, embedding_dim)
        """
        device = t.device
        half_dim = self.embedding_dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = t.float()[:, None] * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)

        # Handle odd embedding dimensions
        if self.embedding_dim % 2 == 1:
            emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=-1)

        return emb


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
            query: (batch_size, query_len, embedding_dim) queries from noised targets
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
