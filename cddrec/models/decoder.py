"""Conditional Denoising Decoder using TransformerDecoderLayer"""

import torch
import torch.nn as nn


class ConditionalDenoisingDecoder(nn.Module):
    """
    Generates target item embeddings conditioned on encoded sequences and timestep.

    UPDATED: Now uses PyTorch's TransformerDecoderLayer to match the authors' implementation.

    Authors' implementation (models.py:32, 218):
        self.decoder = nn.TransformerDecoderLayer(...)
        model_output = self.decoder(conditional_emb, time_emb)

    Where:
        - tgt = conditional_emb: encoded sequence (B, S, D) - flows through to output
        - memory = time_emb: timestep embeddings (B, S, D) - provides conditioning

    The TransformerDecoderLayer:
        1. Self-attention on tgt (conditional_emb attends to itself)
        2. Cross-attention: Q=tgt (after self-attn), K/V=memory (time_emb)
        3. Feed-forward network
        4. Returns refined tgt (same shape)

    This is MORE expressive than our previous simple cross-attention because:
        - Self-attention allows the encoded sequence to refine itself
        - Then cross-attends with time information
        - The decoder does heavy lifting to generate good predictions
    """

    def __init__(
        self,
        embedding_dim: int = 128,
        num_layers: int = 1,
        num_heads: int = 4,
        dropout: float = 0.2,
        num_diffusion_steps: int = 20,
    ):
        """
        Args:
            embedding_dim: Dimension of embeddings
            num_layers: Number of decoder layers (authors use 1)
            num_heads: Number of attention heads (authors use 4)
            dropout: Dropout probability (authors use 0.2 for attention, 0.0 for hidden)
            num_diffusion_steps: Number of diffusion steps (for timestep embedding)
        """
        super().__init__()

        self.embedding_dim = embedding_dim
        self.num_diffusion_steps = num_diffusion_steps

        # Timestep embedding: learnable embedding table (as per paper)
        # "Initially, we acquire a learnable embedding et for the indicator t
        # from a time lookup embedding table" (paper lines 610-611)
        self.time_embed = nn.Embedding(num_diffusion_steps, embedding_dim)

        # TransformerDecoderLayer (matching authors' implementation)
        # Authors use dim_feedforward=hidden_size (1x expansion, not 4x)
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=embedding_dim,
            nhead=num_heads,
            dim_feedforward=embedding_dim,  # Authors use 1x (not 4x like standard)
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )

        # Stack decoder layers (authors use num_layers=1, but support more)
        self.decoder = nn.TransformerDecoder(
            decoder_layer,
            num_layers=num_layers,
        )

        # Layer norm for output (authors use this)
        self.layer_norm = nn.LayerNorm(embedding_dim)

    def forward(
        self,
        t: torch.Tensor,
        encoded_seq: torch.Tensor,
        padding_mask: torch.Tensor | None = None,
        target_seq_len: int | None = None,
    ) -> torch.Tensor:
        """
        Forward pass matching authors' implementation.

        Args:
            t: (batch_size,) timestep indices
            encoded_seq: (batch_size, seq_len, embedding_dim) encoded sequence from encoder
            padding_mask: (batch_size, seq_len) True = valid, False = padding
            target_seq_len: Optional target sequence length (defaults to encoded_seq length)

        Returns:
            x_pred: (batch_size, seq_len, embedding_dim) predicted target embeddings
        """
        batch_size = encoded_seq.size(0)
        seq_len = encoded_seq.size(1)

        if target_seq_len is None:
            target_seq_len = seq_len

        # Embed timestep: (batch_size, embedding_dim)
        t_emb = self.time_embed(t)

        # Expand timestep to sequence length: (batch_size, seq_len, embedding_dim)
        # This matches what authors do: time_ids = t.unsqueeze(1).expand(...)
        time_emb = t_emb.unsqueeze(1).expand(-1, seq_len, -1)

        # Convert padding_mask for PyTorch convention
        # Our convention: True = valid, False = padding
        # PyTorch expects: True = masked (ignored), False = valid
        if padding_mask is not None:
            # Invert: True → False (valid), False → True (masked)
            key_padding_mask = ~padding_mask
        else:
            key_padding_mask = None

        # TransformerDecoder forward:
        # tgt=conditional_emb (encoded sequence)
        # memory=time_emb (timestep embeddings)
        #
        # Flow:
        # 1. Self-attention: conditional_emb attends to itself
        # 2. Cross-attention: Q=conditional_emb', K/V=time_emb
        # 3. FFN
        # 4. Output: refined embeddings (same shape as tgt)
        output = self.decoder(
            tgt=encoded_seq,
            memory=time_emb,
            tgt_key_padding_mask=key_padding_mask,
            memory_key_padding_mask=key_padding_mask,
        )

        # Apply final layer norm (authors do this)
        output = self.layer_norm(output)

        return output
