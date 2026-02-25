"""CDDRec: Main model combining encoder, decoder, and diffuser"""

import torch
import torch.nn as nn

from .encoder import SequenceEncoder
from .decoder import ConditionalDenoisingDecoder
from .diffuser import StepWiseDiffuser


class CDDRec(nn.Module):
    """
    CDDRec: Conditional Denoising Diffusion for Sequential Recommendation

    Combines three components:
    1. SequenceEncoder: Encodes historical interactions
    2. ConditionalDenoisingDecoder: Denoises conditioned on encoded sequence
    3. StepWiseDiffuser: Manages forward and reverse diffusion process
    """

    def __init__(
        self,
        num_items: int,
        embedding_dim: int = 128,
        encoder_layers: int = 2,
        decoder_layers: int = 2,
        num_heads: int = 2,
        dropout: float = 0.2,
        max_seq_len: int = 20,
        num_diffusion_steps: int = 30,
        noise_schedule: str = "linear",
        max_beta: float = 0.1,
        padding_idx: int = 0,
    ):
        """
        Args:
            num_items: Total number of items in the dataset
            embedding_dim: Dimension of embeddings
            encoder_layers: Number of transformer encoder layers
            decoder_layers: Number of cross-attention decoder layers
            num_heads: Number of attention heads
            dropout: Dropout probability
            max_seq_len: Maximum sequence length
            num_diffusion_steps: Number of diffusion steps (T)
            noise_schedule: 'linear' or 'cosine'
            max_beta: Maximum noise variance for linear schedule
            padding_idx: Index used for padding tokens
        """
        super().__init__()

        # Store hyperparameters for serialization/logging
        self.config = {
            'num_items': num_items,
            'embedding_dim': embedding_dim,
            'encoder_layers': encoder_layers,
            'decoder_layers': decoder_layers,
            'num_heads': num_heads,
            'dropout': dropout,
            'max_seq_len': max_seq_len,
            'num_diffusion_steps': num_diffusion_steps,
            'noise_schedule': noise_schedule,
            'max_beta': max_beta,
            'padding_idx': padding_idx,
        }

        self.num_items = num_items
        self.embedding_dim = embedding_dim
        self.num_diffusion_steps = num_diffusion_steps
        self.padding_idx = padding_idx

        # Sequence Encoder
        self.encoder = SequenceEncoder(
            num_items=num_items,
            embedding_dim=embedding_dim,
            num_layers=encoder_layers,
            num_heads=num_heads,
            dropout=dropout,
            max_seq_len=max_seq_len,
            padding_idx=padding_idx,
        )

        # Conditional Denoising Decoder
        self.decoder = ConditionalDenoisingDecoder(
            embedding_dim=embedding_dim,
            num_layers=decoder_layers,
            num_heads=num_heads,
            dropout=dropout,
            num_diffusion_steps=num_diffusion_steps,
        )

        # Step-Wise Diffuser
        self.diffuser = StepWiseDiffuser(
            num_diffusion_steps=num_diffusion_steps,
            schedule_type=noise_schedule,
            max_beta=max_beta,
        )

        # Access item embeddings from encoder for scoring
        self.item_embeddings = self.encoder.item_embedding

    def forward_train(
        self,
        sequence: torch.Tensor,
        timestep: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass for training at a specific diffusion timestep.

        Args:
            sequence: (batch_size, seq_len) full sequence (input + target combined)
            timestep: Diffusion timestep (0 to T-1)

        Returns:
            x_0_pred: (batch_size, seq_len-1, embedding_dim) predicted target embeddings
            x_t: (batch_size, seq_len-1, embedding_dim) noised target embeddings
            input_mask: (batch_size, seq_len-1) True = data, False = padding
            target_mask: (batch_size, seq_len-1) True = data, False = padding
        """
        batch_size = sequence.size(0)
        device = sequence.device

        # Slice sequence: input = [:-1], target = [1:]
        input_seq = sequence[:, :-1]  # (batch_size, seq_len-1)
        target_seq = sequence[:, 1:]  # (batch_size, seq_len-1)

        # Create masks for loss computation: True = data, False = padding
        input_mask = input_seq != self.padding_idx  # (batch_size, seq_len-1)
        target_mask = target_seq != self.padding_idx  # (batch_size, seq_len-1)

        # 1. Encode input sequence
        encoded_seq = self.encoder(input_seq, input_mask)

        # 2. Get target embeddings for all positions
        target_embeddings = self.item_embeddings(target_seq)  # (batch_size, seq_len-1, embedding_dim)

        # 3. Apply forward diffusion at this timestep
        x_t = self.diffuser.forward_diffusion(target_embeddings, timestep)

        # 4. Create timestep tensor for decoder
        timesteps = torch.full((batch_size,), timestep, device=device, dtype=torch.long)

        # 5. Predict x_0 using decoder
        x_0_pred = self.decoder(x_t, timesteps, encoded_seq, input_mask)

        return x_0_pred, x_t, input_mask, target_mask

    @torch.no_grad()
    def forward_inference(
        self,
        item_seq: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass for inference (generation).

        Args:
            item_seq: (batch_size, seq_len) historical item IDs

        Returns:
            scores: (batch_size, num_items) prediction scores for all items
        """
        # Create mask: True = data, False = padding
        padding_mask = item_seq != self.padding_idx

        # 1. Encode historical sequence
        encoded_seq = self.encoder(item_seq, padding_mask)

        # 2. Generate next item embedding via reverse diffusion
        # Generate single prediction for next item
        x_0_pred = self.diffuser.reverse_diffusion(
            decoder=self.decoder,
            encoded_seq=encoded_seq,
            seq_len=1,
            embedding_dim=self.embedding_dim,
        )  # (batch_size, 1, embedding_dim)

        # Squeeze to (batch_size, embedding_dim)
        x_0_pred = x_0_pred.squeeze(1)

        # 3. Compute scores for all items
        # x_0_pred: (batch_size, embedding_dim)
        # item_embeddings: (num_items + 1, embedding_dim)
        all_item_embeddings = self.item_embeddings.weight[1:]  # Skip padding token

        # Compute dot product: (batch_size, num_items)
        scores = torch.matmul(x_0_pred, all_item_embeddings.T)

        return scores

    def get_target_embedding(self, target_items: torch.Tensor) -> torch.Tensor:
        """
        Get embeddings for target items.

        Args:
            target_items: (batch_size,) or (batch_size, seq_len) target item IDs

        Returns:
            embeddings: (batch_size, embedding_dim) or (batch_size, seq_len, embedding_dim)
        """
        return self.item_embeddings(target_items)

    def get_all_item_embeddings(self) -> torch.Tensor:
        """Get embeddings for all items (excluding padding)"""
        return self.item_embeddings.weight[1:]  # Skip padding token

    def forward(
        self,
        sequence: torch.Tensor | None = None,
        item_seq: torch.Tensor | None = None,
        timestep: int | None = None,
        mode: str = "train",
    ):
        """
        Unified forward pass.

        Args:
            sequence: (batch_size, seq_len) full sequence (for training mode)
            item_seq: (batch_size, seq_len) input sequence (for inference mode)
            timestep: Diffusion timestep (for training mode)
            mode: 'train' or 'inference'

        Returns:
            Training mode: (x_0_pred, x_t, input_mask, target_mask)
            Inference mode: scores
        """
        if mode == "train":
            if sequence is None:
                raise ValueError("sequence required for training mode")
            if timestep is None:
                raise ValueError("timestep required for training mode")
            return self.forward_train(sequence, timestep)
        elif mode == "inference":
            if item_seq is None:
                raise ValueError("item_seq required for inference mode")
            return self.forward_inference(item_seq)
        else:
            raise ValueError(f"Unknown mode: {mode}")

    def to_config(self) -> dict:
        """
        Export model hyperparameters for serialization/logging.

        Returns:
            Dictionary containing all model hyperparameters
        """
        return self.config.copy()

    @classmethod
    def from_config(cls, config: dict) -> "CDDRec":
        """
        Reconstruct model from saved configuration.

        Args:
            config: Dictionary containing model hyperparameters

        Returns:
            CDDRec model instance with specified configuration

        Example:
            >>> config = model.to_config()
            >>> # Save config to JSON
            >>> with open("model_config.json", "w") as f:
            ...     json.dump(config, f)
            >>> # Later, reload
            >>> with open("model_config.json", "r") as f:
            ...     config = json.load(f)
            >>> model = CDDRec.from_config(config)
        """
        return cls(**config)
