"""Step-Wise Diffuser for introducing and removing Gaussian noise"""

import torch
import torch.nn as nn
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler


class StepWiseDiffuser(nn.Module):
    """
    Implements the diffusion process for CDDRec.

    Forward diffusion: Progressively adds Gaussian noise to target embeddings
    Reverse diffusion: Iteratively denoises to generate predictions

    Uses diffusers library for noise scheduling (tested implementations).
    """

    def __init__(
        self,
        num_diffusion_steps: int,
        schedule_type: str = "linear",
        max_beta: float = 0.1,
    ):
        """
        Args:
            num_diffusion_steps: Number of diffusion steps (T)
            schedule_type: 'linear' or 'cosine'
            max_beta: Maximum noise variance for linear schedule
        """
        super().__init__()

        self.num_diffusion_steps = num_diffusion_steps

        # Initialize noise scheduler from diffusers
        if schedule_type == "linear":
            self.scheduler = DDPMScheduler(
                num_train_timesteps=num_diffusion_steps,
                beta_schedule="linear",
                beta_start=0.0001,
                beta_end=max_beta,
                clip_sample=False,
            )
        elif schedule_type == "cosine":
            self.scheduler = DDPMScheduler(
                num_train_timesteps=num_diffusion_steps,
                beta_schedule="squaredcos_cap_v2",
                clip_sample=False,
            )
        else:
            raise ValueError(f"Unknown schedule: {schedule_type}")

        # Extract noise schedule parameters
        self.betas = self.scheduler.betas
        self.alphas = self.scheduler.alphas
        self.alphas_cumprod = self.scheduler.alphas_cumprod

    def add_noise(
        self,
        x_0: torch.Tensor,
        t: torch.Tensor,
        noise: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Forward diffusion: Add noise to clean embeddings.

        Following DDPM (Ho et al., 2020), the forward diffusion process is:
            q(x_t | x_0) = N(x_t; sqrt(α̅_t) * x_0, (1 - α̅_t) * I)

        Using the reparameterization trick:
            x_t = sqrt(α̅_t) * x_0 + sqrt(1 - α̅_t) * ε, where ε ~ N(0, I)

        Where:
            α̅_t = ∏_{i=1}^t α_i (cumulative product of alphas)
            α_t = 1 - β_t (one minus beta at step t)
            β_t: noise schedule variance at step t

        Args:
            x_0: (batch_size, seq_len, embedding_dim) clean target embeddings
            t: (batch_size,) timestep indices (0 to T-1)
            noise: Optional pre-generated noise

        Returns:
            x_t: (batch_size, seq_len, embedding_dim) noised embeddings
        """
        if noise is None:
            noise = torch.randn_like(x_0)

        # Get sqrt(α̅_t) and sqrt(1 - α̅_t)
        sqrt_alpha_prod = self.alphas_cumprod[t] ** 0.5
        sqrt_one_minus_alpha_prod = (1 - self.alphas_cumprod[t]) ** 0.5

        # Reshape for broadcasting: (batch_size, 1, 1)
        sqrt_alpha_prod = sqrt_alpha_prod.view(-1, 1, 1)
        sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.view(-1, 1, 1)

        # Reparameterization: x_t = sqrt(α̅_t) * x_0 + sqrt(1 - α̅_t) * ε
        x_t = sqrt_alpha_prod * x_0 + sqrt_one_minus_alpha_prod * noise

        return x_t

    def forward_diffusion(
        self,
        target_embeddings: torch.Tensor,
        timestep: int,
    ) -> torch.Tensor:
        """
        Apply forward diffusion at a specific timestep.

        Args:
            target_embeddings: (batch_size, seq_len, embedding_dim) clean target embeddings
            timestep: Diffusion timestep (0 to T-1)

        Returns:
            noised_embeddings: (batch_size, seq_len, embedding_dim) noised embeddings
        """
        batch_size = target_embeddings.size(0)
        device = target_embeddings.device

        # Create timestep tensor for entire batch
        timesteps = torch.full((batch_size,), timestep, device=device, dtype=torch.long)

        # Add noise at this timestep
        noised_embeddings = self.add_noise(target_embeddings, timesteps)

        return noised_embeddings

    @torch.no_grad()
    def reverse_diffusion(
        self,
        decoder: nn.Module,
        encoded_seq: torch.Tensor,
        seq_len: int,
        embedding_dim: int,
    ) -> torch.Tensor:
        """
        Reverse diffusion for inference: denoise from random noise to predictions.

        Implements the reverse process from DDPM:
            p_θ(x_{t-1} | x_t, es) = N(x_{t-1}; μ_θ(x_t, t, es), Σ_θ(x_t, t))

        Where the posterior mean is computed as:
            μ_θ = (1/sqrt(α_t)) * (x_t - (β_t/sqrt(1-α̅_t)) * (x_t - sqrt(α̅_t) * x̂_0))

        For CDDRec, x̂_0 is predicted by the conditional decoder μ_θ(es, t)
        conditioned on encoded sequence es.

        The process starts from x_T ~ N(0, I) and iteratively denoises
        to generate x_0.

        Args:
            decoder: Conditional denoising decoder model
            encoded_seq: (batch_size, seq_len, embedding_dim) encoded sequence
            seq_len: Sequence length for output
            embedding_dim: Dimension of embeddings

        Returns:
            x_0_pred: (batch_size, seq_len, embedding_dim) predicted target embeddings
        """
        batch_size = encoded_seq.size(0)
        device = encoded_seq.device

        # Start from random Gaussian noise
        x_t = torch.randn(batch_size, seq_len, embedding_dim, device=device)

        # Iteratively denoise from T-1 to 0
        for t in reversed(range(self.num_diffusion_steps)):
            # Create timestep tensor
            timestep = torch.full((batch_size,), t, device=device, dtype=torch.long)

            # Predict x_0 (mean of target embedding) using decoder
            x_0_pred = decoder(x_t, timestep, encoded_seq)

            if t > 0:
                # Compute x_{t-1} using DDPM reverse process
                # This is a simplified version; full DDPM includes variance term
                alpha_t = self.alphas[t]
                alpha_bar_t = self.alphas_cumprod[t]
                beta_t = self.betas[t]

                # Mean of p(x_{t-1} | x_t, x_0)
                coef1 = (1 - alpha_t) / (1 - alpha_bar_t) ** 0.5
                mean = (x_t - coef1 * (x_t - (alpha_bar_t ** 0.5) * x_0_pred)) / (alpha_t ** 0.5)

                # Add noise (posterior variance)
                noise = torch.randn_like(x_t)
                variance = beta_t ** 0.5
                x_t = mean + variance * noise
            else:
                # Final step: no noise
                x_t = x_0_pred

        return x_t
