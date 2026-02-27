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

        # Extract noise schedule parameters and register as buffers
        # Buffers are non-trainable tensors that move with the model (CPU/GPU)
        self.register_buffer("betas", self.scheduler.betas)
        self.register_buffer("alphas", self.scheduler.alphas)
        self.register_buffer("alphas_cumprod", self.scheduler.alphas_cumprod)

        # Precompute sqrt(beta) for efficient sampling in Equation 10
        self.register_buffer("sqrt_betas", self.scheduler.betas ** 0.5)

    def sample_prediction(
        self,
        mean: torch.Tensor,
        t: torch.Tensor,
        noise: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Sample predicted target embeddings from the decoder's output using reparameterization trick.

        From paper Equation 10:
            x̂nt = μ_θn + √(β_t) * ε,  ε ~ N(0, I)

        Where:
            μ_θn: predicted mean from decoder (conditioned on encoded sequence and timestep)
            β_t: variance from noise schedule at timestep t
            x̂nt: sampled prediction that gets compared to the corrupted target in losses

        This is the crucial step that converts the decoder's predicted mean into a
        sampled prediction by adding noise according to the variance schedule.

        Args:
            mean: (batch_size, seq_len, embedding_dim) predicted mean from decoder
            t: (batch_size,) timestep indices (0 to T-1)
            noise: Optional pre-generated noise

        Returns:
            x_pred_sampled: (batch_size, seq_len, embedding_dim) sampled predictions
        """
        if noise is None:
            noise = torch.randn_like(mean)

        # Get precomputed sqrt(β_t) for the given timesteps
        sqrt_beta = self.sqrt_betas[t]

        # Reshape for broadcasting: (batch_size, 1, 1)
        sqrt_beta = sqrt_beta.view(-1, 1, 1)

        # Reparameterization: x̂_t = μ_θ + sqrt(β_t) * ε
        x_pred_sampled = mean + sqrt_beta * noise

        return x_pred_sampled

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

        Using the reparameterization trick (Equation 11 in paper):
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
        padding_mask: torch.Tensor | None,
        target_seq_len: int,
    ) -> torch.Tensor:
        """
        Reverse diffusion for inference: iteratively denoise from random noise.

        NOTE: This iterative denoising approach is currently NOT used in inference.
        Instead, we perform direct prediction at t=0 to avoid accumulating bias
        during step-wise inference. This method is kept for experimentation.

        IMPORTANT: In CDDRec, the decoder does NOT take x_t as input. This method
        implements an experimental alternative inference approach where we:
        1. Predict x̂_0 from (t, es) at each step
        2. Use x̂_0 to compute the posterior mean for x_{t-1}
        3. Sample x_{t-1} and repeat

        This is different from standard DDPM where the network predicts noise from x_t.

        Args:
            decoder: Conditional denoising decoder model (signature: (t, encoded_seq, mask, target_seq_len))
            encoded_seq: (batch_size, seq_len, embedding_dim) encoded sequence
            padding_mask: (batch_size, seq_len) boolean mask for valid positions
            target_seq_len: Length of output sequence

        Returns:
            x_0_pred: (batch_size, target_seq_len, embedding_dim) predicted target embeddings
        """
        batch_size = encoded_seq.size(0)
        device = encoded_seq.device

        # Start from random Gaussian noise
        x_t = torch.randn(batch_size, target_seq_len, self.embedding_dim, device=device)

        # Iteratively denoise from T-1 to 0
        for t in reversed(range(self.num_diffusion_steps)):
            # Create timestep tensor
            timestep = torch.full((batch_size,), t, device=device, dtype=torch.long)

            # Predict x_0 directly from (t, es)
            # Note: decoder does NOT take x_t as input
            x_0_pred = decoder(timestep, encoded_seq, padding_mask, target_seq_len)

            if t > 0:
                # Compute x_{t-1} using DDPM posterior
                # q(x_{t-1} | x_t, x_0) mean formula
                alpha_t = self.alphas[t]
                alpha_bar_t = self.alphas_cumprod[t]
                alpha_bar_t_prev = self.alphas_cumprod[t-1] if t > 0 else torch.tensor(1.0, device=device)
                beta_t = self.betas[t]

                # Posterior mean: μ̃ = (√α̅_{t-1}β_t)/(1-α̅_t) x_0 + (√α_t(1-α̅_{t-1}))/(1-α̅_t) x_t
                coef_x0 = (alpha_bar_t_prev ** 0.5 * beta_t) / (1 - alpha_bar_t)
                coef_xt = (alpha_t ** 0.5 * (1 - alpha_bar_t_prev)) / (1 - alpha_bar_t)
                mean = coef_x0 * x_0_pred + coef_xt * x_t

                # Posterior variance: β̃_t = (1-α̅_{t-1})/(1-α̅_t) β_t
                variance = ((1 - alpha_bar_t_prev) / (1 - alpha_bar_t)) * beta_t
                noise = torch.randn_like(x_t)
                x_t = mean + (variance ** 0.5) * noise
            else:
                # Final step: no noise
                x_t = x_0_pred

        return x_t
