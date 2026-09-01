"""
DDPM (Denoising Diffusion Probabilistic Models) Implementation.

This module implements the core diffusion process:
    - Forward process: gradually add noise to images
    - Reverse process: iteratively denoise to generate images
    - Training: predict noise added to clean images


Two noise schedules are supported:
    1. Linear: β_t increases linearly (original DDPM paper)
    2. Cosine: smoother schedule (Improved DDPM paper) - better quality
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List
import math


class DDPM(nn.Module):
    """
    Denoising Diffusion Probabilistic Model.
    
    Wraps a noise prediction network (U-Net) and implements the forward/reverse
    diffusion process with configurable noise schedules.
    
    Args:
        model: Noise prediction network (U-Net) that takes (x_t, t) → ε_θ
        timesteps: Number of diffusion steps T (e.g., 1000)
        beta_schedule: Type of noise schedule ('linear' or 'cosine')
        beta_start: Starting noise level for linear schedule (e.g., 1e-4)
        beta_end: Ending noise level for linear schedule (e.g., 0.02)
    """
    
    def __init__(
        self,
        model: nn.Module,
        timesteps: int = 1000,
        beta_schedule: str = 'linear',
        beta_start: float = 1e-4,
        beta_end: float = 0.02,
    ):
        super().__init__()
        
        self.model = model
        self.timesteps = timesteps
        self.beta_schedule = beta_schedule
        
        # Get noise schedule β_t
        betas = self._get_beta_schedule(beta_schedule, timesteps, beta_start, beta_end)
        
        # Precompute all derived quantities as buffers
        # These are fixed tensors that move with the model but don't need gradients
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0) 
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)  
        
        # For forward process
        self.register_buffer('betas', betas)
        self.register_buffer('alphas', alphas)
        self.register_buffer('alphas_cumprod', alphas_cumprod)
        self.register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1.0 - alphas_cumprod))
        
        # For reverse process
        self.register_buffer('sqrt_recip_alphas', torch.sqrt(1.0 / alphas))
        
        # Posterior variance:
        posterior_variance = (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod) * betas
        self.register_buffer('posterior_variance', posterior_variance)
        
        # For p_sample: coefficients for the mean
        self.register_buffer(
            'posterior_mean_coef1',
            betas * torch.sqrt(alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        )
        self.register_buffer(
            'posterior_mean_coef2',
            (1.0 - alphas_cumprod_prev) * torch.sqrt(alphas) / (1.0 - alphas_cumprod)
        )
        
        # For numerical stability: clamp to avoid division by zero or log(0)
        self.register_buffer(
            'sqrt_alphas_cumprod_clamped',
            torch.clamp(torch.sqrt(alphas_cumprod), min=1e-4)
        )
    
    def _get_beta_schedule(
        self,
        schedule: str,
        timesteps: int,
        beta_start: float,
        beta_end: float
    ) -> torch.Tensor:
        """
        Compute noise schedule β_t.
        
        Args:
            schedule: 'linear' or 'cosine'
            timesteps: Number of steps T
            beta_start: Starting β (for linear)
            beta_end: Ending β (for linear)
        
        Returns:
            Tensor of shape [T] containing β_t values
        """
        if schedule == 'linear':
            # Linear schedule:
            betas = torch.linspace(beta_start, beta_end, timesteps)
        
        elif schedule == 'cosine':
            # Cosine schedule from "Improved DDPM" paper
            s = 0.008
            steps = torch.arange(timesteps + 1, dtype=torch.float64)
            alpha_bar = torch.cos(((steps / timesteps + s) / (1 + s)) * math.pi / 2) ** 2
            alpha_bar = alpha_bar / alpha_bar[0]  # Normalize
            
            betas = 1 - (alpha_bar[1:] / alpha_bar[:-1])
            betas = torch.clamp(betas, max=0.999)  # Prevent β from being too large
            betas = betas.float()
        
        else:
            raise ValueError(f"Unknown beta schedule: {schedule}. Must be 'linear' or 'cosine'.")
        
        return betas
    
    def q_sample(
        self,
        x_0: torch.Tensor,
        t: torch.Tensor,
        noise: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward process: add noise to clean image at timestep t.
        
        
        This closed-form formula allows us to jump directly to any timestep t
        without simulating all intermediate steps.
        
        Args:
            x_0: Clean images [B, C, H, W]
            t: Timesteps [B] (integers in [0, T-1])
            noise: Optional noise ε ~ N(0, I). If None, sampled randomly.
        
        Returns:
            Noisy images x_t [B, C, H, W]
        """
        if noise is None:
            noise = torch.randn_like(x_0)
        
        # Reshape coefficients for broadcasting: [B] → [B, 1, 1, 1]
        sqrt_alpha_cumprod = self.sqrt_alphas_cumprod[t].view(-1, 1, 1, 1)
        sqrt_one_minus_alpha_cumprod = self.sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1, 1)
        
        return sqrt_alpha_cumprod * x_0 + sqrt_one_minus_alpha_cumprod * noise
    
    def p_sample(self, x_t: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Single reverse step: denoise x_t by one step to get x_{t-1}.
        

        Args:
            x_t: Noisy images [B, C, H, W]
            t: Timesteps [B] (integers in [0, T-1])
        
        Returns:
            Less noisy images x_{t-1} [B, C, H, W]
        """
        # Predict noise using U-Net
        eps_theta = self.model(x_t, t)
        
        # Compute mean of the reverse distribution
        sqrt_recip_alpha = self.sqrt_recip_alphas[t].view(-1, 1, 1, 1)
        posterior_mean_coef1 = self.posterior_mean_coef1[t].view(-1, 1, 1, 1)
        posterior_mean_coef2 = self.posterior_mean_coef2[t].view(-1, 1, 1, 1)
        
        mean = sqrt_recip_alpha * (
            x_t - posterior_mean_coef1 * eps_theta
        )
        
        # Add noise (except at t=0)
        # At t=0, we want the final output to be deterministic
        if t[0] > 0:
            posterior_var = self.posterior_variance[t].view(-1, 1, 1, 1)
            noise = torch.randn_like(x_t)
            return mean + torch.sqrt(posterior_var) * noise
        else:
            # At t=0, no noise added
            return mean
    
    @torch.no_grad()
    def sample(
        self,
        num_samples: int,
        device: torch.device,
        save_intermediate: bool = False,
        save_every: int = 100
    ) -> torch.Tensor:
        """
        Generate images by starting from pure noise and iteratively denoising.
        
        This is the full reverse process: x_T → x_{T-1} → ... → x_0
        
        Args:
            num_samples: Number of images to generate
            device: Device to use
            save_intermediate: If True, save intermediate denoising steps
            save_every: Save every N steps (for visualization)
        
        Returns:
            Generated images [num_samples, C, H, W] in [-1, 1] range
        """
        self.eval()
        
        # Start from pure Gaussian noise: x_T ~ N(0, I)
        # We need to infer image shape from the model
        # Assume model expects 3-channel 32x32 images (CIFAR-10)
        img_shape = (num_samples, self.model.in_channels, 32, 32)
        x = torch.randn(img_shape, device=device)
        
        intermediates = [] if save_intermediate else None
        
        # Iteratively denoise: x_T → x_{T-1} → ... → x_0
        for i in reversed(range(self.timesteps)):
            t = torch.full((num_samples,), i, device=device, dtype=torch.long)
            x = self.p_sample(x, t)
            
            # Save intermediate steps for visualization
            if save_intermediate and i % save_every == 0:
                intermediates.append(x.cpu().clone())
        
        # Save final image
        if save_intermediate:
            intermediates.append(x.cpu().clone())
        
        if save_intermediate:
            return x, intermediates
        else:
            return x
    
    def compute_loss(self, x_0: torch.Tensor) -> torch.Tensor:
        """
        Compute training loss: MSE between predicted noise and actual noise.
        
        
        Args:
            x_0: Clean images [B, C, H, W]
        
        Returns:
            Scalar loss tensor
        """
        batch_size = x_0.shape[0]
        device = x_0.device
        
        # Sample random timesteps for each image in the batch
        t = torch.randint(0, self.timesteps, (batch_size,), device=device)
        
        # Sample noise
        noise = torch.randn_like(x_0)
        
        # Add noise to get x_t
        x_t = self.q_sample(x_0, t, noise)
        
        # Predict noise using U-Net
        noise_pred = self.model(x_t, t)
        
        # MSE loss
        loss = F.mse_loss(noise_pred, noise)
        
        return loss
    
    def forward(self, x_0: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for training (alias for compute_loss).
        
        Args:
            x_0: Clean images [B, C, H, W]
        
        Returns:
            Scalar loss tensor
        """
        return self.compute_loss(x_0)


# =============================================================================
# Testing
# =============================================================================

if __name__ == '__main__':
    from src.models.unet import UNet
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create U-Net
    unet = UNet(
        in_channels=3,
        out_channels=3,
        unet_channels=[64, 128, 256],
        time_emb_dim=128,
        num_res_blocks=2,
        use_attention=True,
        dropout=0.1,
    ).to(device)
    
    # Test both noise schedules
    for schedule in ['linear', 'cosine']:
        print(f"\n{'='*60}")
        print(f"Testing {schedule} schedule")
        print('='*60)
        
        ddpm = DDPM(
            model=unet,
            timesteps=1000,
            beta_schedule=schedule,
            beta_start=1e-4,
            beta_end=0.02,
        ).to(device)
        
        # Check buffers
        print(f"Betas range: [{ddpm.betas[0]:.6f}, {ddpm.betas[-1]:.6f}]")
        print(f"Alphas_cumprod range: [{ddpm.alphas_cumprod[0]:.6f}, {ddpm.alphas_cumprod[-1]:.6f}]")
        
        # Test forward process
        x_0 = torch.randn(2, 3, 32, 32, device=device)
        t = torch.randint(0, 1000, (2,), device=device)
        
        x_t = ddpm.q_sample(x_0, t)
        print(f"Forward process: x_0 shape {x_0.shape} → x_t shape {x_t.shape}")
        
        # Test loss computation
        loss = ddpm.compute_loss(x_0)
        print(f"Loss: {loss.item():.4f}")
        
        # Test sampling (just 10 steps for speed)
        ddpm.timesteps = 10  # Reduce for testing
        x_gen, intermediates = ddpm.sample(
            num_samples=2,
            device=device,
            save_intermediate=True,
            save_every=2
        )
        print(f"Sampling: generated {x_gen.shape[0]} images")
        print(f"Intermediate steps saved: {len(intermediates)}")
        print(f"Output range: [{x_gen.min():.3f}, {x_gen.max():.3f}]")
        
        # Reset timesteps
        ddpm.timesteps = 1000
    
    print("\n✓ All DDPM tests passed!")