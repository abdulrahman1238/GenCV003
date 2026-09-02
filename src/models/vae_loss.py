import torch
import torch.nn.functional as F


def vae_loss(
    x_recon: torch.Tensor, 
    x: torch.Tensor, 
    mu: torch.Tensor, 
    logvar: torch.Tensor,
    recon_loss_type: str = 'bce',
    kl_weight: float = 1.0
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compute VAE loss = Reconstruction Loss + KL Divergence
    
    Args:
        x_recon: Reconstructed images [B, C, H, W]
        x: Original images [B, C, H, W]
        mu: Latent means [B, latent_dim]
        logvar: Log variances [B, latent_dim]
        recon_loss_type: 'bce' or 'mse'
        kl_weight: Weight for KL term (for KL annealing)
    
    Returns:
        total_loss: Combined loss
        recon_loss: Reconstruction component
        kl_loss: KL divergence component
    """
    # Reconstruction loss
    if recon_loss_type == 'bce':
        # Binary Cross-Entropy (expects inputs in [0,1])
        recon_loss = F.binary_cross_entropy(x_recon, x, reduction='mean') 
    elif recon_loss_type == 'mse':
        # Mean Squared Error
        recon_loss = F.mse_loss(x_recon, x, reduction='mean')
    else:
        raise ValueError(f"Unknown recon_loss_type: {recon_loss_type}")
    
    # KL Divergence: -0.5 * sum(1 + log(σ²) - μ² - σ²)
    # This is the analytical KL between N(μ, σ²) and N(0, I)
    kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp()) 
    
    # Total loss
    total_loss = recon_loss + kl_weight * kl_loss
    
    return total_loss, recon_loss, kl_loss
