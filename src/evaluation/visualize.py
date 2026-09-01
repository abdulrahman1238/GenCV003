import torch
import torchvision.utils as vutils
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from typing import Optional


def denormalize_images(images: torch.Tensor, normalize_range: str) -> torch.Tensor:
    """
    Convert images from model output range to [0, 1] for display.
    
    Args:
        images: Images from model
        normalize_range: '01' if already in [0,1], '-11' if in [-1,1]
    
    Returns:
        Images in [0, 1] range, clamped to handle any overflow
    """
    if normalize_range == '01':
        return images.clamp(0, 1)
    elif normalize_range == '-11':
        # Convert from [-1, 1] to [0, 1]
        return ((images + 1.0) / 2.0).clamp(0, 1)
    else:
        raise ValueError(f"Unknown normalize_range: {normalize_range}")


def save_sample_grid(
    model: torch.nn.Module,
    num_samples: int = 64,
    nrow: int = 8,
    device: torch.device = torch.device('cuda'),
    model_normalize_range: str = '01',
    save_path: Optional[Path] = None
) -> plt.Figure:
    """
    Generate and save a grid of random samples.
    
    Args:
        model_normalize_range: Range of model output ('01' or '-11')
    """
    model.eval()
    actual_model = model.module if hasattr(model, 'module') else model
    
    with torch.no_grad():
        samples = actual_model.sample(num_samples, device)
    
    # Denormalize to [0, 1] for display
    samples = denormalize_images(samples, model_normalize_range).cpu()
    grid = vutils.make_grid(samples, nrow=nrow, padding=2, normalize=False)
    
    fig, ax = plt.subplots(figsize=(12, 12))
    ax.imshow(grid.permute(1, 2, 0))
    ax.axis('off')
    ax.set_title('Random Samples', fontsize=16, pad=20)
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=150)
        print(f"Sample grid saved to {save_path}")
    
    return fig


def save_reconstruction_comparison(
    model: torch.nn.Module,
    real_images: torch.Tensor,
    num_images: int = 8,
    device: torch.device = torch.device('cuda'),
    model_normalize_range: str = '01',
    save_path: Optional[Path] = None
) -> plt.Figure:
    """
    Compare real images with their reconstructions.
    
    Args:
        real_images: Real images in [0, 1] range
        model_normalize_range: Range of model output ('01' or '-11')
    """
    model.eval()
    actual_model = model.module if hasattr(model, 'module') else model
    
    # Take first num_images (already in [0, 1])
    real_batch = real_images[:num_images].to(device)
    
    # For models expecting [-1, 1], we need to normalize the input
    if model_normalize_range == '-11':
        model_input = real_batch * 2.0 - 1.0  # [0,1] → [-1,1]
    else:
        model_input = real_batch
    
    with torch.no_grad():
        # VAE returns (recon, mu, logvar); DDPM might return just recon
        output = actual_model(model_input)
        if isinstance(output, tuple):
            recon_batch = output[0]
        else:
            recon_batch = output
    
    # Denormalize reconstruction to [0, 1]
    recon_batch = denormalize_images(recon_batch, model_normalize_range).cpu()
    
    # Combine real and reconstructed
    comparison = torch.cat([real_batch.cpu(), recon_batch], dim=0)
    grid = vutils.make_grid(comparison, nrow=num_images, padding=2, normalize=False)
    
    fig, ax = plt.subplots(figsize=(16, 4))
    ax.imshow(grid.permute(1, 2, 0))
    ax.axis('off')
    ax.set_title('Real (top) vs Reconstruction (bottom)', fontsize=14, pad=10)
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=150)
        print(f"Reconstruction comparison saved to {save_path}")
    
    return fig


def save_latent_interpolation(
    model: torch.nn.Module,
    num_steps: int = 10,
    device: torch.device = torch.device('cuda'),
    model_normalize_range: str = '01',
    save_path: Optional[Path] = None
) -> plt.Figure:
    """
    Show smooth interpolation in the latent space.
    
    NOTE: This is VAE-specific. DDPM doesn't have a structured latent space
    in the same way, so this visualization is only meaningful for VAEs.
    """
    model.eval()
    actual_model = model.module if hasattr(model, 'module') else model
    
    # Check if model has an encoder (VAE-specific)
    if not hasattr(actual_model, 'encoder'):
        print("Warning: Model has no encoder. Skipping latent interpolation.")
        return None
    
    latent_dim = actual_model.encoder.fc_mu.out_features
    
    z1 = torch.randn(1, latent_dim, device=device)
    z2 = torch.randn(1, latent_dim, device=device)
    
    alphas = torch.linspace(0, 1, num_steps, device=device)
    interpolated = []
    
    with torch.no_grad():
        for alpha in alphas:
            z_interp = (1 - alpha) * z1 + alpha * z2
            x_interp = actual_model.decoder(z_interp)
            interpolated.append(x_interp.cpu())
    
    interpolated = torch.cat(interpolated, dim=0)
    interpolated = denormalize_images(interpolated, model_normalize_range)
    grid = vutils.make_grid(interpolated, nrow=num_steps, padding=2, normalize=False)
    
    fig, ax = plt.subplots(figsize=(16, 2))
    ax.imshow(grid.permute(1, 2, 0))
    ax.axis('off')
    ax.set_title('Latent Space Interpolation (z1 → z2)', fontsize=14, pad=10)
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=150)
        print(f"Latent interpolation saved to {save_path}")
    
    return fig


def save_latent_space_walk(
    model: torch.nn.Module,
    latent_dim_to_vary: int = 0,
    num_steps: int = 10,
    range_min: float = -2.0,
    range_max: float = 2.0,
    device: torch.device = torch.device('cuda'),
    model_normalize_range: str = '01',
    save_path: Optional[Path] = None
) -> plt.Figure:
    """
    Walk along a single dimension of the latent space (VAE-specific).
    """
    model.eval()
    actual_model = model.module if hasattr(model, 'module') else model
    
    if not hasattr(actual_model, 'encoder'):
        print("Warning: Model has no encoder. Skipping latent walk.")
        return None
    
    latent_dim = actual_model.encoder.fc_mu.out_features
    z_base = torch.zeros(1, latent_dim, device=device)
    
    values = torch.linspace(range_min, range_max, num_steps, device=device)
    images = []
    
    with torch.no_grad():
        for val in values:
            z = z_base.clone()
            z[0, latent_dim_to_vary] = val
            x = actual_model.decoder(z)
            images.append(x.cpu())
    
    images = torch.cat(images, dim=0)
    images = denormalize_images(images, model_normalize_range)
    grid = vutils.make_grid(images, nrow=num_steps, padding=2, normalize=False)
    
    fig, ax = plt.subplots(figsize=(16, 2))
    ax.imshow(grid.permute(1, 2, 0))
    ax.axis('off')
    ax.set_title(f'Latent Dimension {latent_dim_to_vary} Walk [{range_min}, {range_max}]', 
                 fontsize=14, pad=10)
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=150)
        print(f"Latent walk saved to {save_path}")
    
    return fig


def generate_all_visualizations(
    model: torch.nn.Module,
    real_images: torch.Tensor,
    output_dir: Path,
    device: torch.device = torch.device('cuda'),
    model_normalize_range: str = '01'
):
    """
    Generate all visualization plots.
    
    Args:
        real_images: Real images in [0, 1] range
        model_normalize_range: Range of model output ('01' or '-11')
    """
    print("=" * 60)
    print("Generating Visualizations")
    print("=" * 60)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Random samples
    save_sample_grid(
        model, 
        num_samples=64, 
        device=device,
        model_normalize_range=model_normalize_range,
        save_path=output_dir / 'sample_grid.png'
    )
    plt.close()
    
    # 2. Reconstruction comparison
    save_reconstruction_comparison(
        model,
        real_images,
        num_images=8,
        device=device,
        model_normalize_range=model_normalize_range,
        save_path=output_dir / 'reconstruction_comparison.png'
    )
    plt.close()
    
    # 3. Latent interpolation (VAE-specific)
    actual_model = model.module if hasattr(model, 'module') else model
    if hasattr(actual_model, 'encoder'):
        save_latent_interpolation(
            model,
            num_steps=10,
            device=device,
            model_normalize_range=model_normalize_range,
            save_path=output_dir / 'latent_interpolation.png'
        )
        plt.close()
        
        # 4. Latent dimension walks (show first 4 dimensions)
        latent_dim = actual_model.encoder.fc_mu.out_features
        for dim in range(min(4, latent_dim)):
            save_latent_space_walk(
                model,
                latent_dim_to_vary=dim,
                device=device,
                model_normalize_range=model_normalize_range,
                save_path=output_dir / f'latent_walk_dim_{dim}.png'
            )
            plt.close()
    else:
        print("Skipping latent space visualizations (model has no encoder).")
    
    print(f"All visualizations saved to {output_dir}")