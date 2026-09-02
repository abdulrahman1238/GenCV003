import torch
import numpy as np
from torch.utils.data import DataLoader, Dataset
from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.image.inception import InceptionScore
from tqdm import tqdm
from typing import Tuple, Callable, Optional


class ImageDataset(Dataset):
    """Simple dataset wrapper for a tensor of images."""
    def __init__(self, images: torch.Tensor):
        self.images = images
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        return self.images[idx]


def denormalize_images(images: torch.Tensor, normalize_range: str) -> torch.Tensor:
    """
    Convert images from model output range to [0, 1].
    
    Args:
        images: Images from model
        normalize_range: '01' if already in [0,1], '-11' if in [-1,1]
    
    Returns:
        Images in [0, 1] range
    """
    if normalize_range == '01':
        return images
    elif normalize_range == '-11':
        # Convert from [-1, 1] to [0, 1]
        return (images + 1.0) / 2.0
    else:
        raise ValueError(f"Unknown normalize_range: {normalize_range}")


def compute_fid(
    sample_fn: Callable[[int, torch.device], torch.Tensor],
    real_images: torch.Tensor,
    num_samples: int = 500,
    batch_size: int = 128,
    device: torch.device = torch.device('cuda'),
    model_normalize_range: str = '01'
) -> float:
    """
    Compute Fréchet Inception Distance (FID).
    
    Args:
        sample_fn: Function that takes (num_samples, device) and returns images
                   This abstracts away the model interface.
        real_images: Tensor of real images [N, C, H, W] in [0, 1]
        num_samples: Number of images to generate for comparison
        batch_size: Batch size for generation
        device: Device to use
        model_normalize_range: Range of images returned by sample_fn
                               '01' for [0,1], '-11' for [-1,1]
    
    Returns:
        FID score (float)
    """
    # Initialize FID metric
    # InceptionV3 expects images in [0, 1] with 3 channels
    fid = FrechetInceptionDistance(feature=2048, normalize=True).to(device)
    
    # Add real images to FID (already in [0, 1])
    print("Computing FID: adding real images...")
    real_dataset = ImageDataset(real_images)
    real_loader = DataLoader(real_dataset, batch_size=batch_size, shuffle=False)
    
    with torch.no_grad():
        for batch in tqdm(real_loader, desc="Real images"):
            batch = batch.to(device)
            fid.update(batch, real=True)
    
    # Generate images and add to FID
    print(f"Computing FID: generating {num_samples} images...")
    generated_images = []
    num_batches = (num_samples + batch_size - 1) // batch_size
    
    with torch.no_grad():
        for _ in tqdm(range(num_batches), desc="Generating"):
            samples = sample_fn(batch_size, device)
            # Denormalize to [0, 1] if needed
            samples = denormalize_images(samples, model_normalize_range)
            generated_images.append(samples.cpu())
    
    generated_images = torch.cat(generated_images, dim=0)[:num_samples]
    
    # Add generated images to FID
    gen_dataset = ImageDataset(generated_images)
    gen_loader = DataLoader(gen_dataset, batch_size=batch_size, shuffle=False)
    
    with torch.no_grad():
        for batch in tqdm(gen_loader, desc="Generated images"):
            batch = batch.to(device)
            fid.update(batch, real=False)
    
    fid_score = fid.compute().item()
    print(f"FID Score: {fid_score:.2f}")
    
    return fid_score


def compute_inception_score(
    sample_fn: Callable[[int, torch.device], torch.Tensor],
    num_samples: int = 500,
    batch_size: int = 128,
    device: torch.device = torch.device('cuda'),
    model_normalize_range: str = '01'
) -> Tuple[float, float]:
    """
    Compute Inception Score (IS).
    
    Args:
        sample_fn: Function that takes (num_samples, device) and returns images
        num_samples: Number of images to generate
        batch_size: Batch size for generation
        device: Device to use
        model_normalize_range: Range of images returned by sample_fn
    
    Returns:
        Tuple of (mean IS, std IS)
    """
    # Initialize Inception Score metric
    inception = InceptionScore(normalize=True).to(device)
    
    # Generate images
    print(f"Computing IS: generating {num_samples} images...")
    generated_images = []
    num_batches = (num_samples + batch_size - 1) // batch_size
    
    with torch.no_grad():
        for _ in tqdm(range(num_batches), desc="Generating"):
            samples = sample_fn(batch_size, device)
            # Denormalize to [0, 1] if needed
            samples = denormalize_images(samples, model_normalize_range)
            generated_images.append(samples.cpu())
    
    generated_images = torch.cat(generated_images, dim=0)[:num_samples]
    
    # Compute IS
    gen_dataset = ImageDataset(generated_images)
    gen_loader = DataLoader(gen_dataset, batch_size=batch_size, shuffle=False)
    
    with torch.no_grad():
        for batch in tqdm(gen_loader, desc="Computing IS"):
            batch = batch.to(device)
            inception.update(batch)
    
    mean_is, std_is = inception.compute()
    print(f"Inception Score: {mean_is:.2f} ± {std_is:.2f}")
    
    return mean_is.item(), std_is.item()


def compute_all_metrics(
    model: torch.nn.Module,
    real_images: torch.Tensor,
    device: torch.device = torch.device('cuda'),
    model_normalize_range: str = '01'
) -> dict:
    """
    Compute all evaluation metrics.
    
    Args:
        model: Trained model (VAE or DDPM)
               Must have a .sample(num_samples, device) method
        real_images: Real images in [0, 1]
        device: Device to use
        model_normalize_range: Range of images returned by model.sample()
    
    Returns:
        Dictionary with FID, IS_mean, IS_std
    """
    print("=" * 60)
    print("Computing Evaluation Metrics")
    print("=" * 60)
    
    # Create a sample function that wraps the model
    # This abstracts away the model interface
    def sample_fn(num_samples: int, device: torch.device) -> torch.Tensor:
        return model.sample(num_samples, device)
    
    fid = compute_fid(
        sample_fn, 
        real_images, 
        num_samples=500, 
        device=device,
        model_normalize_range=model_normalize_range
    )
    
    is_mean, is_std = compute_inception_score(
        sample_fn, 
        num_samples=500, 
        device=device,
        model_normalize_range=model_normalize_range
    )
    
    metrics = {
        'fid': fid,
        'inception_score_mean': is_mean,
        'inception_score_std': is_std
    }
    
    print("\n" + "=" * 60)
    print("Final Metrics:")
    print(f"  FID: {fid:.2f}")
    print(f"  Inception Score: {is_mean:.2f} ± {is_std:.2f}")
    print("=" * 60)
    
    return metrics