import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from typing import Tuple


def get_cifar10_dataloaders(
    data_dir: str = './data',
    batch_size: int = 128,
    num_workers: int = 2,
    normalize_range: str = '01'  # '01' for [0,1] or '-11' for [-1,1]
) -> Tuple[DataLoader, DataLoader]:
    """
    Create CIFAR-10 train and test dataloaders.
    
    Args:
        data_dir: Directory to store/load dataset
        batch_size: Batch size
        num_workers: Number of data loading workers
        normalize_range: 
            - '01': Normalize to [0, 1] (for VAE with sigmoid output)
            - '-11': Normalize to [-1, 1] (for DDPM)
    
    Returns:
        Tuple of (train_loader, test_loader)
    """
    
    # Build transform based on normalization range
    transform_list = [transforms.ToTensor()]  # Always start with [0, 1]
    
    if normalize_range == '-11':
        # Normalize from [0, 1] to [-1, 1]
        # Formula: x_normalized = (x - 0.5) / 0.5 = 2*x - 1
        transform_list.append(
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        )
    # If '01', we just use ToTensor() which gives [0, 1]
    
    transform = transforms.Compose(transform_list)
    
    # Download and load datasets
    train_dataset = datasets.CIFAR10(
        root=data_dir,
        train=True,
        download=True,
        transform=transform
    )
    
    test_dataset = datasets.CIFAR10(
        root=data_dir,
        train=False,
        download=True,
        transform=transform
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, test_loader


def get_dataset_stats(loader: DataLoader) -> dict:
    """
    Compute dataset statistics (useful for debugging).
    
    Returns:
        Dictionary with mean, std, min, max per channel
    """
    means = torch.zeros(3)
    stds = torch.zeros(3)
    min_vals = torch.full((3,), float('inf'))
    max_vals = torch.full((3,), float('-inf'))
    count = 0
    
    for images, _ in loader:
        # images shape: [B, C, H, W]
        b, c, h, w = images.shape
        count += b
        
        # Reshape to [B*H*W, C] for per-channel stats
        images_flat = images.permute(0, 2, 3, 1).reshape(-1, c)
        
        means += images_flat.mean(dim=0)
        stds += images_flat.std(dim=0)
        min_vals = torch.min(min_vals, images_flat.min(dim=0)[0])
        max_vals = torch.max(max_vals, images_flat.max(dim=0)[0])
    
    means /= count
    stds /= count
    
    return {
        'mean': means.tolist(),
        'std': stds.tolist(),
        'min': min_vals.tolist(),
        'max': max_vals.tolist()
    }