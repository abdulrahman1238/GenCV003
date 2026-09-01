"""
DDPM Evaluation Script.

Evaluates a trained DDPM by:
    1. Computing quantitative metrics (FID, Inception Score)
    2. Generating qualitative visualizations (samples, denoising trajectory)
    3. Comparing reconstruction quality (optional)

"""

import torch
import yaml
from pathlib import Path
import argparse
import matplotlib.pyplot as plt
import torchvision.utils as vutils
from tqdm import tqdm

from src.models.unet import UNet
from src.models.ddpm import DDPM
from src.data.dataset import get_cifar10_dataloaders
from src.evaluation.metrics import compute_all_metrics
from src.evaluation.visualize import save_sample_grid


def reconstruct_image(
    ddpm: DDPM,
    x_0: torch.Tensor,
    noise_level: float = 0.5,
    device: torch.device = torch.device('cuda')
) -> torch.Tensor:
    """
    Reconstruct an image by adding noise and then denoising.
    
    This demonstrates DDPM's ability to recover from partial corruption.
    
    Args:
        ddpm: Trained DDPM model
        x_0: Clean image [1, C, H, W] in [-1, 1]
        noise_level: Fraction of timesteps to add noise (0.0 to 1.0)
        device: Device to use
    
    Returns:
        Reconstructed image [1, C, H, W] in [-1, 1]
    """
    ddpm.eval()
    
    # Choose timestep based on noise level
    t = int(ddpm.timesteps * noise_level)
    t_tensor = torch.full((1,), t, device=device, dtype=torch.long)
    
    # Add noise to get x_t
    with torch.no_grad():
        x_t = ddpm.q_sample(x_0.to(device), t_tensor)
        
        # Run reverse process from x_t down to x_0
        x_recon = x_t.clone()
        for i in reversed(range(t + 1)):
            t_step = torch.full((1,), i, device=device, dtype=torch.long)
            x_recon = ddpm.p_sample(x_recon, t_step)
    
    return x_recon


def save_denoising_trajectory(
    ddpm: DDPM,
    num_samples: int = 4,
    device: torch.device = torch.device('cuda'),
    save_path: Path = None
) -> plt.Figure:
    """
    Visualize the denoising process from pure noise to clean image.
    
    Shows intermediate steps at regular intervals to demonstrate
    how the model iteratively removes noise.
    
    Args:
        ddpm: Trained DDPM model
        num_samples: Number of images to generate
        device: Device to use
        save_path: Path to save the visualization
    
    Returns:
        matplotlib Figure
    """
    ddpm.eval()
    
    print(f"Generating denoising trajectory for {num_samples} images...")
    print("This will take a few minutes (1000 steps per image)...")
    
    # Generate samples with intermediate steps
    with torch.no_grad():
        samples, intermediates = ddpm.sample(
            num_samples=num_samples,
            device=device,
            save_intermediate=True,
            save_every=100  # Save every 100 steps
        )
    
    # intermediates is a list of tensors, each [num_samples, C, H, W]
    # Each tensor represents the state at a particular timestep
    
    num_steps = len(intermediates)
    
    # Create a grid where:
    # - Rows = different samples
    # - Columns = different timesteps (left = noise, right = clean)
    
    # Denormalize from [-1, 1] to [0, 1] for display
    trajectory = []
    for step in intermediates:
        step_display = (step + 1.0) / 2.0
        step_display = step_display.clamp(0, 1)
        trajectory.append(step_display)
    
    # Stack: [num_steps, num_samples, C, H, W]
    trajectory = torch.stack(trajectory, dim=0)
    
    # Reshape to show as grid
    # We want: rows = samples, cols = timesteps
    # Current shape: [num_steps, num_samples, C, H, W]
    # We need to transpose to: [num_samples, num_steps, C, H, W]
    trajectory = trajectory.permute(1, 0, 2, 3, 4)
    
    # Flatten to: [num_samples * num_steps, C, H, W]
    trajectory_flat = trajectory.reshape(-1, 3, 32, 32)
    
    # Create grid with num_steps columns
    grid = vutils.make_grid(trajectory_flat.cpu(), nrow=num_steps, padding=2)
    
    # Plot
    fig, ax = plt.subplots(figsize=(num_steps * 2, num_samples * 2))
    ax.imshow(grid.permute(1, 2, 0))
    ax.axis('off')
    ax.set_title('Denoising Trajectory\n(Left: pure noise → Right: clean image)', 
                 fontsize=14, pad=20)
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=150)
        print(f"Denoising trajectory saved to {save_path}")
    
    return fig


def save_reconstruction_comparison(
    ddpm: DDPM,
    real_images: torch.Tensor,
    num_images: int = 8,
    noise_levels: list = [0.3, 0.5, 0.7],
    device: torch.device = torch.device('cuda'),
    save_path: Path = None
) -> plt.Figure:
    """
    Compare real images with reconstructions at different noise levels.
    
    Shows how well DDPM can recover images from partial corruption.
    
    Args:
        ddpm: Trained DDPM model
        real_images: Real images [N, C, H, W] in [0, 1]
        num_images: Number of images to reconstruct
        noise_levels: List of noise levels to test (0.0 to 1.0)
        device: Device to use
        save_path: Path to save the visualization
    
    Returns:
        matplotlib Figure
    """
    ddpm.eval()
    
    # Take first num_images and normalize to [-1, 1]
    real_batch = real_images[:num_images].to(device)
    real_batch_norm = real_batch * 2.0 - 1.0  # [0,1] → [-1,1]
    
    # Reconstruct at different noise levels
    reconstructions = []
    for noise_level in noise_levels:
        print(f"Reconstructing with noise level {noise_level}...")
        recon = reconstruct_image(ddpm, real_batch_norm, noise_level, device)
        # Denormalize back to [0, 1]
        recon_display = (recon + 1.0) / 2.0
        recon_display = recon_display.clamp(0, 1)
        reconstructions.append(recon_display.cpu())
    
    # Create comparison grid
    # Row 1: Real images
    # Row 2-4: Reconstructions at different noise levels
    
    all_images = [real_batch.cpu()] + reconstructions
    comparison = torch.cat(all_images, dim=0)
    
    # Create grid with num_images columns
    grid = vutils.make_grid(comparison, nrow=num_images, padding=2, normalize=False)
    
    # Plot
    fig, ax = plt.subplots(figsize=(16, 8))
    ax.imshow(grid.permute(1, 2, 0))
    ax.axis('off')
    
    # Add labels
    y_positions = [0, 32, 64, 96]  # Approximate y positions for each row
    labels = ['Real Images'] + [f'Noise Level {nl}' for nl in noise_levels]
    
    for i, (y_pos, label) in enumerate(zip(y_positions, labels)):
        ax.text(-10, y_pos + 16, label, ha='right', va='center', 
                fontsize=10, fontweight='bold', rotation=90)
    
    ax.set_title('DDPM Reconstruction at Different Noise Levels', 
                 fontsize=14, pad=20)
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=150)
        print(f"Reconstruction comparison saved to {save_path}")
    
    return fig


def evaluate_ddpm(config_path: str, checkpoint_path: str):
    """
    Run full evaluation on a trained DDPM.
    
    Args:
        config_path: Path to config YAML file
        checkpoint_path: Path to trained model checkpoint
    """
    
    # =========================================================================
    # Setup
    # =========================================================================
    
    # Load config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Output directory
    output_dir = Path('outputs/ddpm_evaluation')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # =========================================================================
    # Load Model
    # =========================================================================
    
    print(f"\nLoading model from {checkpoint_path}...")
    
    # Build U-Net architecture
    unet = UNet(
        in_channels=config['channels'],
        out_channels=config['channels'],
        unet_channels=config['unet_channels'],
        time_emb_dim=config['time_emb_dim'],
        num_res_blocks=config['num_res_blocks'],
        use_attention=config['use_attention'],
        dropout=config['dropout'],
    )
    
    # Build DDPM
    ddpm = DDPM(
        model=unet,
        timesteps=config['timesteps'],
        beta_schedule=config['beta_schedule'],
        beta_start=config['beta_start'],
        beta_end=config['beta_end'],
    )
    
    # Load checkpoint
    state_dict = torch.load(checkpoint_path, map_location=device)
    
    # Handle DataParallel checkpoint (remove 'module.' prefix)
    if any(k.startswith('module.') for k in state_dict.keys()):
        new_state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
        ddpm.load_state_dict(new_state_dict)
    else:
        ddpm.load_state_dict(state_dict)
    
    ddpm = ddpm.to(device)
    ddpm.eval()
    
    print(f"Model loaded successfully")
    
    # =========================================================================
    # Load Test Data
    # =========================================================================
    
    print("\nLoading test dataset for evaluation...")
    print("Note: Real images must be in [0, 1] for FID/IS metrics")
    
    # Load test set with [0, 1] normalization for metrics
    _, test_loader = get_cifar10_dataloaders(
        batch_size=config['batch_size'],
        normalize_range='01'  # Always [0, 1] for FID/IS ground truth
    )
    
    # Collect all test images
    all_images = []
    for images, _ in tqdm(test_loader, desc="Loading test images"):
        all_images.append(images)
    real_images = torch.cat(all_images, dim=0)
    print(f"Loaded {len(real_images)} test images in [0, 1] range")
    
    # =========================================================================
    # Generate Visualizations
    # =========================================================================
    
    print("\n" + "="*60)
    print("Generating Visualizations")
    print("="*60)
    
    # 1. Sample grid (reuse shared function)
    print("\n1. Generating sample grid...")
    save_sample_grid(
        ddpm,
        num_samples=64,
        device=device,
        model_normalize_range='-11',  # DDPM outputs [-1, 1]
        save_path=output_dir / 'sample_grid.png'
    )
    plt.close()
    
    # 2. Denoising trajectory (DDPM-specific)
    print("\n2. Generating denoising trajectory...")
    save_denoising_trajectory(
        ddpm,
        num_samples=4,
        device=device,
        save_path=output_dir / 'denoising_trajectory.png'
    )
    plt.close()
    
    # 3. Reconstruction comparison
    print("\n3. Generating reconstruction comparison...")
    save_reconstruction_comparison(
        ddpm,
        real_images,
        num_images=8,
        noise_levels=[0.3, 0.5, 0.7],
        device=device,
        save_path=output_dir / 'reconstruction_comparison.png'
    )
    plt.close()
    
    # =========================================================================
    # Compute Metrics
    # =========================================================================
    
    print("\n" + "="*60)
    print("Computing Quantitative Metrics")
    print("="*60)
    print("This will take ~30 minutes (generating 10,000 images)...")
    
    metrics = compute_all_metrics(
        ddpm,
        real_images,
        device,
        model_normalize_range='-11'  # DDPM outputs [-1, 1]
    )
    
    # Save metrics
    metrics_path = output_dir / 'metrics.yaml'
    with open(metrics_path, 'w') as f:
        yaml.dump(metrics, f)
    print(f"Metrics saved to {metrics_path}")
    
    # Save config used for evaluation
    eval_config = {
        'config_path': config_path,
        'checkpoint_path': checkpoint_path,
        'model_normalize_range': '-11',
        'num_real_images': len(real_images),
        'num_generated_images': 10000,
        **config
    }
    
    config_save_path = output_dir / 'eval_config.yaml'
    with open(config_save_path, 'w') as f:
        yaml.dump(eval_config, f)
    print(f"Eval config saved to {config_save_path}")
    
    # =========================================================================
    # Summary
    # =========================================================================
    
    print("\n" + "="*60)
    print("Evaluation Complete")
    print("="*60)
    print(f"\nQuantitative Metrics:")
    print(f"  FID: {metrics['fid']:.2f}")
    print(f"  Inception Score: {metrics['inception_score_mean']:.2f} ± {metrics['inception_score_std']:.2f}")
    print(f"\nAll outputs saved to: {output_dir}")
    print("="*60)
    
    return metrics


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/ddpm_cifar10.yaml')
    parser.add_argument('--checkpoint', type=str, default='outputs/ddpm/ddpm_final.pth')
    args = parser.parse_args()
    
    evaluate_ddpm(args.config, args.checkpoint)