"""
DDPM Training Script.

Trains a Denoising Diffusion Probabilistic Model on CIFAR-10.

"""

import torch
import torch.optim as optim
from pathlib import Path
import yaml
from tqdm import tqdm
import matplotlib.pyplot as plt
import torchvision.utils as vutils

from src.models.unet import UNet
from src.models.ddpm import DDPM
from src.data.dataset import get_cifar10_dataloaders


def train_ddpm(config_path: str):
    """
    Main training function for DDPM.
    
    Args:
        config_path: Path to YAML config file
    """
    
    # =========================================================================
    # Setup
    # =========================================================================
    
    # Load config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Set seed for reproducibility
    torch.manual_seed(config['seed'])
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config['seed'])
    
    # Device setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs with DataParallel")
    
    # Output directory
    output_dir = Path('outputs/ddpm')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save config for reproducibility
    config_save_path = output_dir / 'ddpm_config.yaml'
    with open(config_save_path, 'w') as f:
        yaml.dump(config, f)
    print(f"Config saved to {config_save_path}")
    
    # =========================================================================
    # Data
    # =========================================================================
    
    print("\nLoading CIFAR-10 dataset...")
    train_loader, test_loader = get_cifar10_dataloaders(
        batch_size=config['batch_size'],
        normalize_range=config['normalize_range']  # '-11' for DDPM
    )
    print(f"Train batches: {len(train_loader)}, Test batches: {len(test_loader)}")
    
    # =========================================================================
    # Model
    # =========================================================================
    
    print("\nBuilding model...")
    
    # Step 1: Build U-Net (noise predictor)
    unet = UNet(
        in_channels=config['channels'],
        out_channels=config['channels'],
        unet_channels=config['unet_channels'],
        time_emb_dim=config['time_emb_dim'],
        num_res_blocks=config['num_res_blocks'],
        use_attention=config['use_attention'],
        dropout=config['dropout'],
    )
    
    # Step 2: Wrap with DDPM (diffusion process)
    ddpm = DDPM(
        model=unet,
        timesteps=config['timesteps'],
        beta_schedule=config['beta_schedule'],
        beta_start=config['beta_start'],
        beta_end=config['beta_end'],
    )
    
    # Use DataParallel for multiple GPUs
    if torch.cuda.device_count() > 1:
        ddpm = torch.nn.DataParallel(ddpm)
    
    ddpm = ddpm.to(device)
    
    # Count parameters
    num_params = sum(p.numel() for p in ddpm.parameters())
    print(f"Model parameters: {num_params:,}")
    
    # =========================================================================
    # Optimizer
    # =========================================================================
    
    optimizer = optim.Adam(ddpm.parameters(), lr=config['learning_rate'])
    
    # =========================================================================
    # Training Loop
    # =========================================================================
    
    print("\n" + "="*60)
    print("Starting Training")
    print("="*60)
    
    # Track training history
    history = {
        'epoch': [],
        'loss': [],
    }
    
    for epoch in range(1, config['epochs'] + 1):
        ddpm.train()
        epoch_loss = 0.0
        num_batches = 0
        
        pbar = tqdm(train_loader, desc=f'Epoch {epoch}/{config["epochs"]}')
        for batch_idx, (data, _) in enumerate(pbar):
            data = data.to(device)
            
            # Forward pass: compute loss
            # ddpm.forward(data) internally calls compute_loss(data)
            loss = ddpm(data)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Track metrics
            epoch_loss += loss.item()
            num_batches += 1
            
            # Update progress bar
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        # Average loss for this epoch
        avg_loss = epoch_loss / num_batches
        history['epoch'].append(epoch)
        history['loss'].append(avg_loss)
        
        print(f'Epoch {epoch}: Loss = {avg_loss:.4f}')
        
        # =====================================================================
        # Sampling and Visualization
        # =====================================================================
        
        if epoch % config['sample_every'] == 0:
            print(f"\nGenerating samples at epoch {epoch}...")
            
            # Get the actual DDPM (unwrap DataParallel if needed)
            actual_ddpm = ddpm.module if hasattr(ddpm, 'module') else ddpm
            
            # Generate samples with intermediate steps
            samples, intermediates = actual_ddpm.sample(
                num_samples=config['num_sample_images'],
                device=device,
                save_intermediate=config['save_intermediate'],
                save_every=100  # Save every 100 steps
            )
            
            # Denormalize from [-1, 1] to [0, 1] for display
            samples_display = (samples + 1.0) / 2.0
            samples_display = samples_display.clamp(0, 1)
            
            # Save sample grid
            grid = vutils.make_grid(samples_display.cpu(), nrow=8, padding=2)
            plt.figure(figsize=(12, 12))
            plt.imshow(grid.permute(1, 2, 0))
            plt.axis('off')
            plt.title(f'DDPM Samples - Epoch {epoch}', fontsize=16, pad=20)
            plt.savefig(output_dir / f'samples_epoch_{epoch}.png', 
                       bbox_inches='tight', dpi=150)
            plt.close()
            print(f"Saved sample grid to {output_dir / f'samples_epoch_{epoch}.png'}")
            
            # Save denoising trajectory (show progression for first 8 images)
            if config['save_intermediate'] and len(intermediates) > 0:
                # Take first 8 images from each intermediate step
                trajectory = []
                for step in intermediates:
                    step_images = step[:8]  # First 8 images
                    step_display = (step_images + 1.0) / 2.0
                    step_display = step_display.clamp(0, 1)
                    trajectory.append(step_display)
                
                # Stack all steps: [num_steps, 8, C, H, W]
                trajectory = torch.stack(trajectory, dim=0)
                
                # Reshape to show as grid: [num_steps * 8, C, H, W]
                # Each row shows one timestep, each column shows one image
                num_steps = trajectory.shape[0]
                num_images = trajectory.shape[1]
                
                # Create a grid where rows = timesteps, cols = images
                trajectory_flat = trajectory.reshape(-1, 3, 32, 32)
                grid = vutils.make_grid(trajectory_flat.cpu(), 
                                       nrow=num_images, padding=2)
                
                plt.figure(figsize=(16, num_steps * 2))
                plt.imshow(grid.permute(1, 2, 0))
                plt.axis('off')
                plt.title(f'Denoising Trajectory - Epoch {epoch}\n'
                         f'(Top: pure noise → Bottom: clean image)', 
                         fontsize=14, pad=20)
                plt.savefig(output_dir / f'denoising_trajectory_epoch_{epoch}.png', 
                           bbox_inches='tight', dpi=150)
                plt.close()
                print(f"Saved denoising trajectory to {output_dir / f'denoising_trajectory_epoch_{epoch}.png'}")
            
            print()
    
    # =========================================================================
    # Save Final Model
    # =========================================================================
    
    print("\n" + "="*60)
    print("Training Complete")
    print("="*60)
    
    # Save model checkpoint
    checkpoint_path = output_dir / 'ddpm_final.pth'
    torch.save(ddpm.state_dict(), checkpoint_path)
    print(f"Model saved to {checkpoint_path}")
    
    # Save training history
    history_path = output_dir / 'training_history.yaml'
    with open(history_path, 'w') as f:
        yaml.dump(history, f)
    print(f"Training history saved to {history_path}")
    
    # Plot training loss
    plt.figure(figsize=(10, 6))
    plt.plot(history['epoch'], history['loss'], 'b-', linewidth=2)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss (MSE)', fontsize=12)
    plt.title('DDPM Training Loss', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.savefig(output_dir / 'training_loss.png', bbox_inches='tight', dpi=150)
    plt.close()
    print(f"Training loss plot saved to {output_dir / 'training_loss.png'}")
    
    print("\n" + "="*60)
    print("All outputs saved to:", output_dir)
    print("="*60)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/ddpm_cifar10.yaml')
    args = parser.parse_args()
    
    train_ddpm(args.config)