import torch
import torch.optim as optim
from pathlib import Path
import yaml
from tqdm import tqdm
import matplotlib.pyplot as plt

from src.models.vae import VAE
from src.models.vae_loss import vae_loss
from src.data.dataset import get_cifar10_dataloaders


def train_vae(config_path: str):
    """Main training function."""
    
    # Load config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Set seed for reproducibility
    torch.manual_seed(config['seed'])
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config['seed'])
    
    # Device setup (use both T4s if available)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs")
    
    # Data
    train_loader, test_loader = get_cifar10_dataloaders(
        batch_size=config['batch_size']
    )
    
    # Model
    model = VAE(
        in_channels=config['channels'],
        latent_dim=config['latent_dim'],
        hidden_dims=config['hidden_dims'],
        image_size=config['image_size']
    )
    
    # Use DataParallel for multiple GPUs
    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)
    
    model = model.to(device)
    
    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'])
    
    # Training loop
    output_dir = Path('outputs/vae')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    for epoch in range(config['epochs']):
        model.train()
        train_loss = 0
        train_recon = 0
        train_kl = 0
        
        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{config["epochs"]}')
        for batch_idx, (data, _) in enumerate(pbar):
            data = data.to(device)
            
            # Forward pass
            optimizer.zero_grad()
            recon, mu, logvar = model(data)
            
            # Compute loss
            loss, recon_loss, kl_loss = vae_loss(
                recon, data, mu, logvar,
                recon_loss_type='bce',
                kl_weight=config['kl_weight']
            )
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            # Track metrics
            train_loss += loss.item()
            train_recon += recon_loss.item()
            train_kl += kl_loss.item()
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f'{loss.item()/len(data):.4f}',
                'recon': f'{recon_loss.item()/len(data):.4f}',
                'kl': f'{kl_loss.item()/len(data):.4f}'
            })
        
        # Average metrics
        n_samples = len(train_loader.dataset)
        train_loss /= n_samples
        train_recon /= n_samples
        train_kl /= n_samples
        
        print(f'Epoch {epoch+1}: Loss={train_loss:.4f}, '
              f'Recon={train_recon:.4f}, KL={train_kl:.4f}')
        
        # Save samples every 10 epochs
        if (epoch + 1) % 10 == 0:
            save_samples(model, epoch + 1, device, output_dir)
    
    # Save final model
    torch.save(model.state_dict(), output_dir / 'vae_final.pth')
    print(f'Model saved to {output_dir / "vae_final.pth"}')


def save_samples(model, epoch, device, output_dir, n_samples=64):
    """Generate and save sample images."""
    model.eval()
    with torch.no_grad():
        # Get the actual model (unwrap DataParallel if needed)
        actual_model = model.module if hasattr(model, 'module') else model
        samples = actual_model.sample(n_samples, device)
        
        # Create grid
        samples = samples.cpu()
        grid = make_grid(samples, nrow=8)
        
        # Save
        plt.figure(figsize=(10, 10))
        plt.imshow(grid.permute(1, 2, 0))
        plt.axis('off')
        plt.savefig(output_dir / f'samples_epoch_{epoch}.png', 
                    bbox_inches='tight', dpi=150)
        plt.close()


def make_grid(images, nrow=8):
    """Create a grid of images."""
    import torchvision.utils as vutils
    return vutils.make_grid(images, nrow=nrow, padding=2, normalize=False)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/vae_cifar10.yaml')
    args = parser.parse_args()
    
    train_vae(args.config)