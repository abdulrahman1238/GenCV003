import torch
import yaml
from pathlib import Path
import argparse

from src.models.vae import VAE
from src.data.dataset import get_cifar10_dataloaders
from src.evaluation.metrics import compute_all_metrics
from src.evaluation.visualize import generate_all_visualizations


def evaluate_vae(config_path: str, checkpoint_path: str):
    """
    Run full evaluation on a trained VAE.
    
    All settings are read from the config file, including:
    - Model architecture parameters
    - Normalization range (should be '01' for VAE)
    - Batch size, etc.
    """
    
    # Load config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Get model's normalize range from config (default to '01' for VAE)
    model_normalize_range = config.get('normalize_range', '01')
    print(f"Model output range: {model_normalize_range}")
    
    # Load model with config parameters
    print(f"Loading model from {checkpoint_path}...")
    model = VAE(
        in_channels=config['channels'],
        latent_dim=config['latent_dim'],
        hidden_dims=config['hidden_dims'],
        image_size=config['image_size']
    )
    
    # Load checkpoint (handle DataParallel prefix)
    state_dict = torch.load(checkpoint_path, map_location=device)
    if any(k.startswith('module.') for k in state_dict.keys()):
        new_state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
        model.load_state_dict(new_state_dict)
    else:
        model.load_state_dict(state_dict)
    
    model = model.to(device)
    model.eval()
    
    # IMPORTANT: For FID/IS, we need real images in [0, 1] regardless of model range.
    # So we always load a separate test loader with '01' normalization for metrics.
    print("Loading test dataset for evaluation (real images in [0, 1])...")
    _, test_loader = get_cifar10_dataloaders(
        batch_size=config['batch_size'],
        normalize_range='01'  # Always [0, 1] for FID/IS ground truth
    )
    
    # Collect all test images
    all_images = []
    for images, _ in test_loader:
        all_images.append(images)
    real_images = torch.cat(all_images, dim=0)
    print(f"Loaded {len(real_images)} test images in [0, 1] range")
    
    # Output directory
    output_dir = Path('outputs/vae_evaluation')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate visualizations (pass model_normalize_range)
    generate_all_visualizations(
        model, 
        real_images, 
        output_dir, 
        device,
        model_normalize_range=model_normalize_range
    )
    
    # Compute metrics (pass model_normalize_range)
    metrics = compute_all_metrics(
        model, 
        real_images, 
        device,
        model_normalize_range=model_normalize_range
    )
    
    # Save metrics to file
    metrics_path = output_dir / 'metrics.yaml'
    with open(metrics_path, 'w') as f:
        yaml.dump(metrics, f)
    print(f"Metrics saved to {metrics_path}")
    
    # Also save the config used for reproducibility
    config_path_out = output_dir / 'eval_config.yaml'
    with open(config_path_out, 'w') as f:
        yaml.dump({
            'config_path': config_path,
            'checkpoint_path': checkpoint_path,
            'model_normalize_range': model_normalize_range,
            'num_real_images': len(real_images),
            **config
        }, f)
    
    return metrics


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/vae_cifar10.yaml')
    parser.add_argument('--checkpoint', type=str, default='outputs/vae/vae_final.pth')
    args = parser.parse_args()
    
    evaluate_vae(args.config, args.checkpoint)