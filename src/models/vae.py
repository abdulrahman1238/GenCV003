import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List


class VAEEncoder(nn.Module):
    def __init__(self, in_channels: int, latent_dim: int, 
                 hidden_dims: List[int], image_size: int):
        super().__init__()
        self.image_size = image_size
        self.num_downsamples = len(hidden_dims)
        
        # Build convolutional layers dynamically
        layers = []
        current_channels = in_channels
        for h_dim in hidden_dims:
            layers.extend([
                nn.Conv2d(current_channels, h_dim, kernel_size=3, stride=2, padding=1),
                nn.BatchNorm2d(h_dim),
                nn.ReLU()
            ])
            current_channels = h_dim
        self.conv_blocks = nn.Sequential(*layers)
        
        # Compute flattened size dynamically
        # Each stride-2 conv halves spatial dims: image_size / 2^num_downsamples
        spatial_size = image_size // (2 ** self.num_downsamples)
        flat_dim = hidden_dims[-1] * spatial_size * spatial_size
        
        self.fc_mu = nn.Linear(flat_dim, latent_dim)
        self.fc_logvar = nn.Linear(flat_dim, latent_dim)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.conv_blocks(x)
        h = h.view(h.size(0), -1)
        return self.fc_mu(h), self.fc_logvar(h)


class VAEDecoder(nn.Module):
    def __init__(self, latent_dim: int, out_channels: int, 
                 hidden_dims: List[int], image_size: int):
        super().__init__()
        self.num_upsamples = len(hidden_dims)
        
        # Compute spatial size at the bottleneck (mirrors encoder)
        spatial_size = image_size // (2 ** self.num_upsamples)
        flat_dim = hidden_dims[-1] * spatial_size * spatial_size
        
        self.fc = nn.Linear(latent_dim, flat_dim)
        self.spatial_size = spatial_size
        
        # Build transposed conv layers in REVERSE order of encoder
        # hidden_dims = [64, 128, 256] → decoder goes 256 → 128 → 64 → out_channels
        layers = []
        reversed_dims = list(reversed(hidden_dims))
        for i in range(len(reversed_dims) - 1):
            layers.extend([
                nn.ConvTranspose2d(reversed_dims[i], reversed_dims[i+1], 
                                   kernel_size=3, stride=2, padding=1, output_padding=1),
                nn.BatchNorm2d(reversed_dims[i+1]),
                nn.ReLU()
            ])
        # Final layer: no BatchNorm, use sigmoid for [0,1] output
        layers.extend([
            nn.ConvTranspose2d(reversed_dims[-1], out_channels, 
                               kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid()
        ])
        self.deconv_blocks = nn.Sequential(*layers)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        h = self.fc(z)
        h = h.view(h.size(0), -1, self.spatial_size, self.spatial_size)
        return self.deconv_blocks(h)


class VAE(nn.Module):
    def __init__(self, in_channels: int, latent_dim: int, 
                 hidden_dims: List[int], image_size: int):
        super().__init__()
        self.encoder = VAEEncoder(in_channels, latent_dim, hidden_dims, image_size)
        self.decoder = VAEDecoder(latent_dim, in_channels, hidden_dims, image_size)

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + std * eps

    def forward(self, x: torch.Tensor):
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        x_recon = self.decoder(z)
        return x_recon, mu, logvar

    def sample(self, num_samples: int, device: torch.device) -> torch.Tensor:
        z = torch.randn(num_samples, self.encoder.fc_mu.out_features, device=device)
        return self.decoder(z)