"""
U-Net Architecture for DDPM (Denoising Diffusion Probabilistic Models).

This module implements the noise prediction network ε_θ(x_t, t) used in DDPM.
Given a noisy image x_t and timestep t, it predicts the noise that was added.

Key design choices:
    - GroupNorm (not BatchNorm): more stable across varying noise levels
    - SiLU activation: standard in modern diffusion models
    - Time injection via addition: simple but effective
    - Upsample via interpolation + conv: avoids checkerboard artifacts
    - Self-attention at lowest resolution: captures global structure cheaply
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional


# =============================================================================
# Time Embedding
# =============================================================================

class SinusoidalPositionEmbedding(nn.Module):
    """
    Sinusoidal positional encoding for timesteps (from "Attention Is All You Need").
    
    Converts an integer timestep t into a continuous embedding vector using
    sinusoidal functions at different frequencies. This allows the network to
    understand the relative ordering and magnitude of timesteps.
    
    Math:
        PE(t, 2i)   = sin(t / 10000^(2i/d))
        PE(t, 2i+1) = cos(t / 10000^(2i/d))
    
    Args:
        embedding_dim: Dimension of the output embedding
    """
    def __init__(self, embedding_dim: int):
        super().__init__()
        self.embedding_dim = embedding_dim
    
    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        Args:
            t: Timesteps [B] (integers)
        
        Returns:
            Embeddings [B, embedding_dim]
        """
        device = t.device
        half_dim = self.embedding_dim // 2
        
        # Compute frequencies: 1 / 10000^(2i/d)
        # Using log-space for numerical stability
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        
        # Outer product: [B, 1] * [half_dim] → [B, half_dim]
        emb = t.float().unsqueeze(1) * emb.unsqueeze(0)
        
        # Concatenate sin and cos: [B, embedding_dim]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
        
        # Pad if embedding_dim is odd
        if self.embedding_dim % 2 == 1:
            emb = F.pad(emb, (0, 1))
        
        return emb


class TimeEmbedding(nn.Module):
    """
    Time embedding module: sinusoidal encoding + MLP.
    
    The sinusoidal encoding gives a unique representation for each timestep,
    and the MLP allows the network to learn a useful transformation of it.
    
    Architecture:
        Sinusoidal(t) → Linear(dim → 4*dim) → SiLU → Linear(4*dim → dim)
    """
    def __init__(self, time_emb_dim: int):
        super().__init__()
        self.sinusoidal = SinusoidalPositionEmbedding(time_emb_dim)
        self.mlp = nn.Sequential(
            nn.Linear(time_emb_dim, time_emb_dim * 4),
            nn.SiLU(),
            nn.Linear(time_emb_dim * 4, time_emb_dim),
        )
    
    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        Args:
            t: Timesteps [B] (integers)
        
        Returns:
            Time embeddings [B, time_emb_dim]
        """
        return self.mlp(self.sinusoidal(t))


# =============================================================================
# Building Blocks
# =============================================================================

class ResBlock(nn.Module):
    """
    Residual block with time conditioning.
    
    
    The time embedding is projected to match the intermediate channel dimension
    and added to the hidden state after the first convolution. This is a simple
    but effective way to condition the network on the timestep.
    
    Args:
        in_channels: Number of input channels (may include concatenated skip)
        out_channels: Number of output channels
        time_emb_dim: Dimension of time embedding
        dropout: Dropout probability
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        time_emb_dim: int,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        # First conv block
        self.norm1 = nn.GroupNorm(num_groups=8, num_channels=in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        
        # Time embedding projection
        self.time_mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_emb_dim, out_channels),
        )
        
        # Second conv block
        self.norm2 = nn.GroupNorm(num_groups=8, num_channels=out_channels)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        
        # Residual connection (1x1 conv if channels change)
        if in_channels != out_channels:
            self.residual_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        else:
            self.residual_conv = nn.Identity()
    
    def forward(self, x: torch.Tensor, time_emb: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor [B, in_channels, H, W]
            time_emb: Time embeddings [B, time_emb_dim]
        
        Returns:
            Output tensor [B, out_channels, H, W]
        """
        h = self.norm1(x)
        h = F.silu(h)
        h = self.conv1(h)
        
        # Inject time embedding: project to [B, out_channels, 1, 1] and add
        t = self.time_mlp(time_emb).unsqueeze(-1).unsqueeze(-1)
        h = h + t
        
        h = self.norm2(h)
        h = F.silu(h)
        h = self.dropout(h)
        h = self.conv2(h)
        
        # Residual connection
        return h + self.residual_conv(x)


class Downsample(nn.Module):
    """
    Spatial downsampling by factor of 2 using strided convolution.
    
    Input:  [B, C, H, W]
    Output: [B, C, H/2, W/2]
    """
    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, kernel_size=3, stride=2, padding=1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class Upsample(nn.Module):
    """
    Spatial upsampling by factor of 2 using interpolation + convolution.
    
    We use nearest-neighbor interpolation followed by a conv instead of
    ConvTranspose2d to avoid checkerboard artifacts.
    
    Input:  [B, C, H, W]
    Output: [B, C, 2H, 2W]
    """
    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, scale_factor=2, mode='nearest')
        return self.conv(x)


class SelfAttention(nn.Module):
    """
    Multi-head self-attention block.
    
    Applied at the lowest resolution (8×8) to capture global structure
    without excessive computation.
    
    
    Args:
        channels: Number of channels
        num_heads: Number of attention heads (must divide channels)
    """
    def __init__(self, channels: int, num_heads: int = 4):
        super().__init__()
        assert channels % num_heads == 0, f"channels ({channels}) must be divisible by num_heads ({num_heads})"
        
        self.channels = channels
        self.num_heads = num_heads
        self.head_dim = channels // num_heads
        
        self.norm = nn.GroupNorm(num_groups=8, num_channels=channels)
        self.qkv = nn.Conv2d(channels, channels * 3, kernel_size=1)
        self.proj_out = nn.Conv2d(channels, channels, kernel_size=1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor [B, C, H, W]
        
        Returns:
            Output tensor [B, C, H, W]
        """
        B, C, H, W = x.shape
        
        h = self.norm(x)
        qkv = self.qkv(h)
        
        # Split into Q, K, V: [B, 3*C, H, W] → 3 × [B, C, H, W]
        q, k, v = qkv.chunk(3, dim=1)
        
        # Reshape for attention: [B, C, H, W] → [B, num_heads, head_dim, H*W]
        q = q.reshape(B, self.num_heads, self.head_dim, H * W)
        k = k.reshape(B, self.num_heads, self.head_dim, H * W)
        v = v.reshape(B, self.num_heads, self.head_dim, H * W)
        
        # Transpose for attention computation
        q = q.permute(0, 1, 3, 2)  # [B, num_heads, H*W, head_dim]
        k = k.permute(0, 1, 2, 3)  # [B, num_heads, head_dim, H*W]
        v = v.permute(0, 1, 3, 2)  # [B, num_heads, H*W, head_dim]
        
        # Scaled dot-product attention
        # [B, num_heads, H*W, head_dim] @ [B, num_heads, head_dim, H*W] → [B, num_heads, H*W, H*W]
        scale = self.head_dim ** -0.5
        attn = torch.matmul(q, k) * scale
        attn = F.softmax(attn, dim=-1)
        
        # Apply attention to values
        # [B, num_heads, H*W, H*W] @ [B, num_heads, H*W, head_dim] → [B, num_heads, H*W, head_dim]
        out = torch.matmul(attn, v)
        
        # Reshape back: [B, num_heads, H*W, head_dim] → [B, C, H, W]
        out = out.permute(0, 1, 3, 2).reshape(B, C, H, W)
        out = self.proj_out(out)
        
        # Residual connection
        return x + out


# =============================================================================
# Full U-Net
# =============================================================================

class UNet(nn.Module):
    """
    U-Net for DDPM noise prediction.
    
    Takes a noisy image x_t and timestep t, predicts the noise ε_θ(x_t, t).
    
    Architecture:
        - Encoder: progressively downsamples with ResBlocks
        - Bottleneck: ResBlock at lowest resolution
        - Decoder: progressively upsamples with ResBlocks + skip connections
        - Self-attention at the lowest resolution (optional)
    
    Args:
        in_channels: Number of input channels (e.g., 3 for RGB)
        out_channels: Number of output channels (usually same as in_channels)
        unet_channels: Channel progression, e.g., [64, 128, 256]
        time_emb_dim: Dimension of time embedding
        num_res_blocks: Number of ResBlocks per resolution level
        use_attention: Whether to use self-attention at lowest resolution
        dropout: Dropout probability
    """
    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        unet_channels: List[int] = [64, 128, 256],
        time_emb_dim: int = 128,
        num_res_blocks: int = 2,
        use_attention: bool = True,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.unet_channels = unet_channels
        self.time_emb_dim = time_emb_dim
        self.num_res_blocks = num_res_blocks
        self.use_attention = use_attention
        
        # Time embedding
        self.time_embed = TimeEmbedding(time_emb_dim)
        
        # Initial convolution
        self.input_conv = nn.Conv2d(in_channels, unet_channels[0], kernel_size=3, padding=1)
        
        # Encoder (downsampling path)
        self.encoder_blocks = nn.ModuleList()
        self.downsamples = nn.ModuleList()
        
        current_channels = unet_channels[0]
        for level, out_ch in enumerate(unet_channels):
            level_blocks = nn.ModuleList()
            for _ in range(num_res_blocks):
                level_blocks.append(
                    ResBlock(current_channels, out_ch, time_emb_dim, dropout)
                )
                current_channels = out_ch
            self.encoder_blocks.append(level_blocks)
            
            # Downsample after each level except the last
            if level < len(unet_channels) - 1:
                self.downsamples.append(Downsample(current_channels))
            else:
                self.downsamples.append(nn.Identity())
        
        # Bottleneck
        self.mid_block1 = ResBlock(current_channels, current_channels, time_emb_dim, dropout)
        if use_attention:
            self.mid_attn = SelfAttention(current_channels)
        else:
            self.mid_attn = nn.Identity()
        self.mid_block2 = ResBlock(current_channels, current_channels, time_emb_dim, dropout)
        
        # Decoder (upsampling path)
        self.decoder_blocks = nn.ModuleList()
        self.upsamples = nn.ModuleList()
        
        for level, out_ch in reversed(list(enumerate(unet_channels))):
            # After concatenation with skip, input channels = current + skip
            skip_channels = unet_channels[level]
            
            level_blocks = nn.ModuleList()
            for i in range(num_res_blocks + 1):  # +1 for the skip connection
                in_ch = current_channels + skip_channels if i == 0 else current_channels
                level_blocks.append(
                    ResBlock(in_ch, out_ch, time_emb_dim, dropout)
                )
                current_channels = out_ch
            
            self.decoder_blocks.append(level_blocks)
            
            # Upsample after each level except the last
            if level > 0:
                self.upsamples.append(Upsample(current_channels))
            else:
                self.upsamples.append(nn.Identity())
        
        # Output
        self.output_norm = nn.GroupNorm(num_groups=8, num_channels=current_channels)
        self.output_conv = nn.Conv2d(current_channels, out_channels, kernel_size=3, padding=1)
    
    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Predict noise given noisy image and timestep.
        
        Args:
            x: Noisy images [B, C, H, W] in [-1, 1]
            t: Timesteps [B] (integers in [0, T-1])
        
        Returns:
            Predicted noise [B, C, H, W] (same shape as input)
        """
        # Time embedding
        time_emb = self.time_embed(t)
        
        # Initial conv
        h = self.input_conv(x)
        
        # Encoder: collect skip connections
        skips = [h]
        for level, (blocks, downsample) in enumerate(zip(self.encoder_blocks, self.downsamples)):
            for block in blocks:
                h = block(h, time_emb)
            skips.append(h)
            h = downsample(h)
        
        # Bottleneck
        h = self.mid_block1(h, time_emb)
        h = self.mid_attn(h)
        h = self.mid_block2(h, time_emb)
        
        # Decoder: use skip connections
        for level, (blocks, upsample) in enumerate(zip(self.decoder_blocks, self.upsamples)):
            for i, block in enumerate(blocks):
                if i == 0:
                    # Concatenate with skip connection
                    skip = skips.pop()
                    h = torch.cat([h, skip], dim=1)
                h = block(h, time_emb)
            h = upsample(h)
        
        # Output
        h = self.output_norm(h)
        h = F.silu(h)
        h = self.output_conv(h)
        
        return h


# =============================================================================
# Testing
# =============================================================================

if __name__ == '__main__':
    # Quick test to verify dimensions
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model = UNet(
        in_channels=3,
        out_channels=3,
        unet_channels=[64, 128, 256],
        time_emb_dim=128,
        num_res_blocks=2,
        use_attention=True,
        dropout=0.1,
    ).to(device)
    
    # Count parameters
    num_params = sum(p.numel() for p in model.parameters())
    print(f"U-Net parameters: {num_params:,}")
    
    # Test forward pass
    x = torch.randn(2, 3, 32, 32, device=device)
    t = torch.randint(0, 1000, (2,), device=device)
    
    with torch.no_grad():
        out = model(x, t)
    
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {out.shape}")
    print(f"Output range: [{out.min():.3f}, {out.max():.3f}]")
    assert out.shape == x.shape, "Output shape must match input shape"
    print("✓ U-Net test passed!")