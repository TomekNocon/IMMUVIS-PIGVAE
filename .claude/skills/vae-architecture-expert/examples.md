# VAE Architecture Examples

## Complete PyTorch Implementation Examples

### 1. Transformer VAE for Images (No Positional Encoding in Encoder)

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from einops.layers.torch import Rearrange

class PatchEmbedding(nn.Module):
    def __init__(self, image_size, patch_size, embed_dim, channels=3):
        super().__init__()
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_patches = (image_size // patch_size) ** 2

        self.projection = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)',
                     p1=patch_size, p2=patch_size),
            nn.LayerNorm(patch_size * patch_size * channels),
            nn.Linear(patch_size * patch_size * channels, embed_dim),
            nn.LayerNorm(embed_dim)
        )

    def forward(self, x):
        return self.projection(x)

class TransformerEncoder(nn.Module):
    def __init__(self, embed_dim, num_layers, num_heads, mlp_ratio=4.0, dropout=0.1):
        super().__init__()
        self.embed_dim = embed_dim

        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=embed_dim,
                nhead=num_heads,
                dim_feedforward=int(embed_dim * mlp_ratio),
                dropout=dropout,
                activation='gelu',
                batch_first=True
            ) for _ in range(num_layers)
        ])

        # Global pooling for spatial invariance
        self.global_pool = nn.AdaptiveAvgPool1d(1)

    def forward(self, x):
        # x shape: (batch, num_patches, embed_dim)
        for layer in self.layers:
            x = layer(x)

        # Global average pooling across patches for invariance
        x = rearrange(x, 'b n d -> b d n')
        x = self.global_pool(x)
        x = rearrange(x, 'b d 1 -> b d')

        return x

class TransformerDecoder(nn.Module):
    def __init__(self, latent_dim, embed_dim, num_layers, num_heads,
                 image_size, patch_size, channels=3, mlp_ratio=4.0, dropout=0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_patches = (image_size // patch_size) ** 2

        # Project latent to patch tokens
        self.latent_to_patches = nn.Linear(latent_dim, embed_dim * self.num_patches)

        # Learnable positional embedding
        self.pos_embedding = nn.Parameter(torch.randn(1, self.num_patches, embed_dim))

        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=embed_dim,
                nhead=num_heads,
                dim_feedforward=int(embed_dim * mlp_ratio),
                dropout=dropout,
                activation='gelu',
                batch_first=True
            ) for _ in range(num_layers)
        ])

        # Project back to pixel space
        self.to_pixels = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, patch_size * patch_size * channels),
            Rearrange('b (h w) (p1 p2 c) -> b c (h p1) (w p2)',
                     h=image_size//patch_size, w=image_size//patch_size,
                     p1=patch_size, p2=patch_size)
        )

    def forward(self, z):
        # z shape: (batch, latent_dim)
        batch_size = z.size(0)

        # Project latent to patch tokens
        x = self.latent_to_patches(z)
        x = rearrange(x, 'b (n d) -> b n d', n=self.num_patches)

        # Add positional encoding
        x = x + self.pos_embedding

        # Apply transformer layers
        for layer in self.layers:
            x = layer(x)

        # Convert to image
        return self.to_pixels(x)

class TransformerVAE(nn.Module):
    def __init__(self, image_size=224, patch_size=16, channels=3,
                 latent_dim=512, embed_dim=768,
                 encoder_layers=12, decoder_layers=16,
                 num_heads=12, mlp_ratio=4.0, dropout=0.1):
        super().__init__()

        self.patch_embed = PatchEmbedding(image_size, patch_size, embed_dim, channels)

        self.encoder = TransformerEncoder(
            embed_dim, encoder_layers, num_heads, mlp_ratio, dropout
        )

        # VAE latent space projection
        self.to_latent = nn.Linear(embed_dim, latent_dim * 2)  # mu and logvar

        self.decoder = TransformerDecoder(
            latent_dim, embed_dim, decoder_layers, num_heads,
            image_size, patch_size, channels, mlp_ratio, dropout
        )

    def encode(self, x):
        x = self.patch_embed(x)
        x = self.encoder(x)
        mu, logvar = self.to_latent(x).chunk(2, dim=-1)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        return recon, mu, logvar
```

### 2. Convolutional VAE with Attention

```python
class ConvAttentionBlock(nn.Module):
    def __init__(self, channels, num_heads=8):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.attention = nn.MultiheadAttention(channels, num_heads, batch_first=True)
        self.norm1 = nn.GroupNorm(8, channels)
        self.norm2 = nn.GroupNorm(8, channels)

    def forward(self, x):
        # Residual connection
        identity = x

        # First conv
        out = F.gelu(self.norm1(self.conv1(x)))
        out = self.norm2(self.conv2(out))

        # Self-attention on spatial features
        b, c, h, w = out.shape
        out_flat = rearrange(out, 'b c h w -> b (h w) c')
        attn_out, _ = self.attention(out_flat, out_flat, out_flat)
        out = rearrange(attn_out, 'b (h w) c -> b c h w', h=h, w=w)

        return F.gelu(out + identity)

class ConvVAEWithAttention(nn.Module):
    def __init__(self, channels=3, latent_dim=256, base_channels=64):
        super().__init__()

        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(channels, base_channels, 4, 2, 1),  # 128x128
            nn.GroupNorm(8, base_channels),
            nn.GELU(),

            ConvAttentionBlock(base_channels),

            nn.Conv2d(base_channels, base_channels*2, 4, 2, 1),  # 64x64
            nn.GroupNorm(8, base_channels*2),
            nn.GELU(),

            ConvAttentionBlock(base_channels*2),

            nn.Conv2d(base_channels*2, base_channels*4, 4, 2, 1),  # 32x32
            nn.GroupNorm(8, base_channels*4),
            nn.GELU(),

            ConvAttentionBlock(base_channels*4),

            nn.Conv2d(base_channels*4, base_channels*8, 4, 2, 1),  # 16x16
            nn.GroupNorm(8, base_channels*8),
            nn.GELU(),
        )

        # Global average pooling for invariance
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.to_latent = nn.Linear(base_channels*8, latent_dim*2)

        # Decoder with positional information
        self.from_latent = nn.Linear(latent_dim, base_channels*8 * 16 * 16)

        self.decoder = nn.Sequential(
            ConvAttentionBlock(base_channels*8),

            nn.ConvTranspose2d(base_channels*8, base_channels*4, 4, 2, 1),  # 32x32
            nn.GroupNorm(8, base_channels*4),
            nn.GELU(),

            ConvAttentionBlock(base_channels*4),

            nn.ConvTranspose2d(base_channels*4, base_channels*2, 4, 2, 1),  # 64x64
            nn.GroupNorm(8, base_channels*2),
            nn.GELU(),

            ConvAttentionBlock(base_channels*2),

            nn.ConvTranspose2d(base_channels*2, base_channels, 4, 2, 1),  # 128x128
            nn.GroupNorm(8, base_channels),
            nn.GELU(),

            ConvAttentionBlock(base_channels),

            nn.ConvTranspose2d(base_channels, channels, 4, 2, 1),  # 256x256
            nn.Sigmoid()
        )

    def encode(self, x):
        x = self.encoder(x)
        x = self.global_pool(x).flatten(1)
        mu, logvar = self.to_latent(x).chunk(2, dim=1)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        x = self.from_latent(z)
        x = x.view(-1, 512, 16, 16)  # Reshape to spatial
        return self.decoder(x)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        return recon, mu, logvar
```

## Training Example

### Complete Training Loop with Advanced Losses

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision.models import vgg16
import lpips

class VAETrainer:
    def __init__(self, model, device='cuda'):
        self.model = model.to(device)
        self.device = device

        # Optimizers with different learning rates
        self.encoder_optimizer = optim.AdamW(
            list(model.patch_embed.parameters()) + list(model.encoder.parameters()) +
            list(model.to_latent.parameters()),
            lr=1e-4, weight_decay=1e-5
        )
        self.decoder_optimizer = optim.AdamW(
            model.decoder.parameters(),
            lr=3e-4, weight_decay=1e-5
        )

        # Schedulers
        self.encoder_scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.encoder_optimizer, T_max=100
        )
        self.decoder_scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.decoder_optimizer, T_max=100
        )

        # Perceptual loss
        self.lpips_loss = lpips.LPIPS(net='vgg').to(device)

        # VGG for perceptual loss
        vgg = vgg16(pretrained=True).features[:16].to(device)
        for param in vgg.parameters():
            param.requires_grad = False
        self.vgg = vgg

    def perceptual_loss(self, input, target):
        input_features = self.vgg(input)
        target_features = self.vgg(target)
        return F.mse_loss(input_features, target_features)

    def ssim_loss(self, input, target, window_size=11):
        return 1 - self.ssim(input, target, window_size)

    def compute_loss(self, recon, target, mu, logvar, epoch, total_epochs):
        # Reconstruction losses
        mse_loss = F.mse_loss(recon, target)
        perceptual_loss = self.perceptual_loss(recon, target)
        lpips_loss = self.lpips_loss(recon, target).mean()

        # Combined reconstruction loss
        recon_loss = 0.4 * mse_loss + 0.3 * perceptual_loss + 0.3 * lpips_loss

        # KL divergence with β-annealing
        beta = min(1.0, epoch / (total_epochs * 0.1))  # 10% warmup
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)
        kl_loss = beta * kl_loss.mean()

        total_loss = recon_loss + kl_loss

        return total_loss, recon_loss, kl_loss, beta

    def train_epoch(self, dataloader, epoch, total_epochs):
        self.model.train()
        total_loss = 0
        total_recon = 0
        total_kl = 0

        for batch_idx, (data, _) in enumerate(dataloader):
            data = data.to(self.device)

            # Forward pass
            recon, mu, logvar = self.model(data)

            # Compute loss
            loss, recon_loss, kl_loss, beta = self.compute_loss(
                recon, data, mu, logvar, epoch, total_epochs
            )

            # Backward pass
            self.encoder_optimizer.zero_grad()
            self.decoder_optimizer.zero_grad()
            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

            self.encoder_optimizer.step()
            self.decoder_optimizer.step()

            total_loss += loss.item()
            total_recon += recon_loss.item()
            total_kl += kl_loss.item()

            if batch_idx % 100 == 0:
                print(f'Epoch {epoch}, Batch {batch_idx}: '
                      f'Loss={loss.item():.4f}, '
                      f'Recon={recon_loss.item():.4f}, '
                      f'KL={kl_loss.item():.4f}, '
                      f'β={beta:.4f}')

        self.encoder_scheduler.step()
        self.decoder_scheduler.step()

        return total_loss / len(dataloader), total_recon / len(dataloader), total_kl / len(dataloader)

# Usage example
def main():
    # Model configuration
    model = TransformerVAE(
        image_size=256,
        patch_size=16,
        channels=3,
        latent_dim=512,
        embed_dim=768,
        encoder_layers=12,
        decoder_layers=16,
        num_heads=12
    )

    # Training
    trainer = VAETrainer(model)

    # Your dataloader here
    # trainer.train_epoch(dataloader, epoch, total_epochs)
```

## Configuration Examples

### Small Model (Fast Training/Inference)
```python
small_config = {
    'image_size': 128,
    'patch_size': 16,
    'latent_dim': 256,
    'embed_dim': 384,
    'encoder_layers': 6,
    'decoder_layers': 8,
    'num_heads': 6,
    'mlp_ratio': 4.0
}
```

### Medium Model (Balanced Quality/Speed)
```python
medium_config = {
    'image_size': 256,
    'patch_size': 16,
    'latent_dim': 512,
    'embed_dim': 512,
    'encoder_layers': 12,
    'decoder_layers': 16,
    'num_heads': 8,
    'mlp_ratio': 4.0
}
```

### Large Model (High Quality)
```python
large_config = {
    'image_size': 512,
    'patch_size': 16,
    'latent_dim': 1024,
    'embed_dim': 768,
    'encoder_layers': 24,
    'decoder_layers': 32,
    'num_heads': 12,
    'mlp_ratio': 4.0
}
```
