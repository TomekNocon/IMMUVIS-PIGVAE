# VAE Architecture Improvements

## Critical Improvements for Better Performance

### 1. Enhanced Encoder with Multi-Scale Attention Pooling

**Current Issue**: Simple global pooling loses important spatial relationships and fine details.

**Improvement**: Replace global average pooling with learned attention pooling:

```python
class AttentionPooling(nn.Module):
    def __init__(self, embed_dim, num_queries=8):
        super().__init__()
        self.num_queries = num_queries
        self.query_embed = nn.Parameter(torch.randn(num_queries, embed_dim))
        self.attention = nn.MultiheadAttention(embed_dim, 8, batch_first=True)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        # x: (batch, num_patches, embed_dim)
        batch_size = x.size(0)
        queries = self.query_embed.unsqueeze(0).expand(batch_size, -1, -1)

        # Cross-attention: queries attend to patch features
        pooled, _ = self.attention(queries, x, x)
        pooled = self.norm(pooled)

        # Aggregate queries (mean or learned combination)
        return pooled.mean(dim=1)  # (batch, embed_dim)

class ImprovedTransformerEncoder(nn.Module):
    def __init__(self, embed_dim, num_layers, num_heads, mlp_ratio=4.0, dropout=0.1):
        super().__init__()
        self.embed_dim = embed_dim

        # Add LayerScale for better training stability
        self.layers = nn.ModuleList([
            LayerScaleTransformerBlock(embed_dim, num_heads, mlp_ratio, dropout)
            for _ in range(num_layers)
        ])

        # Multi-scale feature extraction
        self.multiscale_conv = nn.ModuleList([
            nn.Conv1d(embed_dim, embed_dim, kernel_size=k, padding=k//2, groups=embed_dim)
            for k in [3, 5, 7]  # Different receptive fields
        ])

        # Attention pooling instead of global pooling
        self.attention_pool = AttentionPooling(embed_dim, num_queries=16)

    def forward(self, x):
        # x shape: (batch, num_patches, embed_dim)
        for layer in self.layers:
            x = layer(x)

        # Multi-scale feature fusion
        x_conv = x.transpose(1, 2)  # (batch, embed_dim, num_patches)
        multiscale_features = []
        for conv in self.multiscale_conv:
            multiscale_features.append(conv(x_conv))

        # Combine multi-scale features
        x_combined = torch.stack(multiscale_features, dim=-1).mean(dim=-1)
        x = x_combined.transpose(1, 2) + x  # Residual connection

        # Attention pooling for invariant representation
        return self.attention_pool(x)

class LayerScaleTransformerBlock(nn.Module):
    """Transformer block with LayerScale for better training"""
    def __init__(self, embed_dim, num_heads, mlp_ratio=4.0, dropout=0.1, init_scale=1e-4):
        super().__init__()
        self.attention = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, int(embed_dim * mlp_ratio)),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(int(embed_dim * mlp_ratio), embed_dim),
            nn.Dropout(dropout)
        )
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)

        # LayerScale parameters
        self.gamma1 = nn.Parameter(init_scale * torch.ones(embed_dim))
        self.gamma2 = nn.Parameter(init_scale * torch.ones(embed_dim))

    def forward(self, x):
        # Self-attention with LayerScale
        attn_out, _ = self.attention(x, x, x)
        x = x + self.gamma1 * attn_out
        x = self.norm1(x)

        # MLP with LayerScale
        mlp_out = self.mlp(x)
        x = x + self.gamma2 * mlp_out
        x = self.norm2(x)

        return x
```

### 2. Advanced Decoder with Progressive Reconstruction

**Current Issue**: Direct reconstruction from latent can miss fine details.

**Improvement**: Multi-stage decoder with progressive upsampling:

```python
class ProgressiveDecoder(nn.Module):
    def __init__(self, latent_dim, embed_dim, num_layers, num_heads,
                 image_size, patch_size, channels=3):
        super().__init__()
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_patches = (image_size // patch_size) ** 2

        # Progressive scales: 4x4 -> 8x8 -> 16x16 -> 32x32 patches
        self.scales = [4, 8, 16, max(32, image_size // patch_size)]

        # Initial projection from latent
        self.latent_proj = nn.Linear(latent_dim, embed_dim * 16)  # Start with 4x4

        # Progressive decoders for each scale
        self.progressive_decoders = nn.ModuleList([
            self._make_scale_decoder(embed_dim, num_heads, scale)
            for scale in self.scales
        ])

        # Cross-scale attention for feature fusion
        self.cross_scale_attention = nn.ModuleList([
            CrossScaleAttention(embed_dim) for _ in range(len(self.scales)-1)
        ])

        # Final reconstruction heads for each scale
        self.reconstruction_heads = nn.ModuleList([
            self._make_recon_head(embed_dim, channels, scale)
            for scale in self.scales
        ])

        # Learned positional embeddings for each scale
        self.pos_embeddings = nn.ParameterList([
            nn.Parameter(torch.randn(1, scale*scale, embed_dim))
            for scale in self.scales
        ])

    def _make_scale_decoder(self, embed_dim, num_heads, scale):
        return nn.ModuleList([
            LayerScaleTransformerBlock(embed_dim, num_heads)
            for _ in range(4)  # 4 layers per scale
        ])

    def _make_recon_head(self, embed_dim, channels, scale):
        patch_pixels = (self.image_size // scale) ** 2 * channels
        return nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, patch_pixels),
            nn.Unflatten(1, (channels, self.image_size // scale, self.image_size // scale))
        )

    def forward(self, z):
        # Start from latent
        x = self.latent_proj(z)  # (batch, embed_dim * 16)
        x = x.view(-1, 16, x.size(-1) // 16)  # (batch, 16, embed_dim) for 4x4

        reconstructions = []

        for i, (scale, decoder, pos_embed) in enumerate(zip(
            self.scales, self.progressive_decoders, self.pos_embeddings)):

            # Add positional encoding
            if i > 0:
                # Upsample from previous scale
                x = F.interpolate(x.permute(0, 2, 1).view(
                    -1, x.size(-1), int(math.sqrt(x.size(1))), int(math.sqrt(x.size(1)))
                ), size=(scale, scale), mode='bilinear').flatten(2).permute(0, 2, 1)

                # Cross-scale attention
                if i > 0:
                    x = self.cross_scale_attention[i-1](x, reconstructions[-1])

            x = x + pos_embed[:, :x.size(1), :]

            # Apply decoder layers
            for layer in decoder:
                x = layer(x)

            # Generate reconstruction at this scale
            recon = self.reconstruction_heads[i](x)
            reconstructions.append(recon)

        return reconstructions  # Return all scales for multi-scale loss

class CrossScaleAttention(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.cross_attention = nn.MultiheadAttention(embed_dim, 8, batch_first=True)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, current_features, prev_reconstruction):
        # Use previous reconstruction as context
        batch_size = current_features.size(0)

        # Flatten previous reconstruction and project
        prev_flat = prev_reconstruction.flatten(2).permute(0, 2, 1)
        prev_proj = self.norm(prev_flat)

        # Cross-attention
        enhanced, _ = self.cross_attention(current_features, prev_proj, prev_proj)
        return current_features + enhanced
```

### 3. Advanced Loss Functions for Better Training

**Current Issue**: Simple MSE + KL doesn't capture perceptual quality.

**Improvement**: Multi-scale perceptual loss with adaptive weighting:

```python
class AdvancedVAELoss(nn.Module):
    def __init__(self, device='cuda', scales=[1.0, 0.5, 0.25]):
        super().__init__()
        self.device = device
        self.scales = scales

        # Multi-scale LPIPS
        self.lpips_nets = nn.ModuleList([
            lpips.LPIPS(net='vgg').to(device) for _ in scales
        ])

        # Style loss (Gram matrices)
        self.style_layers = [3, 8, 15, 22]  # VGG layers
        vgg = models.vgg19(pretrained=True).features.to(device).eval()
        for param in vgg.parameters():
            param.requires_grad = False
        self.vgg = vgg

        # Adaptive loss weights
        self.loss_weights = nn.Parameter(torch.ones(5))  # 5 loss components

    def forward(self, reconstructions, target, mu, logvar, epoch, total_epochs):
        device = target.device
        total_loss = 0
        loss_dict = {}

        # 1. Multi-scale reconstruction loss
        recon_loss = 0
        for i, (recon, scale) in enumerate(zip(reconstructions, self.scales)):
            if scale < 1.0:
                target_scaled = F.interpolate(target, scale_factor=scale)
            else:
                target_scaled = target

            # MSE loss
            mse = F.mse_loss(recon, target_scaled)

            # Perceptual loss (LPIPS)
            lpips_loss = self.lpips_nets[i](recon, target_scaled).mean()

            # SSIM loss
            ssim_val = ssim(recon, target_scaled, data_range=1.0, size_average=True)
            ssim_loss = 1 - ssim_val

            scale_loss = 0.4 * mse + 0.4 * lpips_loss + 0.2 * ssim_loss
            recon_loss += scale_loss / len(reconstructions)

        # 2. Style loss (Gram matrix matching)
        style_loss = self.compute_style_loss(reconstructions[-1], target)

        # 3. KL divergence with β-annealing
        beta = min(1.0, epoch / (total_epochs * 0.15))  # 15% warmup
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1).mean()

        # 4. Total Correlation loss for disentanglement
        tc_loss = self.compute_total_correlation(mu, logvar)

        # 5. Spectral regularization
        spectral_loss = self.compute_spectral_regularization(mu)

        # Adaptive weighting
        weights = F.softmax(self.loss_weights, dim=0)
        total_loss = (weights[0] * recon_loss +
                     weights[1] * style_loss +
                     weights[2] * beta * kl_loss +
                     weights[3] * tc_loss +
                     weights[4] * spectral_loss)

        loss_dict = {
            'total': total_loss,
            'reconstruction': recon_loss,
            'style': style_loss,
            'kl': kl_loss,
            'tc': tc_loss,
            'spectral': spectral_loss,
            'beta': beta
        }

        return total_loss, loss_dict

    def compute_style_loss(self, input, target):
        """Compute style loss using Gram matrices"""
        input_features = self.extract_features(input)
        target_features = self.extract_features(target)

        style_loss = 0
        for inp_feat, targ_feat in zip(input_features, target_features):
            # Compute Gram matrices
            inp_gram = self.gram_matrix(inp_feat)
            targ_gram = self.gram_matrix(targ_feat)
            style_loss += F.mse_loss(inp_gram, targ_gram)

        return style_loss / len(input_features)

    def extract_features(self, x):
        """Extract features from VGG"""
        features = []
        for i, layer in enumerate(self.vgg):
            x = layer(x)
            if i in self.style_layers:
                features.append(x)
        return features

    def gram_matrix(self, x):
        """Compute Gram matrix"""
        b, c, h, w = x.size()
        features = x.view(b, c, h * w)
        gram = torch.bmm(features, features.transpose(1, 2))
        return gram.div(c * h * w)

    def compute_total_correlation(self, mu, logvar):
        """Compute total correlation for disentanglement"""
        batch_size, latent_dim = mu.size()

        # Sample from posterior
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std

        # Compute log q(z|x)
        log_qz_condx = (-0.5 * ((z - mu) / std).pow(2) - 0.5 * logvar - 0.5 * math.log(2 * math.pi)).sum(dim=1)

        # Compute log q(z) ≈ log(1/N * Σ q(z|x_i))
        log_qz = torch.logsumexp(
            (-0.5 * (z.unsqueeze(1) - mu.unsqueeze(0)).pow(2) / std.unsqueeze(0).pow(2)
             - 0.5 * logvar.unsqueeze(0) - 0.5 * math.log(2 * math.pi)).sum(dim=2),
            dim=1
        ) - math.log(batch_size)

        # TC = KL(q(z)||∏q(z_j))
        return (log_qz - log_qz_condx).mean()

    def compute_spectral_regularization(self, mu):
        """Regularize latent space to prevent mode collapse"""
        # Encourage diversity in latent codes
        mu_mean = mu.mean(dim=0, keepdim=True)
        mu_centered = mu - mu_mean

        # Compute covariance matrix
        cov = torch.mm(mu_centered.T, mu_centered) / (mu.size(0) - 1)

        # Eigenvalue regularization - encourage full rank
        eigenvals = torch.linalg.eigvals(cov).real
        return -torch.log(eigenvals + 1e-8).mean()
```

### 4. Improved Training Strategy

**Current Issue**: Static training approach doesn't adapt to model needs.

**Improvement**: Dynamic curriculum with adaptive scheduling:

```python
class AdaptiveVAETrainer:
    def __init__(self, model, device='cuda'):
        self.model = model.to(device)
        self.device = device

        # Separate optimizers with different schedules
        self.encoder_optimizer = torch.optim.AdamW(
            self.model.encoder.parameters(),
            lr=1e-4, weight_decay=1e-5, betas=(0.9, 0.95)
        )
        self.decoder_optimizer = torch.optim.AdamW(
            self.model.decoder.parameters(),
            lr=2e-4, weight_decay=1e-5, betas=(0.9, 0.95)
        )

        # Advanced schedulers
        self.encoder_scheduler = torch.optim.lr_scheduler.OneCycleLR(
            self.encoder_optimizer, max_lr=2e-4, total_steps=1000
        )
        self.decoder_scheduler = torch.optim.lr_scheduler.OneCycleLR(
            self.decoder_optimizer, max_lr=4e-4, total_steps=1000
        )

        # EMA for stable training
        self.ema = ExponentialMovingAverage(model.parameters(), decay=0.999)

        # Gradient scaling
        self.scaler = torch.cuda.amp.GradScaler()

        # Loss function
        self.loss_fn = AdvancedVAELoss(device)

    def train_step(self, batch, epoch, total_epochs):
        self.model.train()
        data, _ = batch
        data = data.to(self.device)

        with torch.cuda.amp.autocast():
            # Forward pass
            reconstructions, mu, logvar = self.model(data)

            # Compute loss
            loss, loss_dict = self.loss_fn(
                reconstructions, data, mu, logvar, epoch, total_epochs
            )

        # Backward pass with gradient scaling
        self.encoder_optimizer.zero_grad()
        self.decoder_optimizer.zero_grad()

        self.scaler.scale(loss).backward()

        # Gradient clipping
        self.scaler.unscale_(self.encoder_optimizer)
        self.scaler.unscale_(self.decoder_optimizer)
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

        # Optimizer steps
        self.scaler.step(self.encoder_optimizer)
        self.scaler.step(self.decoder_optimizer)
        self.scaler.update()

        # Update EMA
        self.ema.update()

        # Update schedulers
        self.encoder_scheduler.step()
        self.decoder_scheduler.step()

        return loss_dict

class ExponentialMovingAverage:
    def __init__(self, parameters, decay=0.999):
        self.parameters = list(parameters)
        self.decay = decay
        self.shadow = [p.clone().detach() for p in self.parameters]

    def update(self):
        for param, shadow in zip(self.parameters, self.shadow):
            if param.grad is not None:
                shadow.copy_(self.decay * shadow + (1.0 - self.decay) * param)

    def apply(self):
        for param, shadow in zip(self.parameters, self.shadow):
            param.data.copy_(shadow)
```

## Summary of Key Improvements

1. **Attention Pooling**: Replace global pooling with learned attention for better invariance
2. **LayerScale**: Add LayerScale to transformer blocks for training stability
3. **Multi-Scale Features**: Extract features at multiple scales in encoder
4. **Progressive Decoding**: Reconstruct at multiple resolutions with cross-scale attention
5. **Advanced Loss**: Multi-scale perceptual + style + disentanglement losses
6. **Adaptive Training**: EMA, gradient scaling, and OneCycle scheduling

These improvements will significantly enhance both reconstruction quality and the quality of learned invariant representations.
