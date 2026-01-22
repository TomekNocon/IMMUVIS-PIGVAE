# VAE Architecture Reference

## Mathematical Foundations

### VAE Objective Function

The VAE objective combines reconstruction loss and regularization:

```
L(θ,φ) = E[log p_θ(x|z)] - β * KL(q_φ(z|x) || p(z))

Where:
- θ: decoder parameters
- φ: encoder parameters
- β: regularization weight
- q_φ(z|x): encoder (recognition model)
- p_θ(x|z): decoder (generative model)
- p(z): prior (usually N(0,I))
```

### Advanced Loss Functions

#### 1. β-VAE with Total Correlation
```python
def beta_tc_vae_loss(recon, target, mu, logvar, beta=1.0, gamma=1.0):
    # Reconstruction term
    recon_loss = F.binary_cross_entropy_with_logits(recon, target, reduction='sum')

    # KL divergence components
    batch_size, latent_dim = mu.size()

    # KL(q(z|x)||p(z)) = log_q_z - log_p_z
    log_q_z = log_normal_pdf(z, mu, logvar).sum(dim=1)  # [batch_size]
    log_p_z = log_standard_normal_pdf(z).sum(dim=1)      # [batch_size]

    # Mutual Information: I(z;x) ≈ KL(q(z|x)||q(z))
    # log q(z) ≈ log(1/batch_size * sum_i q(z|x_i))
    log_q_z_product = log_normal_pdf(z.unsqueeze(0), mu.unsqueeze(1), logvar.unsqueeze(1))
    log_q_z_product = torch.logsumexp(log_q_z_product.sum(dim=2), dim=1) - np.log(batch_size)

    mi_loss = (log_q_z - log_q_z_product).mean()

    # Total Correlation: KL(q(z)||∏_j q(z_j))
    log_q_z_j = log_normal_pdf(z, mu, logvar)  # [batch_size, latent_dim]
    log_q_z_j = torch.logsumexp(log_q_z_j, dim=0) - np.log(batch_size)  # [latent_dim]
    tc_loss = (log_q_z_product - log_q_z_j.sum()).mean()

    # Dimension-wise KL: sum_j KL(q(z_j)||p(z_j))
    dw_kl = (log_q_z_j - log_standard_normal_pdf(z).sum(dim=0) + np.log(batch_size)).sum()

    total_loss = recon_loss + beta * mi_loss + gamma * tc_loss + dw_kl

    return total_loss, recon_loss, mi_loss, tc_loss, dw_kl
```

#### 2. Spectral Normalization for Stability
```python
class SpectralNorm(nn.Module):
    def __init__(self, module, name='weight', power_iterations=1):
        super().__init__()
        self.module = module
        self.name = name
        self.power_iterations = power_iterations
        if not self._made_params():
            self._make_params()

    def _update_u_v(self):
        u = getattr(self.module, self.name + "_u")
        v = getattr(self.module, self.name + "_v")
        w = getattr(self.module, self.name + "_bar")

        height = w.data.shape[0]
        for _ in range(self.power_iterations):
            v.data = l2normalize(torch.mv(torch.t(w.view(height,-1).data), u.data))
            u.data = l2normalize(torch.mv(w.view(height,-1).data, v.data))

        sigma = u.dot(w.view(height, -1).mv(v))
        setattr(self.module, self.name, w / sigma.expand_as(w))

    def _made_params(self):
        try:
            u = getattr(self.module, self.name + "_u")
            v = getattr(self.module, self.name + "_v")
            w = getattr(self.module, self.name + "_bar")
            return True
        except AttributeError:
            return False

    def _make_params(self):
        w = getattr(self.module, self.name)

        height = w.data.shape[0]
        width = w.view(height, -1).data.shape[1]

        u = Parameter(w.data.new(height).normal_(0, 1), requires_grad=False)
        v = Parameter(w.data.new(width).normal_(0, 1), requires_grad=False)
        u.data = l2normalize(u.data)
        v.data = l2normalize(v.data)
        w_bar = Parameter(w.data)

        del self.module._parameters[self.name]

        self.module.register_parameter(self.name + "_u", u)
        self.module.register_parameter(self.name + "_v", v)
        self.module.register_parameter(self.name + "_bar", w_bar)

    def forward(self, *args):
        self._update_u_v()
        return self.module.forward(*args)
```

## Advanced Architecture Patterns

### 1. Progressive Growing for High Resolution

```python
class ProgressiveVAE(nn.Module):
    def __init__(self, max_resolution=1024):
        super().__init__()
        self.max_resolution = max_resolution
        self.current_resolution = 4  # Start small

        # Multi-scale encoders and decoders
        self.encoders = nn.ModuleDict()
        self.decoders = nn.ModuleDict()

        resolutions = [4, 8, 16, 32, 64, 128, 256, 512, 1024]
        for res in resolutions:
            if res <= max_resolution:
                self.encoders[str(res)] = self._make_encoder(res)
                self.decoders[str(res)] = self._make_decoder(res)

    def _make_encoder(self, resolution):
        # Build encoder for specific resolution
        num_layers = int(np.log2(resolution)) - 1
        return TransformerEncoder(
            embed_dim=min(768, 64 * num_layers),
            num_layers=max(6, num_layers),
            num_heads=min(12, max(4, num_layers))
        )

    def grow_network(self):
        """Gradually increase resolution during training"""
        current_res = self.current_resolution
        if current_res < self.max_resolution:
            self.current_resolution = min(current_res * 2, self.max_resolution)
            print(f"Growing network to resolution: {self.current_resolution}")

    def forward(self, x, alpha=1.0):
        """Forward pass with smooth transition between resolutions"""
        current_res = str(self.current_resolution)

        if alpha < 1.0 and self.current_resolution > 4:
            # Blend between current and previous resolution
            prev_res = str(self.current_resolution // 2)

            # Process at both resolutions
            x_current = F.interpolate(x, size=(self.current_resolution, self.current_resolution))
            x_prev = F.interpolate(x, size=(self.current_resolution // 2, self.current_resolution // 2))

            mu1, logvar1 = self.encoders[current_res](x_current)
            mu2, logvar2 = self.encoders[prev_res](x_prev)

            # Blend latents
            mu = alpha * mu1 + (1 - alpha) * mu2
            logvar = alpha * logvar1 + (1 - alpha) * logvar2

        else:
            x = F.interpolate(x, size=(self.current_resolution, self.current_resolution))
            mu, logvar = self.encoders[current_res](x)

        z = self.reparameterize(mu, logvar)
        recon = self.decoders[current_res](z)

        return recon, mu, logvar
```

### 2. Hierarchical VAE for Multi-Scale Modeling

```python
class HierarchicalVAE(nn.Module):
    def __init__(self, num_levels=3, base_channels=64):
        super().__init__()
        self.num_levels = num_levels

        # Multi-level encoders
        self.encoders = nn.ModuleList([
            self._make_encoder_level(i, base_channels)
            for i in range(num_levels)
        ])

        # Multi-level decoders
        self.decoders = nn.ModuleList([
            self._make_decoder_level(i, base_channels)
            for i in range(num_levels)
        ])

        # Top-down connections
        self.top_down = nn.ModuleList([
            nn.Linear(256, 256) for _ in range(num_levels - 1)
        ])

    def _make_encoder_level(self, level, base_channels):
        channels = base_channels * (2 ** level)
        return nn.Sequential(
            nn.Conv2d(3 if level == 0 else channels//2, channels, 3, 2, 1),
            nn.GroupNorm(8, channels),
            nn.GELU(),
            ResidualBlock(channels),
            ResidualBlock(channels)
        )

    def encode_hierarchical(self, x):
        """Encode at multiple scales"""
        features = []
        current = x

        for i, encoder in enumerate(self.encoders):
            current = encoder(current)
            features.append(current)

        # Extract latents at each level
        latents = []
        for i, feat in enumerate(features):
            # Global pooling + projection to latent
            pooled = F.adaptive_avg_pool2d(feat, 1).flatten(1)
            mu, logvar = self.to_latent[i](pooled).chunk(2, dim=1)
            latents.append((mu, logvar))

        return latents

    def decode_hierarchical(self, latents):
        """Top-down decoding with skip connections"""
        # Start from top level
        current = self.from_latent[-1](latents[-1])
        current = current.view(-1, 512, 4, 4)  # Reshape

        # Top-down pass
        for i in reversed(range(len(self.decoders) - 1)):
            current = self.decoders[i+1](current)

            # Add skip connection from encoder level
            if i < len(latents) - 1:
                skip = self.from_latent[i](latents[i])
                skip = skip.view(current.shape[0], -1, current.shape[2], current.shape[3])
                current = current + skip

        # Final reconstruction
        return self.decoders[0](current)
```

### 3. Attention-Based Feature Fusion

```python
class CrossScaleAttention(nn.Module):
    def __init__(self, dim, num_heads=8):
        super().__init__()
        self.num_heads = num_heads
        self.dim = dim
        self.scale = dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x_low, x_high):
        """Attend from low-res to high-res features"""
        B, N_low, C = x_low.shape
        B, N_high, C = x_high.shape

        # Queries from low-res, Keys/Values from high-res
        q = self.qkv(x_low).reshape(B, N_low, 3, self.num_heads, C // self.num_heads)[:, :, 0].permute(0, 2, 1, 3)
        kv = self.qkv(x_high).reshape(B, N_high, 3, self.num_heads, C // self.num_heads)[:, :, 1:].permute(0, 2, 3, 1, 4)
        k, v = kv.unbind(2)

        # Scaled dot-product attention
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)

        x = (attn @ v).transpose(1, 2).reshape(B, N_low, C)
        x = self.proj(x)

        return x + x_low  # Residual connection
```

## Training Strategies

### 1. Curriculum Learning Schedule

```python
class CurriculumTrainer:
    def __init__(self, model, stages):
        self.model = model
        self.stages = stages  # [(epochs, resolution, beta, lr)]
        self.current_stage = 0

    def update_stage(self, epoch):
        """Update training stage based on epoch"""
        for i, (stage_epochs, res, beta, lr) in enumerate(self.stages):
            if epoch < sum(s[0] for s in self.stages[:i+1]):
                if i != self.current_stage:
                    self.current_stage = i
                    self._update_model_config(res, beta, lr)
                break

    def _update_model_config(self, resolution, beta, learning_rate):
        """Update model configuration for new stage"""
        if hasattr(self.model, 'set_resolution'):
            self.model.set_resolution(resolution)

        # Update optimizer learning rates
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = learning_rate

        self.beta = beta
        print(f"Stage updated: res={resolution}, β={beta}, lr={learning_rate}")

# Usage
stages = [
    (10, 64, 0.1, 1e-3),    # Stage 1: Low res, low β
    (20, 128, 0.5, 5e-4),   # Stage 2: Medium res, medium β
    (30, 256, 1.0, 1e-4),   # Stage 3: High res, full β
    (40, 512, 1.0, 5e-5),   # Stage 4: Very high res, fine-tune
]
trainer = CurriculumTrainer(model, stages)
```

### 2. Advanced Data Augmentation for Invariance

```python
class GeometricAugmentation:
    def __init__(self, prob=0.5):
        self.prob = prob

    def __call__(self, x):
        if torch.rand(1) < self.prob:
            # Random geometric transformations
            transformations = [
                self.random_rotation,
                self.random_translation,
                self.random_scaling,
                self.random_shear
            ]

            transform = np.random.choice(transformations)
            x = transform(x)

        return x

    def random_rotation(self, x, max_angle=30):
        angle = torch.rand(1) * max_angle - max_angle/2
        return transforms.functional.rotate(x, angle.item())

    def random_translation(self, x, max_shift=0.1):
        h, w = x.shape[-2:]
        shift_h = int(torch.rand(1) * h * max_shift)
        shift_w = int(torch.rand(1) * w * max_shift)
        return transforms.functional.affine(x, 0, [shift_w, shift_h], 1, 0)

class ContrastiveAugmentation:
    """Create multiple views for contrastive learning"""
    def __init__(self):
        self.weak_aug = transforms.Compose([
            transforms.RandomHorizontalFlip(0.5),
            transforms.ColorJitter(0.1, 0.1, 0.1, 0.05)
        ])

        self.strong_aug = transforms.Compose([
            transforms.RandomHorizontalFlip(0.5),
            transforms.ColorJitter(0.4, 0.4, 0.4, 0.2),
            transforms.RandomGrayscale(0.2),
            transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0))
        ])

    def __call__(self, x):
        view1 = self.weak_aug(x)
        view2 = self.strong_aug(x)
        return view1, view2
```

### 3. Disentanglement Evaluation Metrics

```python
def compute_mig(model, dataset_loader, num_samples=10000):
    """Mutual Information Gap for disentanglement"""
    # Sample latents and factors
    latents = []
    factors = []

    with torch.no_grad():
        for data, factor in dataset_loader:
            if len(latents) * data.size(0) >= num_samples:
                break
            mu, _ = model.encode(data.to(model.device))
            latents.append(mu.cpu())
            factors.append(factor)

    latents = torch.cat(latents, 0)[:num_samples]
    factors = torch.cat(factors, 0)[:num_samples]

    # Compute mutual information
    mi_matrix = np.zeros((latents.size(1), factors.size(1)))

    for i in range(latents.size(1)):
        for j in range(factors.size(1)):
            mi_matrix[i, j] = mutual_information(
                latents[:, i].numpy(), factors[:, j].numpy()
            )

    # Compute MIG
    mig = 0
    for j in range(factors.size(1)):
        mi_j = mi_matrix[:, j]
        mig += (np.max(mi_j) - np.partition(mi_j, -2)[-2]) / np.max(mi_j)

    return mig / factors.size(1)

def compute_sap_score(model, dataset_loader):
    """SAP Score for disentanglement"""
    # Similar implementation to MIG but using different metric
    pass

def compute_dci_score(model, dataset_loader):
    """DCI Score: Disentanglement, Completeness, Informativeness"""
    pass
```

## Memory Optimization Techniques

### 1. Gradient Checkpointing
```python
class CheckpointedTransformerLayer(nn.TransformerEncoderLayer):
    def forward(self, src, src_mask=None, src_key_padding_mask=None):
        return checkpoint(super().forward, src, src_mask, src_key_padding_mask)

# Use in model
self.layers = nn.ModuleList([
    CheckpointedTransformerLayer(embed_dim, num_heads, dim_feedforward)
    for _ in range(num_layers)
])
```

### 2. Mixed Precision Training
```python
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()

for data, _ in dataloader:
    optimizer.zero_grad()

    with autocast():
        recon, mu, logvar = model(data)
        loss = compute_loss(recon, data, mu, logvar)

    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
```

### 3. Model Parallelism for Large Models
```python
class ParallelVAE(nn.Module):
    def __init__(self, config):
        super().__init__()
        # Split encoder across multiple GPUs
        self.encoder_1 = FirstHalfEncoder().to('cuda:0')
        self.encoder_2 = SecondHalfEncoder().to('cuda:1')

        # Decoder on another GPU
        self.decoder = Decoder().to('cuda:2')

    def forward(self, x):
        x = x.to('cuda:0')
        x = self.encoder_1(x)

        x = x.to('cuda:1')
        mu, logvar = self.encoder_2(x)

        z = self.reparameterize(mu, logvar).to('cuda:2')
        recon = self.decoder(z)

        return recon, mu.to('cuda:0'), logvar.to('cuda:0')
```
