---
name: vae-architecture-expert
description: Expert guidance on VAE architecture design for optimal reconstruction quality and invariant latent representations. Use when designing VAE models, choosing architectures, selecting loss functions, or optimizing reconstruction performance for computer vision tasks.
---

# VAE Architecture Expert

Expert system for designing Variational Autoencoder (VAE) architectures optimized for high-quality reconstruction and learning invariant latent representations, particularly for transformer-based encoders without positional encoding and decoders with positional encoding.

## Quick Start

For immediate architecture recommendations:

1. **Data type**: Specify your input (images, sequences, etc.)
2. **Goals**: Reconstruction quality vs. disentanglement vs. invariance
3. **Constraints**: Model size, computational budget, latency requirements

## Core Architecture Principles

### Encoder Design (Without Positional Encoding)

**Transformer-based Encoder Guidelines:**

1. **Patch Embedding Strategy**
   - Use learnable patch embeddings (16x16 or 8x8 for images)
   - Apply layer normalization before first attention layer
   - No positional encoding to maintain spatial invariance

2. **Optimal Layer Configuration**
   - **Small models**: 6-8 transformer layers
   - **Medium models**: 12-16 transformer layers
   - **Large models**: 24+ transformer layers
   - Use `dim_feedforward = 4 * d_model` ratio

3. **Attention Mechanisms**
   - Multi-head attention with 8-12 heads
   - Head dimension should divide evenly into model dimension
   - Use scaled dot-product attention with dropout (0.1-0.2)

4. **Activation Functions**
   - **Primary**: GELU (Gaussian Error Linear Unit)
   - **Alternative**: SiLU/Swish for better gradients
   - **Avoid**: ReLU (causes dead neurons in deep networks)

### Decoder Design (With Positional Encoding)

**Transformer-based Decoder Guidelines:**

1. **Positional Encoding Strategy**
   - Use learnable positional embeddings
   - Add positional encoding after latent upsampling
   - Consider 2D positional encoding for spatial data

2. **Progressive Upsampling**
   - Start from compressed latent representation
   - Use transposed convolutions or nearest neighbor + conv
   - Maintain aspect ratios through upsampling stages

3. **Layer Configuration**
   - Mirror encoder depth or use 1.5x encoder layers
   - Cross-attention between latent and spatial features
   - Self-attention for spatial coherence

## Latent Space Design

### Optimal Latent Dimensions

```python
# Rule of thumb for latent dimension sizing
input_size = H * W * C  # For images
compression_ratio = 32  # Typical range: 16-64
latent_dim = max(64, input_size // compression_ratio)

# For invariant representations
latent_dim = min(latent_dim, 512)  # Cap for better disentanglement
```

### Invariance Techniques

1. **Spatial Invariance**
   - Global average pooling before latent projection
   - Attention pooling with learnable queries
   - No positional encoding in encoder

2. **Rotation/Translation Invariance**
   - Data augmentation during training
   - Group equivariant convolutions
   - Spatial transformer networks

## Advanced Architecture Tricks

### Reconstruction Quality Improvements

1. **Progressive Training**
   ```python
   # Start with lower resolution, gradually increase
   resolutions = [64, 128, 256, 512]
   for res in resolutions:
       train_at_resolution(res, epochs=10)
   ```

2. **Multi-Scale Features**
   - Skip connections from encoder to decoder
   - Feature pyramid networks
   - Multi-resolution loss computation

3. **Attention Mechanisms**
   - Channel attention (Squeeze-and-Excitation)
   - Spatial attention maps
   - Cross-attention between scales

### Training Stability Tricks

1. **Gradient Management**
   - Gradient clipping (max_norm=1.0)
   - Spectral normalization for discriminators
   - EMA (Exponential Moving Average) for stable training

2. **Learning Rate Scheduling**
   - Cosine annealing with warm restarts
   - Separate learning rates for encoder/decoder
   - Higher learning rate for decoder (2-5x encoder rate)

## Loss Function Selection

### Primary Losses

1. **Reconstruction Losses**
   ```python
   # For images - perceptual loss combination
   loss_mse = F.mse_loss(recon, target)
   loss_perceptual = perceptual_loss(recon, target)  # VGG features
   loss_ssim = 1 - ssim(recon, target)

   recon_loss = 0.5 * loss_mse + 0.3 * loss_perceptual + 0.2 * loss_ssim
   ```

2. **KL Divergence**
   ```python
   # β-VAE with annealing
   beta = min(1.0, epoch / warmup_epochs)
   kl_loss = beta * torch.mean(0.5 * (mu.pow(2) + logvar.exp() - logvar - 1))
   ```

### Advanced Loss Functions

1. **Adversarial Loss** (VAE-GAN)
   - Add discriminator for realistic outputs
   - Balance reconstruction and adversarial terms
   - Use feature matching loss

2. **Disentanglement Losses**
   - β-VAE (β > 1 for disentanglement)
   - Factor-VAE with Total Correlation penalty
   - β-TCVAE for better disentanglement

3. **Consistency Losses**
   - Cycle consistency for invariance
   - Contrastive learning objectives
   - Mutual information maximization

## Model Size Guidelines

### Small VAE (< 10M parameters)
- Encoder: 6 layers, 384 dim, 6 heads
- Decoder: 8 layers, 384 dim, 6 heads
- Latent dim: 128-256
- Best for: Prototyping, simple datasets

### Medium VAE (10-50M parameters)
- Encoder: 12 layers, 512 dim, 8 heads
- Decoder: 16 layers, 512 dim, 8 heads
- Latent dim: 256-512
- Best for: Complex images, good quality

### Large VAE (50M+ parameters)
- Encoder: 24 layers, 768 dim, 12 heads
- Decoder: 32 layers, 768 dim, 12 heads
- Latent dim: 512-1024
- Best for: High-resolution, state-of-the-art quality

## Training Recommendations

### Hyperparameter Settings

```python
# Proven configurations
config = {
    'learning_rate': 1e-4,  # Conservative, stable
    'encoder_lr': 1e-4,
    'decoder_lr': 3e-4,     # Higher for faster reconstruction learning
    'batch_size': 32,       # Adjust based on memory
    'weight_decay': 1e-5,   # Light regularization
    'dropout': 0.1,         # Transformer layers
    'beta_warmup': 1000,    # KL annealing steps
}
```

### Training Schedule

1. **Phase 1** (Epochs 1-20): Focus on reconstruction
   - β = 0.1, high reconstruction loss weight
   - Learning rate: 1e-4

2. **Phase 2** (Epochs 21-80): Balanced training
   - β = 0.5-1.0, balanced losses
   - Learning rate: 5e-5 with cosine schedule

3. **Phase 3** (Epochs 81+): Fine-tuning
   - β = 1.0, add perceptual losses
   - Learning rate: 1e-5

## Architecture Examples

For specific implementations and code examples, see [examples.md](examples.md).

For detailed mathematical foundations and advanced techniques, see [reference.md](reference.md).

## Validation Checklist

- [ ] Encoder has no positional encoding
- [ ] Decoder uses appropriate positional encoding
- [ ] Latent dimension matches compression requirements
- [ ] Activation functions are modern (GELU/SiLU)
- [ ] Loss combination includes perceptual terms
- [ ] Training uses β-annealing for KL loss
- [ ] Architecture depth matches model size budget
- [ ] Attention heads divide evenly into model dimension
- [ ] Skip connections preserve spatial information
- [ ] Data augmentation supports desired invariances
