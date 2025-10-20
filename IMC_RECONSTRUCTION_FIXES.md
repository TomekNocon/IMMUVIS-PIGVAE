# IMC Reconstruction Quality - Problem Analysis & Fixes

## Problem Summary

Your model works well on MNIST/Fashion-MNIST but produces **blurry, averaged reconstructions** on IMC data despite good MSE metrics. This document explains why and what was fixed.

---

## Root Causes Identified

### 1. **MISSING NORMALIZATION (CRITICAL)** ⚠️

**Problem:**
- IMC features (512-dim vectors from encoder) had **NO normalization**
- MNIST pixels are normalized (0-1 or standardized)
- Different feature dimensions have vastly different scales
- Some dimensions dominate loss, others are ignored
- Model learns unstable representations

**Impact:**
- Unstable training dynamics
- Poor gradient flow
- MSE becomes meaningless (large-scale features dominate)
- Model cannot learn fine-grained structure

**Fix Applied:**
Added channel-wise normalization in `IMCBaseDictTransform`:
```python
# Normalize each channel (feature dim) independently
mean = x.mean(dim=(1, 2), keepdim=True)  # Per-channel mean
std = x.std(dim=(1, 2), keepdim=True)    # Per-channel std
x_norm = (x - mean) / std
```

---

### 2. **VAE POSTERIOR COLLAPSE** ⚠️

**Problem:**
- KLD weight was `0.0001` - **far too low** for feature space
- Free bits at `0.5` - insufficient for 1024-dim latent space
- Model learns to **ignore the latent code** entirely
- Decoder always outputs the same "average" features
- This is THE classic cause of blurry VAE reconstructions

**Why this differs from MNIST:**
- Pixel space (28×28 = 784 dims) vs Feature space (512 dims per node)
- Feature space has more complex distributions
- Needs stronger regularization to maintain meaningful latent codes

**Fix Applied:**
```yaml
kld_loss_scale: 0.001      # Increased 10x from 0.0001
kld_free_bits: 2.0         # Increased 4x from 0.5
```

**Free Bits Explanation:**
- Allows each latent dimension to encode ≥2 bits of information
- Prevents over-compression of latent space
- Balances reconstruction quality vs regularization

---

### 3. **MSE IN FEATURE SPACE ENCOURAGES AVERAGING** 

**Problem:**
- MSE in high-dimensional space (512 dims) minimizes error by **predicting the mean**
- Mathematically: `E[(x - μ)²]` is minimized when predicting `μ = E[x]`
- Results in loss of detail, structure, and high-frequency information
- Good MSE ≠ good perceptual quality in feature space

**Why MNIST works:**
- Pixel space has natural structure
- Lower dimensionality per patch (4 → 128)
- Pixels have clear spatial relationships

**Fix Applied:**
Added **cosine similarity loss** to preserve feature directions:
```python
# Cosine similarity preserves angular relationships
cosine_sim = F.cosine_similarity(pred, target)
cosine_loss = 1.0 - cosine_sim.mean()

total_loss = mse_loss + 0.5 * cosine_loss
```

Benefits:
- Preserves feature vector **directions** (not just magnitudes)
- More robust to scale differences
- Better for semantic feature matching

---

### 4. **INSUFFICIENT STRUCTURE PRESERVATION**

**Problem:**
- Original gradient loss weight was `0.5` - too low for feature space
- Feature space loses spatial structure more easily than pixel space

**Fix Applied:**
```yaml
gradient_loss_weight: 1.0   # Increased from 0.5
```

This preserves high-frequency spatial patterns in the 6×6 grid.

---

## Summary of All Changes

### 1. Data Preprocessing (`src/data/components/graphs_datamodules.py`)
✅ Added channel-wise normalization to `IMCBaseDictTransform`
- Normalizes each of 512 feature dimensions independently
- Ensures balanced gradients across all features
- Controlled via `normalize=True` and `norm_type="channel_wise"`

### 2. Loss Functions (`src/models/components/losses.py`)
✅ Enhanced `GraphReconstructionLoss`:
- Added cosine similarity loss (weight: 0.5)
- Improved gradient loss integration
- Better handling of feature space reconstruction

✅ Improved `KLDLoss`:
- Added free bits mechanism
- Per-dimension regularization control
- Prevents posterior collapse

### 3. Model Configuration (`configs/model/model.yaml`)
✅ Updated VAE hyperparameters:
```yaml
kld_loss_scale: 0.001          # 10x increase
kld_free_bits: 2.0             # 4x increase
gradient_loss_weight: 1.0      # 2x increase
cosine_loss_weight: 0.5        # New
```

### 4. Critic Updates (`src/models/components/model.py`)
✅ Integrated new loss parameters
✅ Proper initialization with free bits and cosine loss

---

## Why It Works on MNIST But Not IMC

| Aspect | MNIST | IMC |
|--------|-------|-----|
| **Input Space** | Pixels (0-255) | Feature embeddings |
| **Dimensionality** | 4 dims → 128 projection | 512 dims (no projection) |
| **Normalization** | ✅ Built-in | ❌ **MISSING** |
| **Scale** | Uniform (0-1) | Unknown, variable |
| **Structure** | Spatial (2×2 patches) | Abstract features |
| **Loss Landscape** | Simple, smooth | Complex, multimodal |
| **MSE Effectiveness** | Good | Poor (encourages averaging) |

---

## Expected Improvements

After these fixes, you should see:

1. **Sharper Reconstructions** - Features preserve structure instead of averaging
2. **Better Diversity** - Model doesn't collapse to mean output
3. **Improved Training Stability** - Normalized gradients across features
4. **Better Permutation Learning** - Model can focus on permutation task with better reconstructions
5. **More Meaningful Latents** - Latent codes actually encode information

---

## Recommended Next Steps

### 1. **Immediate Testing**
```bash
# Train with new settings
python src/train.py
```

### 2. **Monitor These Metrics**
- `kld_loss` - Should be higher now (good! means latent is used)
- `cosine_loss` - Track feature direction preservation
- `gradient_loss` - Monitor structural preservation
- Compare: `node_loss` vs `cosine_loss` vs `gradient_loss` ratios

### 3. **Hyperparameter Tuning (if needed)**

**If reconstructions are still blurry:**
- Increase `kld_loss_scale` to 0.002-0.005
- Increase `kld_free_bits` to 3.0-4.0
- Increase `cosine_loss_weight` to 1.0

**If reconstructions are good but permutation accuracy is low:**
- Increase `perm_loss_scale` from 0.8 to 1.0-1.5
- Decrease KLD weight slightly

**If training is unstable:**
- Lower learning rate to 5e-5
- Add gradient clipping (max_norm=1.0)
- Increase warmup period

### 4. **Ablation Studies (Optional)**

Test individual contributions:
```yaml
# Test A: Normalization only
normalize: true
use_cosine_loss: false
kld_loss_scale: 0.0001  # original

# Test B: Normalization + VAE fixes
normalize: true
use_cosine_loss: false
kld_loss_scale: 0.001
kld_free_bits: 2.0

# Test C: All fixes (current)
normalize: true
use_cosine_loss: true
kld_loss_scale: 0.001
kld_free_bits: 2.0
```

### 5. **Advanced Options (if still needed)**

If problems persist, consider:

**A. Learned Normalization:**
```python
# Per-channel learnable normalization
self.channel_norm = nn.BatchNorm1d(512)
```

**B. Perceptual Loss (if you have access to the original images):**
```python
# Use original encoder for perceptual loss
perceptual_loss = F.mse_loss(
    encoder(reconstruction),
    encoder(target)
)
```

**C. Two-Stage Training:**
```python
# Stage 1: Focus on reconstruction (50 epochs)
kld_loss_scale: 0.0
perm_loss_scale: 0.0

# Stage 2: Add VAE regularization (50 epochs)
kld_loss_scale: 0.001
perm_loss_scale: 0.8
```

---

## Technical Deep Dive

### Why Channel-Wise Normalization?

Feature maps from CNNs have different semantic meanings per channel:
- Channel 0 might detect edges (variance: 100)
- Channel 256 might detect texture (variance: 0.1)
- Channel 511 might be nearly constant (variance: 0.001)

Without normalization:
- Gradient for channel 0: `∂L/∂w₀ = 100 × error`
- Gradient for channel 511: `∂L/∂w₅₁₁ = 0.001 × error`

Result: Model only learns channel 0, ignores others!

With channel-wise normalization:
- All channels have mean=0, std=1
- Equal gradient magnitudes
- Model learns all features equally

### Why Free Bits Matter

KLD loss per dimension: `KLD_dim = -0.5 * (1 + log(σ²) - μ² - σ²)`

Without free bits:
- Model minimizes KLD by setting σ=1, μ=0 (standard normal)
- **Latent codes carry no information!**
- Decoder learns to ignore latent and output average

With free bits (fb=2.0):
- KLD_dim = max(KLD_dim, 2.0)
- Allows each dimension to encode ≥2 bits
- Forces model to use latent code
- Prevents posterior collapse

### Why Cosine Similarity in Feature Space?

MSE: `||a - b||²` - sensitive to magnitude
Cosine: `1 - (a·b)/(||a|| ||b||)` - sensitive to direction

Feature space semantics:
```
Original:  [0.8, 0.1, 0.05, ...] → "cell boundary"
Average:   [0.4, 0.4, 0.4,  ...] → "noise"
Scaled:    [1.6, 0.2, 0.10, ...] → "cell boundary" (same direction!)
```

MSE penalizes "Scaled" equally to "Average"
Cosine recognizes "Scaled" is semantically similar to "Original"

---

## Questions & Debugging

**Q: How do I know if normalization is working?**
```python
# Add to training loop:
print(f"Feature mean: {node_features.mean():.4f}")
print(f"Feature std: {node_features.std():.4f}")
# Should be ~0 and ~1 respectively
```

**Q: How do I know if posterior collapse is fixed?**
```python
# Monitor during training:
print(f"KLD loss: {kld_loss:.4f}")
# Should be > 100 with free_bits=2.0 and 1024 latent dims
# If < 50, increase kld_loss_scale
```

**Q: Reconstructions improved but permutation accuracy dropped?**
- This is expected initially - model was "cheating" before
- Increase `perm_loss_scale` to compensate
- May need more training epochs

**Q: How to visualize if it's working?**
```python
# In evaluation:
# 1. Compare std of latent codes
std_z = z.std(dim=0).mean()  # Should be > 0.5

# 2. Check reconstruction diversity
recon_std = reconstructions.std(dim=0).mean()  # Should be high

# 3. Visualize feature PCA
from sklearn.decomposition import PCA
pca = PCA(n_components=2)
z_2d = pca.fit_transform(z.cpu().numpy())
plt.scatter(z_2d[:, 0], z_2d[:, 1])
# Should see clusters, not single blob
```

---

## References & Further Reading

1. **Posterior Collapse in VAEs:**
   - Bowman et al. "Generating Sentences from a Continuous Space" (2016)
   - He et al. "Lagging Inference Networks and Posterior Collapse in VAEs" (2019)

2. **Free Bits:**
   - Kingma et al. "Improved Variational Inference with Inverse Autoregressive Flow" (2016)

3. **Feature Space Losses:**
   - Johnson et al. "Perceptual Losses for Real-Time Style Transfer" (2016)
   - Zhang et al. "The Unreasonable Effectiveness of Deep Features as a Perceptual Metric" (2018)

4. **Normalization in Deep Learning:**
   - Ioffe & Szegedy "Batch Normalization" (2015)
   - Ulyanov et al. "Instance Normalization" (2016)

---

## Contact & Support

If issues persist after these fixes:
1. Check training logs for loss ratios
2. Visualize intermediate activations
3. Try the ablation studies above
4. Consider adjusting the hyperparameter ranges provided

**Key Success Metrics:**
- ✅ Reconstructions preserve structure (not averaged/blurry)
- ✅ KLD loss > 50 (latent codes are being used)
- ✅ Training is stable (losses decrease smoothly)
- ✅ Permutation accuracy improves with more epochs
- ✅ Different samples produce different reconstructions (diversity)

Good luck! 🚀

