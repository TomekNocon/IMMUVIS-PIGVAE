#!/usr/bin/env python
"""
Diagnostic script to check IMC data statistics and normalization.
Run this to verify that the normalization fixes are working correctly.

Usage:
    python scripts/diagnose_imc_data.py
"""

import torch
import h5py
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.components.graphs_datamodules import IMCBaseDictTransform


def load_sample_data(data_path: str = "/raid/tnocon/data/IMC/train.h5", n_samples: int = 10):
    """Load a few samples from the IMC dataset."""
    with h5py.File(data_path, 'r') as f:
        keys = list(f.keys())
        samples = []
        for i in range(min(n_samples, len(f[keys[0]]))):
            sample = {key: f[key][i] for key in keys if key != 'img_path'}
            samples.append(sample)
    return samples


def analyze_normalization(samples, norm_type="channel_wise"):
    """Compare statistics before and after normalization."""
    print("\n" + "="*80)
    print(f"NORMALIZATION ANALYSIS: {norm_type}")
    print("="*80)
    
    # Without normalization
    transform_no_norm = IMCBaseDictTransform(normalize=False)
    
    # With normalization
    transform_with_norm = IMCBaseDictTransform(
        normalize=True, 
        norm_type=norm_type
    )
    
    stats_before = []
    stats_after = []
    
    for idx, sample in enumerate(samples):
        # Process without normalization
        sample_copy = {k: v.copy() if isinstance(v, np.ndarray) else v 
                      for k, v in sample.items()}
        result_no_norm = transform_no_norm(sample_copy)
        
        # Process with normalization
        sample_copy = {k: v.copy() if isinstance(v, np.ndarray) else v 
                      for k, v in sample.items()}
        result_with_norm = transform_with_norm(sample_copy)
        
        for key in result_no_norm:
            if isinstance(result_no_norm[key], torch.Tensor):
                data_before = result_no_norm[key]
                data_after = result_with_norm[key]
                
                stats_before.append({
                    'sample': idx,
                    'key': key,
                    'mean': data_before.mean().item(),
                    'std': data_before.std().item(),
                    'min': data_before.min().item(),
                    'max': data_before.max().item(),
                    'shape': tuple(data_before.shape)
                })
                
                stats_after.append({
                    'sample': idx,
                    'key': key,
                    'mean': data_after.mean().item(),
                    'std': data_after.std().item(),
                    'min': data_after.min().item(),
                    'max': data_after.max().item(),
                    'shape': tuple(data_after.shape)
                })
    
    # Print summary
    print("\n" + "-"*80)
    print("BEFORE NORMALIZATION:")
    print("-"*80)
    print(f"{'Sample':<8} {'Transform':<15} {'Mean':<12} {'Std':<12} {'Min':<12} {'Max':<12}")
    print("-"*80)
    for stat in stats_before[:5]:  # Show first 5
        print(f"{stat['sample']:<8} {stat['key']:<15} {stat['mean']:>11.4f} {stat['std']:>11.4f} "
              f"{stat['min']:>11.4f} {stat['max']:>11.4f}")
    
    print("\n" + "-"*80)
    print("AFTER NORMALIZATION:")
    print("-"*80)
    print(f"{'Sample':<8} {'Transform':<15} {'Mean':<12} {'Std':<12} {'Min':<12} {'Max':<12}")
    print("-"*80)
    for stat in stats_after[:5]:  # Show first 5
        print(f"{stat['sample']:<8} {stat['key']:<15} {stat['mean']:>11.4f} {stat['std']:>11.4f} "
              f"{stat['min']:>11.4f} {stat['max']:>11.4f}")
    
    # Aggregate statistics
    print("\n" + "="*80)
    print("AGGREGATE STATISTICS:")
    print("="*80)
    
    means_before = [s['mean'] for s in stats_before]
    stds_before = [s['std'] for s in stats_before]
    means_after = [s['mean'] for s in stats_after]
    stds_after = [s['std'] for s in stats_after]
    
    print(f"\nBEFORE Normalization:")
    print(f"  Mean of means: {np.mean(means_before):.6f} ± {np.std(means_before):.6f}")
    print(f"  Mean of stds:  {np.mean(stds_before):.6f} ± {np.std(stds_before):.6f}")
    
    print(f"\nAFTER Normalization:")
    print(f"  Mean of means: {np.mean(means_after):.6f} ± {np.std(means_after):.6f}")
    print(f"  Mean of stds:  {np.mean(stds_after):.6f} ± {np.std(stds_after):.6f}")
    
    print("\n✅ EXPECTED after normalization: Mean ≈ 0, Std ≈ 1")
    print(f"✅ STATUS: {'PASS' if abs(np.mean(means_after)) < 0.1 and 0.8 < np.mean(stds_after) < 1.2 else 'FAIL'}")
    
    return stats_before, stats_after


def visualize_distributions(samples):
    """Visualize feature distributions before and after normalization."""
    print("\n" + "="*80)
    print("GENERATING DISTRIBUTION PLOTS...")
    print("="*80)
    
    transform_no_norm = IMCBaseDictTransform(normalize=False)
    transform_with_norm = IMCBaseDictTransform(normalize=True, norm_type="channel_wise")
    
    # Get first sample
    sample = samples[0]
    
    # Process
    sample_copy1 = {k: v.copy() if isinstance(v, np.ndarray) else v for k, v in sample.items()}
    result_no_norm = transform_no_norm(sample_copy1)
    
    sample_copy2 = {k: v.copy() if isinstance(v, np.ndarray) else v for k, v in sample.items()}
    result_with_norm = transform_with_norm(sample_copy2)
    
    # Get data
    for key in result_no_norm:
        if isinstance(result_no_norm[key], torch.Tensor):
            data_before = result_no_norm[key].flatten().numpy()
            data_after = result_with_norm[key].flatten().numpy()
            
            # Create plots
            fig, axes = plt.subplots(2, 2, figsize=(12, 10))
            fig.suptitle(f'IMC Data Distribution Analysis: {key}', fontsize=16, fontweight='bold')
            
            # Histogram before
            axes[0, 0].hist(data_before, bins=50, alpha=0.7, color='red', edgecolor='black')
            axes[0, 0].set_title('Before Normalization', fontsize=12, fontweight='bold')
            axes[0, 0].set_xlabel('Value')
            axes[0, 0].set_ylabel('Frequency')
            axes[0, 0].axvline(data_before.mean(), color='blue', linestyle='--', 
                              label=f'Mean: {data_before.mean():.2f}')
            axes[0, 0].axvline(data_before.mean() + data_before.std(), color='green', 
                              linestyle='--', label=f'Std: {data_before.std():.2f}')
            axes[0, 0].axvline(data_before.mean() - data_before.std(), color='green', 
                              linestyle='--')
            axes[0, 0].legend()
            axes[0, 0].grid(alpha=0.3)
            
            # Histogram after
            axes[0, 1].hist(data_after, bins=50, alpha=0.7, color='green', edgecolor='black')
            axes[0, 1].set_title('After Normalization', fontsize=12, fontweight='bold')
            axes[0, 1].set_xlabel('Value')
            axes[0, 1].set_ylabel('Frequency')
            axes[0, 1].axvline(data_after.mean(), color='blue', linestyle='--', 
                              label=f'Mean: {data_after.mean():.4f}')
            axes[0, 1].axvline(data_after.mean() + data_after.std(), color='red', 
                              linestyle='--', label=f'Std: {data_after.std():.4f}')
            axes[0, 1].axvline(data_after.mean() - data_after.std(), color='red', 
                              linestyle='--')
            axes[0, 1].legend()
            axes[0, 1].grid(alpha=0.3)
            
            # Q-Q plot before
            from scipy import stats
            stats.probplot(data_before, dist="norm", plot=axes[1, 0])
            axes[1, 0].set_title('Q-Q Plot Before', fontsize=12, fontweight='bold')
            axes[1, 0].grid(alpha=0.3)
            
            # Q-Q plot after
            stats.probplot(data_after, dist="norm", plot=axes[1, 1])
            axes[1, 1].set_title('Q-Q Plot After (should be more linear)', 
                                fontsize=12, fontweight='bold')
            axes[1, 1].grid(alpha=0.3)
            
            plt.tight_layout()
            
            # Save
            output_path = Path("imc_normalization_diagnostic.png")
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            print(f"\n✅ Plot saved to: {output_path.absolute()}")
            
            # Show if possible
            try:
                plt.show()
            except:
                print("   (Unable to display plot - running in non-interactive mode)")
            
            break  # Only plot first tensor


def check_per_channel_variance(samples):
    """Check variance across different feature channels."""
    print("\n" + "="*80)
    print("PER-CHANNEL VARIANCE ANALYSIS:")
    print("="*80)
    
    transform_no_norm = IMCBaseDictTransform(normalize=False, apply_center_crop=False)
    
    sample = samples[0]
    sample_copy = {k: v.copy() if isinstance(v, np.ndarray) else v for k, v in sample.items()}
    
    # Get raw data (before reshaping)
    for key, embedding in sample_copy.items():
        if isinstance(embedding, np.ndarray) and key != 'img_path':
            embedding = torch.from_numpy(embedding)
            c, h, w = embedding.shape
            
            print(f"\nTransform: {key}")
            print(f"Shape: {c} channels × {h} × {w}")
            
            # Compute per-channel statistics
            channel_means = embedding.mean(dim=(1, 2)).numpy()
            channel_stds = embedding.std(dim=(1, 2)).numpy()
            
            print(f"\nChannel statistics (first 10 channels):")
            print(f"{'Channel':<10} {'Mean':<15} {'Std':<15}")
            print("-" * 40)
            for i in range(min(10, c)):
                print(f"{i:<10} {channel_means[i]:>14.6f} {channel_stds[i]:>14.6f}")
            
            # Summary
            print(f"\nAll {c} channels:")
            print(f"  Mean range: [{channel_means.min():.4f}, {channel_means.max():.4f}]")
            print(f"  Std range:  [{channel_stds.min():.4f}, {channel_stds.max():.4f}]")
            print(f"  Std ratio (max/min): {channel_stds.max() / (channel_stds.min() + 1e-8):.2f}x")
            
            if channel_stds.max() / (channel_stds.min() + 1e-8) > 100:
                print(f"\n⚠️  WARNING: Huge variance differences across channels!")
                print(f"    Some channels will dominate gradients without normalization.")
            else:
                print(f"\n✅ Channel variance is relatively balanced.")
            
            break


def main():
    print("\n" + "="*80)
    print("IMC DATA NORMALIZATION DIAGNOSTIC TOOL")
    print("="*80)
    print("\nThis script analyzes your IMC data to verify normalization is working correctly.")
    
    # Load data
    data_path = Path("/raid/tnocon/data/IMC/train.h5")
    if not data_path.exists():
        print(f"\n❌ ERROR: Data file not found at {data_path}")
        print("   Please update the path in the script or ensure data exists.")
        return
    
    print(f"\nLoading data from: {data_path}")
    samples = load_sample_data(str(data_path), n_samples=10)
    print(f"✅ Loaded {len(samples)} samples")
    
    # Run diagnostics
    check_per_channel_variance(samples)
    stats_before, stats_after = analyze_normalization(samples, norm_type="channel_wise")
    
    try:
        visualize_distributions(samples)
    except Exception as e:
        print(f"\n⚠️  Could not generate plots: {e}")
    
    print("\n" + "="*80)
    print("DIAGNOSTIC COMPLETE!")
    print("="*80)
    print("\nKEY TAKEAWAYS:")
    print("1. Check that AFTER normalization: Mean ≈ 0, Std ≈ 1")
    print("2. Per-channel variance should be more balanced after normalization")
    print("3. Distribution should be more Gaussian (check Q-Q plot)")
    print("\nIf normalization is working correctly, you should see:")
    print("  ✅ Mean ≈ 0 (within ±0.1)")
    print("  ✅ Std ≈ 1 (within 0.8-1.2)")
    print("  ✅ More linear Q-Q plot after normalization")
    print("\n" + "="*80 + "\n")


if __name__ == "__main__":
    main()

