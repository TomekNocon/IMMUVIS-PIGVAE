import torch
import torch.nn as nn
import pytorch_lightning as pl
from lightning.pytorch.callbacks import Callback
from typing import Dict, List, Optional, Any
import matplotlib.pyplot as plt
import os
from collections import defaultdict
import numpy as np
from src.data.components.graphs_datamodules import DenseGraphBatch

class InterpretabilityCallback(Callback):
    """
    PyTorch Lightning Callback for transformer interpretability analysis.
    This is the recommended approach - keeps interpretability separate from model logic.
    """

    def __init__(
        self,
        save_dir: str = "./interpretability_outputs",
        analyze_every_n_epochs: int = 5,
        analyze_every_n_batches: int = 100,
        max_samples_per_epoch: int = 3,
        enable_chefer_methods: bool = True,
        enable_attention_analysis: bool = True,
        enable_gradient_analysis: bool = True
    ):
        super().__init__()
        self.save_dir = save_dir
        self.analyze_every_n_epochs = analyze_every_n_epochs
        self.analyze_every_n_batches = analyze_every_n_batches
        self.max_samples_per_epoch = max_samples_per_epoch
        self.enable_chefer_methods = enable_chefer_methods
        self.enable_attention_analysis = enable_attention_analysis
        self.enable_gradient_analysis = enable_gradient_analysis

        # Storage for analysis results
        self.attention_storage = defaultdict(list)
        self.gradient_storage = defaultdict(list)
        self.analysis_results = {}

        # Hooks for attention tracking
        self.hooks = []

        os.makedirs(self.save_dir, exist_ok=True)

    def state_dict(self):
        """Return state dict for checkpointing - exclude non-serializable data"""
        return {
            'save_dir': self.save_dir,
            'analyze_every_n_epochs': self.analyze_every_n_epochs,
            'analyze_every_n_batches': self.analyze_every_n_batches,
            'max_samples_per_epoch': self.max_samples_per_epoch,
            'enable_chefer_methods': self.enable_chefer_methods,
            'enable_attention_analysis': self.enable_attention_analysis,
            'enable_gradient_analysis': self.enable_gradient_analysis,
            # Don't save hooks, layer_names, module_to_name - they need to be re-registered
        }

    def load_state_dict(self, state_dict):
        """Load state dict from checkpoint"""
        for key, value in state_dict.items():
            if hasattr(self, key):
                setattr(self, key, value)

        # Reinitialize collections that shouldn't be saved
        if not hasattr(self, 'attention_storage'):
            self.attention_storage = defaultdict(list)
        if not hasattr(self, 'gradient_storage'):
            self.gradient_storage = defaultdict(list)
        if not hasattr(self, 'analysis_results'):
            self.analysis_results = {}
        if not hasattr(self, 'hooks'):
            self.hooks = []
        if not hasattr(self, 'layer_names'):
            self.layer_names = {}
        if not hasattr(self, 'module_to_name'):
            self.module_to_name = {}

    def on_train_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        """Initialize interpretability tracking when training starts"""
        self._register_hooks(pl_module)
        print(f"Interpretability callback initialized. Results will be saved to {self.save_dir}")

    def on_train_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        """Clean up hooks when training ends"""
        self._remove_hooks()
        print("Interpretability analysis completed.")

    def _register_hooks(self, model: pl.LightningModule):
        """Register hooks to capture attention weights and gradients from SelfAttention layers"""

        # Store layer names for the hooks
        self.layer_names = {}
        self.module_to_name = {}  # Store mapping from module id to name

        # Register hooks specifically for SelfAttention modules
        for name, module in model.named_modules():
            if module.__class__.__name__ == 'SelfAttention':
                # Store the mapping
                self.layer_names[id(module)] = name
                self.module_to_name[id(module)] = name

                # Use partial functions with explicit module id mapping
                forward_hook = module.register_forward_hook(self._forward_hook_wrapper)
                backward_hook = module.register_backward_hook(self._backward_hook_wrapper)
                self.hooks.extend([forward_hook, backward_hook])
                print(f"Registered hooks for SelfAttention layer: {name}")

    def _forward_hook_wrapper(self, module, inp, out):
        """Wrapper for forward hook that uses module id to find layer name"""
        module_id = id(module)
        layer_name = self.module_to_name.get(module_id, f"unknown_{module_id}")
        return self._attention_forward_hook(module, inp, out, layer_name)

    def _backward_hook_wrapper(self, module, grad_inp, grad_out):
        """Wrapper for backward hook that uses module id to find layer name"""
        module_id = id(module)
        layer_name = self.module_to_name.get(module_id, f"unknown_{module_id}")
        return self._attention_backward_hook(module, grad_inp, grad_out, layer_name)

    def _attention_forward_hook(self, module, input_data, output, layer_name):
        """Forward hook to capture attention weights - now a proper method"""
        if not self.enable_attention_analysis:
            return

        if len(self.attention_storage[layer_name]) > 5:
            self.attention_storage[layer_name].pop(0)

        # Store input to attention layer for later analysis
        self.attention_storage[layer_name].append({
            'input': input_data[0].detach().cpu(),
            'output': output.detach().cpu(),
            'layer_name': layer_name
        })

    def _attention_backward_hook(self, module, grad_input, grad_output, layer_name):
        """Backward hook to capture gradients - now a proper method"""
        if not self.enable_gradient_analysis or grad_output[0] is None:
            return

        if len(self.gradient_storage[layer_name]) > 5:
            self.gradient_storage[layer_name].pop(0)
        self.gradient_storage[layer_name].append(grad_output[0].detach().cpu())

    def _remove_hooks(self):
        """Remove all registered hooks"""
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()

    def on_train_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs: Any,
        batch: Any,
        batch_idx: int
    ) -> None:
        """Analyze interpretability every N training batches"""

        if batch_idx % self.analyze_every_n_batches == 0:
            self._log_attention_statistics(trainer, pl_module, batch_idx, 'train')

    def on_validation_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs: Any,
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0
    ) -> None:
        """Store validation samples for detailed analysis"""

        if batch_idx < self.max_samples_per_epoch:
            # batch is DenseGraphBatch, not (x, target)
            graph_batch = batch

            # Store sample for end-of-epoch analysis
            self.analysis_results[f'val_epoch_{trainer.current_epoch}_batch_{batch_idx}'] = {
                'graph_batch': graph_batch,
                'node_features': graph_batch.node_features[:1].cpu(),  # First graph only
                'target': graph_batch.y[:1].cpu() if hasattr(graph_batch, 'y') and graph_batch.y is not None else None,
                'attention_data': {
                    name: attn_data[-1].copy() if attn_data else None
                    for name, attn_data in self.attention_storage.items()
                }
            }

    def on_validation_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        """Generate detailed interpretability analysis at epoch end"""

        if trainer.current_epoch % self.analyze_every_n_epochs != 0:
            return

        print(f"Generating interpretability analysis for epoch {trainer.current_epoch}...")

        # Analyze stored validation samples
        for sample_key, sample_data in self.analysis_results.items():
            if f'epoch_{trainer.current_epoch}' in sample_key:
                self._analyze_sample(
                    trainer,
                    pl_module,
                    sample_data,
                    sample_key
                )

        # Clear analysis results to save memory
        self.analysis_results = {
            k: v for k, v in self.analysis_results.items()
            if f'epoch_{trainer.current_epoch}' not in k
        }

    def _analyze_sample(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        sample_data: Dict,
        sample_key: str
    ):
        """Perform detailed analysis on a single sample"""

        graph_batch = sample_data['graph_batch']
        node_features = sample_data['node_features'].to(pl_module.device)

        try:
            # 1. Basic attention analysis
            if self.enable_attention_analysis:
                attention_analysis = self._analyze_graph_attention_patterns(sample_data['attention_data'])
                self._save_attention_visualizations(attention_analysis, sample_key)

            # 2. Gradient-based analysis for graphs
            if self.enable_gradient_analysis:
                gradient_analysis = self._analyze_graph_gradients(pl_module, graph_batch)
                self._save_gradient_visualizations(gradient_analysis, sample_key)

            # 3. Chefer methods adapted for graphs
            if self.enable_chefer_methods:
                chefer_analysis = self._apply_graph_chefer_methods(pl_module, graph_batch)
                self._save_chefer_visualizations(chefer_analysis, sample_key)

            # 4. Log metrics to logger
            self._log_interpretability_metrics(trainer, pl_module, sample_key)

        except Exception as e:
            print(f"Error in interpretability analysis for {sample_key}: {e}")

    def _analyze_graph_attention_patterns(self, attention_data: Dict) -> Dict:
        """Analyze attention patterns"""

        analysis = {}

        for layer_name, attention in attention_data.items():
            if attention is None:
                continue

            # attention is a dictionary containing 'input', 'output', etc.
            # We need to extract actual attention weights from the output or input
            if isinstance(attention, dict):
                # Try to get output tensor which should contain attention-like information
                if 'output' in attention and attention['output'] is not None:
                    attention_tensor = attention['output']
                elif 'input' in attention and attention['input'] is not None:
                    attention_tensor = attention['input']
                else:
                    continue
            else:
                attention_tensor = attention

            # Convert to numpy for analysis
            if isinstance(attention_tensor, torch.Tensor):
                attn_np = attention_tensor.numpy()
            else:
                attn_np = attention_tensor

            # Ensure attn_np is a numpy array
            if not isinstance(attn_np, np.ndarray):
                continue

            # Handle different attention shapes
            if attn_np.ndim == 4:  # [batch, heads, seq, seq]
                attn_np = attn_np[0]  # First batch
            elif attn_np.ndim == 3 and attn_np.shape[0] == 1:  # [1, seq, seq]
                attn_np = attn_np[0]

            # Compute attention statistics
            layer_analysis = {
                'attention_matrix': attn_np,
                'entropy': self._compute_attention_entropy(attn_np),
                'sparsity': self._compute_attention_sparsity(attn_np),
                'head_similarity': self._compute_head_similarity(attn_np) if attn_np.ndim == 3 else None
            }

            analysis[layer_name] = layer_analysis

        return analysis

    def _analyze_graph_gradients(self, model: pl.LightningModule, graph_batch: DenseGraphBatch) -> Dict:
        """Analyze gradient-based attributions for graph data"""

        model.eval()

        # Create a copy with requires_grad for node features
        node_features = graph_batch.node_features.clone().requires_grad_(True)

        # Create a new graph batch with gradient-enabled features
        graph_for_grad = DenseGraphBatch(
            node_features=node_features,
            edge_features=graph_batch.edge_features,
            y=graph_batch.y,
            argsort_augmented_features=graph_batch.argsort_augmented_features
        )

        # Forward pass through the model
        tau = 1.0  # Fixed temperature for analysis
        graph_emb, graph_pred, soft_probs, perm, mu, logvar = model(
            graph=graph_for_grad, training=False, tau=tau
        )

        # Compute gradients w.r.t. graph embedding or reconstruction loss
        target = graph_emb.sum()  # Or use reconstruction loss
        target.backward()

        # Get gradients w.r.t. node features
        node_gradients = node_features.grad

        # Compute gradient-based attributions
        analysis = {
            'node_gradients': node_gradients.detach().cpu(),
            'node_x_gradient': (node_features * node_gradients).detach().cpu(),
            'gradient_norm': torch.norm(node_gradients, dim=-1).detach().cpu(),
            'graph_embedding': graph_emb.detach().cpu()
        }

        return analysis

    def _apply_graph_chefer_methods(self, model: pl.LightningModule, graph_batch: DenseGraphBatch) -> Dict:
        """Apply Chefer et al. interpretability methods for graphs"""

        analysis = {}

        try:
            # Implement attention rollout for graphs
            attention_matrices = []

            # Collect attention matrices from all layers
            for layer_name, attention_data_list in self.attention_storage.items():
                if attention_data_list and 'attention_weights' in attention_data_list[-1]:
                    attention_weights = attention_data_list[-1]['attention_weights']
                    if attention_weights.dim() == 4:  # [batch, heads, nodes, nodes]
                        # Average over heads for simplicity
                        avg_attention = attention_weights.mean(dim=1)  # [batch, nodes, nodes]
                        attention_matrices.append(avg_attention[0])  # First batch

            if attention_matrices:
                # Compute attention rollout (simplified version)
                rollout = self._compute_attention_rollout(attention_matrices)
                analysis['attention_rollout'] = rollout.cpu()

                # Compute LRP-like attribution (simplified)
                if len(attention_matrices) > 0:
                    lrp_scores = self._compute_simple_lrp(attention_matrices[-1])  # Last layer
                    analysis['lrp_attribution'] = lrp_scores.cpu()

        except Exception as e:
            print(f"Error in graph Chefer methods: {e}")

        return analysis

    def _compute_attention_rollout(self, attention_matrices: List[torch.Tensor]) -> torch.Tensor:
        """Compute attention rollout as in Chefer et al."""
        if not attention_matrices:
            return torch.empty(0)

        # Start with identity + first attention
        rollout = attention_matrices[0]

        # Add identity to account for residual connections
        eye = torch.eye(rollout.size(0), device=rollout.device)
        rollout = 0.5 * rollout + 0.5 * eye

        # Roll out through subsequent layers
        for attention in attention_matrices[1:]:
            attention_with_identity = 0.5 * attention + 0.5 * eye
            rollout = torch.matmul(attention_with_identity, rollout)

        return rollout

    def _compute_simple_lrp(self, attention_matrix: torch.Tensor) -> torch.Tensor:
        """Compute simplified LRP scores for nodes"""
        # Sum attention weights received by each node (simplified relevance)
        lrp_scores = attention_matrix.sum(dim=0)  # Sum over source nodes
        return lrp_scores

    def _compute_attention_entropy(self, attention: np.ndarray) -> float:
        """Compute attention entropy"""
        # Flatten attention and compute entropy
        attention_flat = attention.flatten()
        attention_flat = np.abs(attention_flat) + 1e-8  # Ensure positive values and avoid log(0)
        attention_sum = attention_flat.sum()

        if attention_sum == 0:
            return 0.0

        attention_flat = attention_flat / attention_sum

        # Only compute log for positive values
        log_vals = np.log(attention_flat)
        entropy = -(attention_flat * log_vals).sum()
        return float(entropy)

    def _compute_attention_sparsity(self, attention: np.ndarray) -> float:
        """Compute attention sparsity (Gini coefficient)"""
        attention_flat = attention.flatten()
        attention_sorted = np.sort(attention_flat)
        n = len(attention_sorted)
        cumsum = np.cumsum(attention_sorted)

        gini = (n + 1 - 2 * np.sum(cumsum) / cumsum[-1]) / n
        return float(gini)

    def _compute_head_similarity(self, attention: np.ndarray) -> float:
        """Compute similarity between attention heads"""
        if attention.ndim != 3:  # Not multi-head
            return 0.0

        # Flatten each head and compute pairwise correlations
        heads_flat = attention.reshape(attention.shape[0], -1)
        correlations = np.corrcoef(heads_flat)

        # Average correlation between different heads
        mask = ~np.eye(correlations.shape[0], dtype=bool)
        avg_correlation = correlations[mask].mean()

        return float(avg_correlation)

    def _save_attention_visualizations(self, analysis: Dict, sample_key: str):
        """Save attention pattern visualizations"""

        if not analysis:
            print(f"No attention data to visualize for {sample_key}")
            return

        n_plots = min(3, len(analysis))
        if n_plots == 0:
            return

        fig, axes = plt.subplots(2, n_plots, figsize=(15, 10))
        if n_plots == 1:
            axes = axes.reshape(-1, 1)
        elif len(axes.shape) == 1:
            axes = axes.reshape(-1, n_plots)

        for idx, (layer_name, layer_analysis) in enumerate(list(analysis.items())[:3]):
            attention_matrix = layer_analysis['attention_matrix']

            # Plot attention heatmap
            if attention_matrix.ndim == 3:  # Multi-head
                attention_to_plot = attention_matrix.mean(axis=0)  # Average over heads
            else:
                attention_to_plot = attention_matrix

            im = axes[0, idx].imshow(attention_to_plot, cmap='Blues', aspect='auto')
            plt.colorbar(im, ax=axes[0, idx])
            axes[0, idx].set_title(f'{layer_name.split(".")[-1]} - Attention')

            # Plot attention statistics
            stats = [
                layer_analysis['entropy'],
                layer_analysis['sparsity'],
                layer_analysis['head_similarity'] or 0
            ]
            stat_names = ['Entropy', 'Sparsity', 'Head Sim']

            axes[1, idx].bar(stat_names, stats)
            axes[1, idx].set_title(f'{layer_name.split(".")[-1]} - Stats')
            axes[1, idx].set_ylim(0, max(stats) * 1.1)

        plt.tight_layout()
        save_path = os.path.join(self.save_dir, f'{sample_key}_attention.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

    def _save_gradient_visualizations(self, analysis: Dict, sample_key: str):
        """Save gradient-based visualizations"""

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        # Plot node gradients
        node_grads = analysis['node_gradients'][0]  # First graph
        if node_grads.dim() > 1:
            node_grads = node_grads.mean(dim=-1)  # Average over features

        axes[0].plot(node_grads.numpy())
        axes[0].set_title('Node Feature Gradients')
        axes[0].set_xlabel('Node Index')

        # Plot gradient * input
        grad_x_input = analysis['node_x_gradient'][0]
        if grad_x_input.dim() > 1:
            grad_x_input = grad_x_input.mean(dim=-1)

        axes[1].plot(grad_x_input.numpy())
        axes[1].set_title('Gradient × Node Features')
        axes[1].set_xlabel('Node Index')

        # Plot gradient norm
        grad_norm = analysis['gradient_norm'][0]
        axes[2].plot(grad_norm.numpy())
        axes[2].set_title('Gradient Norm per Node')
        axes[2].set_xlabel('Node Index')

        plt.tight_layout()
        save_path = os.path.join(self.save_dir, f'{sample_key}_gradients.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

    def _save_chefer_visualizations(self, analysis: Dict, sample_key: str):
        """Save Chefer method visualizations"""

        if not analysis:
            return

        fig, axes = plt.subplots(1, len(analysis), figsize=(5*len(analysis), 5))
        if len(analysis) == 1:
            axes = [axes]

        for idx, (method_name, result) in enumerate(analysis.items()):
            if isinstance(result, torch.Tensor):
                result = result.numpy()

            # Handle different result shapes
            if result.ndim > 2:
                result = result.mean(axis=tuple(range(result.ndim-2)))  # Reduce to 2D

            if result.ndim == 2:
                im = axes[idx].imshow(result, cmap='RdBu_r', aspect='auto')
                plt.colorbar(im, ax=axes[idx])
            elif result.ndim == 1:
                axes[idx].plot(result)

            axes[idx].set_title(method_name.replace('_', ' ').title())

        plt.tight_layout()
        save_path = os.path.join(self.save_dir, f'{sample_key}_chefer.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

    def _log_attention_statistics(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        batch_idx: int,
        stage: str
    ):
        """Log attention statistics to tensorboard/wandb"""

        try:
            metrics = {}

            for layer_name, attention_data_list in self.attention_storage.items():
                if attention_data_list:
                    attention_data = attention_data_list[-1]

                    # Extract attention patterns from stored data
                    layer_input = attention_data['input']
                    layer_output = attention_data['output']

                    # Compute basic statistics
                    input_np = layer_input.numpy() if isinstance(layer_input, torch.Tensor) else layer_input
                    output_np = layer_output.numpy() if isinstance(layer_output, torch.Tensor) else layer_output

                    # Compute attention-like metrics from input/output
                    entropy = self._compute_attention_entropy(output_np)
                    sparsity = self._compute_attention_sparsity(output_np)

                    layer_short_name = layer_name.split('.')[-1]
                    metrics[f'{stage}_attention_entropy/{layer_short_name}'] = entropy
                    metrics[f'{stage}_attention_sparsity/{layer_short_name}'] = sparsity

            # Log to trainer's logger
            pl_module.log_dict(metrics, on_step=True, on_epoch=False)

        except Exception as e:
            print(f"Error logging attention statistics: {e}")

    def _log_interpretability_metrics(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        sample_key: str
    ):
        """Log interpretability metrics"""

        # Log to tensorboard if available
        if hasattr(trainer.logger, 'experiment'):
            try:
                # Log file paths for generated visualizations
                metrics = {
                    'interpretability_analysis_completed': 1.0,
                    'analysis_epoch': float(trainer.current_epoch)
                }
                pl_module.log_dict(metrics)

            except Exception as e:
                print(f"Error logging interpretability metrics: {e}")


# Usage example with your existing training pipeline
# Add this to your configs/callbacks/ directory as a YAML file:
#
# _target_: src.utils.callbacks.transformer_explainability.InterpretabilityCallback
# save_dir: "./interpretability_results"
# analyze_every_n_epochs: 5
# analyze_every_n_batches: 100
# max_samples_per_epoch: 3
# enable_chefer_methods: true
# enable_attention_analysis: true
# enable_gradient_analysis: true

# Then add it to your default.yaml callback config

if __name__ == "__main__":
    print("InterpretabilityCallback ready for integration with your training pipeline!")
    print("Add the callback config to your configs/callbacks/ directory")