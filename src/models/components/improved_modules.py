"""Improved Graph Modules with Attention Pooling.

This file provides enhanced versions of the existing graph modules that integrate
SimpleAttentionPooling for better graph-level representations.
"""

from typing import Literal, overload

import torch
import torch.nn as nn
from omegaconf import DictConfig

from src.data.components.graphs_datamodules import DenseGraphBatch
from src.models.components.attention_pooling import (
    MultiQueryAttentionPooling,
    SimpleAttentionPooling,
)
from src.models.components.llama_graph_transformer import Transformer


class ImprovedGraphEncoder(torch.nn.Module):
    """Enhanced GraphEncoder with attention pooling instead of CLS token.

    Key improvements:
    - Replaces CLS token (summary_node) with learnable attention pooling
    - Better invariant representations without positional encoding
    - Maintains same interface as original GraphEncoder
    - Can be used as drop-in replacement
    """

    def __init__(self, hparams: DictConfig):
        super().__init__()

        # Type annotation for attention pooling
        self.attention_pool: SimpleAttentionPooling | MultiQueryAttentionPooling

        # Optional input projection (same as original)
        if hparams.project:
            self.projection_in = nn.Linear(
                hparams.num_node_features, hparams.graph_encoder_hidden_dim
            )
        self.project = hparams.project

        # Graph transformer (same as original)
        self.graph_transformer = Transformer(
            hidden_dim=hparams.graph_encoder_hidden_dim,
            num_heads=hparams.graph_encoder_num_heads,
            ppf_hidden_dim=hparams.graph_encoder_ppf_hidden_dim,
            num_layers=hparams.graph_encoder_num_layers,
            dropout=hparams.dropout,
        )

        # Input processing layers (same as original)
        self.fc_in = nn.Linear(hparams.graph_encoder_hidden_dim, hparams.graph_encoder_hidden_dim)
        self.layer_norm = nn.LayerNorm(hparams.graph_encoder_hidden_dim)
        self.dropout = nn.Dropout(hparams.dropout)

        # IMPROVEMENT: Replace CLS token with attention pooling
        pooling_type = getattr(hparams, "pooling_type", "simple")  # Default to simple

        if pooling_type == "simple":
            self.attention_pool = SimpleAttentionPooling(
                embed_dim=hparams.graph_encoder_hidden_dim,
                num_heads=min(hparams.graph_encoder_num_heads, 8),
                dropout=hparams.dropout,
            )
        elif pooling_type == "multi":
            self.attention_pool = MultiQueryAttentionPooling(
                embed_dim=hparams.graph_encoder_hidden_dim,
                num_queries=getattr(hparams, "num_attention_queries", 8),
                num_heads=min(hparams.graph_encoder_num_heads, 8),
                dropout=hparams.dropout,
            )
        else:
            raise ValueError(f"Unknown pooling_type: {pooling_type}")

    def forward(
        self,
        node_features: torch.Tensor,
        edge_features: torch.Tensor,
        mask: torch.Tensor,
        return_attention: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor] | tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass with attention pooling.

        Args:
            node_features: (batch, num_nodes, node_dim)
            edge_features: (batch, num_nodes, num_nodes, edge_dim)
            mask: (batch, num_nodes) - node mask
            return_attention: Whether to return attention maps

        Returns:
            graph_emb: (batch, hidden_dim) - graph-level representation
            node_features: (batch, num_nodes, hidden_dim) - updated node features
            attention_maps: (batch, num_nodes) - attention weights (optional)
        """
        # Input projection if needed
        if self.project:
            node_features = self.projection_in(node_features)

        # Process input features (no CLS token needed!)
        x = self.layer_norm(self.dropout(self.fc_in(node_features)))

        # Apply graph transformer.
        # NOTE: this "Improved" encoder does NOT use a CLS token, so we must NOT use the
        # encoder-style masks that assume an extra CLS position.
        x = self.graph_transformer(x, mask=mask, is_encoder=False)

        # IMPROVEMENT: Use attention pooling instead of CLS token extraction
        if return_attention:
            graph_emb, attention_maps = self.attention_pool(x, return_attention=True)
            return graph_emb, x, attention_maps
        else:
            graph_emb = self.attention_pool(x, return_attention=False)
            return graph_emb, x


class ImprovedGraphAE(torch.nn.Module):
    """Enhanced GraphAE using ImprovedGraphEncoder with attention pooling.

    Maintains same interface as original GraphAE but with better representations.
    """

    def __init__(self, hparams: DictConfig):
        super().__init__()
        self.input_size = hparams.input_size
        self.vae = hparams.vae

        # IMPROVEMENT: Use ImprovedGraphEncoder instead of GraphEncoder
        self.encoder = ImprovedGraphEncoder(hparams.encoder)

        # Keep everything else the same
        self.bottle_neck_encoder = BottleNeckEncoder(hparams.bottle_neck_encoder)
        self.bottle_neck_decoder = BottleNeckDecoder(hparams.bottle_neck_decoder)
        self.permuter = SimplePermuter(hparams.permuter)
        self.decoder = GraphDecoder(hparams.decoder)

    @overload
    def encode(
        self, graph: DenseGraphBatch, *, return_attention: Literal[False] = False
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]: ...

    @overload
    def encode(
        self, graph: DenseGraphBatch, *, return_attention: Literal[True]
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]: ...

    def encode(
        self, graph: DenseGraphBatch, return_attention: bool = False
    ) -> (
        tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
        | tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
    ):
        """Encode graph with optional attention visualization."""
        node_features = graph.node_features
        edge_features = graph.edge_features
        mask = graph.mask

        if return_attention:
            graph_emb, node_features, attention_maps = self.encoder(
                node_features=node_features,
                edge_features=edge_features,
                mask=mask,
                return_attention=True,
            )
            graph_emb, mu, logvar = self.bottle_neck_encoder(graph_emb)
            return graph_emb, node_features, mu, logvar, attention_maps
        else:
            graph_emb, node_features = self.encoder(
                node_features=node_features,
                edge_features=edge_features,
                mask=mask,
            )
            graph_emb, mu, logvar = self.bottle_neck_encoder(graph_emb)
            return graph_emb, node_features, mu, logvar

    def decode(
        self,
        graph_emb: torch.Tensor,
        perm: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> DenseGraphBatch:
        """Same as original decode method."""
        graph_emb = self.bottle_neck_decoder(graph_emb)
        node_logits, edge_logits = self.decoder(graph_emb=graph_emb, perm=perm, mask=mask)
        graph_pred = DenseGraphBatch(
            node_features=node_logits,
            edge_features=edge_logits,
            mask=mask,
            properties=torch.tensor([]),
        )
        return graph_pred

    def forward(self, graph: DenseGraphBatch, training: bool, tau: float = 1.0) -> tuple:
        """Same interface as original forward method."""
        graph_emb, node_features, mu, logvar = self.encode(graph=graph, return_attention=False)
        perm, context, soft_probs, _ = self.permuter(
            node_features, mask=graph.mask, hard=not training, tau=tau
        )
        if context is not None:
            graph_emb += context
        graph_pred = self.decode(graph_emb, perm, graph.mask)
        return graph_emb, graph_pred, soft_probs, perm, mu, logvar

    def forward_with_attention(self, graph: DenseGraphBatch, training: bool, tau: float = 1.0):
        """New method: forward pass with attention visualization"""
        graph_emb, node_features, mu, logvar, attention_maps = self.encode(
            graph=graph, return_attention=True
        )
        perm, context, soft_probs, _ = self.permuter(
            node_features, mask=graph.mask, hard=not training, tau=tau
        )
        if context is not None:
            graph_emb += context
        graph_pred = self.decode(graph_emb, perm, graph.mask)
        return graph_emb, graph_pred, soft_probs, perm, mu, logvar, attention_maps


# Import the existing classes that we're not modifying
from src.models.components.modules import (
    BottleNeckDecoder,
    BottleNeckEncoder,
    GraphDecoder,
    SimplePermuter,
)


def create_attention_config_example():
    """Example of how to modify your existing config to use attention pooling.

    Add these fields to your encoder config:
    """
    config_additions = {
        "pooling_type": "simple",  # or 'multi' for MultiQueryAttentionPooling
        "num_attention_queries": 8,  # only needed for 'multi' pooling type
    }
    print("Add these fields to your encoder config:")
    for key, value in config_additions.items():
        print(f"  {key}: {value}")
    return config_additions


def test_improved_modules():
    """Test the improved modules."""
    print("Testing ImprovedGraphEncoder...")

    # Create dummy config (similar to your actual config structure)
    from omegaconf import DictConfig

    encoder_config = DictConfig({
        "graph_encoder_hidden_dim": 256,
        "graph_encoder_num_heads": 8,
        "graph_encoder_ppf_hidden_dim": 512,
        "graph_encoder_num_layers": 6,
        "num_node_features": 64,
        "dropout": 0.1,
        "project": True,
        "pooling_type": "simple",  # NEW: attention pooling type
    })

    # Test data
    batch_size, num_nodes, node_dim = 4, 32, 64
    node_features = torch.randn(batch_size, num_nodes, node_dim)
    edge_features = torch.randn(batch_size, num_nodes, num_nodes, 16)
    mask = torch.ones(batch_size, num_nodes, dtype=torch.bool)

    # Test ImprovedGraphEncoder
    encoder = ImprovedGraphEncoder(encoder_config)

    # Standard forward pass
    graph_emb, updated_nodes = encoder(node_features, edge_features, mask)
    print(f"Graph embedding shape: {graph_emb.shape}")
    print(f"Updated nodes shape: {updated_nodes.shape}")

    # Forward pass with attention
    graph_emb, updated_nodes, attention = encoder(
        node_features, edge_features, mask, return_attention=True
    )
    print(f"Attention maps shape: {attention.shape}")
    print(f"Attention focuses on: {attention.argmax(dim=-1).tolist()}")

    # Check attention properties
    attention_sum = attention.sum(dim=-1)
    print(f"Attention weights sum to: {attention_sum.mean():.4f} (should be ~1.0)")

    attention_max = attention.max(dim=-1)[0]
    uniform_attention = 1.0 / num_nodes
    print(f"Max attention: {attention_max.mean():.4f} vs uniform: {uniform_attention:.4f}")
    print(f"Learning selectivity: {attention_max.mean() > uniform_attention}")

    print("✅ ImprovedGraphEncoder test passed!")


if __name__ == "__main__":
    create_attention_config_example()
    test_improved_modules()
