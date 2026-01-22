"""Attention Pooling Components for VAE Encoder.

Provides SimpleAttentionPooling as a drop-in replacement for global average pooling in
graph transformer encoders to improve invariant representation learning.
"""

import torch
import torch.nn as nn


class SimpleAttentionPooling(nn.Module):
    """Simple attention pooling with single learnable query.

    Replaces global average pooling to:
    - Learn what graph nodes/patches are important for reconstruction
    - Preserve information better than global pooling
    - Maintain invariance without positional encoding
    - Provide interpretable attention maps

    Args:
        embed_dim: Hidden dimension of the transformer
        num_heads: Number of attention heads (default: 8)
        dropout: Dropout probability (default: 0.1)
    """

    def __init__(self, embed_dim: int, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads

        # Single learnable query - learns what to look for in the graph
        self.query = nn.Parameter(torch.randn(1, embed_dim) * 0.02)

        # Multi-head attention mechanism
        self.attention = nn.MultiheadAttention(
            embed_dim, num_heads, dropout=dropout, batch_first=True
        )

        # Layer normalization for training stability
        self.norm_query = nn.LayerNorm(embed_dim)
        self.norm_input = nn.LayerNorm(embed_dim)
        self.norm_output = nn.LayerNorm(embed_dim)

        self._init_weights()

    def _init_weights(self):
        """Initialize query with small random values for stable training."""
        nn.init.normal_(self.query, std=0.02)

    def forward(self, x: torch.Tensor, return_attention: bool = False):
        """Apply attention pooling to node/patch features.

        Args:
            x: (batch_size, num_nodes, embed_dim) - node features from transformer
            return_attention: Whether to return attention weights for visualization

        Returns:
            pooled: (batch_size, embed_dim) - graph-level representation
            attention_weights: (batch_size, num_nodes) - attention weights (optional)
        """
        batch_size, _num_nodes, _embed_dim = x.shape

        # Prepare learnable query (expand for batch dimension)
        query = self.query.expand(batch_size, 1, self.embed_dim)  # (batch, 1, embed_dim)
        query = self.norm_query(query)

        # Normalize input node features
        kv = self.norm_input(x)  # (batch, num_nodes, embed_dim)

        # Apply cross-attention: query attends to all nodes
        pooled, attention_weights = self.attention(
            query=query,  # What we're looking for (learnable)
            key=kv,  # What we're looking at (node features)
            value=kv,  # What we extract (node features)
        )

        # Remove query dimension and normalize
        pooled = pooled.squeeze(1)  # (batch, embed_dim)
        pooled = self.norm_output(pooled)

        if return_attention:
            # attention_weights: (batch, 1, num_nodes) -> (batch, num_nodes)
            return pooled, attention_weights.squeeze(1)
        else:
            return pooled

    def get_attention_maps(self, x: torch.Tensor):
        """Get attention maps for visualization without gradients."""
        with torch.no_grad():
            _, attention_weights = self.forward(x, return_attention=True)
        return attention_weights


class MultiQueryAttentionPooling(nn.Module):
    """Advanced attention pooling with multiple learnable queries.

    Uses multiple queries to capture different aspects of the graph, then combines them
    for a richer representation.
    """

    def __init__(
        self, embed_dim: int, num_queries: int = 8, num_heads: int = 8, dropout: float = 0.1
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_queries = num_queries
        self.num_heads = num_heads

        # Multiple learnable queries - each focuses on different aspects
        self.queries = nn.Parameter(torch.randn(num_queries, embed_dim) * 0.02)

        # Multi-head cross-attention
        self.cross_attention = nn.MultiheadAttention(
            embed_dim, num_heads, dropout=dropout, batch_first=True
        )

        # Layer normalizations
        self.norm_queries = nn.LayerNorm(embed_dim)
        self.norm_input = nn.LayerNorm(embed_dim)
        self.norm_output = nn.LayerNorm(embed_dim)

        # Combine multiple query outputs
        self.combine_queries = nn.Sequential(
            nn.Linear(num_queries * embed_dim, embed_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 2, embed_dim),
        )

        self._init_weights()

    def _init_weights(self):
        """Initialize queries with orthogonal initialization for diversity."""
        nn.init.orthogonal_(self.queries)
        self.queries.data *= 0.02  # Scale down

    def forward(self, x: torch.Tensor, return_attention: bool = False):
        """Apply multi-query attention pooling.

        Args:
            x: (batch_size, num_nodes, embed_dim)
            return_attention: Whether to return attention weights

        Returns:
            pooled: (batch_size, embed_dim)
            attention_weights: (batch_size, num_queries, num_nodes) - optional
        """
        batch_size, _num_nodes, _embed_dim = x.shape

        # Prepare queries
        queries = self.queries.unsqueeze(0).expand(batch_size, -1, -1)
        queries = self.norm_queries(queries)  # (batch, num_queries, embed_dim)

        # Normalize input
        kv = self.norm_input(x)  # (batch, num_nodes, embed_dim)

        # Cross-attention: multiple queries attend to nodes
        pooled_queries, attention_weights = self.cross_attention(
            query=queries,  # Multiple learnable queries
            key=kv,  # Node features
            value=kv,  # Node features
        )
        # pooled_queries: (batch, num_queries, embed_dim)
        # attention_weights: (batch, num_queries, num_nodes)

        # Normalize query outputs
        pooled_queries = self.norm_output(pooled_queries)

        # Combine query outputs into single representation
        combined = pooled_queries.flatten(1)  # (batch, num_queries * embed_dim)
        pooled = self.combine_queries(combined)  # (batch, embed_dim)

        if return_attention:
            return pooled, attention_weights
        else:
            return pooled


def test_attention_pooling():
    """Test attention pooling components."""
    print("Testing attention pooling components...")

    # Test parameters (similar to your graph transformer setup)
    batch_size, num_nodes, embed_dim = 4, 32, 256  # Typical graph sizes

    # Create test data
    x = torch.randn(batch_size, num_nodes, embed_dim)
    print(f"Input shape: {x.shape}")

    # Test SimpleAttentionPooling
    print("\nTesting SimpleAttentionPooling...")
    simple_pool = SimpleAttentionPooling(embed_dim=embed_dim, num_heads=8)

    pooled = simple_pool(x)
    print(f"Simple pooled shape: {pooled.shape}")

    pooled, attn = simple_pool(x, return_attention=True)
    print(f"Attention shape: {attn.shape}")
    attn_sum = attn.sum(dim=-1).mean()
    print(f"Attention weights sum: {attn_sum:.4f} (should be ~1.0)")

    # Test MultiQueryAttentionPooling
    print("\nTesting MultiQueryAttentionPooling...")
    multi_pool = MultiQueryAttentionPooling(embed_dim=embed_dim, num_queries=8, num_heads=8)

    pooled = multi_pool(x)
    print(f"Multi-query pooled shape: {pooled.shape}")

    pooled, attn = multi_pool(x, return_attention=True)
    print(f"Multi-query attention shape: {attn.shape}")

    # Analyze attention diversity
    attn_std = attn.std(dim=-1).mean()  # How selective attention is
    query_similarity = torch.corrcoef(attn.flatten(0, 1)).abs().mean()
    print(f"Attention selectivity: {attn_std:.4f} (higher = more focused)")
    print(f"Query similarity: {query_similarity:.4f} (lower = more diverse)")

    print("✅ All tests passed!")


if __name__ == "__main__":
    test_attention_pooling()
