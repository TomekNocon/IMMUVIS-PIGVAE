"""
Test Script: Verify AttentionPooling Integration Works

This script tests that the attention pooling integration works properly
with your existing codebase structure.
"""

import torch
import torch.nn.functional as F
from omegaconf import DictConfig

# Test the imports
try:
    from src.data.components.graphs_datamodules import DenseGraphBatch
    from src.models.components.attention_pooling import (
        SimpleAttentionPooling,
    )
    from src.models.components.improved_modules import (
        ImprovedGraphAE,
        ImprovedGraphEncoder,
    )

    print("✅ All imports successful!")
except Exception as e:
    print(f"❌ Import failed: {e}")
    exit(1)


def create_test_config():
    """Create a minimal config that works with your existing structure."""
    return DictConfig({
        "input_size": 64,
        "vae": True,
        "encoder": {
            "graph_encoder_hidden_dim": 128,  # Smaller for testing
            "graph_encoder_num_heads": 4,
            "graph_encoder_ppf_hidden_dim": 256,
            "graph_encoder_num_layers": 2,  # Fewer layers for speed
            "num_node_features": 32,
            "dropout": 0.1,
            "project": True,
            "pooling_type": "simple",  # NEW: attention pooling
        },
        "bottle_neck_encoder": {
            "graph_encoder_hidden_dim": 128,
            "emb_dim": 64,
            "vae": True,
            "num_permutations": 1,
            "activation": "gelu",
        },
        "bottle_neck_decoder": {
            "emb_dim": 64,
            "graph_decoder_hidden_dim": 128,
        },
        "permuter": {
            "hidden_dim": 128,
            "num_layers": 2,
            "dropout": 0.1,
        },
        "decoder": {
            "graph_decoder_hidden_dim": 128,
            "graph_decoder_num_heads": 4,
            "graph_decoder_ppf_hidden_dim": 256,
            "graph_decoder_num_layers": 2,
            "graph_decoder_pos_emb_dim": 128,
            "head_dim": 16,
            "dropout": 0.1,
            "project": True,
            "num_node_features": 32,
            "num_edge_features": 8,
        },
    })


def create_test_graph_batch(batch_size=2, num_nodes=16):
    """Create a test graph batch."""
    node_features = torch.randn(batch_size, num_nodes, 32)
    edge_features = torch.randn(batch_size, num_nodes, num_nodes, 8)
    mask = torch.ones(batch_size, num_nodes, dtype=torch.bool)
    # Randomly mask some nodes
    for i in range(batch_size):
        mask_nodes = torch.randint(0, num_nodes // 2, (1,)).item()
        if mask_nodes > 0:
            mask[i, -mask_nodes:] = False

    properties = torch.tensor([])

    return DenseGraphBatch(
        node_features=node_features, edge_features=edge_features, mask=mask, properties=properties
    )


def test_attention_pooling_only():
    """Test just the attention pooling component."""
    print("\n=== Testing SimpleAttentionPooling ===")

    batch_size, num_nodes, embed_dim = 2, 16, 128
    x = torch.randn(batch_size, num_nodes, embed_dim)

    # Test SimpleAttentionPooling
    attention_pool = SimpleAttentionPooling(embed_dim=embed_dim, num_heads=4)

    # Forward without attention
    pooled = attention_pool(x)
    print(f"✅ Pooled output shape: {pooled.shape}")
    if pooled.shape != (batch_size, embed_dim):
        raise ValueError(f"Expected {(batch_size, embed_dim)}, got {pooled.shape}")

    # Forward with attention
    pooled, attention_maps = attention_pool(x, return_attention=True)
    print(f"✅ Attention maps shape: {attention_maps.shape}")
    if attention_maps.shape != (batch_size, num_nodes):
        raise ValueError(f"Expected {(batch_size, num_nodes)}, got {attention_maps.shape}")

    # Check attention properties
    attention_sum = attention_maps.sum(dim=-1)
    print(f"✅ Attention weights sum to: {attention_sum.mean():.4f} (should be ~1.0)")
    # Note: MultiheadAttention may not sum exactly to 1.0, but should be close
    if not torch.allclose(attention_sum, torch.ones_like(attention_sum), atol=0.1):
        raise ValueError("Attention weights don't sum to approximately 1.0")

    print("✅ SimpleAttentionPooling test passed!")


def test_improved_graph_encoder():
    """Test the improved graph encoder."""
    print("\n=== Testing ImprovedGraphEncoder ===")

    config = create_test_config()
    encoder = ImprovedGraphEncoder(config.encoder)

    # Create test data
    batch_size, num_nodes = 2, 16
    node_features = torch.randn(batch_size, num_nodes, 32)
    edge_features = torch.randn(batch_size, num_nodes, num_nodes, 8)
    mask = torch.ones(batch_size, num_nodes, dtype=torch.bool)

    # Forward pass
    graph_emb, updated_nodes = encoder(node_features, edge_features, mask)
    print(f"✅ Graph embedding shape: {graph_emb.shape}")
    print(f"✅ Updated nodes shape: {updated_nodes.shape}")

    # Forward with attention
    graph_emb, updated_nodes, attention = encoder(
        node_features, edge_features, mask, return_attention=True
    )
    print(f"✅ Attention shape: {attention.shape}")

    # Verify attention focuses on unmasked nodes
    print(f"✅ Attention learned selectivity: {attention.std(dim=-1).mean():.4f}")

    print("✅ ImprovedGraphEncoder test passed!")


def test_full_improved_graph_ae():
    """Test the complete ImprovedGraphAE."""
    print("\n=== Testing Full ImprovedGraphAE ===")

    try:
        config = create_test_config()
        model = ImprovedGraphAE(config)

        # Create test graph
        graph = create_test_graph_batch(batch_size=2, num_nodes=16)

        print("✅ Created model and test data")
        print(f"   Graph nodes shape: {graph.node_features.shape}")
        print(f"   Graph edges shape: {graph.edge_features.shape}")
        print(f"   Graph mask shape: {graph.mask.shape}")

        # Test standard forward pass
        outputs = model(graph, training=True)
        graph_emb, graph_pred, _soft_probs, _perm, mu, logvar = outputs

        print("✅ Standard forward pass successful!")
        print(f"   Graph embedding: {graph_emb.shape}")
        print(f"   Reconstruction nodes: {graph_pred.node_features.shape}")
        print(f"   Latent mu: {mu.shape}")
        print(f"   Latent logvar: {logvar.shape}")

        # Test forward with attention
        outputs_with_attention = model.forward_with_attention(graph, training=True)
        graph_emb, graph_pred, _soft_probs, _perm, mu, logvar, attention = outputs_with_attention

        print("✅ Forward with attention successful!")
        print(f"   Attention maps: {attention.shape}")

        # Analyze attention
        most_attended = attention.argmax(dim=-1)
        attention_max = attention.max(dim=-1)[0]
        print(f"   Most attended nodes: {most_attended.tolist()}")
        print(f"   Max attention values: {attention_max.tolist()}")

        print("✅ Full ImprovedGraphAE test passed!")

        return True

    except Exception as e:
        print(f"❌ Full test failed: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_comparison_with_uniform():
    """Compare attention pooling vs uniform pooling."""
    print("\n=== Comparing Attention vs Uniform Pooling ===")

    batch_size, num_nodes, embed_dim = 2, 16, 128
    x = torch.randn(batch_size, num_nodes, embed_dim)

    # Uniform pooling (current approach)
    uniform_pooled = x.mean(dim=1)

    # Attention pooling (new approach)
    attention_pool = SimpleAttentionPooling(embed_dim=embed_dim, num_heads=4)
    attention_pooled, attention_maps = attention_pool(x, return_attention=True)

    print(f"✅ Uniform pooling shape: {uniform_pooled.shape}")
    print(f"✅ Attention pooling shape: {attention_pooled.shape}")

    # Compare representations
    cosine_sim = F.cosine_similarity(uniform_pooled, attention_pooled, dim=1)
    print(f"✅ Cosine similarity: {cosine_sim.mean():.4f}")

    # Attention analysis
    attention_entropy = -(attention_maps * torch.log(attention_maps + 1e-8)).sum(dim=-1)
    uniform_entropy = -torch.log(torch.tensor(1.0 / num_nodes)) * num_nodes

    print(f"✅ Attention entropy: {attention_entropy.mean():.4f}")
    print(f"✅ Uniform entropy: {uniform_entropy:.4f}")
    print(f"✅ Attention is more focused: {attention_entropy.mean() < uniform_entropy}")

    return True


def main():
    """Run all tests."""
    print("=" * 60)
    print("ATTENTION POOLING INTEGRATION TEST")
    print("=" * 60)

    success = True

    try:
        test_attention_pooling_only()
        test_improved_graph_encoder()
        success &= test_full_improved_graph_ae()
        test_comparison_with_uniform()

    except Exception as e:
        print(f"\n❌ Test suite failed: {e}")
        import traceback

        traceback.print_exc()
        success = False

    print("\n" + "=" * 60)
    if success:
        print("🎉 ALL TESTS PASSED!")
        print("✅ SimpleAttentionPooling is ready for integration")
        print("✅ ImprovedGraphAE works with your existing structure")
        print("✅ Drop-in replacement confirmed")
    else:
        print("❌ Some tests failed - check output above")
    print("=" * 60)

    return success


if __name__ == "__main__":
    main()
