import torch
import pytest
from omegaconf import OmegaConf

from src.data.components.graphs_datamodules import DenseGraphBatch


def make_graph_ae_hparams():
    return OmegaConf.create({
        "input_size": 64,
        "num_heads": 2,
        "num_layers": 1,
        "emb_dim": 32,
        "vae": True,
        "dropout": 0.0,
        "encoder": {
            "graph_encoder_hidden_dim": 64,
            "num_node_features": 8,
            "num_edge_features": 0,
            "graph_encoder_num_heads": 2,
            "graph_encoder_ppf_hidden_dim": 128,
            "graph_encoder_num_layers": 1,
            "emb_dim": 32,
            "dropout": 0.0,
            "grid_size": 4,
            "project": True,
        },
        "decoder": {
            "graph_decoder_hidden_dim": 64,
            "graph_decoder_pos_emb_dim": 64,
            "graph_decoder_num_heads": 2,
            "graph_decoder_ppf_hidden_dim": 128,
            "graph_decoder_num_layers": 1,
            "dropout": 0.0,
            "head_dim": 32,
            "num_embeddings": 64,
            "num_node_features": 8,
            "num_edge_features": 0,
            "project": True,
            "grid_size": 4,
        },
        "bottle_neck_encoder": {
            "graph_encoder_hidden_dim": 64,
            "emb_dim": 32,
            "vae": True,
            "activation": "silu",
            "num_permutations": 1,  # still needed until Task 5 removes it
        },
        "bottle_neck_decoder": {
            "emb_dim": 32,
            "graph_decoder_hidden_dim": 64,
            "num_nodes": 16,
        },
        "property_predictor": {
            "emb_dim": 32,
            "property_predictor_hidden_dim": 64,
            "num_properties": 1,
        },
    })


def make_batch(B: int = 2, N: int = 16, C: int = 8):
    node_features = torch.randn(B, N, C)
    mask = torch.ones(B, N, dtype=torch.bool)
    return DenseGraphBatch(
        node_features=node_features,
        edge_features=torch.empty(0),
        mask=mask,
    )


class TestGraphAENoPermuter:
    def test_forward_returns_4_tuple(self):
        from src.models.components.modules import GraphAE
        hparams = make_graph_ae_hparams()
        ae = GraphAE(hparams)
        graph = make_batch(B=2, N=16, C=8)
        out = ae(graph)
        assert len(out) == 4, f"expected 4-tuple, got {len(out)}"
        graph_emb, graph_pred, mu, logvar = out
        assert graph_emb.shape[0] == 2
        assert mu.shape == (2, 32)

    def test_no_permuter_attribute(self):
        from src.models.components.modules import GraphAE
        hparams = make_graph_ae_hparams()
        ae = GraphAE(hparams)
        assert not hasattr(ae, "permuter"), "permuter should be removed"
