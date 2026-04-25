import torch
import pytest
from omegaconf import OmegaConf

from src.models.components.modules import BottleNeckEncoder


def make_hparams(vae=True):
    return OmegaConf.create({
        "graph_encoder_hidden_dim": 64,
        "emb_dim": 32,
        "vae": vae,
        "activation": "silu",
    })


class TestBottleNeckEncoderSingleView:
    def test_no_num_permutations_attr(self):
        enc = BottleNeckEncoder(make_hparams())
        assert not hasattr(enc, "num_permutations"), "num_permutations should be removed"

    def test_different_samples_get_different_eps(self):
        """Two forward passes with the same input must give different z (standard VAE)."""
        enc = BottleNeckEncoder(make_hparams())
        x = torch.randn(4, 64)
        z1, mu1, _ = enc(x)
        z2, mu2, _ = enc(x)
        assert torch.allclose(mu1, mu2, atol=1e-5)
        assert not torch.allclose(z1, z2), "z should differ due to different eps"

    def test_output_shapes_vae(self):
        enc = BottleNeckEncoder(make_hparams(vae=True))
        x = torch.randn(4, 64)
        z, mu, logvar = enc(x)
        assert z.shape == (4, 32)
        assert mu.shape == (4, 32)
        assert logvar.shape == (4, 32)

    def test_output_shapes_ae(self):
        enc = BottleNeckEncoder(make_hparams(vae=False))
        x = torch.randn(4, 64)
        z, mu, logvar = enc(x)
        assert z.shape == (4, 32)
        assert mu is None
        assert logvar is None
