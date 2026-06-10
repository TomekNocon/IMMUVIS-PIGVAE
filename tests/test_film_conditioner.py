import torch

from src.models.components.modules import FiLMConditioner


def test_film_identity_at_init():
    """Zero-init projections -> gamma=beta=0 -> FiLM is identity, bounded or not."""
    for bound in (False, True):
        film = FiLMConditioner(z_dim=16, hidden_dim=32, num_layers=4, bound=bound)
        params = film(torch.randn(8, 16))
        assert len(params) == 4
        for gamma, beta in params:
            assert gamma.shape == (8, 32) and beta.shape == (8, 32)
            assert float(gamma.abs().max()) == 0.0
            assert float(beta.abs().max()) == 0.0


def test_film_unbounded_can_grow_large():
    """Without bounding, large projection weights produce large gamma/beta."""
    film = FiLMConditioner(z_dim=16, hidden_dim=32, num_layers=4, bound=False)
    with torch.no_grad():
        for p in film.layer_projs:
            p.weight.normal_(0, 5.0)
            p.bias.normal_(0, 5.0)
    gamma, _ = film(torch.randn(8, 16))[0]
    assert float(gamma.abs().max()) > 1.0  # unbounded: escapes [-1, 1]


def test_film_bounded_stays_in_tanh_range():
    """With bound=True, gamma/beta are tanh-squashed into (-1, 1) no matter the weights."""
    film = FiLMConditioner(z_dim=16, hidden_dim=32, num_layers=4, bound=True)
    with torch.no_grad():
        for p in film.layer_projs:
            p.weight.normal_(0, 5.0)
            p.bias.normal_(0, 5.0)
    for gamma, beta in film(torch.randn(8, 16)):
        assert float(gamma.abs().max()) <= 1.0
        assert float(beta.abs().max()) <= 1.0
