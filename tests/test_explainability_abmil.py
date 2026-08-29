import torch

from src.downstream.abmil.model import GatedABMIL
from src.explainability.abmil_attribution import abmil_attention, abmil_ig, crop_importance
from src.explainability.faithfulness import deletion_curve, faithfulness_score


def _abmil():
    torch.manual_seed(0)
    return GatedABMIL(emb_dim=16, hidden_dim=8, num_heads=1, num_classes=2).eval()


def test_abmil_attention_softmax_over_crops():
    model = _abmil()
    x = torch.randn(2, 5, 16)  # 2 bags, 5 crops
    a = abmil_attention(model, x)
    assert a.shape == (2, 5, 1)
    # attention is a distribution over the 5 crops (per head)
    assert torch.allclose(a.sum(dim=1), torch.ones(2, 1), atol=1e-5)


def test_abmil_ig_shape_and_reduction():
    model = _abmil()
    x = torch.randn(2, 5, 16)
    attr = abmil_ig(model, x, target=0, n_steps=8)
    assert attr.shape == (2, 5, 16)
    assert torch.isfinite(attr).all()
    imp = crop_importance(attr)
    assert imp.shape == (2, 5)


def test_faithfulness_rewards_correct_importance():
    # Linear score = sum of feature-0 across instances; the true importance is feature-0.
    torch.manual_seed(1)
    x = torch.rand(6, 4) + 1.0  # positive contributions
    forward_fn = lambda bag: bag[:, 0].sum()
    good = x[:, 0]
    bad = -x[:, 0]  # reversed ranking

    # deletion with correct importance is monotonically non-increasing (remove positives)
    dele = deletion_curve(forward_fn, x, good)
    assert torch.all(dele[1:] - dele[:-1] <= 1e-6)

    f_good = faithfulness_score(forward_fn, x, good)["faithfulness"]
    f_bad = faithfulness_score(forward_fn, x, bad)["faithfulness"]
    assert f_good > f_bad  # a faithful map scores higher than the reversed one
