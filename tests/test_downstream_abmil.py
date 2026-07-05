import torch


def test_gated_abmil_shapes_and_masking():
    from src.downstream.abmil.model import GatedABMIL

    m = GatedABMIL(emb_dim=16, hidden_dim=8, num_heads=2, num_classes=2)
    x = torch.randn(3, 5, 16)
    mask = torch.zeros(3, 5, dtype=torch.bool)
    mask[:, 4] = True
    logits, pooled = m(x, mask=mask)
    assert logits.shape == (3, 1) and pooled.shape == (3, 2 * 16)

    # Regression check: `mask` uses the True == padding convention, so
    # perturbing only the masked (padded) instances must leave the pooled
    # representation and logits unchanged. This fails if the mask sense is
    # ever inverted (e.g. True treated as "valid" or masked_fill(~mask, ...)).
    m.eval()
    with torch.no_grad():
        assert (~mask).any(), "at least one instance must remain unmasked"
        logits1, pooled1 = m(x, mask=mask)

        x_perturbed = x.clone()
        x_perturbed[mask] = x_perturbed[mask] + 1000.0

        logits2, pooled2 = m(x_perturbed, mask=mask)

        assert torch.allclose(pooled1, pooled2, atol=1e-5)
        assert torch.allclose(logits1, logits2, atol=1e-5)


def test_mil_collate_pads_and_masks():
    import numpy as np, torch
    from src.downstream.abmil.data import MILDataset, mil_collate
    ds = MILDataset([np.ones((2,4),"float32"), np.ones((5,4),"float32")], torch.tensor([0,1]))
    bags, masks, labels = mil_collate([ds[0], ds[1]])
    assert bags.shape == (2,5,4) and masks.shape == (2,5)
    assert masks[0,2] and not masks[0,1] and not masks[1,4]
    assert list(labels) == [0,1]


def test_build_image_bags_groups_and_labels(tmp_path):
    import numpy as np, pandas as pd
    from src.downstream.abmil.data import build_image_bags
    emb = str(tmp_path/"e.npy"); np.save(emb, np.arange(6*4, dtype="float32").reshape(6,4))
    df = pd.DataFrame({
        "img_path": ["a","a","a","b","b","b"],
        "embeddings_file": [emb]*6, "embedding_idx": [0,1,2,3,4,5],
        "feature_value": ["pos","pos","pos","neg","neg","neg"]})
    bags, labels = build_image_bags(df, {"neg":0,"pos":1})
    assert len(bags) == 2 and bags[0].shape == (3,4)
    assert list(labels) == [1,0]


def test_build_image_bags_drops_unmapped_and_stays_aligned(tmp_path):
    import numpy as np, pandas as pd
    from src.downstream.abmil.data import build_image_bags
    emb = str(tmp_path/"e.npy"); np.save(emb, np.arange(7*4, dtype="float32").reshape(7,4))
    df = pd.DataFrame({
        "img_path": ["valid","valid","valid","nanlabel","nanlabel","unmapped","unmapped"],
        "embeddings_file": [emb]*7,
        "embedding_idx": [0,1,2,3,4,5,6],
        "feature_value": ["pos","pos","pos",np.nan,np.nan,"unknown_class","unknown_class"]})

    bags, labels = build_image_bags(df, {"neg":0,"pos":1})

    # Both the NaN-labelled image and the unmapped-label image must be
    # dropped from BOTH outputs, keeping bags and labels aligned 1:1.
    assert len(bags) == 1 and len(labels) == 1
    assert bags[0].shape == (3, 4)
    assert np.array_equal(bags[0], np.arange(3*4, dtype="float32").reshape(3,4))
    assert list(labels) == [1]


def test_abmil_lit_overfits_toy():
    import torch, numpy as np
    from torch.utils.data import DataLoader
    import pytorch_lightning as pl
    from src.downstream.abmil.lit import AbmilLitModule
    from src.downstream.abmil.data import MILDataset, mil_collate
    g = torch.Generator().manual_seed(0)
    pos = [np.ones((3,8),"float32") for _ in range(8)]; neg = [(-np.ones((3,8),"float32")) for _ in range(8)]
    ds = MILDataset(pos+neg, torch.tensor([1]*8+[0]*8))
    dl = DataLoader(ds, batch_size=4, shuffle=True, collate_fn=mil_collate)
    m = AbmilLitModule(emb_dim=8, hidden_dim=8, num_heads=1, num_classes=2, lr=1e-2)
    tr = pl.Trainer(max_epochs=30, enable_progress_bar=False, logger=False, enable_checkpointing=False, accelerator="cpu")
    tr.fit(m, dl)
    # after fit, predictions on the training bags are (near) perfect
    with torch.no_grad():
        b, mk, y = mil_collate([ds[i] for i in range(len(ds))])
        pred = (torch.sigmoid(m.model(b, mk)[0]).squeeze(1) > 0.5).long()
    assert (pred == y).float().mean() > 0.9
