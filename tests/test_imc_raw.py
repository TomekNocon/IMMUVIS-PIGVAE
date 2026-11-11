import numpy as np

from src.data.imc_raw import CROP, crops, preprocess
from src.models.immuvis import DEFAULT_CODE_ROOT, channel_ids


def test_crops_row_major_and_drop_remainder():
    img = np.zeros((2, 300, 270), np.float32)
    out = crops(img)
    assert [ij for ij, _ in out] == [(0, 0), (0, 128), (128, 0), (128, 128)]
    assert all(c.shape == (2, CROP, CROP) for _, c in out)


def test_crops_too_small_is_empty():
    assert crops(np.zeros((1, 127, 500), np.float32)) == []


def test_preprocess_range_and_dtype():
    rng = np.random.default_rng(0)
    img = rng.poisson(20, size=(3, 64, 64)).astype(np.uint16)
    out = preprocess(img)
    assert out.dtype == np.float32 and out.shape == (3, 64, 64)
    assert out.min() >= 0 and out.max() <= 1


def test_cords_channel_ids_from_submodule():
    configs = DEFAULT_CODE_ROOT / "configs"
    ids, names = channel_ids(
        configs / "all_panels_config.yaml", configs / "all_markers_tokenizer.yaml"
    )
    assert len(names) == 43 and ids.shape == (43,)
    assert names[16] == "CD8a" and names[34] == "panCK"
