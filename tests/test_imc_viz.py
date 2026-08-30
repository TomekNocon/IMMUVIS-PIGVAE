"""Unit + smoke tests for the IMC input visualiser (src/explainability/imc_viz.py).

All synthetic (no raid data, no GPU): validates the pure logic (orientation, tiling, contrast,
composite blending) and smoke-tests that the two renderers write a non-empty PNG.
"""

import numpy as np
import pytest

from src.explainability import imc_viz


@pytest.fixture
def img_chw():
    rng = np.random.default_rng(0)
    return rng.random((40, 256, 256)).astype(np.float32)


def test_to_chw_from_hwc(img_chw):
    hwc = np.moveaxis(img_chw, 0, -1)  # [H, W, C]
    out = imc_viz._to_chw(hwc)
    assert out.shape == (40, 256, 256)
    assert np.allclose(out, img_chw)


def test_to_chw_passthrough(img_chw):
    assert imc_viz._to_chw(img_chw).shape == (40, 256, 256)


def test_to_chw_rejects_non_3d():
    with pytest.raises(ValueError):
        imc_viz._to_chw(np.zeros((10, 10)))


def test_get_all_patches_coords_and_shape(img_chw):
    patches, coords = imc_viz.get_all_patches(img_chw, patch_size=128)
    assert len(patches) == 4  # 256/128 = 2x2
    assert patches[0].shape == (40, 128, 128)
    assert coords[0] == [(0, 0), (128, 128)]
    assert coords[-1] == [(128, 128), (256, 256)]


def test_get_all_patches_drops_remainder():
    img = np.zeros((40, 300, 300), dtype=np.float32)  # 300 -> only 2 full 128 tiles per axis
    patches, coords = imc_viz.get_all_patches(img, patch_size=128)
    assert len(patches) == 4
    assert all(p.shape == (40, 128, 128) for p in patches)
    assert coords[-1] == [(128, 128), (256, 256)]  # remainder [256:300] dropped


def test_percentile_clip_range(img_chw):
    out = imc_viz._percentile_clip(img_chw[0], 1, 99)
    assert out.min() >= 0.0 and out.max() <= 1.0


def test_percentile_clip_flat_channel():
    out = imc_viz._percentile_clip(np.full((16, 16), 3.0, dtype=np.float32))
    assert np.all(out == 0.0)  # no dynamic range -> all zeros, no NaN


def test_load_npy_applies_arcsinh(tmp_path):
    arr = np.full((40, 128, 128), 10.0, dtype=np.float32)
    p = tmp_path / "patch.npy"
    np.save(p, arr)
    out = imc_viz.load_imc_image(str(p), arcsinh_scale=5.0)
    assert out.shape == (40, 128, 128)
    assert np.allclose(out, np.arcsinh(10.0 / 5.0), atol=1e-5)


def test_load_npy_no_arcsinh(tmp_path):
    arr = np.full((40, 128, 128), 2.0, dtype=np.float32)
    p = tmp_path / "patch.npy"
    np.save(p, arr)
    out = imc_viz.load_imc_image(str(p), arcsinh_scale=None)
    assert np.allclose(out, 2.0)


def test_load_tiff_roundtrip(tmp_path):
    tifffile = pytest.importorskip("tifffile")
    arr = (np.random.default_rng(1).random((42, 120, 110)) * 5).astype(np.float32)
    p = tmp_path / "whole.tiff"
    tifffile.imwrite(p, arr)
    out = imc_viz.load_imc_image(str(p), arcsinh_scale=5.0)
    assert out.shape == (42, 120, 110)


def test_resolve_channels_by_name():
    names = ["DNA1", "CD8", "panCK"]
    assert imc_viz._resolve_channels(names, [0, "CD8", "panCK"]) == [0, 1, 2]


def test_default_composite_by_name():
    names = ["SMA", "DNA1", "CD8a", "Foo", "CD68"]
    picked = imc_viz.default_composite_channels(names, k=3)
    assert picked[0] == 1  # DNA lands first (blue slot)
    assert 2 in picked      # CD8


def test_default_composite_by_index_fallback():
    picked = imc_viz.default_composite_channels(marker_names=None, num_channels=40, k=5)
    assert picked[0] == 0 and picked[-1] == 39 and len(picked) == 5


def test_compose_false_colour_shape_and_range(img_chw):
    rgb = imc_viz.compose_false_colour(img_chw, [0, 1, 2])
    assert rgb.shape == (256, 256, 3)
    assert rgb.min() >= 0.0 and rgb.max() <= 1.0


def test_compose_false_colour_needs_enough_colours(img_chw):
    with pytest.raises(ValueError):
        imc_viz.compose_false_colour(img_chw, [0, 1], colours=[(1, 0, 0)])


def test_butterworth_denoise_smooths(img_chw):
    pytest.importorskip("skimage")
    out = imc_viz.butterworth_denoise(img_chw[:3], cutoff_frequency_ratio=0.2)
    assert out.shape == (3, 256, 256)
    # low-pass filtering a noise field must reduce its high-frequency variance
    for c in range(3):
        assert out[c].std() < img_chw[c].std()


def test_plot_marker_grid_writes_png(img_chw, tmp_path):
    out = tmp_path / "grid.png"
    imc_viz.plot_marker_grid(img_chw[:6], out_path=str(out), ncols=3, title="t")
    assert out.exists() and out.stat().st_size > 0


def test_plot_false_colour_writes_png_and_returns_rgb(img_chw, tmp_path):
    out = tmp_path / "comp.png"
    names = [f"m{i}" for i in range(40)]
    rgb = imc_viz.plot_false_colour(img_chw, [0, "m1", 2], marker_names=names,
                                    out_path=str(out), pixel_size_um=1.0)
    assert out.exists() and out.stat().st_size > 0
    assert rgb.shape == (256, 256, 3)
