import base64
import io

import numpy as np
import pytest
from PIL import Image

from src.deploy.preview import UnknownMarkerError, render_preview, resolve_markers
from tests.test_deploy_pipeline import MARKERS, make_pipeline, raw_image


def _decode(preview):
    return Image.open(io.BytesIO(base64.b64decode(preview["png_base64"])))


def test_resolve_markers_errors():
    assert resolve_markers(["panCK", "CD45", "DNA1"], MARKERS) == [34, 13, 19]
    with pytest.raises(UnknownMarkerError, match="unknown markers"):
        resolve_markers(["panCK", "CD45", "nope"], MARKERS)
    with pytest.raises(UnknownMarkerError, match="exactly 3"):
        resolve_markers(["panCK"], MARKERS)


def test_render_small_image_keeps_size():
    pre = np.random.default_rng(0).random((43, 300, 270), dtype=np.float32)
    p = render_preview(pre, ["panCK", "CD45", "DNA1"], MARKERS)
    img = _decode(p)
    assert (p["width"], p["height"], p["scale"]) == (270, 300, 1.0)
    assert img.size == (270, 300) and img.mode == "RGB"


def test_render_large_non_square_image_downscales_consistently():
    pre = np.zeros((43, 2000, 1000), np.float32)  # empty channels must not divide by zero
    p = render_preview(pre, ["panCK", "CD45", "DNA1"], MARKERS, max_side=1024)
    assert p["height"] == 1024 and p["width"] == 512
    assert p["scale"] == pytest.approx(1024 / 2000)
    assert _decode(p).size == (512, 1024)


def test_predict_with_preview():
    out = make_pipeline().predict(raw_image(), ig_steps=2, include_preview=True)
    assert out["preview"]["width"] == 270 and out["preview"]["markers"] == ["panCK", "CD45", "DNA1"]
    assert "preview" in out["timing_ms"]
    assert "preview" not in make_pipeline().predict(raw_image(), ig_steps=0)


def test_predict_bad_preview_marker_fails_before_inference():
    with pytest.raises(UnknownMarkerError):
        make_pipeline().predict(raw_image(), include_preview=True, preview_markers=["a", "b", "c"])
