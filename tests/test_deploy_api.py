import io

import numpy as np
import pytest
import tifffile
from fastapi.testclient import TestClient

from src.deploy.api import create_app
from tests.test_deploy_pipeline import make_pipeline, raw_image


@pytest.fixture(scope="module")
def client():
    pipe = make_pipeline()
    return TestClient(create_app(lambda: pipe))


def _tiff(img: np.ndarray) -> bytes:
    buf = io.BytesIO()
    tifffile.imwrite(buf, img)
    return buf.getvalue()


def _npy(img: np.ndarray) -> bytes:
    buf = io.BytesIO()
    np.save(buf, img)
    return buf.getvalue()


def test_healthz_and_info(client):
    assert client.get("/healthz").json() == {"status": "ok"}
    assert client.get("/info").json()["n_markers"] == 43


@pytest.mark.parametrize("encode", [_tiff, _npy])
def test_predict_tiff_and_npy(client, encode):
    r = client.post(
        "/predict", files={"file": ("img", encode(raw_image()))}, params={"ig_steps": 4}
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["image"]["n_crops"] == 4 and body["crops"][0]["ig_importance"] is not None


def test_embed(client):
    r = client.post("/embed", files={"file": ("img.tiff", _tiff(raw_image()))})
    assert r.status_code == 200
    assert len(r.json()["crops"][0]["z_global"]) == 16


def test_wrong_channels_is_422(client):
    r = client.post(
        "/predict", files={"file": ("x.tiff", _tiff(np.zeros((3, 200, 200), np.uint16)))}
    )
    assert r.status_code == 422 and "43 channels" in r.json()["detail"]


def test_not_an_image_is_422(client):
    r = client.post("/predict", files={"file": ("x.txt", b"hello")})
    assert r.status_code == 422


def test_bad_ig_steps_is_422(client):
    r = client.post(
        "/predict", files={"file": ("x.tiff", _tiff(raw_image()))}, params={"ig_steps": -1}
    )
    assert r.status_code == 422
