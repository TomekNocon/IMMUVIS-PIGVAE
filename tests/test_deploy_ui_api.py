import io
import sys

import numpy as np
import pytest
import tifffile
from fastapi.testclient import TestClient

from src.deploy.api import predict_envelope
from src.deploy.auth import LoginLimiter
from src.deploy.ui_api import create_ui_app
from tests.test_deploy_pipeline import make_pipeline, raw_image

ORIGIN = "http://localhost:5173"


class Clock:
    def __init__(self):
        self.t = 1_000_000.0

    def __call__(self):
        return self.t


def _tiff(img: np.ndarray) -> bytes:
    buf = io.BytesIO()
    tifffile.imwrite(buf, img)
    return buf.getvalue()


def _app(password="pw", clock=None, limiter=None):  # noqa: S107 - test password
    pipe = make_pipeline()
    return TestClient(
        create_ui_app(
            info=pipe.info,
            predict=lambda d, s, m: predict_envelope(pipe, d, s, m),
            password=password,
            signing_key=b"k" * 32,
            allowed_origins=[ORIGIN],
            token_ttl_s=60,
            clock=clock or Clock(),
            limiter=limiter,
        )
    )


@pytest.fixture(scope="module")
def client():
    return _app()


@pytest.fixture(scope="module")
def auth(client):
    r = client.post("/api/login", json={"password": "pw"})
    assert r.status_code == 200, r.text
    return {"Authorization": f"Bearer {r.json()['token']}"}


def test_ui_api_imports_no_torch():
    # The gateway image has no torch; ui_api must stay importable there.
    import subprocess

    code = "import sys, src.deploy.ui_api; print('torch' in sys.modules)"
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert out.stdout.strip() == "False"


def test_routes_need_token(client):
    assert client.get("/api/healthz").status_code == 200
    assert client.get("/api/info").status_code == 401
    assert client.get("/api/info", headers={"Authorization": "Bearer junk"}).status_code == 401
    r = client.post("/api/predict", files={"file": ("x.tiff", b"")})
    assert r.status_code == 401


def test_info_hides_manifest(client, auth):
    body = client.get("/api/info", headers=auth).json()
    assert body["n_markers"] == 43 and body["classes"] == ["class0", "class1"]
    assert "manifest" not in body


def test_predict_with_preview(client, auth):
    r = client.post(
        "/api/predict",
        files={"file": ("img.tiff", _tiff(raw_image()))},
        params={"ig_steps": 2},
        headers=auth,
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["preview"]["width"] == 270 and len(body["crops"]) == 4
    assert "z_global" not in body["crops"][0]


def test_predict_input_errors_are_422(client, auth):
    bad = _tiff(np.zeros((3, 200, 200), np.uint16))
    r = client.post("/api/predict", files={"file": ("x.tiff", bad)}, headers=auth)
    assert r.status_code == 422 and "43 channels" in r.json()["detail"]
    r = client.post(
        "/api/predict",
        files={"file": ("x.tiff", _tiff(raw_image()))},
        params={"preview_markers": "panCK,CD45,nope", "ig_steps": 0},
        headers=auth,
    )
    assert r.status_code == 422 and "unknown markers" in r.json()["detail"]


def test_token_expires():
    clock = Clock()
    c = _app(clock=clock)
    token = c.post("/api/login", json={"password": "pw"}).json()["token"]
    h = {"Authorization": f"Bearer {token}"}
    assert c.get("/api/info", headers=h).status_code == 200
    clock.t += 61
    assert c.get("/api/info", headers=h).status_code == 401


def test_wrong_password_then_rate_limited():
    c = _app(limiter=LoginLimiter(max_failures=2, window_s=600, max_global_failures=100))
    assert c.post("/api/login", json={"password": "x"}).status_code == 401
    assert c.post("/api/login", json={"password": "x"}).status_code == 401
    assert c.post("/api/login", json={"password": "pw"}).status_code == 429


def test_unset_password_rejects_everything():
    c = _app(password="")
    assert c.post("/api/login", json={"password": ""}).status_code == 401


def test_cors_only_for_allowed_origin(client):
    pre = {
        "Access-Control-Request-Method": "POST",
        "Access-Control-Request-Headers": "authorization",
    }
    ok = client.options("/api/predict", headers={"Origin": ORIGIN, **pre})
    assert ok.headers.get("access-control-allow-origin") == ORIGIN
    bad = client.options("/api/predict", headers={"Origin": "https://evil.example", **pre})
    assert "access-control-allow-origin" not in bad.headers
