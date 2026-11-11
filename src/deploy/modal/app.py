"""Modal deployment of the IMC API (``src/deploy/api.py`` over ``src/deploy/pipeline.py``).

Two web endpoints:

- ``IMCServer.web``: the script API (``/healthz``, ``/info``, ``/predict``, ``/embed``). Callers
  need a Modal proxy-auth token (``Modal-Key`` / ``Modal-Secret`` headers); Modal rejects other
  requests before a container starts.
- ``api_gateway``: the browser API (``/api/*``, ``src/deploy/ui_api.py``) in a small gateway container
  without torch. It checks the UI password and session tokens itself and calls ``IMCServer`` only
  for logged-in requests, so anonymous traffic never starts the model container. Its password,
  signing key and allowed browser origins come from the Modal Secret ``pigvae-imc-ui``
  (``UI_PASSWORD``, ``UI_SIGNING_KEY``, ``UI_ORIGINS`` comma-separated).

The model container loads ImmuVis, the PCA, PIGVAE and the ABMIL head once from the bundle
Volume. At most one container of each runs, and they stop after 2 idle minutes.

One-time setup (from the repo root)::

    uv run modal setup                                              # log in
    uv run modal volume create pigvae-imc-bundle
    uv run modal volume put pigvae-imc-bundle <out>/bundle_DX.name /  # scripts/deploy/export_bundle.py
    uv run modal secret create pigvae-imc-ui UI_PASSWORD=... UI_SIGNING_KEY=... UI_ORIGINS=...
    uv run modal deploy src/deploy/modal/app.py

Then create a proxy-auth token in the Modal dashboard (Settings -> Proxy Auth Tokens) and call the
API with ``scripts/deploy/query_modal.py``.
"""

import os
from pathlib import Path

import modal

APP_NAME = "pigvae-imc"
VOLUME_NAME = "pigvae-imc-bundle"
BUNDLE_DIR = "/bundle"  # Volume mount
BUNDLE = f"{BUNDLE_DIR}/bundle_DX.name"  # the uploaded bundle directory inside the Volume
# GPU type; `PIGVAE_GPU=none modal deploy ...` serves on CPU (slower, no payment method needed).
_gpu = os.environ.get("PIGVAE_GPU", "L4")
GPU = None if _gpu.lower() == "none" else _gpu
UI_SECRET = "pigvae-imc-ui"  # noqa: S105 - the Secret's name, not its value
IMMUVIS_CODE = "/root/immuvis"  # holds the multiplex_model package from the submodule

# Local paths only matter where the image is built; inside the container app.py is /root/app.py.
REPO = Path(__file__).resolve().parents[3] if modal.is_local() else Path("/")

# Versions follow uv.lock. The default PyPI torch wheel ships CUDA 12, which runs on the L4.
image = (
    modal.Image
    .debian_slim(python_version="3.12")
    .uv_pip_install(
        "torch==2.7.1",
        "torchvision==0.22.1",
        "numpy==2.2.6",
        "scikit-image==0.25.2",
        "scikit-learn==1.7.1",
        "networkx==3.4.2",
        "h5py==3.16.0",
        "joblib==1.5.1",
        "einops==0.8.1",
        "omegaconf==2.3.0",
        "pyyaml==6.0.2",
        "captum==0.9.0",
        "tifffile==2025.5.10",
        "fastapi==0.162.4",
        "python-multipart==0.0.33",
        "pillow==11.3.0",
    )
    .add_local_dir(
        REPO / "external" / "multiplex-image-model" / "multiplex_model",
        f"{IMMUVIS_CODE}/multiplex_model",
        ignore=["**/__pycache__"],
    )
    .add_local_python_source("src")
)

# The browser gateway: web framework only, starts in a second or two.
gateway_image = (
    modal.Image
    .debian_slim(python_version="3.12")
    .uv_pip_install("fastapi==0.162.4", "python-multipart==0.0.33", "pydantic==2.11.7")
    .add_local_python_source("src")
)

app = modal.App(APP_NAME)
bundle = modal.Volume.from_name(VOLUME_NAME)


@app.cls(
    image=image,
    gpu=GPU,
    cpu=4.0,
    memory=8192,
    volumes={BUNDLE_DIR: bundle},
    max_containers=1,  # caps the cost even if the token leaks
    scaledown_window=120,  # seconds idle before the container stops (idle time is billed)
    timeout=600,
)
class IMCServer:
    @modal.enter()
    def load(self) -> None:
        import torch

        from src.deploy.pipeline import IMCPipeline

        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.pipeline = IMCPipeline.from_bundle(BUNDLE, device=device, code_root=IMMUVIS_CODE)

    @modal.asgi_app(requires_proxy_auth=True)
    def web(self):
        from src.deploy.api import create_app

        return create_app(lambda: self.pipeline)

    @modal.method()
    def info(self) -> dict:
        return self.pipeline.info()

    @modal.method()
    def predict_bytes(
        self, data: bytes, ig_steps: int, preview_markers: list[str] | None
    ) -> tuple[int, dict]:
        from src.deploy.api import predict_envelope

        return predict_envelope(self.pipeline, data, ig_steps, preview_markers)


@app.function(
    image=gateway_image,
    cpu=0.25,
    memory=1024,  # holds one upload (capped at 300 MB) in memory
    secrets=[modal.Secret.from_name(UI_SECRET)],
    max_containers=1,  # the login rate limiter lives in this one container's memory
    scaledown_window=120,
    timeout=600,
)
@modal.concurrent(max_inputs=20)
@modal.asgi_app()
def api_gateway():
    from src.deploy.ui_api import create_ui_app

    server = IMCServer()
    origins = [o.strip() for o in os.environ.get("UI_ORIGINS", "").split(",") if o.strip()]
    return create_ui_app(
        info=lambda: server.info.remote(),
        predict=lambda data, steps, markers: tuple(
            server.predict_bytes.remote(data, steps, markers)
        ),
        password=os.environ.get("UI_PASSWORD", ""),
        signing_key=os.environ.get("UI_SIGNING_KEY", "").encode(),
        allowed_origins=origins,
    )
