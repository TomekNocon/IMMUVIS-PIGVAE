"""Modal deployment of the IMC API (``src/deploy/api.py`` over ``src/deploy/pipeline.py``).

One GPU container loads ImmuVis, the PCA, PIGVAE and the ABMIL head once from the bundle Volume
and serves ``/healthz``, ``/info``, ``/predict`` and ``/embed``. Callers need a Modal proxy-auth
token (``Modal-Key`` / ``Modal-Secret`` headers); unauthenticated requests are rejected by Modal
before a container starts. At most one container runs, and it stops after 5 idle minutes.

One-time setup (from the repo root)::

    uv run modal setup                                              # log in
    uv run modal volume create pigvae-imc-bundle
    uv run modal volume put pigvae-imc-bundle <out>/bundle_DX.name /  # scripts/deploy/export_bundle.py
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
    )
    .add_local_dir(
        REPO / "external" / "multiplex-image-model" / "multiplex_model",
        f"{IMMUVIS_CODE}/multiplex_model",
        ignore=["**/__pycache__"],
    )
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
    scaledown_window=300,  # seconds idle before the container stops
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
