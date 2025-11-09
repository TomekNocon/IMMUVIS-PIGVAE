from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import modal


APP_NAME = "pigvae-onnx-modal"

# Configure the container image
image = modal.Image.debian_slim().pip_install(
    [
        # Match your repo's versions where possible
        "numpy>=1.26",
        "onnxruntime==1.23.2",
        "fastapi>=0.115.0",
        "pydantic>=2.7",
        "requests>=2.32",
        "wandb>=0.17",
        "pyyaml>=6.0.1",
    ]
)

# Optionally bake local files into the image at build time instead of using Mount
_local_onnx_for_build = os.environ.get("LOCAL_ONNX_PATH")
if _local_onnx_for_build:
    image = image.add_local_file(_local_onnx_for_build, "/weights/model.onnx")

_local_cfg_for_build = os.environ.get("LOCAL_CONFIG_PATH")
if _local_cfg_for_build:
    image = image.add_local_file(_local_cfg_for_build, "/config/app.yaml")

app = modal.App(APP_NAME)


def _default_onnx_remote_path() -> str:
    return "/weights/model.onnx"


def _default_config_remote_path() -> str:
    return "/config/app.yaml"


_mounts: list = []  # Mount is not required when files are baked into the image


@app.cls(
    image=image,
)
@modal.concurrent(max_inputs=4)
class ONNXPredictor:
    session: Any
    output_names: list[str]

    def __enter__(self) -> None:
        # Load optional config file (YAML/JSON) if present
        cfg_path = os.environ.get("CONFIG_PATH", _default_config_remote_path())
        cfg: dict[str, Any] = {}
        if Path(cfg_path).exists():
            text = Path(cfg_path).read_text()
            try:
                import yaml  # type: ignore

                cfg = yaml.safe_load(text) or {}
            except Exception:
                try:
                    import json as _json

                    cfg = _json.loads(text)
                except Exception:
                    cfg = {}

        onnx_path = os.environ.get(
            "ONNX_PATH", cfg.get("onnx_path", _default_onnx_remote_path())
        )
        onnx_url = os.environ.get("ONNX_URL", cfg.get("onnx_url"))
        wandb_artifact = os.environ.get(
            "WANDB_ONNX_ARTIFACT", cfg.get("wandb_onnx_artifact")
        )

        # Ensure the ONNX file exists; if not, try to fetch it
        if not Path(onnx_path).exists():
            Path(onnx_path).parent.mkdir(parents=True, exist_ok=True)
            if onnx_url:
                import requests

                resp = requests.get(onnx_url, timeout=60)
                resp.raise_for_status()
                Path(onnx_path).write_bytes(resp.content)
            elif wandb_artifact:
                import shutil
                import wandb

                run = wandb.init(
                    job_type="deploy", settings=wandb.Settings(start_method="thread")
                )
                art = run.use_artifact(wandb_artifact, type="model")
                dl_dir = Path(art.download())
                candidates = list(dl_dir.rglob("*.onnx"))
                if not candidates:
                    raise FileNotFoundError("No .onnx found in W&B artifact")
                shutil.copyfile(candidates[0], onnx_path)

        # Import onnxruntime inside the container context
        import onnxruntime as ort  # type: ignore

        providers = ["CPUExecutionProvider"]
        self.session = ort.InferenceSession(onnx_path, providers=providers)
        self.output_names = [o.name for o in self.session.get_outputs()]

    @modal.method()
    def infer(self, payload: dict[str, Any]) -> dict[str, Any]:
        # Import numpy at call time so local import isn't required
        import numpy as np  # type: ignore

        node_features = np.asarray(payload["node_features"], dtype=np.float32)
        mask = np.asarray(payload["mask"], dtype=bool)
        outputs = self.session.run(
            self.output_names,
            {"node_features": node_features, "mask": mask},
        )
        # The export script names outputs ["node_logits", "graph_emb"] in that order
        node_logits, graph_emb = outputs[0], outputs[1]
        return {
            "node_logits": node_logits.tolist(),
            "graph_emb": graph_emb.tolist(),
        }

    @modal.method()
    def io_meta(self) -> dict[str, Any]:
        inputs = []
        for i in self.session.get_inputs():
            # onnxruntime dim may be int or None for dynamic
            shape = [int(d) if isinstance(d, int) else None for d in i.shape]
            inputs.append({"name": i.name, "shape": shape, "type": i.type})
        outputs = []
        for o in self.session.get_outputs():
            shape = [int(d) if isinstance(d, int) else None for d in o.shape]
            outputs.append({"name": o.name, "shape": shape, "type": o.type})
        return {"inputs": inputs, "outputs": outputs}


@app.function(image=image)
@modal.asgi_app()
def fastapi_app():
    # Defer heavy imports to runtime
    from fastapi import FastAPI, Query  # type: ignore

    api = FastAPI(title=APP_NAME)
    predictor = ONNXPredictor()

    @api.get("/healthz")
    async def _healthz() -> dict[str, str]:
        return {"status": "ok"}

    @api.get("/meta")
    async def _meta() -> dict[str, Any]:
        return await predictor.io_meta.remote()

    @api.post("/infer")
    async def _infer(body: dict[str, Any], inp: str = Query(None)) -> dict[str, Any]:
        # If inp query param is provided, use it; otherwise use body
        if inp:
            import json

            payload = json.loads(inp)
        else:
            payload = body
        result = await predictor.infer.remote(payload)
        return result

    return api


# Local-only helper: when running `modal run` you can mount your ONNX file into the container.
def local_mount_for_onnx(onnx_local_path: str | os.PathLike[str]) -> modal.Mount:
    onnx_local_path = str(onnx_local_path)
    if not Path(onnx_local_path).exists():
        raise FileNotFoundError(f"ONNX not found: {onnx_local_path}")
    return modal.Mount.from_local_file(
        onnx_local_path, remote_path=_default_onnx_remote_path()
    )
