from typing import Any

import modal

try:
    from pydantic import BaseModel
except Exception:

    class BaseModel:  # type: ignore
        pass


# Name your Modal application
app = modal.App("immuvis-onnx-service")

# Absolute path to your exported ONNX model
LOCAL_MODEL_PATH = "/Users/tomasznocon/Documents/MIM/Repositories/Master thesis/immuvis/logs/export_onnx/runs/2025-10-25_22-10-24/model.onnx"
REMOTE_MODEL_PATH = "/model/model.onnx"

# Build a lightweight image with required runtime deps and bake the model file inside
image = (
    modal.Image.debian_slim()
    .pip_install(
        "onnxruntime==1.23.2",
        "numpy>=1.26.0",
        "fastapi",
        "pydantic>=2.0.0",
    )
    .add_local_file(LOCAL_MODEL_PATH, REMOTE_MODEL_PATH)
)


SESSION = None
OUTPUT_NAMES = None
INPUT_META = None  # name -> {"shape": list, "type": str}


def _get_session():
    global SESSION, OUTPUT_NAMES, INPUT_META
    if SESSION is None:
        import onnxruntime as ort

        SESSION = ort.InferenceSession(REMOTE_MODEL_PATH, providers=["CPUExecutionProvider"])
        OUTPUT_NAMES = [o.name for o in SESSION.get_outputs()]
        INPUT_META = {
            i.name: {
                "shape": list(i.shape) if hasattr(i, "shape") else None,
                "type": getattr(i, "type", None),
            }
            for i in SESSION.get_inputs()
        }
    return SESSION, OUTPUT_NAMES


def _infer_impl(payload: dict[str, Any]) -> dict[str, Any]:
    import numpy as np

    if "node_features" not in payload or "mask" not in payload:
        raise ValueError("Request must include 'node_features' and 'mask'.")

    session, output_names = _get_session()

    node_features = np.asarray(payload["node_features"], dtype=np.float32)
    mask_raw = np.asarray(payload["mask"])

    # Validate shapes vs ONNX input metadata (dims 1.. should match; dim 0 is batch)
    meta = INPUT_META or {}
    nf_meta = meta.get("node_features")
    m_meta = meta.get("mask")
    if nf_meta and nf_meta["shape"]:
        exp = nf_meta["shape"]
        if len(exp) != 3:
            raise ValueError(f"node_features rank must be 3, model expects {exp}")
        if node_features.ndim != 3:
            raise ValueError(f"node_features rank must be 3, got {node_features.shape}")
        if exp[1] not in (None, "None") and node_features.shape[1] != exp[1]:
            raise ValueError(
                f"node_features dim1 (N) must be {exp[1]}, got {node_features.shape[1]}"
            )
        if exp[2] not in (None, "None") and node_features.shape[2] != exp[2]:
            raise ValueError(
                f"node_features dim2 (Din) must be {exp[2]}, got {node_features.shape[2]}"
            )
    if m_meta and m_meta["shape"]:
        exp = m_meta["shape"]
        if len(exp) != 2:
            raise ValueError(f"mask rank must be 2, model expects {exp}")
        if mask_raw.ndim != 2:
            raise ValueError(f"mask rank must be 2, got {mask_raw.shape}")
        if exp[1] not in (None, "None") and mask_raw.shape[1] != exp[1]:
            raise ValueError(f"mask dim1 (N) must be {exp[1]}, got {mask_raw.shape[1]}")

    # Cast mask to ONNX-expected dtype
    mask_type = (m_meta or {}).get("type", "") or ""
    if "bool" in mask_type:
        mask = mask_raw.astype(np.bool_)
    elif "int64" in mask_type:
        mask = mask_raw.astype(np.int64)
    else:
        mask = mask_raw.astype(np.bool_)

    outputs = session.run(
        output_names,
        {
            "node_features": node_features,
            "mask": mask,
        },
    )

    # Export names are ["node_logits", "graph_emb"] per exporter
    node_logits, graph_emb = outputs
    return {
        "node_logits": node_logits.tolist(),
        "graph_emb": graph_emb.tolist(),
    }


class InferenceRequest(BaseModel):
    node_features: list[list[list[float]]]
    mask: list[list[bool]]


class InferenceResponse(BaseModel):
    node_logits: list
    graph_emb: list


@app.function(image=image, timeout=600)
def infer(node_features: list[list[list[float]]], mask: list[list[bool]]) -> dict[str, Any]:
    return _infer_impl({"node_features": node_features, "mask": mask})


@app.function(image=image, timeout=600)
@modal.fastapi_endpoint(method="POST")
def http(request: InferenceRequest) -> InferenceResponse:
    result = _infer_impl({"node_features": request.node_features, "mask": request.mask})
    return InferenceResponse(**result)


# Optional: direct callable for quick testing via:
# modal run src/deploy/modal/app.py::call_infer --payload-json "$(cat src/deploy/modal/payload.json)"
@app.function(image=image, timeout=600)
def call_infer(payload_json: str) -> dict[str, Any]:
    import json

    payload = json.loads(payload_json)
    return _infer_impl(payload)
