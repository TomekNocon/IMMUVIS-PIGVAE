"""HTTP API over :class:`~src.deploy.pipeline.IMCPipeline` (FastAPI, no Modal dependency).

Endpoints: ``GET /healthz``, ``GET /info``, ``POST /predict``, ``POST /embed``. Images are sent as
a multipart ``file``: a TIFF (or ``.npy``) of raw ion counts, ``[43, H, W]`` in cords marker order
(see ``/info``; channels-last is accepted). Invalid images give HTTP 422.
"""

import io
from collections.abc import Callable
from typing import Annotated

import numpy as np
from fastapi import FastAPI, File, HTTPException, Query, UploadFile

from src.deploy.pipeline import IMCPipeline, InvalidImageError
from src.deploy.preview import UnknownMarkerError

MAX_UPLOAD_BYTES = 1 << 30  # 1 GiB


def read_image(data: bytes) -> np.ndarray:
    """Decode an uploaded TIFF or ``.npy`` file into an array."""
    if data[:6] == b"\x93NUMPY":
        return np.load(io.BytesIO(data), allow_pickle=False)
    import tifffile

    try:
        return tifffile.imread(io.BytesIO(data))
    except Exception as exc:  # tifffile raises several types for non-TIFF input
        raise InvalidImageError(f"could not read the file as a TIFF or .npy: {exc}") from exc


def load_upload(file: UploadFile) -> np.ndarray:
    """Read an uploaded image; 413 when too large, 422 when not a TIFF/.npy."""
    data = file.file.read(MAX_UPLOAD_BYTES + 1)
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(413, f"file larger than {MAX_UPLOAD_BYTES} bytes")
    try:
        return read_image(data)
    except InvalidImageError as exc:
        raise HTTPException(422, str(exc)) from exc


def predict_envelope(
    pipeline: IMCPipeline, data: bytes, ig_steps: int, preview_markers: list[str] | None
) -> tuple[int, dict]:
    """Run ``/api/predict`` for the browser gateway: ``(status, body)``, never raising on bad input.

    Exceptions do not cross the Modal call (the gateway cannot import torch), so input errors
    come back as ``(422, {"detail": ...})``.
    """
    kwargs = {"preview_markers": preview_markers} if preview_markers else {}
    try:
        return 200, pipeline.predict(
            read_image(data), ig_steps=ig_steps, include_preview=True, **kwargs
        )
    except (InvalidImageError, UnknownMarkerError) as exc:
        return 422, {"detail": str(exc)}


def create_app(get_pipeline: Callable[[], IMCPipeline]) -> FastAPI:
    """Build the API; ``get_pipeline`` returns the loaded pipeline (loaded once by the caller)."""
    api = FastAPI(
        title="PIGVAE IMC API",
        description="Raw cords IMC image -> DX.name prediction with per-crop explanations.",
    )

    @api.get("/healthz")
    def healthz() -> dict:
        return {"status": "ok"}

    @api.get("/info")
    def info() -> dict:
        return get_pipeline().info()

    # Plain `def` handlers run in FastAPI's thread pool, so inference does not block the loop.
    @api.post("/predict")
    def predict(
        file: Annotated[UploadFile, File()],
        ig_steps: Annotated[
            int, Query(ge=0, le=500, description="Integrated-Gradients steps; 0 = off")
        ] = 50,
        include_embeddings: Annotated[bool, Query(description="add each crop's z_global")] = False,
    ) -> dict:
        img = load_upload(file)
        try:
            return get_pipeline().predict(
                img, ig_steps=ig_steps, include_embeddings=include_embeddings
            )
        except InvalidImageError as exc:
            raise HTTPException(422, str(exc)) from exc

    @api.post("/embed")
    def embed(file: Annotated[UploadFile, File()]) -> dict:
        img = load_upload(file)
        try:
            return get_pipeline().embed_json(img)
        except InvalidImageError as exc:
            raise HTTPException(422, str(exc)) from exc

    return api
