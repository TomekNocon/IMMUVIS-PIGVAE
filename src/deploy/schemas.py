"""Response and request models of the browser API (``/api/*``).

These are the contract with the UI: ``scripts/deploy/export_openapi.py`` writes them to
``src/deploy/openapi.json``, from which ``ui/`` generates its TypeScript types. Pydantic only (no
torch), so the light gateway container can import them.
"""

from pydantic import BaseModel, Field


class ErrorResponse(BaseModel):
    detail: str = Field(description="Human-readable reason")


class LoginRequest(BaseModel):
    password: str = Field(max_length=256)


class LoginResponse(BaseModel):
    token: str = Field(description="Send as `Authorization: Bearer <token>`")
    expires_at: int = Field(description="Unix seconds")


class ModelInfo(BaseModel):
    label: str = Field(description="Clinical variable the head predicts, e.g. DX.name")
    fold: int = Field(description="Cross-validation fold the head was trained on")
    fold_val_auroc: float = Field(description="Validation AUROC of that fold")
    bundle: str | None = Field(description="Id of the deployed model bundle")


class InfoResponse(ModelInfo):
    classes: list[str] = Field(description="Class names; IG sign refers to classes[1]")
    markers: list[str] = Field(description="Expected channel order of the uploaded image")
    n_markers: int
    tile_size: int = Field(description="Crop side in image pixels")
    max_side: int = Field(description="Largest accepted image side in pixels")
    device: str = Field(description="cpu or cuda")


class ImageInfo(BaseModel):
    height: int
    width: int
    tile_size: int
    n_crops: int


class Crop(BaseModel):
    row: int = Field(description="Top-left pixel row in the original image")
    col: int = Field(description="Top-left pixel column in the original image")
    attention: float = Field(description="ABMIL attention weight; sums to 1 over crops")
    ig_importance: float | None = Field(
        description="Signed Integrated-Gradients contribution; > 0 pushes toward classes[1]"
    )


class Preview(BaseModel):
    png_base64: str = Field(description="RGB PNG, base64")
    width: int
    height: int
    scale: float = Field(description="Preview pixels per image pixel")
    markers: list[str] = Field(description="Markers shown as red, green, blue")


class PredictResponse(BaseModel):
    label: str
    prediction: str
    probabilities: dict[str, float]
    logit: float | list[float]
    image: ImageInfo
    crops: list[Crop]
    preview: Preview
    model: ModelInfo
    timing_ms: dict[str, float]
