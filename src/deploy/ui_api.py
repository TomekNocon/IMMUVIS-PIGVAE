"""Browser API (``/api/*``): password login, then info and predict with a Bearer token.

Runs in a light gateway container with no torch; the model work happens behind the ``info`` and
``predict`` callables (on Modal: calls into the GPU class). Unauthenticated requests therefore
never start the model container.
"""

import os
import time
from collections.abc import Callable
from typing import Annotated

from fastapi import Depends, FastAPI, File, HTTPException, Query, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from src.deploy.auth import LoginLimiter, check_password, issue_token, verify_token
from src.deploy.schemas import (
    ErrorResponse,
    InfoResponse,
    LoginRequest,
    LoginResponse,
    PredictResponse,
)

UI_MAX_UPLOAD_BYTES = 300 * 1024**2  # the gateway holds the upload in memory
TOKEN_TTL_S = 8 * 3600

# (data, ig_steps, preview_markers) -> (HTTP status, JSON body)
PredictFn = Callable[[bytes, int, list[str] | None], tuple[int, dict]]


def _errors(*codes: int) -> dict:
    return {c: {"model": ErrorResponse} for c in codes}


def create_ui_app(
    info: Callable[[], dict],
    predict: PredictFn,
    password: str,
    signing_key: bytes,
    allowed_origins: list[str],
    token_ttl_s: int = TOKEN_TTL_S,
    clock: Callable[[], float] = time.time,
    limiter: LoginLimiter | None = None,
) -> FastAPI:
    """Build the browser API.

    An empty ``password`` rejects every login. An empty ``signing_key`` is replaced by a random
    one (tokens then do not survive a restart).
    """
    key = signing_key or os.urandom(32)
    limiter = limiter or LoginLimiter()
    bearer = HTTPBearer(auto_error=False)

    api = FastAPI(
        title="PIGVAE IMC browser API",
        description="Password login, then raw cords IMC image -> DX.name prediction.",
        version="1",
    )
    api.add_middleware(
        CORSMiddleware,
        allow_origins=allowed_origins,
        allow_methods=["GET", "POST"],
        allow_headers=["Authorization", "Content-Type"],
    )

    def require_token(
        creds: Annotated[HTTPAuthorizationCredentials | None, Depends(bearer)],
    ) -> None:
        if creds is None or not verify_token(key, creds.credentials, clock()):
            raise HTTPException(
                401, "missing or expired session", headers={"WWW-Authenticate": "Bearer"}
            )

    authed = [Depends(require_token)]

    @api.get("/api/healthz")
    def healthz() -> dict:
        """Gateway liveness; does not start the model."""
        return {"status": "ok"}

    @api.post("/api/login", responses=_errors(401, 429))
    def login(body: LoginRequest, request: Request) -> LoginResponse:
        # The first X-Forwarded-For hop can be spoofed; the limiter's global cap covers that.
        forwarded = request.headers.get("x-forwarded-for", "")
        client = forwarded.split(",")[0].strip() or (request.client.host if request.client else "?")
        now = clock()
        if limiter.blocked(client, now):
            raise HTTPException(429, "too many failed attempts, try again in a few minutes")
        if not check_password(body.password, password):
            limiter.record_failure(client, now)
            raise HTTPException(401, "wrong password")
        token, expires_at = issue_token(key, token_ttl_s, now)
        return LoginResponse(token=token, expires_at=expires_at)

    @api.get("/api/info", dependencies=authed, responses=_errors(401))
    def get_info() -> InfoResponse:
        """Model card and expected input; also wakes the model container."""
        return InfoResponse(**info())

    @api.post("/api/predict", dependencies=authed, responses=_errors(401, 413, 422))
    def post_predict(
        file: Annotated[UploadFile, File(description="TIFF or .npy, [43, H, W] raw ion counts")],
        ig_steps: Annotated[
            int, Query(ge=0, le=500, description="Integrated-Gradients steps; 0 = off")
        ] = 50,
        preview_markers: Annotated[
            str | None, Query(description="Three comma-separated marker names for R,G,B")
        ] = None,
    ) -> PredictResponse:
        data = file.file.read(UI_MAX_UPLOAD_BYTES + 1)
        if len(data) > UI_MAX_UPLOAD_BYTES:
            raise HTTPException(413, f"file larger than {UI_MAX_UPLOAD_BYTES // 1024**2} MB")
        markers = [m.strip() for m in preview_markers.split(",")] if preview_markers else None
        status, body = predict(data, ig_steps, markers)
        if status != 200:
            return JSONResponse(status_code=status, content=body)
        return PredictResponse(**body)

    return api
