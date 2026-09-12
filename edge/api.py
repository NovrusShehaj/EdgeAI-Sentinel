"""
FastAPI server wrapping the ONNX inference engine.

Endpoints:
  GET  /live     — process liveness
  GET  /ready    — model readiness (503 unless the engine is loaded)
  GET  /health   — human diagnostic payload
  GET  /info     — model configuration
  POST /infer    — authenticated inference
  GET  /metrics  — Prometheus scrape (token or monitoring CIDR)
"""

from __future__ import annotations

import ipaddress
import logging
import os
import secrets
import time
from contextlib import asynccontextmanager
from typing import Optional

import cv2
import numpy as np
import psutil
from fastapi import FastAPI, File, Header, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, PlainTextResponse, Response
from pydantic import BaseModel, Field
from starlette.middleware.trustedhost import TrustedHostMiddleware

from edge.config import resolve_class_names
from edge.inference import InferenceResult, InferenceTimeout, ONNXInferenceEngine

logger = logging.getLogger(__name__)

ALLOWED_CONTENT_TYPES = {"image/jpeg", "image/png", "image/bmp"}
PRIVATE_CIDRS = (
    "127.0.0.1/32",
    "::1/128",
    "10.0.0.0/8",
    "172.16.0.0/12",
    "192.168.0.0/16",
)

_engine: Optional[ONNXInferenceEngine] = None
_startup_time: float = 0.0
_load_error: Optional[str] = None


def _env_int(name: str, default: int) -> int:
    return int(os.getenv(name, str(default)))


def _env_float(name: str, default: float) -> float:
    return float(os.getenv(name, str(default)))


def max_upload_bytes() -> int:
    return _env_int("MAX_UPLOAD_BYTES", 5 * 1024 * 1024)


def max_image_side() -> int:
    return _env_int("MAX_IMAGE_SIDE", 4096)


def api_token() -> str:
    return os.getenv("API_TOKEN", "")


def metrics_token() -> str:
    return os.getenv("METRICS_TOKEN") or api_token()


def infer_timeout_s() -> float:
    return _env_float("INFER_TIMEOUT_S", 5.0)


def allowed_hosts() -> list[str]:
    raw = os.getenv("ALLOWED_HOSTS", "*")
    return [part.strip() for part in raw.split(",") if part.strip()] or ["*"]


def cors_origins() -> list[str]:
    raw = os.getenv("CORS_ALLOW_ORIGINS", "")
    return [part.strip() for part in raw.split(",") if part.strip()]


def metrics_cidrs() -> list[str]:
    raw = os.getenv("METRICS_ALLOWED_CIDRS", ",".join(PRIVATE_CIDRS))
    return [part.strip() for part in raw.split(",") if part.strip()]


def _tokens_match(provided: Optional[str], expected: str) -> bool:
    if not provided or not expected:
        return False
    provided_bytes = provided.encode("utf-8")
    expected_bytes = expected.encode("utf-8")
    if len(provided_bytes) != len(expected_bytes):
        secrets.compare_digest(expected_bytes, expected_bytes)
        return False
    return secrets.compare_digest(provided_bytes, expected_bytes)


def _extract_bearer(authorization: Optional[str]) -> Optional[str]:
    if not authorization:
        return None
    scheme, _, value = authorization.partition(" ")
    if scheme.lower() == "bearer" and value:
        return value.strip()
    return None


def _client_ip(request: Request) -> str:
    if request.client and request.client.host:
        return request.client.host
    return ""


def _ip_allowed(ip: str, cidrs: list[str]) -> bool:
    try:
        address = ipaddress.ip_address(ip)
    except ValueError:
        return False
    for cidr in cidrs:
        try:
            if address in ipaddress.ip_network(cidr, strict=False):
                return True
        except ValueError:
            continue
    return False


def require_infer_auth(
    x_api_token: Optional[str] = None,
    authorization: Optional[str] = None,
) -> None:
    expected = api_token()
    provided = x_api_token or _extract_bearer(authorization)
    if not expected:
        raise HTTPException(status_code=401, detail="API_TOKEN is not configured")
    if not _tokens_match(provided, expected):
        raise HTTPException(status_code=401, detail="Invalid or missing API token")


def allow_metrics_access(
    request: Request,
    x_api_token: Optional[str] = None,
    authorization: Optional[str] = None,
) -> None:
    expected = metrics_token()
    provided = x_api_token or _extract_bearer(authorization)
    if expected and _tokens_match(provided, expected):
        return
    if _ip_allowed(_client_ip(request), metrics_cidrs()):
        return
    raise HTTPException(status_code=401, detail="Metrics access denied")


class DetectionOut(BaseModel):
    x1: float
    y1: float
    x2: float
    y2: float
    confidence: float = Field(..., ge=0.0, le=1.0)
    class_id: int
    class_name: str


class InferenceResponse(BaseModel):
    frame_id: int
    inference_time_ms: float
    detections: list[DetectionOut]
    num_detections: int
    image_width: int
    image_height: int


class HealthResponse(BaseModel):
    status: str
    uptime_seconds: float
    model_loaded: bool
    ready: bool
    device: str
    cpu_pct: float
    ram_used_mb: float
    load_error: Optional[str] = None


class ModelInfoResponse(BaseModel):
    model_path: str
    class_names: list[str]
    conf_threshold: float
    iou_threshold: float
    num_threads: int
    input_size: int
    timeout_s: float


def _device_name() -> str:
    try:
        with open("/proc/cpuinfo") as handle:
            info = handle.read()
        if "Raspberry Pi" in info or "BCM" in info:
            return "raspberry_pi"
        return "x86_64"
    except OSError:
        return "unknown"


def _load_engine() -> Optional[ONNXInferenceEngine]:
    global _load_error
    model_path = os.getenv("MODEL_PATH", "models/best.onnx")
    class_names = resolve_class_names()
    try:
        engine = ONNXInferenceEngine(
            model_path=model_path,
            class_names=class_names,
            conf_threshold=_env_float("CONF_THRESHOLD", 0.45),
            iou_threshold=_env_float("IOU_THRESHOLD", 0.45),
            num_threads=_env_int("NUM_THREADS", 4),
            timeout_s=infer_timeout_s(),
            manifest_path=os.getenv("MODEL_MANIFEST_PATH") or None,
        )
        _load_error = None
        logger.info("Model loaded successfully")
        return engine
    except FileNotFoundError:
        _load_error = f"Model file not found: {model_path}"
        logger.error(_load_error)
        logger.error("Mount a versioned ONNX model and set MODEL_PATH")
        return None
    except ValueError as exc:
        _load_error = str(exc)
        logger.error("Invalid model or class configuration: %s", exc)
        raise


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _engine, _startup_time
    _startup_time = time.time()
    logger.info("Starting EdgeAI Sentinel API server...")
    _engine = _load_engine()
    yield
    logger.info("Shutting down EdgeAI Sentinel API server")
    if _engine is not None:
        _engine.close()
        _engine = None


app = FastAPI(
    title="EdgeAI Sentinel",
    description="Low-SWaP edge inference API for object detection",
    version="0.2.0",
    lifespan=lifespan,
)

_hosts = allowed_hosts()
app.add_middleware(TrustedHostMiddleware, allowed_hosts=_hosts)

_origins = cors_origins()
if _origins:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=_origins,
        allow_methods=["GET", "POST"],
        allow_headers=["X-API-Token", "Authorization", "Content-Type"],
        allow_credentials=False,
    )


def _engine_ready() -> bool:
    return _engine is not None and _engine.is_ready()


@app.get("/live", tags=["System"])
async def live():
    """Process liveness probe. Returns 200 while the server is running."""
    return {"status": "alive", "uptime_seconds": round(time.time() - _startup_time, 1)}


@app.get("/ready", tags=["System"])
async def ready():
    """Traffic readiness probe. 503 unless a verified model is loaded."""
    payload = {
        "ready": _engine_ready(),
        "model_loaded": _engine is not None,
        "model_path": os.getenv("MODEL_PATH", "models/best.onnx"),
        "class_names": resolve_class_names() if _engine is None else _engine.class_names,
        "reason": None if _engine_ready() else (_load_error or "model not loaded"),
    }
    status = 200 if payload["ready"] else 503
    return JSONResponse(status_code=status, content=payload)


@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health():
    """Human diagnostic endpoint. Not a substitute for /live or /ready."""
    memory = psutil.virtual_memory()
    return HealthResponse(
        status="ok" if _engine_ready() else "degraded",
        uptime_seconds=round(time.time() - _startup_time, 1),
        model_loaded=_engine is not None,
        ready=_engine_ready(),
        device=_device_name(),
        cpu_pct=round(psutil.cpu_percent(interval=None), 1),
        ram_used_mb=round(memory.used / 1e6, 1),
        load_error=_load_error,
    )


@app.get("/info", response_model=ModelInfoResponse, tags=["System"])
async def model_info():
    if not _engine_ready():
        raise HTTPException(status_code=503, detail="Model not loaded")
    return ModelInfoResponse(
        model_path=str(_engine.model_path),
        class_names=_engine.class_names,
        conf_threshold=_engine.conf_threshold,
        iou_threshold=_engine.iou_threshold,
        num_threads=_env_int("NUM_THREADS", 4),
        input_size=_engine.input_size,
        timeout_s=_engine.timeout_s,
    )


@app.post("/infer", response_model=InferenceResponse, tags=["Inference"])
async def run_inference(
    request: Request,
    file: UploadFile = File(...),
    x_api_token: Optional[str] = Header(default=None, alias="X-API-Token"),
    authorization: Optional[str] = Header(default=None),
):
    require_infer_auth(x_api_token, authorization)
    if not _engine_ready():
        raise HTTPException(status_code=503, detail="Model not loaded")

    content_length = request.headers.get("content-length")
    limit = max_upload_bytes()
    if content_length:
        try:
            if int(content_length) > limit:
                raise HTTPException(status_code=413, detail="Upload exceeds size limit")
        except ValueError:
            raise HTTPException(status_code=400, detail="Invalid Content-Length") from None

    if file.content_type not in ALLOWED_CONTENT_TYPES:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported image type: {file.content_type}. Use JPEG, PNG, or BMP.",
        )

    contents = await file.read(limit + 1)
    if len(contents) > limit:
        raise HTTPException(status_code=413, detail="Upload exceeds size limit")

    image = cv2.imdecode(np.frombuffer(contents, np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise HTTPException(status_code=400, detail="Could not decode image")

    height, width = image.shape[:2]
    side_limit = max_image_side()
    if height > side_limit or width > side_limit:
        raise HTTPException(
            status_code=400,
            detail=f"Image dimensions {width}x{height} exceed {side_limit}px limit",
        )

    try:
        result: InferenceResult = _engine.infer(image)
    except InferenceTimeout as exc:
        logger.error("Inference timed out: %s", exc)
        raise HTTPException(status_code=504, detail="Inference timed out") from exc
    except Exception as exc:
        logger.error("Inference failed: %s", exc)
        raise HTTPException(status_code=500, detail="Inference error") from exc

    return InferenceResponse(
        frame_id=result.frame_id,
        inference_time_ms=result.inference_time_ms,
        detections=[
            DetectionOut(
                x1=det.x1,
                y1=det.y1,
                x2=det.x2,
                y2=det.y2,
                confidence=det.confidence,
                class_id=det.class_id,
                class_name=det.class_name,
            )
            for det in result.detections
        ],
        num_detections=len(result.detections),
        image_width=width,
        image_height=height,
    )


@app.get("/metrics", tags=["Observability"])
async def metrics(
    request: Request,
    x_api_token: Optional[str] = Header(default=None, alias="X-API-Token"),
    authorization: Optional[str] = Header(default=None),
):
    allow_metrics_access(request, x_api_token, authorization)
    try:
        from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

        return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)
    except ImportError:
        return PlainTextResponse("# prometheus_client not installed\n", status_code=200)
