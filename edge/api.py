"""
edge/api.py
───────────
FastAPI REST server wrapping the ONNX inference engine.

Exposes:
  POST /infer         — run inference on an uploaded image
  GET  /health        — liveness probe (for Kubernetes / Ansible health checks)
  GET  /metrics       — Prometheus metrics endpoint
  GET  /info          — model and device information

This server is containerized and deployed to the Raspberry Pi fleet
via the Ansible playbook in orchestration/ansible/deploy_edge.yml.

Usage:
    uvicorn edge.api:app --host 0.0.0.0 --port 8080
    # Or via Docker:
    docker run -p 8080:8080 sentinel-edge
"""

import base64
import io
import logging
import os
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import psutil
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import PlainTextResponse, Response
from pydantic import BaseModel, Field

from edge.inference import ONNXInferenceEngine, InferenceResult

logger = logging.getLogger(__name__)

# ── Configuration from environment variables (12-factor app) ──────────────────
MODEL_PATH = os.getenv("MODEL_PATH", "models/best.onnx")
CLASS_NAMES = os.getenv("CLASS_NAMES", "person,vehicle,equipment").split(",")
CONF_THRESHOLD = float(os.getenv("CONF_THRESHOLD", "0.45"))
IOU_THRESHOLD = float(os.getenv("IOU_THRESHOLD", "0.45"))
NUM_THREADS = int(os.getenv("NUM_THREADS", "4"))
PORT = int(os.getenv("PORT", "8080"))

# Global engine instance (loaded once at startup)
_engine: Optional[ONNXInferenceEngine] = None
_startup_time: float = 0.0


# ── Pydantic models ────────────────────────────────────────────────────────────

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
    device: str
    cpu_pct: float
    ram_used_mb: float


class ModelInfoResponse(BaseModel):
    model_path: str
    class_names: list[str]
    conf_threshold: float
    iou_threshold: float
    num_threads: int
    input_size: int


# ── App lifecycle ──────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the model at startup, release resources at shutdown."""
    global _engine, _startup_time
    _startup_time = time.time()

    logger.info("Starting EdgeAI Sentinel API server...")
    try:
        _engine = ONNXInferenceEngine(
            model_path=MODEL_PATH,
            class_names=CLASS_NAMES,
            conf_threshold=CONF_THRESHOLD,
            iou_threshold=IOU_THRESHOLD,
            num_threads=NUM_THREADS,
        )
        logger.info("Model loaded successfully")
    except FileNotFoundError:
        logger.error(f"Model file not found: {MODEL_PATH}")
        logger.error("Set MODEL_PATH env var or ensure models/best.onnx exists")
        # Continue startup — /health will report model_loaded=false
    except Exception as e:
        logger.error(f"Failed to load model: {e}")

    yield

    logger.info("Shutting down EdgeAI Sentinel API server")


app = FastAPI(
    title="EdgeAI Sentinel",
    description="Low-SWaP edge inference API for AI/ML object detection",
    version="1.0.0",
    lifespan=lifespan,
)


# ── Routes ─────────────────────────────────────────────────────────────────────

@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health():
    """
    Liveness/readiness probe.
    Used by Kubernetes, Ansible, and load balancers to verify the service.
    """
    uptime = time.time() - _startup_time
    mem = psutil.virtual_memory()
    cpu = psutil.cpu_percent(interval=0.1)

    # Try to identify the platform
    device = "unknown"
    try:
        with open("/proc/cpuinfo") as f:
            info = f.read()
        if "Raspberry Pi" in info or "BCM" in info:
            device = "raspberry_pi"
        else:
            device = "x86_64"
    except Exception:
        device = "unknown"

    return HealthResponse(
        status="ok" if _engine is not None else "degraded",
        uptime_seconds=round(uptime, 1),
        model_loaded=_engine is not None,
        device=device,
        cpu_pct=round(cpu, 1),
        ram_used_mb=round(mem.used / 1e6, 1),
    )


@app.get("/info", response_model=ModelInfoResponse, tags=["System"])
async def model_info():
    """Return model configuration."""
    if _engine is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return ModelInfoResponse(
        model_path=MODEL_PATH,
        class_names=CLASS_NAMES,
        conf_threshold=CONF_THRESHOLD,
        iou_threshold=IOU_THRESHOLD,
        num_threads=NUM_THREADS,
        input_size=_engine.input_size,
    )


@app.post("/infer", response_model=InferenceResponse, tags=["Inference"])
async def run_inference(file: UploadFile = File(...)):
    """
    Run object detection on an uploaded image.

    Accepts: JPEG, PNG, BMP
    Returns: bounding boxes, class names, confidence scores, and latency
    """
    if _engine is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    # Validate content type
    if file.content_type not in ("image/jpeg", "image/png", "image/bmp"):
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported image type: {file.content_type}. Use JPEG, PNG, or BMP."
        )

    # Decode image
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if image is None:
        raise HTTPException(status_code=400, detail="Could not decode image")

    h, w = image.shape[:2]

    try:
        result: InferenceResult = _engine.infer(image)
    except Exception as e:
        logger.error(f"Inference failed: {e}")
        raise HTTPException(status_code=500, detail="Inference error")

    return InferenceResponse(
        frame_id=result.frame_id,
        inference_time_ms=result.inference_time_ms,
        detections=[
            DetectionOut(
                x1=d.x1, y1=d.y1, x2=d.x2, y2=d.y2,
                confidence=d.confidence,
                class_id=d.class_id,
                class_name=d.class_name,
            )
            for d in result.detections
        ],
        num_detections=len(result.detections),
        image_width=w,
        image_height=h,
    )


@app.get("/metrics", response_class=PlainTextResponse, tags=["Observability"])
async def metrics():
    """
    Prometheus metrics endpoint.
    Scraped by Prometheus every 15s (see monitoring/prometheus/prometheus.yml).
    """
    try:
        from prometheus_client import generate_latest, CONTENT_TYPE_LATEST
        return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)
    except ImportError:
        return PlainTextResponse("# prometheus_client not installed\n", status_code=200)
