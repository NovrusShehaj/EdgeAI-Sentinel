"""Contract tests for liveness, readiness, auth, and upload limits."""

from __future__ import annotations

import io
import os

import cv2
import numpy as np


def _png_file(width: int = 64, height: int = 64) -> dict:
    image = np.zeros((height, width, 3), dtype=np.uint8)
    ok, buffer = cv2.imencode(".png", image)
    assert ok
    return {"file": ("test.png", io.BytesIO(buffer.tobytes()), "image/png")}


def test_live_without_model(api_client_no_model):
    response = api_client_no_model.get("/live")
    assert response.status_code == 200
    assert response.json()["status"] == "alive"


def test_ready_without_model_is_503(api_client_no_model):
    response = api_client_no_model.get("/ready")
    assert response.status_code == 503
    payload = response.json()
    assert payload["ready"] is False
    assert payload["model_loaded"] is False


def test_health_is_diagnostic_without_model(api_client_no_model):
    response = api_client_no_model.get("/health")
    assert response.status_code == 200
    payload = response.json()
    assert payload["model_loaded"] is False
    assert payload["ready"] is False
    assert payload["status"] == "degraded"


def test_live_and_ready_with_model(api_client):
    live = api_client.get("/live")
    ready = api_client.get("/ready")
    assert live.status_code == 200
    assert ready.status_code == 200
    assert ready.json()["ready"] is True
    assert ready.json()["class_names"] == ["object"]


def test_infer_requires_token(api_client):
    response = api_client.post("/infer", files=_png_file())
    assert response.status_code == 401


def test_infer_rejects_empty_api_token(demo_onnx, monkeypatch):
    monkeypatch.setenv("MODEL_PATH", demo_onnx)
    monkeypatch.setenv("CLASS_NAMES", "object")
    monkeypatch.setenv("API_TOKEN", "")
    monkeypatch.setenv("ALLOWED_HOSTS", "testserver,localhost,127.0.0.1")
    monkeypatch.setenv("METRICS_ALLOWED_CIDRS", "")

    import importlib

    import edge.api as api_module

    importlib.reload(api_module)
    from fastapi.testclient import TestClient

    with TestClient(api_module.app) as client:
        response = client.post("/infer", files=_png_file())
    assert response.status_code == 401
    assert "API_TOKEN" in response.json()["detail"]


def test_infer_rejects_invalid_token(api_client):
    response = api_client.post(
        "/infer",
        files=_png_file(),
        headers={"X-API-Token": "wrong-token"},
    )
    assert response.status_code == 401


def test_infer_with_valid_token(api_client, api_token):
    response = api_client.post(
        "/infer",
        files=_png_file(640, 640),
        headers={"X-API-Token": api_token},
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["num_detections"] >= 1
    assert payload["detections"][0]["class_name"] == "object"
    assert payload["image_width"] == 640
    assert payload["image_height"] == 640


def test_infer_rejects_bad_type(api_client, api_token):
    response = api_client.post(
        "/infer",
        files={"file": ("test.txt", io.BytesIO(b"not an image"), "text/plain")},
        headers={"X-API-Token": api_token},
    )
    assert response.status_code == 400


def test_oversized_upload_returns_413(api_client, api_token, monkeypatch):
    monkeypatch.setenv("MAX_UPLOAD_BYTES", "128")
    response = api_client.post(
        "/infer",
        files={"file": ("big.png", io.BytesIO(os.urandom(4096)), "image/png")},
        headers={"X-API-Token": api_token},
    )
    assert response.status_code == 413


def test_excessive_dimensions_rejected(api_client, api_token, monkeypatch):
    monkeypatch.setenv("MAX_IMAGE_SIDE", "32")
    response = api_client.post(
        "/infer",
        files=_png_file(64, 64),
        headers={"X-API-Token": api_token},
    )
    assert response.status_code == 400
    assert "exceed" in response.json()["detail"]


def test_metrics_requires_auth_from_untrusted_client(api_client):
    response = api_client.get("/metrics")
    assert response.status_code == 401


def test_metrics_with_token(api_client, api_token):
    response = api_client.get("/metrics", headers={"X-API-Token": api_token})
    assert response.status_code == 200
    assert (
        "sentinel_inferences_total" in response.text
        or response.text.startswith("#")
        or "python" in response.text
    )


def test_info_requires_ready_engine(api_client_no_model):
    response = api_client_no_model.get("/info")
    assert response.status_code == 503
