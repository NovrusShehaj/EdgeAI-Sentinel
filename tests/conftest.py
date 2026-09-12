"""Shared fixtures for EdgeAI Sentinel tests."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from prometheus_client import CollectorRegistry

from edge.demo_model import write_demo_artifact, write_yolo_constant_onnx


@pytest.fixture
def isolated_registry() -> CollectorRegistry:
    return CollectorRegistry()


@pytest.fixture
def demo_onnx(tmp_path: Path) -> str:
    model_path = tmp_path / "demo.onnx"
    write_demo_artifact(model_path, manifest_path=tmp_path / "model_manifest.json")
    return str(model_path)


@pytest.fixture
def two_class_onnx(tmp_path: Path) -> str:
    model_path = tmp_path / "two_class.onnx"
    write_yolo_constant_onnx(
        model_path,
        num_classes=2,
        detections=((320.0, 320.0, 40.0, 40.0, 0.9, 1),),
    )
    return str(model_path)


@pytest.fixture
def dummy_image() -> np.ndarray:
    return np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)


@pytest.fixture
def api_token() -> str:
    return "test-api-token"


@pytest.fixture
def api_client(demo_onnx, api_token, monkeypatch):
    monkeypatch.setenv("MODEL_PATH", demo_onnx)
    monkeypatch.setenv("CLASS_NAMES", "object")
    monkeypatch.setenv("API_TOKEN", api_token)
    monkeypatch.setenv("ALLOWED_HOSTS", "testserver,localhost,127.0.0.1")
    monkeypatch.setenv("METRICS_ALLOWED_CIDRS", "")
    monkeypatch.setenv("INFER_TIMEOUT_S", "5")

    import importlib

    import edge.api as api_module

    importlib.reload(api_module)
    from fastapi.testclient import TestClient

    with TestClient(api_module.app) as client:
        yield client


@pytest.fixture
def api_client_no_model(api_token, monkeypatch):
    monkeypatch.setenv("MODEL_PATH", "/nonexistent/model.onnx")
    monkeypatch.setenv("CLASS_NAMES", "object")
    monkeypatch.setenv("API_TOKEN", api_token)
    monkeypatch.setenv("ALLOWED_HOSTS", "testserver,localhost,127.0.0.1")
    monkeypatch.setenv("METRICS_ALLOWED_CIDRS", "")

    import importlib

    import edge.api as api_module

    importlib.reload(api_module)
    from fastapi.testclient import TestClient

    with TestClient(api_module.app) as client:
        yield client
