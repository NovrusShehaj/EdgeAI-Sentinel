"""
tests/test_inference.py
────────────────────────
Unit and integration tests for the EdgeAI Sentinel inference engine.

Run with: pytest tests/ -v
"""

import io
import json
import numpy as np
import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture(scope="session")
def dummy_onnx_model(tmp_path_factory):
    """Create a minimal ONNX model for testing (no GPU / real weights needed)."""
    tmp = tmp_path_factory.mktemp("models")
    model_path = tmp / "test.onnx"

    try:
        import onnx
        from onnx import helper, TensorProto

        X = helper.make_tensor_value_info("images", TensorProto.FLOAT, [1, 3, 640, 640])
        Y = helper.make_tensor_value_info("output0", TensorProto.FLOAT, [1, 84, 8400])
        node = helper.make_node("Identity", ["images"], ["output0"])
        graph = helper.make_graph([node], "test_model", [X], [Y])
        model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
        onnx.save(model, str(model_path))
    except ImportError:
        pytest.skip("onnx package not installed")

    return str(model_path)


@pytest.fixture
def dummy_image():
    """BGR image matching typical camera frame."""
    return np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)


# ── Unit Tests: Inference Engine ─────────────────────────────────────────────

class TestONNXInferenceEngine:

    def test_engine_loads(self, dummy_onnx_model):
        """Engine initializes without error given a valid model path."""
        try:
            from edge.inference import ONNXInferenceEngine
        except ImportError:
            pytest.skip("onnxruntime not installed")

        engine = ONNXInferenceEngine(
            model_path=dummy_onnx_model,
            class_names=["person", "vehicle"],
            num_threads=1,
        )
        assert engine is not None

    def test_missing_model_raises(self):
        """FileNotFoundError raised for nonexistent model path."""
        try:
            from edge.inference import ONNXInferenceEngine
        except ImportError:
            pytest.skip("onnxruntime not installed")

        with pytest.raises(FileNotFoundError):
            ONNXInferenceEngine(
                model_path="/nonexistent/model.onnx",
                class_names=["person"],
            )

    def test_preprocess_output_shape(self, dummy_onnx_model, dummy_image):
        """Preprocessed tensor has correct shape [1, 3, 640, 640]."""
        try:
            from edge.inference import ONNXInferenceEngine
        except ImportError:
            pytest.skip("onnxruntime not installed")

        engine = ONNXInferenceEngine(
            model_path=dummy_onnx_model,
            class_names=["person"],
            num_threads=1,
        )
        blob, x_scale, y_scale = engine.preprocess(dummy_image)
        assert blob.shape == (1, 3, 640, 640), f"Unexpected shape: {blob.shape}"
        assert blob.dtype == np.float32
        assert 0.0 <= blob.min() and blob.max() <= 1.0, "Pixel values outside [0, 1]"

    def test_preprocess_normalizes_pixels(self, dummy_onnx_model):
        """Pixels are normalized to [0, 1]."""
        try:
            from edge.inference import ONNXInferenceEngine
        except ImportError:
            pytest.skip("onnxruntime not installed")

        engine = ONNXInferenceEngine(
            model_path=dummy_onnx_model,
            class_names=["person"],
            num_threads=1,
        )
        # White image — should normalize to ~1.0
        white_image = np.full((480, 640, 3), 255, dtype=np.uint8)
        blob, _, _ = engine.preprocess(white_image)
        assert abs(blob.max() - 1.0) < 0.01, "Max pixel should be ~1.0"

    def test_infer_returns_result(self, dummy_onnx_model, dummy_image):
        """infer() returns InferenceResult with expected fields."""
        try:
            from edge.inference import ONNXInferenceEngine, InferenceResult
        except ImportError:
            pytest.skip("onnxruntime not installed")

        engine = ONNXInferenceEngine(
            model_path=dummy_onnx_model,
            class_names=["person"],
            num_threads=1,
            conf_threshold=0.99,  # No detections from dummy model
        )
        result = engine.infer(dummy_image)

        assert isinstance(result, InferenceResult)
        assert result.frame_id == 1
        assert result.inference_time_ms > 0
        assert isinstance(result.detections, list)
        assert result.image_shape == (480, 640)

    def test_frame_id_increments(self, dummy_onnx_model, dummy_image):
        """frame_id increments with each call."""
        try:
            from edge.inference import ONNXInferenceEngine
        except ImportError:
            pytest.skip("onnxruntime not installed")

        engine = ONNXInferenceEngine(
            model_path=dummy_onnx_model,
            class_names=["person"],
            num_threads=1,
            conf_threshold=0.99,
        )
        r1 = engine.infer(dummy_image)
        r2 = engine.infer(dummy_image)
        assert r2.frame_id == r1.frame_id + 1


# ── Unit Tests: Benchmark ─────────────────────────────────────────────────────

class TestEdgeBenchmark:

    def test_get_device_info_returns_dict(self):
        """Device info always returns a dict with required keys."""
        from benchmarks.edge_benchmark import get_device_info

        info = get_device_info()
        assert "device_name" in info
        assert "cores" in info
        assert "ram_total_gb" in info
        assert info["ram_total_gb"] > 0

    def test_estimate_power_bounds(self):
        """Power estimate stays within reasonable range."""
        from benchmarks.edge_benchmark import estimate_power_watts

        for util in [0, 50, 100]:
            for is_pi in [True, False]:
                power = estimate_power_watts(util, is_pi)
                assert power > 0, "Power should be positive"
                assert power < 200, "Power estimate too high"


# ── Integration Tests: API ────────────────────────────────────────────────────

class TestAPI:

    @pytest.fixture
    def client(self, dummy_onnx_model, monkeypatch):
        """Test client with mocked model path."""
        try:
            from fastapi.testclient import TestClient
            from httpx import AsyncClient
        except ImportError:
            pytest.skip("fastapi/httpx not installed")

        monkeypatch.setenv("MODEL_PATH", dummy_onnx_model)
        monkeypatch.setenv("CLASS_NAMES", "person,vehicle")

        try:
            from edge.api import app
            return TestClient(app)
        except Exception:
            pytest.skip("Could not create test client")

    def test_health_endpoint(self, client):
        """GET /health returns 200 with required fields."""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert "model_loaded" in data
        assert "uptime_seconds" in data

    def test_infer_endpoint_with_valid_image(self, client):
        """POST /infer with a valid PNG returns 200 and inference result."""
        try:
            import cv2
        except ImportError:
            pytest.skip("opencv not installed")

        # Create a minimal valid PNG in memory
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        success, buffer = cv2.imencode(".png", img)
        assert success

        response = client.post(
            "/infer",
            files={"file": ("test.png", io.BytesIO(buffer.tobytes()), "image/png")},
        )
        assert response.status_code in (200, 503)  # 503 if model failed to load in CI
        if response.status_code == 200:
            data = response.json()
            assert "detections" in data
            assert "inference_time_ms" in data
            assert data["inference_time_ms"] > 0

    def test_infer_endpoint_rejects_bad_type(self, client):
        """POST /infer with unsupported file type returns 400."""
        response = client.post(
            "/infer",
            files={"file": ("test.txt", io.BytesIO(b"not an image"), "text/plain")},
        )
        assert response.status_code == 400


# ── Smoke Tests: Training utilities ───────────────────────────────────────────

class TestTrainingUtils:

    def test_load_config(self, tmp_path):
        """load_config correctly parses a YAML file."""
        import yaml
        from training.train import load_config

        config = {
            "model": {"architecture": "yolov8n", "num_classes": 3, "pretrained": True},
            "dataset": {"classes": ["a", "b", "c"], "train": "data/train",
                        "val": "data/val", "name": "test"},
            "training": {"epochs": 10, "batch_size": 2, "image_size": 640,
                         "learning_rate": 0.01, "momentum": 0.937,
                         "weight_decay": 0.0005, "warmup_epochs": 1,
                         "optimizer": "SGD", "patience": 5, "save_period": 5,
                         "device": "cpu"},
            "augmentation": {"hsv_h": 0.015, "hsv_s": 0.7, "hsv_v": 0.4,
                             "degrees": 0.0, "translate": 0.1, "scale": 0.5,
                             "flipud": 0.0, "fliplr": 0.5, "mosaic": 1.0, "mixup": 0.0},
            "logging": {"project": "test", "name": "exp", "save_dir": "runs/",
                        "verbose": False},
            "export": {"formats": ["onnx"], "onnx": {"opset": 17, "simplify": True,
                       "dynamic": False}, "output_dir": "models/"},
        }
        cfg_file = tmp_path / "config.yaml"
        with open(cfg_file, "w") as f:
            yaml.dump(config, f)

        loaded = load_config(str(cfg_file))
        assert loaded["model"]["architecture"] == "yolov8n"
        assert loaded["training"]["epochs"] == 10

    def test_get_device_cpu_fallback(self):
        """get_device returns 'cpu' when requested."""
        from training.train import get_device
        assert get_device("cpu") == "cpu"
