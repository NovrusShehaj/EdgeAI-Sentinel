"""Unit tests for the ONNX inference engine."""

from __future__ import annotations

import threading
from unittest.mock import MagicMock

import numpy as np
import pytest
from prometheus_client import generate_latest

from edge.demo_model import write_yolo_constant_onnx
from edge.inference import InferenceTimeout, ONNXInferenceEngine


class TestONNXInferenceEngine:
    def test_engine_loads(self, demo_onnx, isolated_registry):
        engine = ONNXInferenceEngine(
            model_path=demo_onnx,
            class_names=["object"],
            num_threads=1,
            registry=isolated_registry,
        )
        assert engine is not None
        assert engine.class_names == ["object"]

    def test_missing_model_raises(self):
        with pytest.raises(FileNotFoundError):
            ONNXInferenceEngine(model_path="/nonexistent/model.onnx", class_names=["object"])

    def test_class_count_mismatch_fails(self, two_class_onnx, isolated_registry):
        with pytest.raises(ValueError, match="does not match the model head"):
            ONNXInferenceEngine(
                model_path=two_class_onnx,
                class_names=["object"],
                num_threads=1,
                registry=isolated_registry,
            )

    def test_preprocess_output_shape(self, demo_onnx, dummy_image, isolated_registry):
        engine = ONNXInferenceEngine(
            model_path=demo_onnx,
            class_names=["object"],
            num_threads=1,
            registry=isolated_registry,
        )
        blob, transform = engine.preprocess(dummy_image)
        assert blob.shape == (1, 3, 640, 640)
        assert blob.dtype == np.float32
        assert 0.0 <= blob.min() <= blob.max() <= 1.0
        assert transform.orig_w == 640
        assert transform.orig_h == 480

    def test_preprocess_normalizes_pixels(self, demo_onnx, isolated_registry):
        engine = ONNXInferenceEngine(
            model_path=demo_onnx,
            class_names=["object"],
            num_threads=1,
            registry=isolated_registry,
        )
        white = np.full((480, 640, 3), 255, dtype=np.uint8)
        blob, _ = engine.preprocess(white)
        assert abs(float(blob.max()) - 1.0) < 0.01

    def test_infer_returns_known_class(self, demo_onnx, isolated_registry):
        engine = ONNXInferenceEngine(
            model_path=demo_onnx,
            class_names=["object"],
            num_threads=1,
            conf_threshold=0.1,
            registry=isolated_registry,
        )
        image = np.zeros((640, 640, 3), dtype=np.uint8)
        result = engine.infer(image)
        assert result.frame_id == 1
        assert result.inference_time_ms > 0
        assert result.image_shape == (640, 640)
        assert result.detections
        assert result.detections[0].class_name == "object"
        assert result.detections[0].class_id == 0

    def test_frame_id_increments(self, demo_onnx, dummy_image, isolated_registry):
        engine = ONNXInferenceEngine(
            model_path=demo_onnx,
            class_names=["object"],
            num_threads=1,
            registry=isolated_registry,
        )
        first = engine.infer(dummy_image)
        second = engine.infer(dummy_image)
        assert second.frame_id == first.frame_id + 1

    def test_two_engines_do_not_collide(self, demo_onnx):
        first = ONNXInferenceEngine(
            model_path=demo_onnx,
            class_names=["object"],
            num_threads=1,
        )
        second = ONNXInferenceEngine(
            model_path=demo_onnx,
            class_names=["object"],
            num_threads=1,
        )
        image = np.zeros((64, 64, 3), dtype=np.uint8)
        first.infer(image)
        second.infer(image)
        assert first.is_ready()
        assert second.is_ready()

    def test_timeout(self, demo_onnx, dummy_image, isolated_registry):
        engine = ONNXInferenceEngine(
            model_path=demo_onnx,
            class_names=["object"],
            num_threads=1,
            timeout_s=0.01,
            registry=isolated_registry,
        )

        def slow_run(*_args, **_kwargs):
            import time

            time.sleep(0.2)
            return [np.zeros((1, 5, 1), dtype=np.float32)]

        engine._run_session = slow_run
        with pytest.raises(InferenceTimeout):
            engine.infer(dummy_image)
        output = generate_latest(isolated_registry).decode()
        assert "sentinel_inferences_total" in output
        assert 'status="timeout"' in output

    def test_error_counter(self, demo_onnx, dummy_image, isolated_registry):
        engine = ONNXInferenceEngine(
            model_path=demo_onnx,
            class_names=["object"],
            num_threads=1,
            registry=isolated_registry,
        )
        engine._run_session = MagicMock(side_effect=RuntimeError("boom"))
        with pytest.raises(RuntimeError):
            engine.infer(dummy_image)
        output = generate_latest(isolated_registry).decode()
        assert 'status="error"' in output

    def test_concurrent_requests_are_serialized(self, demo_onnx, dummy_image, isolated_registry):
        engine = ONNXInferenceEngine(
            model_path=demo_onnx,
            class_names=["object"],
            num_threads=1,
            registry=isolated_registry,
        )
        results = []
        errors = []

        def worker():
            try:
                results.append(engine.infer(dummy_image))
            except Exception as exc:  # pragma: no cover - test failure path
                errors.append(exc)

        threads = [threading.Thread(target=worker) for _ in range(4)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        assert not errors
        assert len(results) == 4
        assert {item.frame_id for item in results} == {1, 2, 3, 4}

    def test_unavailable_after_startup(self, demo_onnx, dummy_image, isolated_registry, tmp_path):
        engine = ONNXInferenceEngine(
            model_path=demo_onnx,
            class_names=["object"],
            num_threads=1,
            registry=isolated_registry,
        )
        engine.model_path = tmp_path / "deleted.onnx"
        with pytest.raises(RuntimeError, match="unavailable"):
            engine.infer(dummy_image)
        assert not engine.is_ready()


class TestDemoOnnxFixture:
    def test_demo_model_is_not_identity(self, demo_onnx):
        import onnx

        model = onnx.load(demo_onnx)
        onnx.checker.check_model(model)
        kinds = {node.op_type for node in model.graph.node}
        assert "Identity" not in kinds
        assert "Constant" in kinds
        assert model.graph.output[0].type.tensor_type.shape.dim[1].dim_value == 5

    def test_constant_model_roundtrip(self, tmp_path, isolated_registry):
        path = tmp_path / "known.onnx"
        write_yolo_constant_onnx(
            path,
            num_classes=1,
            detections=((320.0, 320.0, 100.0, 80.0, 0.93, 0),),
        )
        engine = ONNXInferenceEngine(
            model_path=str(path),
            class_names=["object"],
            num_threads=1,
            conf_threshold=0.1,
            registry=isolated_registry,
        )
        result = engine.infer(np.zeros((640, 640, 3), dtype=np.uint8))
        assert result.detections[0].class_name == "object"
        assert result.detections[0].x1 == pytest.approx(270.0, abs=0.5)


class TestEdgeBenchmark:
    def test_get_device_info_returns_dict(self):
        from benchmarks.edge_benchmark import get_device_info

        info = get_device_info()
        assert "device_name" in info
        assert "cores" in info
        assert "ram_total_gb" in info
        assert info["ram_total_gb"] > 0

    def test_estimate_power_bounds(self):
        from benchmarks.edge_benchmark import estimate_power_watts

        for util in (0, 50, 100):
            for is_pi in (True, False):
                power = estimate_power_watts(util, is_pi)
                assert power > 0
                assert power < 200
