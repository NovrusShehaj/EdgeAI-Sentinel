"""
ONNX Runtime inference engine for edge deployment.

The engine letterbox-resizes frames, runs a single serialized ONNX session,
inverts preprocessing with the original scale and padding, and records
Prometheus metrics on an injectable registry.
"""

from __future__ import annotations

import argparse
import logging
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FuturesTimeout
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from prometheus_client import CollectorRegistry, Counter, Gauge, Histogram

from edge.config import assert_class_contract, resolve_class_names
from edge.model_manifest import resolve_manifest_path, verify_manifest

logger = logging.getLogger(__name__)

_SHARED_METRICS = None
_SHARED_METRICS_LOCK = threading.Lock()


class InferenceTimeout(TimeoutError):
    """Raised when session.run exceeds the configured timeout."""


@dataclass(frozen=True)
class LetterboxTransform:
    """Exact letterbox parameters used to invert model coordinates."""

    scale: float
    pad_x: float
    pad_y: float
    orig_w: int
    orig_h: int
    new_w: int
    new_h: int
    target: int

    @property
    def x_scale(self) -> float:
        return self.orig_w / self.new_w if self.new_w else 1.0

    @property
    def y_scale(self) -> float:
        return self.orig_h / self.new_h if self.new_h else 1.0


@dataclass
class Detection:
    x1: float
    y1: float
    x2: float
    y2: float
    confidence: float
    class_id: int
    class_name: str


@dataclass
class InferenceResult:
    detections: list[Detection]
    inference_time_ms: float
    frame_id: int
    image_shape: tuple


@dataclass
class _EngineMetrics:
    inferences: Counter
    latency: Histogram
    detections: Histogram
    confidence: Gauge
    ready: Gauge


def letterbox_image(image: np.ndarray, target: int) -> tuple[np.ndarray, LetterboxTransform]:
    """Resize with aspect-preserving letterbox padding to a square canvas."""
    orig_h, orig_w = image.shape[:2]
    scale = min(target / orig_h, target / orig_w)
    new_w, new_h = int(orig_w * scale), int(orig_h * scale)
    if new_w < 1 or new_h < 1:
        raise ValueError("Letterbox produced an empty resized image")

    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    canvas = np.full((target, target, 3), 114, dtype=np.uint8)
    pad_x = (target - new_w) // 2
    pad_y = (target - new_h) // 2
    canvas[pad_y : pad_y + new_h, pad_x : pad_x + new_w] = resized
    transform = LetterboxTransform(
        scale=scale,
        pad_x=float(pad_x),
        pad_y=float(pad_y),
        orig_w=orig_w,
        orig_h=orig_h,
        new_w=new_w,
        new_h=new_h,
        target=target,
    )
    return canvas, transform


def invert_letterbox_xyxy(
    x1: np.ndarray,
    y1: np.ndarray,
    x2: np.ndarray,
    y2: np.ndarray,
    transform: LetterboxTransform,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Invert letterbox coordinates using the original scale and padding.

    x = (x_pred - pad_x) * x_scale
    y = (y_pred - pad_y) * y_scale
    """
    inv_x1 = (x1 - transform.pad_x) * transform.x_scale
    inv_x2 = (x2 - transform.pad_x) * transform.x_scale
    inv_y1 = (y1 - transform.pad_y) * transform.y_scale
    inv_y2 = (y2 - transform.pad_y) * transform.y_scale
    return (
        inv_x1.clip(0, transform.orig_w),
        inv_y1.clip(0, transform.orig_h),
        inv_x2.clip(0, transform.orig_w),
        inv_y2.clip(0, transform.orig_h),
    )


def decode_yolo_output(
    raw_output: np.ndarray,
    transform: LetterboxTransform,
    class_names: list[str],
    conf_threshold: float,
    iou_threshold: float,
) -> list[Detection]:
    """Convert a YOLOv8 [1, 4+nc, anchors] tensor into clipped image-space boxes."""
    if raw_output.ndim == 3:
        predictions = raw_output[0]
    else:
        predictions = raw_output

    if predictions.shape[0] < 5 and predictions.shape[-1] >= 5:
        predictions = predictions.T

    boxes = predictions[:4].T
    scores = predictions[4:].T
    if scores.ndim == 1:
        scores = scores.reshape(-1, 1)

    class_ids = np.argmax(scores, axis=1)
    confidences = scores[np.arange(len(scores)), class_ids]
    mask = confidences > conf_threshold
    boxes = boxes[mask]
    confidences = confidences[mask]
    class_ids = class_ids[mask]
    if len(boxes) == 0:
        return []

    x1 = boxes[:, 0] - boxes[:, 2] / 2
    y1 = boxes[:, 1] - boxes[:, 3] / 2
    x2 = boxes[:, 0] + boxes[:, 2] / 2
    y2 = boxes[:, 1] + boxes[:, 3] / 2
    x1, y1, x2, y2 = invert_letterbox_xyxy(x1, y1, x2, y2, transform)
    xyxy = np.stack([x1, y1, x2, y2], axis=1)

    keep = cv2.dnn.NMSBoxes(
        bboxes=xyxy.tolist(),
        scores=confidences.tolist(),
        score_threshold=conf_threshold,
        nms_threshold=iou_threshold,
    )
    if keep is None or len(keep) == 0:
        return []

    keep = np.array(keep).flatten()
    detections = []
    for index in keep:
        class_id = int(class_ids[index])
        name = class_names[class_id] if class_id < len(class_names) else str(class_id)
        detections.append(
            Detection(
                x1=float(xyxy[index, 0]),
                y1=float(xyxy[index, 1]),
                x2=float(xyxy[index, 2]),
                y2=float(xyxy[index, 3]),
                confidence=float(confidences[index]),
                class_id=class_id,
                class_name=name,
            )
        )
    return detections


def _create_metrics(registry: Optional[CollectorRegistry]) -> _EngineMetrics:
    kwargs = {"registry": registry} if registry is not None else {}
    return _EngineMetrics(
        inferences=Counter(
            "sentinel_inferences_total",
            "Total inference calls",
            ["status"],
            **kwargs,
        ),
        latency=Histogram(
            "sentinel_inference_latency_ms",
            "Inference latency in milliseconds",
            buckets=(10, 20, 50, 100, 150, 200, 300, 500, 1000),
            **kwargs,
        ),
        detections=Histogram(
            "sentinel_detections_per_frame",
            "Number of detections per frame",
            buckets=(0, 1, 2, 5, 10, 20, 50),
            **kwargs,
        ),
        confidence=Gauge(
            "sentinel_avg_confidence",
            "Average detection confidence (rolling)",
            **kwargs,
        ),
        ready=Gauge(
            "sentinel_model_ready",
            "1 when the inference engine can serve traffic",
            **kwargs,
        ),
    )


def _shared_metrics() -> _EngineMetrics:
    global _SHARED_METRICS
    with _SHARED_METRICS_LOCK:
        if _SHARED_METRICS is None:
            _SHARED_METRICS = _create_metrics(None)
        return _SHARED_METRICS


class ONNXInferenceEngine:
    """
    Thread-safe ONNX Runtime engine.

    Concurrency model: one process, one ONNX session, one worker. `session.run`
    is serialized with a lock. API deployments should use a single Uvicorn
    worker. If the model file disappears after startup, readiness fails and
    inference is rejected until the process is restarted with a valid artifact.
    """

    def __init__(
        self,
        model_path: str,
        class_names: Optional[list[str]] = None,
        conf_threshold: float = 0.45,
        iou_threshold: float = 0.45,
        num_threads: int = 4,
        input_size: int = 640,
        timeout_s: float = 5.0,
        registry: Optional[CollectorRegistry] = None,
        manifest_path: Optional[str] = None,
        metrics_enabled: bool = True,
    ):
        self.model_path = Path(model_path)
        self.class_names = list(class_names) if class_names is not None else resolve_class_names()
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.input_size = input_size
        self.timeout_s = timeout_s
        self.manifest: dict = {}
        self._lock = threading.Lock()
        self._frame_counter = 0
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="onnx-infer")

        if not self.model_path.exists():
            raise FileNotFoundError(f"ONNX model not found: {model_path}")

        resolved_manifest = resolve_manifest_path(
            self.model_path,
            Path(manifest_path) if manifest_path else None,
        )
        if resolved_manifest is not None:
            self.manifest = verify_manifest(resolved_manifest, self.model_path, self.class_names)

        self._session = self._load_session(num_threads)
        self._input_name = self._session.get_inputs()[0].name
        output_shape = self._session.get_outputs()[0].shape
        assert_class_contract(self.class_names, output_shape)

        self._metrics_enabled = metrics_enabled
        self._metrics = None
        if metrics_enabled:
            self._metrics = _create_metrics(registry) if registry is not None else _shared_metrics()
            self._metrics.ready.set(1)

        logger.info("Inference engine ready — model: %s", model_path)
        logger.info("  Input:     %s — %s", self._input_name, self._session.get_inputs()[0].shape)
        logger.info("  Threads:   %s", num_threads)
        logger.info("  Timeout:   %ss", timeout_s)
        logger.info("  Conf thr:  %s", conf_threshold)
        logger.info("  Classes:   %s", self.class_names)

    def _load_session(self, num_threads: int):
        try:
            import onnxruntime as ort
        except ImportError as exc:
            raise ImportError("onnxruntime not installed. Run: pip install onnxruntime") from exc

        opts = ort.SessionOptions()
        opts.intra_op_num_threads = num_threads
        opts.inter_op_num_threads = 1
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        opts.enable_profiling = False
        return ort.InferenceSession(
            str(self.model_path),
            sess_options=opts,
            providers=["CPUExecutionProvider"],
        )

    def is_ready(self) -> bool:
        return self._session is not None and self.model_path.exists()

    def mark_unavailable(self) -> None:
        if self._metrics_enabled and self._metrics is not None:
            self._metrics.ready.set(0)

    def preprocess(self, image: np.ndarray) -> tuple[np.ndarray, LetterboxTransform]:
        canvas, transform = letterbox_image(image, self.input_size)
        blob = canvas[:, :, ::-1].transpose(2, 0, 1)[np.newaxis].astype(np.float32) / 255.0
        return blob, transform

    def postprocess(
        self,
        raw_output: np.ndarray,
        transform: LetterboxTransform,
    ) -> list[Detection]:
        return decode_yolo_output(
            raw_output,
            transform,
            self.class_names,
            self.conf_threshold,
            self.iou_threshold,
        )

    def _run_session(self, blob: np.ndarray):
        return self._session.run(None, {self._input_name: blob})

    def _record(
        self,
        status: str,
        latency_ms: Optional[float] = None,
        detections: Optional[list[Detection]] = None,
    ) -> None:
        if not self._metrics_enabled or self._metrics is None:
            return
        self._metrics.inferences.labels(status=status).inc()
        if latency_ms is not None and status == "success":
            self._metrics.latency.observe(latency_ms)
        if detections is not None:
            self._metrics.detections.observe(len(detections))
            if detections:
                self._metrics.confidence.set(float(np.mean([det.confidence for det in detections])))

    def infer(self, image: np.ndarray) -> InferenceResult:
        """
        Run inference on a single BGR image.

        session.run is serialized. Concurrent API requests queue on the engine
        lock rather than sharing an unsafe ORT session.
        """
        with self._lock:
            if not self.is_ready():
                self.mark_unavailable()
                self._record("unavailable")
                raise RuntimeError("Model is unavailable after startup")

            self._frame_counter += 1
            frame_id = self._frame_counter
            height, width = image.shape[:2]
            started = time.perf_counter()
            try:
                blob, transform = self.preprocess(image)
                future = self._executor.submit(self._run_session, blob)
                raw_output = future.result(timeout=self.timeout_s)
                detections = self.postprocess(raw_output[0], transform)
            except FuturesTimeout as exc:
                self._record("timeout")
                raise InferenceTimeout(f"Inference exceeded {self.timeout_s}s") from exc
            except Exception:
                self._record("error")
                raise

            latency_ms = (time.perf_counter() - started) * 1000
            self._record("success", latency_ms, detections)
            return InferenceResult(
                detections=detections,
                inference_time_ms=round(latency_ms, 2),
                frame_id=frame_id,
                image_shape=(height, width),
            )

    def draw_detections(self, image: np.ndarray, result: InferenceResult) -> np.ndarray:
        vis = image.copy()
        for det in result.detections:
            x1, y1, x2, y2 = int(det.x1), int(det.y1), int(det.x2), int(det.y2)
            color = tuple(
                int(channel)
                for channel in (
                    (det.class_id * 67 + 50) % 256,
                    (det.class_id * 113 + 100) % 256,
                    (det.class_id * 157 + 30) % 256,
                )
            )
            cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
            label = f"{det.class_name} {det.confidence:.2f}"
            (label_w, label_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(vis, (x1, y1 - label_h - 6), (x1 + label_w, y1), color, -1)
            cv2.putText(
                vis,
                label,
                (x1, y1 - 4),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1,
            )
        fps = 1000 / result.inference_time_ms if result.inference_time_ms > 0 else 0
        cv2.putText(
            vis,
            f"FPS: {fps:.1f}  |  {len(result.detections)} det",
            (10, 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2,
        )
        return vis

    def close(self) -> None:
        self.mark_unavailable()
        self._executor.shutdown(wait=False)


def run_camera_loop(engine: ONNXInferenceEngine, source, headless: bool = False) -> None:
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        logger.error("Cannot open video source: %s", source)
        return

    logger.info("Streaming from source: %s", source)
    if not headless:
        logger.info("Press 'q' to quit")

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            result = engine.infer(frame)
            if headless:
                logger.info(
                    "frame=%s detections=%s latency_ms=%.2f",
                    result.frame_id,
                    len(result.detections),
                    result.inference_time_ms,
                )
                continue
            vis = engine.draw_detections(frame, result)
            cv2.imshow("EdgeAI Sentinel", vis)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        cap.release()
        if not headless:
            cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(description="EdgeAI Sentinel — Edge Inference")
    parser.add_argument("--model", type=str, default=os.getenv("MODEL_PATH", "models/best.onnx"))
    parser.add_argument("--source", type=str, default="0", help="Camera index or video path")
    parser.add_argument("--conf", type=float, default=0.45)
    parser.add_argument("--iou", type=float, default=0.45)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--timeout", type=float, default=float(os.getenv("INFER_TIMEOUT_S", "5")))
    parser.add_argument("--classes", nargs="+", default=None)
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Log detections without opening a GUI window",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    engine = ONNXInferenceEngine(
        model_path=args.model,
        class_names=resolve_class_names(explicit=args.classes),
        conf_threshold=args.conf,
        iou_threshold=args.iou,
        num_threads=args.threads,
        timeout_s=args.timeout,
    )
    source = int(args.source) if args.source.isdigit() else args.source
    run_camera_loop(engine, source, headless=args.headless)


if __name__ == "__main__":
    main()
