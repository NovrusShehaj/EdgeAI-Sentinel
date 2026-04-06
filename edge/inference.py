"""
edge/inference.py
─────────────────
Production-grade ONNX Runtime inference engine for edge deployment.

Features:
  - ONNX Runtime with CPU execution provider (optimized for ARM64)
  - Non-maximum suppression (NMS) post-processing
  - Prometheus metrics export (latency, throughput, errors)
  - Thread-safe singleton pattern for multi-threaded API serving
  - Configurable confidence and IoU thresholds

This is the core of what runs on the Raspberry Pi in a deployed system.

Usage (standalone):
    python edge/inference.py --model models/best.onnx --source 0
    python edge/inference.py --model models/best.onnx --source video.mp4
"""

import argparse
import logging
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class Detection:
    """A single object detection result."""
    x1: float
    y1: float
    x2: float
    y2: float
    confidence: float
    class_id: int
    class_name: str


@dataclass
class InferenceResult:
    """Full result from one inference call."""
    detections: list[Detection]
    inference_time_ms: float
    frame_id: int
    image_shape: tuple


class ONNXInferenceEngine:
    """
    Thread-safe ONNX Runtime inference engine.

    Designed to run efficiently on Low-SWaP hardware:
      - Single session with pre-allocated buffers
      - Configurable thread count (tune for your Pi model)
      - Minimal Python overhead in hot path
    """

    def __init__(
        self,
        model_path: str,
        class_names: list[str],
        conf_threshold: float = 0.45,
        iou_threshold: float = 0.45,
        num_threads: int = 4,
        input_size: int = 640,
    ):
        self.model_path = Path(model_path)
        self.class_names = class_names
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.input_size = input_size
        self._lock = threading.Lock()
        self._frame_counter = 0

        if not self.model_path.exists():
            raise FileNotFoundError(f"ONNX model not found: {model_path}")

        self._session = self._load_session(num_threads)
        self._input_name = self._session.get_inputs()[0].name
        self._setup_metrics()

        logger.info(f"Inference engine ready — model: {model_path}")
        logger.info(f"  Input:     {self._input_name} — {self._session.get_inputs()[0].shape}")
        logger.info(f"  Threads:   {num_threads}")
        logger.info(f"  Conf thr:  {conf_threshold}")
        logger.info(f"  Classes:   {class_names}")

    def _load_session(self, num_threads: int):
        """Initialize ONNX Runtime session with edge-optimized settings."""
        try:
            import onnxruntime as ort
        except ImportError:
            raise ImportError("onnxruntime not installed. Run: pip install onnxruntime")

        opts = ort.SessionOptions()
        opts.intra_op_num_threads = num_threads
        opts.inter_op_num_threads = 1
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        # Disable telemetry for edge/secure environments
        opts.enable_profiling = False

        providers = ["CPUExecutionProvider"]
        return ort.InferenceSession(str(self.model_path), sess_options=opts, providers=providers)

    def _setup_metrics(self) -> None:
        """Initialize Prometheus metrics for observability."""
        try:
            from prometheus_client import Counter, Histogram, Gauge

            self._metric_inferences = Counter(
                "sentinel_inferences_total",
                "Total inference calls",
                ["status"],
            )
            self._metric_latency = Histogram(
                "sentinel_inference_latency_ms",
                "Inference latency in milliseconds",
                buckets=[10, 20, 50, 100, 150, 200, 300, 500, 1000],
            )
            self._metric_detections = Histogram(
                "sentinel_detections_per_frame",
                "Number of detections per frame",
                buckets=[0, 1, 2, 5, 10, 20, 50],
            )
            self._metric_confidence = Gauge(
                "sentinel_avg_confidence",
                "Average detection confidence (rolling)",
            )
            self._metrics_enabled = True
            logger.info("Prometheus metrics initialized")
        except ImportError:
            logger.warning("prometheus_client not installed — metrics disabled")
            self._metrics_enabled = False

    def preprocess(self, image: np.ndarray) -> tuple[np.ndarray, float, float]:
        """
        Preprocess image for YOLOv8 inference.
        Letterbox resize to preserve aspect ratio.

        Returns: (preprocessed_tensor, x_scale, y_scale)
        """
        h, w = image.shape[:2]
        target = self.input_size

        # Compute scale factor (letterbox — pad to square without stretching)
        scale = min(target / h, target / w)
        new_w, new_h = int(w * scale), int(h * scale)

        resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        # Pad to square
        canvas = np.full((target, target, 3), 114, dtype=np.uint8)
        pad_x = (target - new_w) // 2
        pad_y = (target - new_h) // 2
        canvas[pad_y:pad_y + new_h, pad_x:pad_x + new_w] = resized

        # BGR → RGB, HWC → NCHW, normalize to [0, 1]
        blob = canvas[:, :, ::-1].transpose(2, 0, 1)[np.newaxis].astype(np.float32) / 255.0

        # Return scale factors for coordinate recovery
        x_scale = w / new_w
        y_scale = h / new_h

        return blob, x_scale, y_scale

    def postprocess(
        self,
        raw_output: np.ndarray,
        x_scale: float,
        y_scale: float,
        orig_h: int,
        orig_w: int,
    ) -> list[Detection]:
        """
        Post-process YOLOv8 output tensor to Detection objects.

        YOLOv8 output shape: [1, 4+num_classes, num_anchors]
        Format: [x_center, y_center, width, height, cls_score_0, ..., cls_score_n]
        """
        target = self.input_size
        predictions = raw_output[0]  # [4+nc, 8400]

        # Separate boxes and class scores
        boxes = predictions[:4].T      # [8400, 4] — cx, cy, w, h
        scores = predictions[4:].T     # [8400, nc]

        # Get best class and confidence per anchor
        class_ids = np.argmax(scores, axis=1)
        confidences = scores[np.arange(len(scores)), class_ids]

        # Filter by confidence threshold
        mask = confidences > self.conf_threshold
        boxes = boxes[mask]
        confidences = confidences[mask]
        class_ids = class_ids[mask]

        if len(boxes) == 0:
            return []

        # Convert cx, cy, w, h → x1, y1, x2, y2
        pad_x = (target - orig_w / x_scale) / 2
        pad_y = (target - orig_h / y_scale) / 2
        scale = target / (orig_w * x_scale)  # effective scale back to orig coords

        x1 = ((boxes[:, 0] - boxes[:, 2] / 2 - pad_x) / scale).clip(0, orig_w)
        y1 = ((boxes[:, 1] - boxes[:, 3] / 2 - pad_y) / scale).clip(0, orig_h)
        x2 = ((boxes[:, 0] + boxes[:, 2] / 2 - pad_x) / scale).clip(0, orig_w)
        y2 = ((boxes[:, 1] + boxes[:, 3] / 2 - pad_y) / scale).clip(0, orig_h)

        xyxy = np.stack([x1, y1, x2, y2], axis=1)

        # Non-Maximum Suppression
        keep = cv2.dnn.NMSBoxes(
            bboxes=xyxy.tolist(),
            scores=confidences.tolist(),
            score_threshold=self.conf_threshold,
            nms_threshold=self.iou_threshold,
        )
        if len(keep) == 0:
            return []

        keep = keep.flatten()
        detections = []
        for i in keep:
            cls = int(class_ids[i])
            name = self.class_names[cls] if cls < len(self.class_names) else str(cls)
            detections.append(Detection(
                x1=float(xyxy[i, 0]), y1=float(xyxy[i, 1]),
                x2=float(xyxy[i, 2]), y2=float(xyxy[i, 3]),
                confidence=float(confidences[i]),
                class_id=cls,
                class_name=name,
            ))

        return detections

    def infer(self, image: np.ndarray) -> InferenceResult:
        """
        Run inference on a single BGR image (OpenCV format).
        Thread-safe — can be called from multiple API workers.
        """
        with self._lock:
            self._frame_counter += 1
            frame_id = self._frame_counter

        h, w = image.shape[:2]

        t0 = time.perf_counter()
        blob, x_scale, y_scale = self.preprocess(image)
        raw_output = self._session.run(None, {self._input_name: blob})
        detections = self.postprocess(raw_output[0], x_scale, y_scale, h, w)
        t1 = time.perf_counter()

        inference_ms = (t1 - t0) * 1000

        if self._metrics_enabled:
            self._metric_latency.observe(inference_ms)
            self._metric_inferences.labels(status="success").inc()
            self._metric_detections.observe(len(detections))
            if detections:
                avg_conf = np.mean([d.confidence for d in detections])
                self._metric_confidence.set(avg_conf)

        return InferenceResult(
            detections=detections,
            inference_time_ms=round(inference_ms, 2),
            frame_id=frame_id,
            image_shape=(h, w),
        )

    def draw_detections(self, image: np.ndarray, result: InferenceResult) -> np.ndarray:
        """Draw bounding boxes and labels on the frame (for display/debug)."""
        vis = image.copy()
        for det in result.detections:
            x1, y1, x2, y2 = int(det.x1), int(det.y1), int(det.x2), int(det.y2)
            # Color by class (BGR)
            color = tuple(int(c) for c in (
                (det.class_id * 67 + 50) % 256,
                (det.class_id * 113 + 100) % 256,
                (det.class_id * 157 + 30) % 256,
            ))
            cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
            label = f"{det.class_name} {det.confidence:.2f}"
            (lw, lh), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(vis, (x1, y1 - lh - 6), (x1 + lw, y1), color, -1)
            cv2.putText(vis, label, (x1, y1 - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # Overlay stats
        fps = 1000 / result.inference_time_ms if result.inference_time_ms > 0 else 0
        cv2.putText(vis, f"FPS: {fps:.1f}  |  {len(result.detections)} det",
                    (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        return vis


def run_camera_loop(engine: ONNXInferenceEngine, source) -> None:
    """
    Run real-time inference from a camera or video file.
    Press 'q' to quit.
    """
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        logger.error(f"Cannot open video source: {source}")
        return

    logger.info(f"Streaming from source: {source}")
    logger.info("Press 'q' to quit")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            result = engine.infer(frame)
            vis = engine.draw_detections(frame, result)

            cv2.imshow("EdgeAI Sentinel", vis)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(description="EdgeAI Sentinel — Edge Inference")
    parser.add_argument("--model", type=str, default="models/best.onnx")
    parser.add_argument("--source", type=str, default="0",
                        help="Camera index (0, 1) or video file path")
    parser.add_argument("--conf", type=float, default=0.45)
    parser.add_argument("--iou", type=float, default=0.45)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--classes", nargs="+", default=["person", "vehicle", "equipment"])
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(message)s")

    engine = ONNXInferenceEngine(
        model_path=args.model,
        class_names=args.classes,
        conf_threshold=args.conf,
        iou_threshold=args.iou,
        num_threads=args.threads,
    )

    source = int(args.source) if args.source.isdigit() else args.source
    run_camera_loop(engine, source)


if __name__ == "__main__":
    main()
