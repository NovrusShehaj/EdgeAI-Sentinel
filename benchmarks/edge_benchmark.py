"""
benchmarks/edge_benchmark.py
─────────────────────────────
Benchmark ONNX Runtime inference on Low-SWaP edge devices (Raspberry Pi 4/5).

Profiles:
  - Inference latency and throughput on ARM64 CPU
  - RAM usage during inference
  - CPU utilization per thread
  - Estimated power draw (via vcgencmd on Pi, or estimation)
  - Thread count optimization

This directly maps to the "Low-SWaP deployment" and "Small Board Computers"
requirements in the Lockheed Martin job description.

Usage:
    # Run on a Raspberry Pi or any ARM/x86 machine
    python benchmarks/edge_benchmark.py --model models/best.onnx
    python benchmarks/edge_benchmark.py --model models/best.onnx --threads 1 2 4
"""

import argparse
import json
import logging
import subprocess
import threading
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional

import numpy as np
import psutil

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


@dataclass
class EdgeBenchmarkResult:
    """Structured result for edge device inference benchmark."""
    model_path: str
    device_name: str
    num_threads: int
    image_size: int
    batch_size: int
    # Latency
    latency_mean_ms: float
    latency_p50_ms: float
    latency_p95_ms: float
    latency_p99_ms: float
    latency_std_ms: float
    # Throughput
    throughput_fps: float
    # Resources
    ram_used_mb: float
    ram_total_mb: float
    cpu_utilization_pct: float
    # Power (Pi-specific)
    cpu_temp_c: float
    estimated_power_w: float


def get_device_info() -> dict:
    """
    Collect system information for the edge device.
    Detects Raspberry Pi via /proc/cpuinfo.
    """
    info = {
        "device_name": "Unknown",
        "cpu_model": "Unknown",
        "cores": psutil.cpu_count(logical=False),
        "threads": psutil.cpu_count(logical=True),
        "ram_total_gb": psutil.virtual_memory().total / 1e9,
        "is_raspberry_pi": False,
    }

    try:
        with open("/proc/cpuinfo") as f:
            cpuinfo = f.read()
        if "Raspberry Pi" in cpuinfo or "BCM" in cpuinfo:
            info["is_raspberry_pi"] = True
            for line in cpuinfo.split("\n"):
                if "Model" in line:
                    info["device_name"] = line.split(":")[1].strip()
                    break
            if not info["device_name"] or info["device_name"] == "Unknown":
                info["device_name"] = "Raspberry Pi"
        else:
            # Non-Pi device (development machine)
            try:
                with open("/proc/cpuinfo") as f2:
                    for line in f2:
                        if "model name" in line.lower():
                            info["cpu_model"] = line.split(":")[1].strip()
                            info["device_name"] = info["cpu_model"][:40]
                            break
            except Exception:
                pass
    except Exception:
        info["device_name"] = "Unknown (non-Linux)"

    return info


def get_cpu_temperature() -> float:
    """
    Read CPU temperature.
    Pi: /sys/class/thermal/thermal_zone0/temp
    Linux generic fallback via psutil.
    """
    try:
        with open("/sys/class/thermal/thermal_zone0/temp") as f:
            return float(f.read().strip()) / 1000.0
    except Exception:
        pass
    try:
        temps = psutil.sensors_temperatures()
        if "cpu_thermal" in temps:
            return temps["cpu_thermal"][0].current
        if "coretemp" in temps:
            return temps["coretemp"][0].current
    except Exception:
        pass
    return -1.0


def estimate_power_watts(cpu_utilization: float, is_pi: bool) -> float:
    """
    Rough power estimation.
    Pi 4B idle ~3W, full load ~6W.
    Pi 5 idle ~4W, full load ~8W.
    These are estimates — use a USB power meter for real measurements.
    """
    if is_pi:
        # Linear interpolation between idle and max
        idle = 3.0
        max_draw = 6.0
    else:
        idle = 10.0
        max_draw = 65.0
    return idle + (max_draw - idle) * (cpu_utilization / 100.0)


class ResourceMonitor:
    """
    Background thread that samples CPU and RAM during inference.
    Uses threading to avoid blocking the main benchmark loop.
    """

    def __init__(self, interval_s: float = 0.1):
        self.interval = interval_s
        self.cpu_samples = []
        self.ram_samples = []
        self._stop = threading.Event()

    def start(self):
        self._thread = threading.Thread(target=self._monitor, daemon=True)
        self._thread.start()

    def stop(self):
        self._stop.set()
        self._thread.join()

    def _monitor(self):
        while not self._stop.is_set():
            self.cpu_samples.append(psutil.cpu_percent(interval=None))
            self.ram_samples.append(psutil.virtual_memory().used / 1e6)
            time.sleep(self.interval)

    @property
    def avg_cpu(self) -> float:
        return float(np.mean(self.cpu_samples)) if self.cpu_samples else 0.0

    @property
    def max_ram_mb(self) -> float:
        return float(np.max(self.ram_samples)) if self.ram_samples else 0.0


def run_onnx_benchmark(
    model_path: str,
    num_threads: int,
    image_size: int,
    batch_size: int,
    num_warmup: int = 10,
    num_iters: int = 50,
) -> tuple[list[float], ResourceMonitor]:
    """
    Run ONNX Runtime inference benchmark.

    ONNX Runtime is the recommended inference engine for edge devices —
    it supports CPU, ARM, and GPU execution providers.
    """
    try:
        import onnxruntime as ort
    except ImportError:
        logger.error("onnxruntime not installed. Run: pip install onnxruntime")
        raise

    # Session options — critical for edge performance
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = num_threads
    opts.inter_op_num_threads = 1
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL

    # On Raspberry Pi, CPU is the only provider
    providers = ["CPUExecutionProvider"]

    logger.info(f"Loading ONNX model: {model_path}")
    session = ort.InferenceSession(model_path, sess_options=opts, providers=providers)

    input_name = session.get_inputs()[0].name
    input_shape = (batch_size, 3, image_size, image_size)
    dummy = np.random.randn(*input_shape).astype(np.float32)

    # Warm up
    logger.info(f"  Warm-up ({num_warmup} iters)...")
    for _ in range(num_warmup):
        session.run(None, {input_name: dummy})

    # Benchmark with resource monitoring
    monitor = ResourceMonitor()
    latencies = []

    monitor.start()
    logger.info(f"  Benchmarking ({num_iters} iters)...")
    for _ in range(num_iters):
        t0 = time.perf_counter()
        session.run(None, {input_name: dummy})
        t1 = time.perf_counter()
        latencies.append((t1 - t0) * 1000)
    monitor.stop()

    return latencies, monitor


def benchmark_edge(
    model_path: str,
    thread_counts: list[int],
    image_size: int = 640,
    batch_size: int = 1,
) -> list[EdgeBenchmarkResult]:
    """Run edge benchmarks across different thread configurations."""

    device_info = get_device_info()
    logger.info("=" * 60)
    logger.info("EdgeAI Sentinel — Edge Device Benchmark")
    logger.info("=" * 60)
    logger.info(f"Device:    {device_info['device_name']}")
    logger.info(f"Cores:     {device_info['cores']} physical, {device_info['threads']} logical")
    logger.info(f"RAM:       {device_info['ram_total_gb']:.1f} GB")
    logger.info(f"Model:     {model_path}")
    logger.info(f"Img size:  {image_size}x{image_size}")
    logger.info("=" * 60)

    results = []
    for num_threads in thread_counts:
        logger.info(f"\nThreads: {num_threads}")

        latencies, monitor = run_onnx_benchmark(
            model_path=model_path,
            num_threads=num_threads,
            image_size=image_size,
            batch_size=batch_size,
        )

        lats = np.array(latencies)
        throughput = (1000.0 / np.mean(lats)) * batch_size  # FPS
        cpu_temp = get_cpu_temperature()
        estimated_power = estimate_power_watts(monitor.avg_cpu, device_info["is_raspberry_pi"])

        result = EdgeBenchmarkResult(
            model_path=model_path,
            device_name=device_info["device_name"],
            num_threads=num_threads,
            image_size=image_size,
            batch_size=batch_size,
            latency_mean_ms=round(float(np.mean(lats)), 2),
            latency_p50_ms=round(float(np.percentile(lats, 50)), 2),
            latency_p95_ms=round(float(np.percentile(lats, 95)), 2),
            latency_p99_ms=round(float(np.percentile(lats, 99)), 2),
            latency_std_ms=round(float(np.std(lats)), 2),
            throughput_fps=round(throughput, 2),
            ram_used_mb=round(monitor.max_ram_mb, 1),
            ram_total_mb=round(psutil.virtual_memory().total / 1e6, 1),
            cpu_utilization_pct=round(monitor.avg_cpu, 1),
            cpu_temp_c=round(cpu_temp, 1),
            estimated_power_w=round(estimated_power, 2),
        )
        results.append(result)

        logger.info(f"  FPS:          {result.throughput_fps:.1f}")
        logger.info(f"  Latency mean: {result.latency_mean_ms:.1f} ms  "
                    f"p95={result.latency_p95_ms:.1f} ms")
        logger.info(f"  RAM peak:     {result.ram_used_mb:.0f} MB")
        logger.info(f"  CPU avg:      {result.cpu_utilization_pct:.1f}%")
        if cpu_temp > 0:
            logger.info(f"  CPU temp:     {result.cpu_temp_c:.1f} °C")
        logger.info(f"  Power est.:   {result.estimated_power_w:.1f} W")

    return results


def main():
    parser = argparse.ArgumentParser(description="EdgeAI Sentinel — Edge Benchmark")
    parser.add_argument("--model", type=str, default="models/best.onnx",
                        help="Path to ONNX model")
    parser.add_argument("--threads", nargs="+", type=int, default=[1, 2, 4],
                        help="Thread counts to test")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--output", type=str, default="results/edge_benchmark.json")
    args = parser.parse_args()

    if not Path(args.model).exists():
        logger.warning(f"Model not found: {args.model}")
        logger.warning("Generating dummy ONNX model for demonstration...")
        _generate_dummy_onnx(args.model)

    results = benchmark_edge(
        model_path=args.model,
        thread_counts=args.threads,
        image_size=args.imgsz,
    )

    if results:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        with open(output, "w") as f:
            json.dump([asdict(r) for r in results], f, indent=2)
        logger.info(f"\nResults saved to {output}")

        # Recommend optimal thread count
        best = min(results, key=lambda r: r.latency_mean_ms)
        logger.info(f"\nOptimal threads: {best.num_threads} "
                    f"({best.latency_mean_ms:.1f} ms, {best.throughput_fps:.1f} FPS)")


def _generate_dummy_onnx(output_path: str) -> None:
    """Generate a minimal valid ONNX model for testing without a real checkpoint."""
    try:
        import onnx
        from onnx import helper, TensorProto

        X = helper.make_tensor_value_info("images", TensorProto.FLOAT, [1, 3, 640, 640])
        Y = helper.make_tensor_value_info("output0", TensorProto.FLOAT, [1, 84, 8400])
        identity = helper.make_node("Identity", ["images"], ["output0"])
        graph = helper.make_graph([identity], "sentinel", [X], [Y])
        model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])

        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        onnx.save(model, output_path)
        logger.info(f"Dummy ONNX model saved to {output_path}")
    except ImportError:
        logger.error("onnx package needed for dummy model. pip install onnx")


if __name__ == "__main__":
    main()
