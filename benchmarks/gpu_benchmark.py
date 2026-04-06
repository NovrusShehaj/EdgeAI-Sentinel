"""
benchmarks/gpu_benchmark.py
────────────────────────────
Comprehensive GPU benchmarking for AI/ML workloads.

Profiles:
  - Inference throughput (images/sec) across batch sizes
  - Memory utilization (peak VRAM per batch)
  - Latency percentiles (p50, p95, p99)
  - GPU compute utilization
  - Comparison across model sizes (n, s, m)

This script directly demonstrates the "GPU Computing / HPC" skill
required for the AI/ML Hardware Engineer role.

Usage:
    python benchmarks/gpu_benchmark.py
    python benchmarks/gpu_benchmark.py --model yolov8n --batch-sizes 1 4 8 16 32
    python benchmarks/gpu_benchmark.py --compare-models --output results/gpu_bench.csv
"""

import argparse
import csv
import json
import logging
import time
import warnings
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional

import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)
warnings.filterwarnings("ignore")


@dataclass
class BenchmarkResult:
    """Structured result for a single benchmark run."""
    model: str
    device: str
    batch_size: int
    image_size: int
    # Throughput
    throughput_imgs_per_sec: float
    # Latency (ms)
    latency_mean_ms: float
    latency_p50_ms: float
    latency_p95_ms: float
    latency_p99_ms: float
    latency_std_ms: float
    # Memory
    vram_peak_mb: float
    vram_allocated_mb: float
    # System
    gpu_name: str
    cuda_version: str
    num_warmup: int
    num_iterations: int


def get_gpu_info() -> dict:
    """Collect GPU hardware information for reporting."""
    info = {
        "gpu_name": "CPU (no CUDA)",
        "cuda_version": "N/A",
        "vram_total_gb": 0.0,
        "driver_version": "N/A",
    }
    try:
        import torch

        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(0)
            info["gpu_name"] = props.name
            info["vram_total_gb"] = props.total_memory / 1e9
            info["cuda_version"] = torch.version.cuda or "unknown"
            info["driver_version"] = "via torch"
            info["compute_capability"] = f"{props.major}.{props.minor}"
            info["sm_count"] = props.multi_processor_count
    except ImportError:
        pass

    try:
        import GPUtil
        gpus = GPUtil.getGPUs()
        if gpus:
            info["driver_version"] = gpus[0].driver
    except Exception:
        pass

    return info


def run_inference_benchmark(
    model,
    device: str,
    batch_size: int,
    image_size: int,
    num_warmup: int = 20,
    num_iters: int = 100,
) -> tuple[list[float], float, float]:
    """
    Run a single benchmark configuration.

    Returns: (latency_list_ms, peak_vram_mb, allocated_vram_mb)
    """
    import torch

    # Build a dummy input tensor — same shape as real camera frames
    dummy_input = torch.randn(batch_size, 3, image_size, image_size)

    if device != "cpu":
        dummy_input = dummy_input.to(device)
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()

    # Warm up — essential to get accurate measurements
    # (GPU clocks ramp up, JIT compiles, cuDNN kernels cached)
    logger.info(f"  Warming up ({num_warmup} iters)...")
    with torch.no_grad():
        for _ in range(num_warmup):
            _ = model(dummy_input, verbose=False)
    if device != "cpu":
        torch.cuda.synchronize()

    # Benchmark loop
    latencies = []
    logger.info(f"  Benchmarking ({num_iters} iters)...")
    with torch.no_grad():
        for _ in range(num_iters):
            if device != "cpu":
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            _ = model(dummy_input, verbose=False)
            if device != "cpu":
                torch.cuda.synchronize()  # Wait for GPU to finish
            t1 = time.perf_counter()
            latencies.append((t1 - t0) * 1000)  # ms

    peak_vram = 0.0
    allocated_vram = 0.0
    if device != "cpu":
        peak_vram = torch.cuda.max_memory_reserved(0) / 1e6
        allocated_vram = torch.cuda.memory_allocated(0) / 1e6

    return latencies, peak_vram, allocated_vram


def benchmark_model(
    model_name: str,
    batch_sizes: list[int],
    image_size: int = 640,
    device: str = "auto",
    output_dir: Optional[Path] = None,
) -> list[BenchmarkResult]:
    """
    Full benchmark suite for one model across all batch sizes.
    """
    try:
        import torch
        from ultralytics import YOLO
    except ImportError:
        logger.error("torch and ultralytics required. pip install torch ultralytics")
        return []

    # Resolve device
    if device == "auto":
        device = "0" if torch.cuda.is_available() else "cpu"

    device_label = f"cuda:{device}" if device != "cpu" else "cpu"
    gpu_info = get_gpu_info()

    logger.info("=" * 60)
    logger.info(f"Benchmarking {model_name}")
    logger.info(f"Device:    {gpu_info['gpu_name']}")
    logger.info(f"CUDA:      {gpu_info['cuda_version']}")
    logger.info(f"VRAM:      {gpu_info.get('vram_total_gb', 0):.1f} GB")
    logger.info(f"Image size: {image_size}x{image_size}")
    logger.info("=" * 60)

    # Load model to target device
    model_file = f"{model_name}.pt"
    model = YOLO(model_file)
    if device != "cpu":
        import torch as _torch
        model.to(_torch.device(device_label))

    results = []
    for bs in batch_sizes:
        logger.info(f"\nBatch size: {bs}")
        try:
            latencies, peak_vram, alloc_vram = run_inference_benchmark(
                model=model,
                device=device_label,
                batch_size=bs,
                image_size=image_size,
            )

            lats = np.array(latencies)
            total_images = len(latencies) * bs
            total_time_s = sum(latencies) / 1000
            throughput = total_images / total_time_s

            result = BenchmarkResult(
                model=model_name,
                device=gpu_info["gpu_name"],
                batch_size=bs,
                image_size=image_size,
                throughput_imgs_per_sec=round(throughput, 2),
                latency_mean_ms=round(float(np.mean(lats)), 2),
                latency_p50_ms=round(float(np.percentile(lats, 50)), 2),
                latency_p95_ms=round(float(np.percentile(lats, 95)), 2),
                latency_p99_ms=round(float(np.percentile(lats, 99)), 2),
                latency_std_ms=round(float(np.std(lats)), 2),
                vram_peak_mb=round(peak_vram, 1),
                vram_allocated_mb=round(alloc_vram, 1),
                gpu_name=gpu_info["gpu_name"],
                cuda_version=gpu_info["cuda_version"],
                num_warmup=20,
                num_iterations=100,
            )
            results.append(result)

            # Pretty-print result row
            logger.info(f"  Throughput: {result.throughput_imgs_per_sec:.1f} img/s")
            logger.info(f"  Latency:    {result.latency_mean_ms:.1f} ms mean  "
                        f"| p95={result.latency_p95_ms:.1f} ms "
                        f"| p99={result.latency_p99_ms:.1f} ms")
            if peak_vram > 0:
                logger.info(f"  VRAM peak:  {result.vram_peak_mb:.0f} MB")

        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                logger.warning(f"  OOM at batch_size={bs} — skipping")
                if device != "cpu":
                    import torch as _torch
                    _torch.cuda.empty_cache()
            else:
                logger.error(f"  Error at batch_size={bs}: {e}")

    return results


def save_results(results: list[BenchmarkResult], output_path: Path) -> None:
    """Save benchmark results to CSV and JSON."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # CSV
    csv_path = output_path.with_suffix(".csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(asdict(results[0]).keys()))
        writer.writeheader()
        writer.writerows([asdict(r) for r in results])
    logger.info(f"Results saved to {csv_path}")

    # JSON (easier to parse programmatically)
    json_path = output_path.with_suffix(".json")
    with open(json_path, "w") as f:
        json.dump([asdict(r) for r in results], f, indent=2)
    logger.info(f"Results saved to {json_path}")


def print_summary_table(results: list[BenchmarkResult]) -> None:
    """Print a formatted comparison table to stdout."""
    try:
        from tabulate import tabulate
        rows = [
            [r.model, r.batch_size, f"{r.throughput_imgs_per_sec:.1f}",
             f"{r.latency_mean_ms:.1f}", f"{r.latency_p95_ms:.1f}",
             f"{r.vram_peak_mb:.0f}"]
            for r in results
        ]
        headers = ["Model", "Batch", "Throughput (img/s)", "Latency mean (ms)",
                   "Latency p95 (ms)", "VRAM peak (MB)"]
        print("\n" + tabulate(rows, headers=headers, tablefmt="rounded_outline"))
    except ImportError:
        for r in results:
            print(f"{r.model} | bs={r.batch_size} | "
                  f"{r.throughput_imgs_per_sec:.1f} img/s | "
                  f"{r.latency_mean_ms:.1f} ms")


def main():
    parser = argparse.ArgumentParser(description="EdgeAI Sentinel — GPU Benchmarking")
    parser.add_argument("--model", type=str, default="yolov8n",
                        help="Model to benchmark (yolov8n, yolov8s, ...)")
    parser.add_argument("--batch-sizes", nargs="+", type=int,
                        default=[1, 4, 8, 16],
                        help="Batch sizes to benchmark")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", type=str, default="auto",
                        help="Device: auto | 0 | 1 | cpu")
    parser.add_argument("--compare-models", action="store_true",
                        help="Run across yolov8 n/s/m variants")
    parser.add_argument("--output", type=str, default="results/gpu_benchmark",
                        help="Output path (without extension)")
    args = parser.parse_args()

    all_results = []

    if args.compare_models:
        for model_variant in ["yolov8n", "yolov8s", "yolov8m"]:
            results = benchmark_model(
                model_name=model_variant,
                batch_sizes=args.batch_sizes,
                image_size=args.imgsz,
                device=args.device,
            )
            all_results.extend(results)
    else:
        all_results = benchmark_model(
            model_name=args.model,
            batch_sizes=args.batch_sizes,
            image_size=args.imgsz,
            device=args.device,
        )

    if all_results:
        print_summary_table(all_results)
        save_results(all_results, Path(args.output))
        logger.info("Benchmark complete. Use Grafana or notebooks/ to visualize results.")


if __name__ == "__main__":
    main()
