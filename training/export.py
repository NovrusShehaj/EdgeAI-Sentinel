"""
training/export.py
──────────────────
Export a trained YOLOv8 .pt checkpoint to ONNX and TensorRT formats
for edge device deployment.

Why this matters for Lockheed/edge AI roles:
  - ONNX provides hardware-agnostic inference across CPU, GPU, NPU
  - TensorRT squeezes maximum performance from NVIDIA Jetson and desktop GPUs
  - Fixed input shapes enable deterministic latency (critical for real-time systems)

Usage:
    python training/export.py --checkpoint runs/train/exp/weights/best.pt
    python training/export.py --checkpoint models/best.pt --format onnx trt
"""

import argparse
import logging
import shutil
import sys
import time
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def export_onnx(model, output_dir: Path, imgsz: int, opset: int, simplify: bool) -> Path:
    """
    Export to ONNX format.
    ONNX is the primary format for Raspberry Pi / edge deployment via ONNX Runtime.
    """
    logger.info("Exporting to ONNX...")
    start = time.time()

    # model.export() saves to checkpoint directory and returns the path
    exported_path = model.export(
        format="onnx",
        imgsz=imgsz,
        opset=opset,
        simplify=simplify,
        dynamic=False,          # Static shapes = predictable edge latency
    )

    # Copy exported file to output directory
    output_path = output_dir / "best.onnx"
    exported_path = Path(exported_path) if isinstance(exported_path, str) else exported_path
    if exported_path.exists():
        shutil.copy(str(exported_path), str(output_path))
    else:
        logger.warning(f"Exported file not found at {exported_path}, using checkpoint location")
        output_path = exported_path

    elapsed = time.time() - start
    logger.info(f"ONNX export complete in {elapsed:.1f}s → {output_path}")

    # Validate the exported model
    if output_path.exists():
        _validate_onnx(output_path)
    return output_path


def export_torchscript(model, output_dir: Path, imgsz: int) -> Path:
    """
    Export to TorchScript — useful for deployment without Python dependency.
    """
    logger.info("Exporting to TorchScript...")

    # model.export() saves to checkpoint directory and returns the path
    exported_path = model.export(format="torchscript", imgsz=imgsz)

    # Copy exported file to output directory
    output_path = output_dir / "best.torchscript"
    exported_path = Path(exported_path) if isinstance(exported_path, str) else exported_path
    if exported_path.exists():
        shutil.copy(str(exported_path), str(output_path))
    else:
        output_path = exported_path

    logger.info(f"TorchScript export → {output_path}")
    return output_path


def export_tensorrt(model, output_dir: Path, imgsz: int, fp16: bool) -> Path:
    """
    Export to TensorRT engine.
    Requires NVIDIA GPU + TensorRT installed.
    Provides 2-5x speedup over ONNX on NVIDIA hardware.

    Note: TRT engines are device-specific — an engine built on a desktop
    GPU will NOT run on a Jetson Nano. Always build on the target device.
    """
    try:
        import tensorrt  # noqa: F401
    except ImportError:
        logger.warning("TensorRT not installed. Skipping TRT export.")
        logger.warning("Install: https://docs.nvidia.com/deeplearning/tensorrt/install-guide/")
        return None

    logger.info(f"Exporting to TensorRT (fp16={fp16})...")

    # model.export() saves to checkpoint directory and returns the path
    exported_path = model.export(format="engine", imgsz=imgsz, half=fp16)

    # Copy exported file to output directory
    output_path = output_dir / "best.engine"
    exported_path = Path(exported_path) if isinstance(exported_path, str) else exported_path
    if exported_path.exists():
        shutil.copy(str(exported_path), str(output_path))
    else:
        output_path = exported_path

    logger.info(f"TensorRT export → {output_path}")
    return output_path


def _validate_onnx(onnx_path: Path) -> None:
    """Run ONNX model checker to catch graph errors before deployment."""
    try:
        import onnx

        model = onnx.load(str(onnx_path))
        onnx.checker.check_model(model)
        logger.info("ONNX model validation passed ✓")

        # Log graph inputs/outputs for documentation
        graph = model.graph
        for inp in graph.input:
            shape = [d.dim_value for d in inp.type.tensor_type.shape.dim]
            logger.info(f"  Input:  {inp.name} — shape {shape}")
        for out in graph.output:
            shape = [d.dim_value for d in out.type.tensor_type.shape.dim]
            logger.info(f"  Output: {out.name} — shape {shape}")

    except ImportError:
        logger.warning("onnx package not installed — skipping validation")
    except Exception as e:
        logger.error(f"ONNX validation failed: {e}")
        sys.exit(1)


def print_model_info(checkpoint_path: Path) -> None:
    """Display model size and parameter count."""
    try:
        import torch

        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        if "model" in checkpoint:
            state = checkpoint["model"].float().state_dict()
            params = sum(p.numel() for p in state.values())
            logger.info(f"Model parameters: {params:,}")
        size_mb = checkpoint_path.stat().st_size / 1e6
        logger.info(f"Checkpoint size: {size_mb:.1f} MB")
    except Exception as e:
        logger.warning(f"Could not read model info: {e}")


def main():
    parser = argparse.ArgumentParser(description="EdgeAI Sentinel — Model Export")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to trained .pt checkpoint")
    parser.add_argument("--format", nargs="+", default=["onnx"],
                        choices=["onnx", "torchscript", "trt"],
                        help="Export format(s)")
    parser.add_argument("--imgsz", type=int, default=640,
                        help="Input image size (must match training size)")
    parser.add_argument("--output-dir", type=str, default="models/",
                        help="Directory to save exported models")
    parser.add_argument("--opset", type=int, default=17,
                        help="ONNX opset version")
    parser.add_argument("--simplify", action="store_true", default=True,
                        help="Run onnx-simplifier on exported graph")
    parser.add_argument("--fp16", action="store_true",
                        help="Use FP16 quantization for TensorRT export")
    args = parser.parse_args()

    checkpoint = Path(args.checkpoint)
    if not checkpoint.exists():
        logger.error(f"Checkpoint not found: {checkpoint}")
        sys.exit(1)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        from ultralytics import YOLO
    except ImportError:
        logger.error("ultralytics not installed. Run: pip install ultralytics")
        sys.exit(1)

    logger.info(f"Loading model from {checkpoint}")
    print_model_info(checkpoint)
    model = YOLO(str(checkpoint))

    exported = {}
    for fmt in args.format:
        if fmt == "onnx":
            exported["onnx"] = export_onnx(model, output_dir, args.imgsz,
                                            args.opset, args.simplify)
        elif fmt == "torchscript":
            exported["torchscript"] = export_torchscript(model, output_dir, args.imgsz)
        elif fmt == "trt":
            exported["trt"] = export_tensorrt(model, output_dir, args.imgsz, args.fp16)

    logger.info("=" * 50)
    logger.info("Export summary:")
    for fmt, path in exported.items():
        if path and Path(path).exists():
            size_mb = Path(path).stat().st_size / 1e6
            logger.info(f"  [{fmt:12s}] {path}  ({size_mb:.1f} MB)")
    logger.info("=" * 50)
    logger.info("Next step: Deploy to edge device or run edge/inference.py")


if __name__ == "__main__":
    main()
