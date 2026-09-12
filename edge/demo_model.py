"""
Build a tiny YOLO-shaped ONNX graph for CI and local smoke tests.

This is not a trained detector. The graph emits a constant YOLOv8-style
tensor so ONNX Runtime, class-count checks, and postprocess tests can run
without shipping weights or using an invalid Identity graph.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Optional

import numpy as np

from edge.model_manifest import sha256_file, write_manifest

DEMO_SOURCE = (
    "Generated Constant YOLOv8-shaped ONNX graph for CI/demo only. "
    "Not a trained detector and not valid for accuracy claims."
)


def yolo_output_tensor(
    num_classes: int = 1,
    num_anchors: int = 8400,
    detections: Optional[Iterable[tuple[float, float, float, float, float, int]]] = None,
) -> np.ndarray:
    """
    Return a [1, 4 + num_classes, num_anchors] tensor.

    Each detection is (cx, cy, w, h, score, class_id) in letterbox/input space.
    """
    if num_classes < 1:
        raise ValueError("num_classes must be >= 1")
    output = np.zeros((1, 4 + num_classes, num_anchors), dtype=np.float32)
    if not detections:
        return output
    for index, (cx, cy, width, height, score, class_id) in enumerate(detections):
        if index >= num_anchors:
            break
        if class_id < 0 or class_id >= num_classes:
            raise ValueError(f"class_id {class_id} outside 0..{num_classes - 1}")
        output[0, 0, index] = cx
        output[0, 1, index] = cy
        output[0, 2, index] = width
        output[0, 3, index] = height
        output[0, 4 + class_id, index] = score
    return output


def write_yolo_constant_onnx(
    output_path: Path,
    num_classes: int = 1,
    num_anchors: int = 8400,
    input_size: int = 640,
    detections: Optional[Iterable[tuple[float, float, float, float, float, int]]] = None,
    opset: int = 17,
) -> Path:
    import onnx
    from onnx import TensorProto, helper, numpy_helper

    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)

    tensor = yolo_output_tensor(num_classes, num_anchors, detections)
    images = helper.make_tensor_value_info(
        "images",
        TensorProto.FLOAT,
        [1, 3, input_size, input_size],
    )
    output = helper.make_tensor_value_info(
        "output0",
        TensorProto.FLOAT,
        list(tensor.shape),
    )
    const = numpy_helper.from_array(tensor, name="yolo_constant")
    node = helper.make_node("Constant", inputs=[], outputs=["output0"], value=const)
    graph = helper.make_graph([node], "sentinel_demo_yolo", [images], [output])
    model = helper.make_model(
        graph,
        opset_imports=[helper.make_opsetid("", opset)],
        ir_version=8,
    )
    model.ir_version = 8
    onnx.checker.check_model(model)
    onnx.save(model, str(target))
    return target


def write_demo_artifact(
    output_path: Path,
    manifest_path: Optional[Path] = None,
    class_names: Optional[list[str]] = None,
    detections: Optional[Iterable[tuple[float, float, float, float, float, int]]] = None,
) -> dict:
    names = list(class_names or ["object"])
    if detections is None:
        detections = ((320.0, 320.0, 100.0, 80.0, 0.92, 0),)
    model_path = write_yolo_constant_onnx(
        output_path,
        num_classes=len(names),
        detections=detections,
    )
    payload = {
        "model_id": "sentinel-yolo-demo",
        "version": "0.0.0-demo",
        "format": "onnx",
        "source": DEMO_SOURCE,
        "class_names": names,
        "input_size": 640,
        "filename": model_path.name,
        "sha256": sha256_file(model_path),
        "synthetic": True,
        "not_for_production": True,
    }
    if manifest_path is not None:
        write_manifest(manifest_path, payload)
    return payload


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate a CI/demo YOLO-shaped ONNX graph (not a trained model)."
    )
    parser.add_argument("--output", type=str, default="models/demo.onnx")
    parser.add_argument("--manifest", type=str, default="models/model_manifest.json")
    parser.add_argument("--classes", nargs="+", default=["object"])
    args = parser.parse_args()

    payload = write_demo_artifact(
        Path(args.output),
        manifest_path=Path(args.manifest),
        class_names=args.classes,
    )
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
