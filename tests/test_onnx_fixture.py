"""ONNX checker coverage for the demo fixture."""

from __future__ import annotations

import onnx

from edge.demo_model import write_yolo_constant_onnx


def test_onnx_checker_accepts_demo_graph(tmp_path):
    path = tmp_path / "demo.onnx"
    write_yolo_constant_onnx(path, num_classes=1)
    model = onnx.load(str(path))
    onnx.checker.check_model(model)
    output_dim = model.graph.output[0].type.tensor_type.shape.dim[1].dim_value
    assert output_dim == 5
