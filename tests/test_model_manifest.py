"""Model checksum and manifest tests."""

from __future__ import annotations

import json

import pytest

from edge.demo_model import write_demo_artifact
from edge.inference import ONNXInferenceEngine
from edge.model_manifest import sha256_file, verify_manifest, write_manifest


def test_checksum_and_class_metadata_verified(tmp_path, isolated_registry):
    model_path = tmp_path / "demo.onnx"
    manifest_path = tmp_path / "model_manifest.json"
    payload = write_demo_artifact(model_path, manifest_path=manifest_path)
    assert payload["sha256"] == sha256_file(model_path)
    assert payload["class_names"] == ["object"]

    engine = ONNXInferenceEngine(
        model_path=str(model_path),
        class_names=["object"],
        num_threads=1,
        registry=isolated_registry,
        manifest_path=str(manifest_path),
    )
    assert engine.manifest["sha256"] == payload["sha256"]


def test_checksum_mismatch_fails(tmp_path):
    model_path = tmp_path / "demo.onnx"
    manifest_path = tmp_path / "model_manifest.json"
    write_demo_artifact(model_path, manifest_path=manifest_path)
    write_manifest(
        manifest_path,
        {
            "filename": "demo.onnx",
            "class_names": ["object"],
            "sha256": "0" * 64,
        },
    )
    with pytest.raises(ValueError, match="checksum mismatch"):
        verify_manifest(manifest_path, model_path, ["object"])


def test_manifest_class_mismatch_fails(tmp_path):
    model_path = tmp_path / "demo.onnx"
    manifest_path = tmp_path / "model_manifest.json"
    write_demo_artifact(model_path, manifest_path=manifest_path)
    data = json.loads(manifest_path.read_text())
    data["class_names"] = ["person"]
    manifest_path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="class names"):
        verify_manifest(manifest_path, model_path, ["object"])
