"""
Model artifact manifest and integrity checks.

A mounted ONNX model should be accompanied by a JSON manifest that records
version, source, class names, and a SHA-256 checksum. This is not a substitute
for real training provenance; it only verifies that the file on disk is the
artifact the operator intended to load.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Optional

MANIFEST_FILENAME = "model_manifest.json"


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def load_manifest(path: Path) -> dict[str, Any]:
    with Path(path).open() as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"Model manifest must be a JSON object: {path}")
    return data


def resolve_manifest_path(model_path: Path, explicit: Optional[Path] = None) -> Optional[Path]:
    if explicit is not None:
        return Path(explicit)
    candidate = Path(model_path).with_name(MANIFEST_FILENAME)
    return candidate if candidate.is_file() else None


def verify_manifest(
    manifest_path: Path,
    model_path: Path,
    class_names: list[str],
) -> dict[str, Any]:
    manifest = load_manifest(manifest_path)
    expected_hash = manifest.get("sha256")
    if expected_hash:
        actual = sha256_file(model_path)
        if actual.lower() != str(expected_hash).lower():
            raise ValueError(
                f"Model checksum mismatch for {model_path}: "
                f"manifest={expected_hash} actual={actual}"
            )

    manifest_classes = manifest.get("class_names")
    if manifest_classes:
        names = [str(name) for name in manifest_classes]
        if names != list(class_names):
            raise ValueError(
                "Manifest class names do not match configured class names: "
                f"manifest={names!r} configured={list(class_names)!r}"
            )

    filename = manifest.get("filename")
    if filename and Path(model_path).name != filename:
        raise ValueError(f"Manifest filename {filename!r} does not match {Path(model_path).name!r}")

    return manifest


def write_manifest(path: Path, payload: dict[str, Any]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")
