"""
Shared class-name and runtime configuration.

`configs/data.yaml` is the source of truth for class names. Serving, CLI,
and deployment manifests must resolve to the same list unless CLASS_NAMES
is an explicit runtime override (still validated against the model head).
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable, Optional

DEFAULT_CLASS_NAMES = ["object"]
REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATA_YAML = REPO_ROOT / "configs" / "data.yaml"


def parse_class_names(value: str) -> list[str]:
    names = [part.strip() for part in value.split(",") if part.strip()]
    return names or list(DEFAULT_CLASS_NAMES)


def load_class_names_from_yaml(path: Optional[Path] = None) -> list[str]:
    yaml_path = Path(path) if path is not None else DEFAULT_DATA_YAML
    if not yaml_path.is_file():
        return list(DEFAULT_CLASS_NAMES)

    try:
        import yaml
    except ImportError:
        return list(DEFAULT_CLASS_NAMES)

    with yaml_path.open() as handle:
        data = yaml.safe_load(handle) or {}

    names = data.get("names")
    if isinstance(names, dict):
        names = [names[key] for key in sorted(names, key=lambda item: int(item))]
    if isinstance(names, list) and names:
        return [str(name) for name in names]
    return list(DEFAULT_CLASS_NAMES)


def resolve_class_names(
    explicit: Optional[Iterable[str]] = None,
    env_value: Optional[str] = None,
    data_yaml: Optional[Path] = None,
) -> list[str]:
    if explicit is not None:
        names = [str(name).strip() for name in explicit if str(name).strip()]
        if names:
            return names

    raw_env = env_value if env_value is not None else os.getenv("CLASS_NAMES")
    if raw_env and raw_env.strip():
        return parse_class_names(raw_env)

    return load_class_names_from_yaml(data_yaml)


def expected_yolo_channels(class_names: list[str]) -> int:
    return 4 + len(class_names)


def class_count_from_yolo_shape(shape) -> Optional[int]:
    """
    Infer class count from a YOLOv8-style output shape.

    Typical export: [batch, 4 + num_classes, num_anchors].
    Some graphs transpose to [batch, num_anchors, 4 + num_classes].
    """
    if shape is None or len(shape) < 2:
        return None

    static = []
    for dim in shape[1:]:
        if isinstance(dim, int) and dim > 0:
            static.append(dim)
        elif hasattr(dim, "__int__"):
            try:
                value = int(dim)
            except (TypeError, ValueError):
                continue
            if value > 0:
                static.append(value)

    if not static:
        return None

    channel_candidates = [dim for dim in static if 5 <= dim <= 256]
    if len(channel_candidates) == 1:
        return channel_candidates[0] - 4
    if len(static) >= 2:
        channel = min(static)
        if 5 <= channel <= 256:
            return channel - 4
    return None


def assert_class_contract(class_names: list[str], output_shape) -> None:
    if not class_names:
        raise ValueError("At least one class name is required")
    inferred = class_count_from_yolo_shape(output_shape)
    if inferred is None:
        return
    if inferred != len(class_names):
        raise ValueError(
            "Class-name count does not match the model head: "
            f"configured {len(class_names)} {class_names!r}, "
            f"model reports {inferred} classes from shape {list(output_shape)}"
        )
