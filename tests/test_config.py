"""Class-contract and configuration tests."""

from __future__ import annotations

import textwrap

import pytest
import yaml

from edge.config import (
    DEFAULT_CLASS_NAMES,
    assert_class_contract,
    class_count_from_yolo_shape,
    load_class_names_from_yaml,
    parse_class_names,
    resolve_class_names,
)


def test_data_yaml_is_single_object_class():
    names = load_class_names_from_yaml()
    assert names == ["object"]
    assert DEFAULT_CLASS_NAMES == ["object"]


def test_api_default_matches_data_yaml():
    assert resolve_class_names(explicit=None, env_value="") == load_class_names_from_yaml()


def test_env_override(monkeypatch):
    monkeypatch.setenv("CLASS_NAMES", "object")
    assert resolve_class_names() == ["object"]


def test_parse_class_names_strips_blanks():
    assert parse_class_names(" object , ") == ["object"]


def test_class_count_from_yolo_shape():
    assert class_count_from_yolo_shape([1, 5, 8400]) == 1
    assert class_count_from_yolo_shape([1, 84, 8400]) == 80
    assert class_count_from_yolo_shape([1, "N", "M"]) is None


def test_class_count_mismatch_fails_clearly():
    with pytest.raises(ValueError, match="does not match the model head"):
        assert_class_contract(["object", "vehicle"], [1, 5, 8400])


def test_matching_class_count_passes():
    assert_class_contract(["object"], [1, 5, 8400])


def test_yaml_dict_names(tmp_path):
    path = tmp_path / "data.yaml"
    path.write_text(textwrap.dedent("""
            names:
              0: object
            nc: 1
            """))
    assert load_class_names_from_yaml(path) == ["object"]


def test_train_config_classes_match_data_yaml():
    from pathlib import Path

    train_cfg = yaml.safe_load(Path("configs/train_config.yaml").read_text())
    assert train_cfg["dataset"]["classes"] == load_class_names_from_yaml()
    assert train_cfg["model"]["num_classes"] == 1
