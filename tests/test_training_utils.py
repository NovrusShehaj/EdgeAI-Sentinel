"""Training utility tests that do not require the full training stack."""

from __future__ import annotations

import pytest
import yaml

from training.train import load_config


def test_load_config(tmp_path):
    config = {
        "model": {"architecture": "yolov8n", "num_classes": 1, "pretrained": True},
        "dataset": {
            "classes": ["object"],
            "train": "data/train",
            "val": "data/val",
            "name": "test",
        },
        "training": {
            "epochs": 10,
            "batch_size": 2,
            "image_size": 640,
            "learning_rate": 0.01,
            "momentum": 0.937,
            "weight_decay": 0.0005,
            "warmup_epochs": 1,
            "optimizer": "SGD",
            "patience": 5,
            "save_period": 5,
            "device": "cpu",
        },
    }
    cfg_file = tmp_path / "config.yaml"
    cfg_file.write_text(yaml.dump(config))
    loaded = load_config(str(cfg_file))
    assert loaded["model"]["architecture"] == "yolov8n"
    assert loaded["dataset"]["classes"] == ["object"]


@pytest.mark.training
def test_get_device_cpu_fallback():
    torch = pytest.importorskip("torch")
    from training.train import get_device

    assert get_device("cpu") == "cpu"
    assert torch.device("cpu").type == "cpu"
