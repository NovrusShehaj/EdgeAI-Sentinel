"""Synthetic dataset and script-guard checks."""

from __future__ import annotations

from pathlib import Path

from scripts.test_data import generate_synthetic_dataset


def test_synthetic_script_has_main_guard():
    source = Path("scripts/test_data.py").read_text()
    assert 'if __name__ == "__main__":' in source
    assert "synthetic" in source.lower()


def test_label_counts_and_optional_images(tmp_path):
    counts = generate_synthetic_dataset(root=tmp_path, train_count=3, val_count=2)
    train_images = list((tmp_path / "train" / "images").glob("*.jpg"))
    train_labels = list((tmp_path / "train" / "labels").glob("*.txt"))
    val_images = list((tmp_path / "val" / "images").glob("*.jpg"))
    val_labels = list((tmp_path / "val" / "labels").glob("*.txt"))
    assert counts["train"] == len(train_images) == len(train_labels) == 3
    assert counts["val"] == len(val_images) == len(val_labels) == 2
    assert all(label.read_text().startswith("0 ") for label in train_labels + val_labels)


def test_committed_labels_are_single_class():
    train_labels = list(Path("data/train/labels").glob("*.txt"))
    val_labels = list(Path("data/val/labels").glob("*.txt"))
    assert len(train_labels) == 50
    assert len(val_labels) == 10
    for path in train_labels + val_labels:
        class_id = path.read_text().strip().split()[0]
        assert class_id == "0"
