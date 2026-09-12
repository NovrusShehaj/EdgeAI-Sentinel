"""
Generate SYNTHETIC smoke-test images (white rectangles on a black canvas).

This is not a real-world dataset. Labels and images produced here are for CI
and short CPU smoke-training only. Do not publish accuracy metrics from this
data as production results.

Usage:
    python scripts/test_data.py
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

DATASET_NOTE = "synthetic-white-rectangle-smoke"


def create_sample(path_img: str, path_lbl: str, seed: int) -> None:
    rng = np.random.default_rng(seed)
    image = np.zeros((640, 640, 3), dtype=np.uint8)
    x1, y1 = rng.integers(50, 300, size=2)
    x2, y2 = x1 + 100, y1 + 100
    cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), (255, 255, 255), -1)

    xc = ((x1 + x2) / 2) / 640
    yc = ((y1 + y2) / 2) / 640
    width = (x2 - x1) / 640
    height = (y2 - y1) / 640

    Path(path_lbl).parent.mkdir(parents=True, exist_ok=True)
    Path(path_img).parent.mkdir(parents=True, exist_ok=True)
    Path(path_lbl).write_text(f"0 {xc} {yc} {width} {height}\n")
    cv2.imwrite(path_img, image)


def generate_synthetic_dataset(
    root: Path = Path("data"),
    train_count: int = 50,
    val_count: int = 10,
) -> dict[str, int]:
    for index in range(train_count):
        create_sample(
            str(root / "train" / "images" / f"{index}.jpg"),
            str(root / "train" / "labels" / f"{index}.txt"),
            seed=index,
        )
    for index in range(val_count):
        create_sample(
            str(root / "val" / "images" / f"{index}.jpg"),
            str(root / "val" / "labels" / f"{index}.txt"),
            seed=1000 + index,
        )
    return {"train": train_count, "val": val_count, "dataset": DATASET_NOTE}


def main() -> None:
    counts = generate_synthetic_dataset()
    print(
        "Generated synthetic smoke images only "
        f"(train={counts['train']} val={counts['val']}). "
        "Not valid for production accuracy claims."
    )


if __name__ == "__main__":
    main()
