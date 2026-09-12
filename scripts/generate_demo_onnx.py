#!/usr/bin/env python3
"""CLI wrapper for the CI/demo YOLO-shaped ONNX generator."""

from __future__ import annotations

from edge.demo_model import main


def run() -> None:
    main()


if __name__ == "__main__":
    run()
