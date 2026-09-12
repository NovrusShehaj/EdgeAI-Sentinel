"""Known-coordinate tests for letterbox inversion."""

from __future__ import annotations

import numpy as np
import pytest

from edge.inference import (
    LetterboxTransform,
    decode_yolo_output,
    invert_letterbox_xyxy,
    letterbox_image,
)


def _forward_letterbox_box(x1, y1, x2, y2, transform: LetterboxTransform):
    return (
        x1 / transform.x_scale + transform.pad_x,
        y1 / transform.y_scale + transform.pad_y,
        x2 / transform.x_scale + transform.pad_x,
        y2 / transform.y_scale + transform.pad_y,
    )


@pytest.mark.parametrize(
    ("height", "width"),
    [
        (640, 640),
        (720, 1280),
        (640, 480),
    ],
)
def test_letterbox_inverse_roundtrip(height, width):
    image = np.zeros((height, width, 3), dtype=np.uint8)
    _, transform = letterbox_image(image, 640)
    x1, y1, x2, y2 = 40.0, 50.0, 180.0, 200.0
    lb_x1, lb_y1, lb_x2, lb_y2 = _forward_letterbox_box(x1, y1, x2, y2, transform)
    inv_x1, inv_y1, inv_x2, inv_y2 = invert_letterbox_xyxy(
        np.array([lb_x1]),
        np.array([lb_y1]),
        np.array([lb_x2]),
        np.array([lb_y2]),
        transform,
    )
    assert inv_x1[0] == pytest.approx(x1, abs=1e-4)
    assert inv_y1[0] == pytest.approx(y1, abs=1e-4)
    assert inv_x2[0] == pytest.approx(x2, abs=1e-4)
    assert inv_y2[0] == pytest.approx(y2, abs=1e-4)


def test_square_640_known_box():
    image = np.zeros((640, 640, 3), dtype=np.uint8)
    _, transform = letterbox_image(image, 640)
    assert transform.pad_x == 0
    assert transform.pad_y == 0
    assert transform.x_scale == pytest.approx(1.0)

    raw = np.zeros((1, 5, 1), dtype=np.float32)
    raw[0, :, 0] = [320.0, 320.0, 100.0, 80.0, 0.95]
    detections = decode_yolo_output(raw, transform, ["object"], 0.4, 0.5)
    assert len(detections) == 1
    assert detections[0].class_name == "object"
    assert detections[0].x1 == pytest.approx(270.0)
    assert detections[0].y1 == pytest.approx(280.0)
    assert detections[0].x2 == pytest.approx(370.0)
    assert detections[0].y2 == pytest.approx(360.0)


def test_widescreen_1280x720_known_box():
    image = np.zeros((720, 1280, 3), dtype=np.uint8)
    _, transform = letterbox_image(image, 640)
    assert transform.pad_x == 0
    assert transform.pad_y == pytest.approx(140.0)
    assert transform.x_scale == pytest.approx(2.0)
    assert transform.y_scale == pytest.approx(2.0)

    # Original box (200, 100, 400, 300) maps to letterbox (100, 190, 200, 290)
    raw = np.zeros((1, 5, 1), dtype=np.float32)
    raw[0, :, 0] = [150.0, 240.0, 100.0, 100.0, 0.99]
    detections = decode_yolo_output(raw, transform, ["object"], 0.4, 0.5)
    assert len(detections) == 1
    assert detections[0].x1 == pytest.approx(200.0)
    assert detections[0].y1 == pytest.approx(100.0)
    assert detections[0].x2 == pytest.approx(400.0)
    assert detections[0].y2 == pytest.approx(300.0)


def test_portrait_480x640_known_box():
    image = np.zeros((640, 480, 3), dtype=np.uint8)
    _, transform = letterbox_image(image, 640)
    assert transform.pad_x == pytest.approx(80.0)
    assert transform.pad_y == 0
    assert transform.x_scale == pytest.approx(1.0)
    assert transform.y_scale == pytest.approx(1.0)

    raw = np.zeros((1, 5, 1), dtype=np.float32)
    raw[0, :, 0] = [180.0, 200.0, 40.0, 60.0, 0.88]
    detections = decode_yolo_output(raw, transform, ["object"], 0.4, 0.5)
    assert detections[0].x1 == pytest.approx(80.0)
    assert detections[0].y1 == pytest.approx(170.0)
    assert detections[0].x2 == pytest.approx(120.0)
    assert detections[0].y2 == pytest.approx(230.0)


def test_coordinates_are_clipped_to_original_frame():
    transform = LetterboxTransform(
        scale=1.0,
        pad_x=0,
        pad_y=0,
        orig_w=100,
        orig_h=80,
        new_w=100,
        new_h=80,
        target=640,
    )
    x1, y1, x2, y2 = invert_letterbox_xyxy(
        np.array([-20.0]),
        np.array([-10.0]),
        np.array([400.0]),
        np.array([300.0]),
        transform,
    )
    assert x1[0] == 0
    assert y1[0] == 0
    assert x2[0] == 100
    assert y2[0] == 80
