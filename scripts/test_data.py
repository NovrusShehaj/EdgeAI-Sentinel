import cv2
import numpy as np
import os

def create_sample(path_img, path_lbl, i):
    img = np.zeros((640, 640, 3), dtype=np.uint8)

    # random rectangle (fake object)
    x1, y1 = np.random.randint(50, 300, 2)
    x2, y2 = x1 + 100, y1 + 100

    cv2.rectangle(img, (x1, y1), (x2, y2), (255, 255, 255), -1)

    # YOLO format (class x_center y_center width height)
    xc = ((x1 + x2) / 2) / 640
    yc = ((y1 + y2) / 2) / 640
    w = (x2 - x1) / 640
    h = (y2 - y1) / 640

    with open(path_lbl, "w") as f:
        f.write(f"0 {xc} {yc} {w} {h}")

    cv2.imwrite(path_img, img)

for i in range(50):
    create_sample(f"data/train/images/{i}.jpg", f"data/train/labels/{i}.txt", i)

for i in range(10):
    create_sample(f"data/val/images/{i}.jpg", f"data/val/labels/{i}.txt", i)
