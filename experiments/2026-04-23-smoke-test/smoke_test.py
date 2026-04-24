"""
Smoke test: YOLOv8n inference on a synthetic frame.
Verifies the local pipeline (detection + tracking) works without a GPU.
"""

import numpy as np
import cv2
from ultralytics import YOLO
from loguru import logger


def make_test_frame() -> np.ndarray:
    """Create a simple 640x480 BGR frame with a white rectangle (stand-in for a person)."""
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    cv2.rectangle(frame, (200, 100), (440, 460), (200, 200, 200), -1)
    return frame


def run_smoke_test():
    logger.info("Loading YOLOv8n model...")
    model = YOLO("yolov8n.pt")  # downloads ~6MB on first run
    logger.info(f"Model loaded: {model.info()}")

    frame = make_test_frame()
    logger.info(f"Test frame shape: {frame.shape}")

    logger.info("Running inference...")
    results = model(frame, verbose=False)

    boxes = results[0].boxes
    logger.info(f"Detections: {len(boxes)}")
    for box in boxes:
        cls_id = int(box.cls[0])
        cls_name = model.names[cls_id]
        conf = float(box.conf[0])
        xyxy = box.xyxy[0].tolist()
        logger.info(f"  [{cls_name}] conf={conf:.2f}  box={[round(x) for x in xyxy]}")

    logger.info("Smoke test passed.")


if __name__ == "__main__":
    run_smoke_test()
