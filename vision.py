from ultralytics import YOLO
import cv2
import numpy as np
from collections import Counter

# Load YOLO model only once
model = YOLO("yolov8n.pt")


def detect_image(image):
    """
    Detect objects in an uploaded image.

    Returns:
    annotated_image
    object_counts
    detections
    """

    results = model(image, imgsz=640, verbose=False)

    result = results[0]

    annotated = result.plot()

    detections = []

    names = []

    for box in result.boxes:

        cls = int(box.cls[0])

        conf = float(box.conf[0])

        name = model.names[cls]

        detections.append({
            "object": name,
            "confidence": round(conf * 100, 2)
        })

        names.append(name)

    object_counts = dict(Counter(names))

    return annotated, object_counts, detections


def detect_frame(frame):
    """
    Detect objects in webcam frame.
    """

    results = model(frame, imgsz=320, verbose=False)

    result = results[0]

    annotated = result.plot()

    detections = []

    names = []

    for box in result.boxes:

        cls = int(box.cls[0])

        conf = float(box.conf[0])

        name = model.names[cls]

        detections.append({
            "object": name,
            "confidence": round(conf * 100, 2)
        })

        names.append(name)

    object_counts = dict(Counter(names))

    return annotated, object_counts, detections


def draw_summary(image, object_counts):
    """
    Draw object summary on image.
    """

    y = 30

    for obj, count in object_counts.items():

        cv2.putText(
            image,
            f"{obj}: {count}",
            (10, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2
        )

        y += 30

    return image