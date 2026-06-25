import sqlite3
from datetime import datetime
from ultralytics import YOLO
import cv2

model = YOLO("yolov8n.pt")
DB_NAME = "detections.db"

conn = sqlite3.connect(DB_NAME)
cursor = conn.cursor()

cursor.execute("""
CREATE TABLE IF NOT EXISTS detections (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    object_name TEXT,
    detected_at TIMESTAMP
)
""")

conn.commit()
conn.close()

camera = cv2.VideoCapture(0)

while True:
    success, frame = camera.read()

    if not success:
        break

    results = model(frame, imgsz=320, verbose=False)

    annotated_frame = results[0].plot()

    object_counts = {}

    for box in results[0].boxes:
        class_id = int(box.cls[0])
        class_name = model.names[class_id]

        if class_name not in object_counts:
            object_counts[class_name] = 0

        object_counts[class_name] += 1
        conn = sqlite3.connect(DB_NAME)
        cursor = conn.cursor()

        cursor.execute(
            "INSERT INTO detections (object_name, detected_at) VALUES (?, ?)",
            (class_name, datetime.now())
        )

        conn.commit()
        conn.close()

    y_position = 30

    for obj, count in object_counts.items():
        cv2.putText(
            annotated_frame,
            f"{obj}: {count}",
            (10, y_position),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 255),
            2
        )

        y_position += 30

    cv2.imshow("VisionAssist AI - YOLO Detection", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

camera.release()
cv2.destroyAllWindows()