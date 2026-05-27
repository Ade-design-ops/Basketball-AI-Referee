import math
import numpy as np

try:
    from ultralytics import YOLO
    ML_AVAILABLE = True
except ImportError:
    ML_AVAILABLE = False


class PlayerTracker:
    def __init__(self):
        self.prev_centers: dict = {}
        self.velocities: dict = {}
        self.frame_count = 0

        if ML_AVAILABLE:
            self.model = YOLO("yolov8n.pt")
            print("[✓] YOLOv8n loaded")
        else:
            self.model = None
            print("[!] Demo mode — simulated player tracking")

    def track(self, frame: np.ndarray) -> tuple[list[dict], dict | None]:
        self.frame_count += 1
        if self.model is None:
            return self._demo_detections(frame)

        results = self.model.track(frame, persist=True, conf=0.4, classes=[0, 32], verbose=False, imgsz=320)
        players = []
        ball = None

        if results[0].boxes is None:
            return players, ball

        # Extract ball from all detections (no persistent ID needed)
        for box in results[0].boxes:
            if int(box.cls[0]) == 32:
                x1, y1, x2, y2 = [int(v) for v in box.xyxy[0].tolist()]
                ball = {
                    "bbox": [x1, y1, x2, y2],
                    "center": [(x1 + x2) // 2, (y1 + y2) // 2],
                    "conf": float(box.conf[0]),
                }
                break  # take highest-confidence ball

        # Extract players (need persistent IDs for velocity)
        if results[0].boxes.id is not None:
            for box, tid in zip(results[0].boxes, results[0].boxes.id):
                if int(box.cls[0]) != 0:
                    continue
                x1, y1, x2, y2 = [int(v) for v in box.xyxy[0].tolist()]
                tid = int(tid)
                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

                if tid in self.prev_centers:
                    px, py = self.prev_centers[tid]
                    vx, vy = cx - px, cy - py
                    prev_v = self.velocities.get(tid, [0.0, 0.0])
                    vx = 0.6 * vx + 0.4 * prev_v[0]
                    vy = 0.6 * vy + 0.4 * prev_v[1]
                else:
                    vx, vy = 0.0, 0.0

                self.prev_centers[tid] = (cx, cy)
                self.velocities[tid] = [vx, vy]
                players.append({
                    "track_id": tid,
                    "bbox": [x1, y1, x2, y2],
                    "conf": float(box.conf[0]),
                    "velocity": [vx, vy],
                })

        return players, ball

    def _demo_detections(self, frame: np.ndarray) -> tuple[list[dict], dict | None]:
        h, w = frame.shape[:2]
        t = self.frame_count * 0.05
        players = [
            {
                "track_id": 1,
                "bbox": [int(w * 0.3 + 30 * math.sin(t)), int(h * 0.4),
                         int(w * 0.3 + 30 * math.sin(t) + 80), int(h * 0.4 + 180)],
                "conf": 0.9,
                "velocity": [30 * math.cos(t) * 0.05, 0.0],
            },
            {
                "track_id": 2,
                "bbox": [int(w * 0.5 + 20 * math.cos(t * 0.8)), int(h * 0.4),
                         int(w * 0.5 + 20 * math.cos(t * 0.8) + 80), int(h * 0.4 + 180)],
                "conf": 0.9,
                "velocity": [-20 * math.sin(t * 0.8) * 0.05, 0.0],
            },
            {
                "track_id": 3,
                "bbox": [int(w * 0.65), int(h * 0.4), int(w * 0.65 + 80), int(h * 0.4 + 180)],
                "conf": 0.85,
                "velocity": [0.0, 0.0],
            },
        ]
        bx = int(w * 0.4 + 60 * math.sin(t * 1.5))
        by = int(h * 0.55 + 20 * abs(math.cos(t * 2)))
        ball = {
            "bbox": [bx - 12, by - 12, bx + 12, by + 12],
            "center": [bx, by],
            "conf": 0.9,
        }
        return players, ball
