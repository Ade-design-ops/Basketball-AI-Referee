"""
tracker.py — Player + Ball tracking with YOLOv8m and pose estimation.

Changes from original:
- Upgraded from yolov8n → yolov8m for better accuracy
- Added BallTracker class with Kalman-style smoothing + trajectory history
- Ball detection runs as a separate low-threshold pass (conf=0.20)
- Ball handler assigned by bbox containment → proximity, NOT speed
- Pose estimation runs on full frame once every N frames (efficient)
- Keypoints matched to tracked players by IoU
- ByteTrack specified explicitly for robust multi-person tracking
"""

import math
import numpy as np
from collections import deque
from typing import Optional

try:
    from ultralytics import YOLO
    ML_AVAILABLE = True
except ImportError:
    ML_AVAILABLE = False


def _box_iou(a: list, b: list) -> float:
    ix1 = max(a[0], b[0])
    iy1 = max(a[1], b[1])
    ix2 = min(a[2], b[2])
    iy2 = min(a[3], b[3])
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    if inter == 0:
        return 0.0
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    return inter / (area_a + area_b - inter + 1e-6)


class BallTracker:
    """Smoothed ball tracker with occlusion tolerance."""

    SMOOTH_ALPHA = 0.6
    MAX_MISSING  = 10

    def __init__(self, history_len: int = 30):
        self.position: Optional[list[int]] = None
        self.center:   Optional[tuple[int, int]] = None
        self.velocity: list[float] = [0.0, 0.0]
        self.history:  deque = deque(maxlen=history_len)
        self.frames_since_seen: int = 0
        self._smooth_cx: Optional[float] = None
        self._smooth_cy: Optional[float] = None
        self._prev_center: Optional[tuple] = None

    def update(self, raw_box: Optional[list[int]]) -> None:
        if raw_box is not None:
            x1, y1, x2, y2 = raw_box
            cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0

            if self._smooth_cx is None:
                self._smooth_cx, self._smooth_cy = cx, cy
            else:
                self._smooth_cx = self.SMOOTH_ALPHA * cx + (1 - self.SMOOTH_ALPHA) * self._smooth_cx
                self._smooth_cy = self.SMOOTH_ALPHA * cy + (1 - self.SMOOTH_ALPHA) * self._smooth_cy

            if self._prev_center is not None:
                self.velocity = [
                    self._smooth_cx - self._prev_center[0],
                    self._smooth_cy - self._prev_center[1],
                ]
            self._prev_center = (self._smooth_cx, self._smooth_cy)

            icx, icy = int(self._smooth_cx), int(self._smooth_cy)
            r = max(1, (x2 - x1 + y2 - y1) // 4)
            self.position = [icx - r, icy - r, icx + r, icy + r]
            self.center = (icx, icy)
            self.frames_since_seen = 0
            self.history.append((icx, icy))
        else:
            self.frames_since_seen += 1
            if self.frames_since_seen > self.MAX_MISSING:
                self.position = None
                self.center = None
                self._smooth_cx = None
                self._smooth_cy = None
                self._prev_center = None

    @property
    def speed(self) -> float:
        return float(np.linalg.norm(self.velocity))

    @property
    def trajectory(self) -> list[tuple[int, int]]:
        return list(self.history)

    def as_dict(self) -> Optional[dict]:
        if self.center is None:
            return None
        return {
            "bbox":   self.position,
            "center": list(self.center),
            "conf":   1.0,
        }


class PlayerTracker:
    """
    Tracks players + ball using YOLOv8m + ByteTrack.
    Runs YOLOv8m-pose on full frame every POSE_EVERY_N frames.
    Assigns ball handler by bbox containment then proximity.
    """

    BALL_HANDLER_DIST = 130
    POSE_EVERY_N      = 2

    def __init__(self):
        self.prev_centers: dict[int, tuple] = {}
        self.velocities:   dict[int, list]  = {}
        self.bboxes:       dict[int, list]  = {}
        self.frame_count:  int = 0
        self.ball = BallTracker()
        self._cached_keypoints: dict[int, np.ndarray] = {}

        if ML_AVAILABLE:
            self.det_model  = YOLO("yolov8m.pt")
            self.pose_model = YOLO("yolov8m-pose.pt")
            print("[✓] YOLOv8m (detection) loaded")
            print("[✓] YOLOv8m-pose loaded")
        else:
            self.det_model  = None
            self.pose_model = None
            print("[!] Demo mode — ML unavailable")

    def track(self, frame: np.ndarray) -> tuple[list[dict], dict | None]:
        self.frame_count += 1
        if self.det_model is None:
            return self._demo_detections(frame)

        # 1. Player detection + ByteTrack
        det_results = self.det_model.track(
            frame,
            persist=True,
            conf=0.45,
            iou=0.5,
            classes=[0],
            tracker="bytetrack.yaml",
            verbose=False,
        )

        detections: list[dict] = []

        if (
            det_results[0].boxes is not None
            and det_results[0].boxes.id is not None
        ):
            for box, tid in zip(det_results[0].boxes, det_results[0].boxes.id):
                x1, y1, x2, y2 = [int(v) for v in box.xyxy[0].tolist()]
                tid = int(tid)
                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

                if tid in self.prev_centers:
                    px, py = self.prev_centers[tid]
                    raw_vx, raw_vy = cx - px, cy - py
                    prev_v = self.velocities.get(tid, [0.0, 0.0])
                    vx = 0.6 * raw_vx + 0.4 * prev_v[0]
                    vy = 0.6 * raw_vy + 0.4 * prev_v[1]
                else:
                    vx, vy = 0.0, 0.0

                self.prev_centers[tid] = (cx, cy)
                self.velocities[tid]   = [vx, vy]
                self.bboxes[tid]       = [x1, y1, x2, y2]

                detections.append({
                    "track_id":        tid,
                    "bbox":            [x1, y1, x2, y2],
                    "conf":            float(box.conf[0]),
                    "velocity":        [vx, vy],
                    "keypoints":       None,
                    "is_ball_handler": False,
                    "dist_to_ball":    9999.0,
                })

        # 2. Ball detection — separate low-threshold pass
        ball_results = self.det_model(
            frame,
            conf=0.20,
            classes=[32],
            verbose=False,
        )

        raw_ball_box = None
        if ball_results[0].boxes is not None and len(ball_results[0].boxes) > 0:
            best = max(ball_results[0].boxes, key=lambda b: float(b.conf[0]))
            bx1, by1, bx2, by2 = [int(v) for v in best.xyxy[0].tolist()]
            raw_ball_box = [bx1, by1, bx2, by2]

        self.ball.update(raw_ball_box)

        # 3. Assign ball handler: containment → proximity
        if self.ball.center and detections:
            bcx, bcy = self.ball.center
            for det in detections:
                x1, y1, x2, y2 = det["bbox"]
                det["dist_to_ball"] = float(
                    np.linalg.norm([
                        (x1 + x2) / 2 - bcx,
                        (y1 + y2) / 2 - bcy,
                    ])
                )

            containing = [
                d for d in detections
                if d["bbox"][0] <= bcx <= d["bbox"][2]
                and d["bbox"][1] <= bcy <= d["bbox"][3]
            ]
            if containing:
                handler = min(containing, key=lambda d: d["dist_to_ball"])
            else:
                closest = min(detections, key=lambda d: d["dist_to_ball"])
                handler = closest if closest["dist_to_ball"] < self.BALL_HANDLER_DIST else None

            if handler:
                handler["is_ball_handler"] = True

        # 4. Pose estimation — full frame, every N frames
        if self.frame_count % self.POSE_EVERY_N == 0:
            self._cached_keypoints = self._run_pose(frame, detections)

        for det in detections:
            det["keypoints"] = self._cached_keypoints.get(det["track_id"])

        return detections, self.ball.as_dict()

    def _run_pose(
        self, frame: np.ndarray, detections: list[dict]
    ) -> dict[int, np.ndarray]:
        if self.pose_model is None or not detections:
            return {}

        pose_results = self.pose_model(frame, verbose=False)
        if pose_results[0].keypoints is None or pose_results[0].boxes is None:
            return {}

        pose_boxes = pose_results[0].boxes.xyxy.cpu().numpy()
        pose_kps   = pose_results[0].keypoints.data.cpu().numpy()

        matched: dict[int, np.ndarray] = {}

        for det in detections:
            best_iou = 0.0
            best_kps = None

            for i, pb in enumerate(pose_boxes):
                iou = _box_iou(det["bbox"], pb.tolist())
                if iou > best_iou:
                    best_iou = iou
                    best_kps = pose_kps[i]

            if best_kps is not None and best_iou > 0.25:
                x1, y1, x2, y2 = det["bbox"]
                bw = max(1, x2 - x1)
                bh = max(1, y2 - y1)
                norm = best_kps.copy()
                norm[:, 0] = (best_kps[:, 0] - x1) / bw
                norm[:, 1] = (best_kps[:, 1] - y1) / bh
                matched[det["track_id"]] = norm

        return matched

    def _demo_detections(
        self, frame: np.ndarray
    ) -> tuple[list[dict], dict | None]:
        h, w = frame.shape[:2]
        t = self.frame_count * 0.05

        players = [
            {
                "track_id":        1,
                "bbox":            [
                    int(w * 0.3 + 30 * math.sin(t)), int(h * 0.4),
                    int(w * 0.3 + 30 * math.sin(t) + 80), int(h * 0.4 + 180),
                ],
                "conf":            0.9,
                "velocity":        [30 * math.cos(t) * 0.05, 0.0],
                "keypoints":       None,
                "is_ball_handler": True,
                "dist_to_ball":    0.0,
            },
            {
                "track_id":        2,
                "bbox":            [
                    int(w * 0.55 + 20 * math.cos(t * 0.8)), int(h * 0.4),
                    int(w * 0.55 + 20 * math.cos(t * 0.8) + 80), int(h * 0.4 + 180),
                ],
                "conf":            0.9,
                "velocity":        [-20 * math.sin(t * 0.8) * 0.05, 0.0],
                "keypoints":       None,
                "is_ball_handler": False,
                "dist_to_ball":    200.0,
            },
        ]

        bx = int(w * 0.4 + 60 * math.sin(t * 1.5))
        by = int(h * 0.55 + 20 * abs(math.cos(t * 2)))
        ball_dict = {
            "bbox":   [bx - 12, by - 12, bx + 12, by + 12],
            "center": [bx, by],
            "conf":   0.9,
        }
        return players, ball_dict
