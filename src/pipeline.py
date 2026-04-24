import base64
import cv2
import numpy as np
from dataclasses import asdict

from src.tracking.tracker import PlayerTracker, ML_AVAILABLE
from src.pose.estimator import PoseEstimator
from src.violations.detector import RuleBasedFoulDetector, PlayerState, FoulEvent

# COCO 17-keypoint skeleton edges for drawing
SKELETON_EDGES = [
    (0, 1), (0, 2), (1, 3), (2, 4), (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),
    (5, 11), (6, 12), (11, 12), (11, 13), (13, 15), (12, 14), (14, 16),
]

FOUL_BGR = {
    "blocking":       (0, 100, 255),
    "charging":       (0, 215, 255),
    "hand_check":     (255, 200, 0),
    "shooting_foul":  (0, 50, 255),
    "reach_in":       (0, 255, 150),
    "illegal_screen": (200, 0, 255),
}


class FrameProcessor:
    def __init__(self):
        self.tracker = PlayerTracker()
        self.estimator = PoseEstimator() if ML_AVAILABLE else None
        self.detector = RuleBasedFoulDetector()
        self.frame_count = 0
        self.foul_log: list[FoulEvent] = []
        self.POSE_EVERY_N = 3  # run pose estimation every 3rd frame for performance

    def process(self, frame: np.ndarray) -> dict:
        self.frame_count += 1

        raw_detections = self.tracker.track(frame)

        players: list[PlayerState] = []
        for det in raw_detections:
            kps = None
            if self.estimator and (self.frame_count % self.POSE_EVERY_N == 0):
                kps_arr = self.estimator.get_keypoints(frame, det["bbox"])
                if kps_arr is not None:
                    kps = kps_arr.tolist()

            speed = abs(det["velocity"][0])
            players.append(PlayerState(
                track_id=det["track_id"],
                bbox=det["bbox"],
                keypoints=kps or [],
                velocity=det["velocity"],
                is_ball_handler=(
                    speed == max((abs(d["velocity"][0]) for d in raw_detections), default=0)
                ),
            ))

        foul_events = self.detector.update(players)
        self.foul_log.extend(foul_events)

        annotated = self._draw_overlays(frame.copy(), players, foul_events)
        _, buf = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 70])
        frame_b64 = base64.b64encode(buf).decode()

        return {
            "frame": frame_b64,
            "player_count": len(players),
            "fouls": [asdict(ev) for ev in foul_events],
            "foul_log": [asdict(ev) for ev in self.foul_log[-20:]],
            "frame_number": self.frame_count,
            "demo_mode": not ML_AVAILABLE,
        }

    def _draw_overlays(
        self,
        frame: np.ndarray,
        players: list[PlayerState],
        foul_events: list[FoulEvent],
    ) -> np.ndarray:
        active_foul_ids = {pid for ev in foul_events for pid in ev.player_ids}

        for player in players:
            x1, y1, x2, y2 = player.bbox
            is_foul = player.track_id in active_foul_ids
            color = (0, 60, 255) if is_foul else (0, 220, 120)
            thickness = 3 if is_foul else 2

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)

            label = f"#{player.track_id}" + (" [BALL]" if player.is_ball_handler else "")
            cv2.putText(frame, label, (x1, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)

            if player.keypoints:
                bw, bh = x2 - x1, y2 - y1
                kp_pixels = [
                    (int(x1 + kp[0] * bw), int(y1 + kp[1] * bh), kp[2] if len(kp) > 2 else 1.0)
                    for kp in player.keypoints
                ]
                for a, b in SKELETON_EDGES:
                    if a < len(kp_pixels) and b < len(kp_pixels):
                        if kp_pixels[a][2] > 0.3 and kp_pixels[b][2] > 0.3:
                            cv2.line(frame,
                                     kp_pixels[a][:2], kp_pixels[b][:2],
                                     (180, 180, 255), 1)
                for px, py, vis in kp_pixels:
                    if vis > 0.3:
                        cv2.circle(frame, (px, py), 3, (255, 255, 100), -1)

        for i, ev in enumerate(foul_events):
            foul_color = FOUL_BGR.get(ev.foul_type, (0, 0, 255))
            banner_h = 54
            y_off = 12 + i * (banner_h + 8)
            overlay = frame.copy()
            cv2.rectangle(overlay, (10, y_off), (500, y_off + banner_h), (10, 10, 30), -1)
            cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)
            cv2.rectangle(frame, (10, y_off), (500, y_off + banner_h), foul_color, 2)
            cv2.putText(frame, f"FOUL: {ev.foul_type.upper().replace('_', ' ')}",
                        (20, y_off + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.65, foul_color, 2)
            cv2.putText(frame, f"{int(ev.confidence * 100)}% confidence",
                        (20, y_off + 40), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (200, 200, 255), 1)

        return frame

    def reset_log(self):
        self.foul_log.clear()
        self.detector = RuleBasedFoulDetector()
