"""
pipeline.py — Frame orchestration (compatible with this repo's WebSocket API).

Changes from original:
- Reads is_ball_handler + keypoints directly from tracker output (no recompute)
- Ball trajectory trail drawn on screen
- Foul banners persist for FOUL_DISPLAY_FRAMES and fade out
- HUD at bottom showing frame/player/foul stats
- Returns raw numpy 'annotated' key for local run_camera.py use
- Traveling + double_dribble foul colors added
"""

import base64
import cv2
import numpy as np
from dataclasses import asdict

from src.tracking.tracker import PlayerTracker, ML_AVAILABLE
from src.violations.detector import RuleBasedFoulDetector, PlayerState, FoulEvent

# COCO 17-keypoint skeleton edges
SKELETON_EDGES = [
    (0, 1), (0, 2), (1, 3), (2, 4),
    (5, 6),
    (5, 7), (7, 9),
    (6, 8), (8, 10),
    (5, 11), (6, 12),
    (11, 12),
    (11, 13), (13, 15),
    (12, 14), (14, 16),
]

FOUL_BGR = {
    "blocking":       (0,  100, 255),
    "charging":       (0,  215, 255),
    "hand_check":     (255, 200,  0),
    "shooting_foul":  (0,   50, 255),
    "reach_in":       (0,  255, 150),
    "illegal_screen": (200,  0, 255),
    "traveling":      (50, 255,  50),
    "double_dribble": (255,  50,  50),
}

FOUL_DISPLAY_FRAMES = 60   # ~2 sec @ 30fps


class FrameProcessor:
    def __init__(self):
        self.tracker  = PlayerTracker()
        self.detector = RuleBasedFoulDetector()
        self.frame_count = 0
        self.foul_log: list[FoulEvent] = []

        # (FoulEvent, frames_remaining)
        self._active_banners: list[tuple[FoulEvent, int]] = []

    # ── Public API ─────────────────────────────────────────────────────────

    def process(self, frame: np.ndarray) -> dict:
        self.frame_count += 1

        # 1. Track — keypoints pre-attached by tracker
        raw_detections, ball = self.tracker.track(frame)

        # 2. Build PlayerState list
        players: list[PlayerState] = []
        for det in raw_detections:
            kps = det.get("keypoints")
            kps_list = kps.tolist() if kps is not None else []
            players.append(PlayerState(
                track_id=det["track_id"],
                bbox=det["bbox"],
                keypoints=kps_list,
                velocity=det["velocity"],
                is_ball_handler=det.get("is_ball_handler", False),
            ))

        # 3. Detect violations
        foul_events = self.detector.update(players, self.frame_count)
        self.foul_log.extend(foul_events)

        # 4. Update banners
        for ev in foul_events:
            self._active_banners.append((ev, FOUL_DISPLAY_FRAMES))
        self._active_banners = [
            (ev, remaining - 1)
            for ev, remaining in self._active_banners
            if remaining > 1
        ]

        # 5. Draw
        annotated = self._draw_overlays(frame.copy(), players, ball)

        # 6. Encode for WebSocket transport
        _, buf = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 80])
        frame_b64 = base64.b64encode(buf).decode()

        return {
            "frame":         frame_b64,
            "annotated":     annotated,      # raw numpy — used by run_camera.py
            "player_count":  len(players),
            "ball_detected": self.tracker.ball.center is not None,
            "ball_center":   list(self.tracker.ball.center) if self.tracker.ball.center else None,
            "fouls":         [asdict(ev) for ev in foul_events],
            "foul_log":      [asdict(ev) for ev in self.foul_log[-20:]],
            "frame_number":  self.frame_count,
            "demo_mode":     not ML_AVAILABLE,
        }

    def reset_log(self) -> None:
        self.foul_log.clear()
        self._active_banners.clear()
        self.detector = RuleBasedFoulDetector()

    # ── Drawing ────────────────────────────────────────────────────────────

    def _draw_overlays(
        self,
        frame: np.ndarray,
        players: list[PlayerState],
        ball: dict | None,
    ) -> np.ndarray:

        active_foul_ids = {
            pid
            for ev, _ in self._active_banners
            for pid in ev.player_ids
        }

        # Players
        for player in players:
            x1, y1, x2, y2 = player.bbox
            is_foul   = player.track_id in active_foul_ids
            is_ball_h = player.is_ball_handler

            if is_foul:
                color, thickness = (0, 60, 255), 3
            elif is_ball_h:
                color, thickness = (0, 215, 255), 2   # gold = ball handler
            else:
                color, thickness = (0, 220, 120), 2

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)

            label = f"#{player.track_id}"
            if is_ball_h:
                label += " [BALL]"
            cv2.putText(frame, label, (x1, y1 - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)

            # Skeleton
            if player.keypoints:
                bw, bh = x2 - x1, y2 - y1
                kp_pixels = [
                    (
                        int(x1 + kp[0] * bw),
                        int(y1 + kp[1] * bh),
                        kp[2] if len(kp) > 2 else 1.0,
                    )
                    for kp in player.keypoints
                ]
                for a, b in SKELETON_EDGES:
                    if a < len(kp_pixels) and b < len(kp_pixels):
                        if kp_pixels[a][2] > 0.35 and kp_pixels[b][2] > 0.35:
                            cv2.line(frame, kp_pixels[a][:2], kp_pixels[b][:2],
                                     (180, 180, 255), 1)
                for px, py, vis in kp_pixels:
                    if vis > 0.35:
                        cv2.circle(frame, (px, py), 3, (255, 255, 100), -1)

        # Ball + trajectory trail
        ball_tracker = self.tracker.ball
        if ball_tracker.center:
            trail = ball_tracker.trajectory
            for i in range(1, len(trail)):
                alpha = i / len(trail)
                trail_color = (
                    int(0   * alpha),
                    int(140 * alpha),
                    int(255 * alpha),
                )
                cv2.line(frame, trail[i - 1], trail[i], trail_color, 2)

            bx, by = ball_tracker.center
            cv2.circle(frame, (bx, by), 16, (0, 140, 255), 2)
            cv2.circle(frame, (bx, by),  3, (0, 200, 255), -1)
            cv2.putText(
                frame,
                f"BALL  {ball_tracker.speed:.0f}px/f",
                (bx + 18, by + 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 200, 255), 1,
            )

        # Foul banners (fade out)
        for i, (ev, remaining) in enumerate(self._active_banners):
            foul_color = FOUL_BGR.get(ev.foul_type, (0, 0, 255))
            banner_h = 56
            y_off = 12 + i * (banner_h + 8)
            alpha = min(1.0, remaining / 15.0)

            overlay = frame.copy()
            cv2.rectangle(overlay, (10, y_off), (520, y_off + banner_h), (10, 10, 30), -1)
            cv2.addWeighted(overlay, 0.75 * alpha, frame, 1 - 0.75 * alpha, 0, frame)
            cv2.rectangle(frame, (10, y_off), (520, y_off + banner_h), foul_color, 2)
            cv2.putText(
                frame,
                f"FOUL: {ev.foul_type.upper().replace('_', ' ')}",
                (20, y_off + 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.68, foul_color, 2,
            )
            cv2.putText(
                frame,
                f"{int(ev.confidence * 100)}% conf  |  {ev.explanation[:55]}",
                (20, y_off + 44),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, (200, 200, 255), 1,
            )

        # HUD
        h, w = frame.shape[:2]
        cv2.putText(
            frame,
            f"Frame {self.frame_count}  |  Players: {len(players)}"
            f"  |  Fouls: {len(self.foul_log)}"
            f"  |  {'DEMO' if not ML_AVAILABLE else 'LIVE'}",
            (10, h - 12),
            cv2.FONT_HERSHEY_SIMPLEX, 0.42, (160, 160, 160), 1,
        )

        return frame
