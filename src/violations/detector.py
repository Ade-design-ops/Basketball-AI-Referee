"""
detector.py — Rule-based foul + violation detection.

New violations vs original:
- traveling:      ankle keypoint step-counting with bbox displacement fallback
- double_dribble: full state machine (dribbling → held → dribble again = foul)

Changes vs original:
- update() now accepts frame_num as parameter (cleaner than internal counter)
- _make_event() helper removes repetitive FoulEvent construction
- All keypoint thresholds in pixels (not normalized) for consistency
- Cooldown moved to per-type dict with explicit frame tracking
"""

import time
import numpy as np
from collections import deque
from dataclasses import dataclass
from typing import Optional


# ── Foul catalogue ────────────────────────────────────────────────────────────

FOUL_TYPES = [
    "no_foul", "blocking", "charging", "hand_check",
    "shooting_foul", "reach_in", "illegal_screen",
    "traveling", "double_dribble",
]

FOUL_EXPLANATIONS = {
    "blocking":        "Defender was NOT in legal guarding position at moment of contact.",
    "charging":        "Offensive player ran into a stationary, legally-positioned defender.",
    "hand_check":      "Defender used hand or forearm to impede the ball-handler's movement.",
    "shooting_foul":   "Contact was made with the shooter's arm/body during the shooting motion.",
    "reach_in":        "Defender reached in and made illegal contact on the ball or ball-handler.",
    "illegal_screen":  "Screener moved, widened stance, or extended arms to create illegal contact.",
    "traveling":       "Ball-handler took more than 2 steps without dribbling.",
    "double_dribble":  "Player dribbled, picked up the ball, then dribbled again.",
}

FOUL_SEVERITY = {
    "blocking": 2, "charging": 2, "hand_check": 1,
    "shooting_foul": 3, "reach_in": 1, "illegal_screen": 2,
    "traveling": 2, "double_dribble": 2,
}

COOLDOWN_FRAMES = 45   # ~1.5s @ 30fps before same foul type fires again

# COCO-17 keypoint indices
KP_L_SHOULDER = 5;  KP_R_SHOULDER = 6
KP_L_ELBOW    = 7;  KP_R_ELBOW    = 8
KP_L_WRIST    = 9;  KP_R_WRIST    = 10
KP_L_HIP      = 11; KP_R_HIP      = 12
KP_L_KNEE     = 13; KP_R_KNEE     = 14
KP_L_ANKLE    = 15; KP_R_ANKLE    = 16


# ── Data classes ──────────────────────────────────────────────────────────────

@dataclass
class PlayerState:
    track_id: int
    bbox: list[int]
    keypoints: list          # list of [x_norm, y_norm, conf] — 17 COCO keypoints
    velocity: list[float]    # [vx, vy] pixels/frame
    is_ball_handler: bool = False


@dataclass
class FoulEvent:
    foul_type: str
    confidence: float
    explanation: str
    severity: int
    timestamp: float
    player_ids: list[int]
    frame_number: int


# ── Detector ──────────────────────────────────────────────────────────────────

class RuleBasedFoulDetector:

    def __init__(self, history_len: int = 30):
        self.history: deque = deque(maxlen=history_len)
        self.cooldown: dict[str, int] = {}
        self.frame_num: int = 0

        # Per-player state for ball-handling violations
        # {track_id: {"dribbling": bool, "steps": int, "held_frames": int,
        #             "dribbled_before_hold": bool, "prev_ankle_y": float|None}}
        self._player_state: dict[int, dict] = {}

    # ── Public ────────────────────────────────────────────────────────────────

    def update(
        self,
        players: list[PlayerState],
        frame_num: Optional[int] = None,
    ) -> list[FoulEvent]:
        if frame_num is not None:
            self.frame_num = frame_num
        else:
            self.frame_num += 1

        self.history.append(players)
        events: list[FoulEvent] = []

        if len(self.history) < 5:
            return events

        for rule_fn in [
            self._check_blocking,
            self._check_charging,
            self._check_hand_check,
            self._check_shooting_foul,
            self._check_reach_in,
            self._check_illegal_screen,
            self._check_traveling,
            self._check_double_dribble,
        ]:
            ev = rule_fn(players)
            if ev and self._cooldown_ok(ev.foul_type):
                events.append(ev)
                self.cooldown[ev.foul_type] = self.frame_num

        return events

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _cooldown_ok(self, foul_type: str) -> bool:
        return (self.frame_num - self.cooldown.get(foul_type, -9999)) > COOLDOWN_FRAMES

    def _make_event(
        self,
        foul_type: str,
        confidence: float,
        player_ids: list[int],
    ) -> FoulEvent:
        return FoulEvent(
            foul_type=foul_type,
            confidence=min(0.97, confidence),
            explanation=FOUL_EXPLANATIONS[foul_type],
            severity=FOUL_SEVERITY[foul_type],
            timestamp=time.time(),
            player_ids=player_ids,
            frame_number=self.frame_num,
        )

    def _get_kp(self, player: PlayerState, idx: int) -> Optional[np.ndarray]:
        """Return absolute pixel keypoint (x,y) if confidence > 0.30, else None."""
        if not player.keypoints or len(player.keypoints) <= idx:
            return None
        kp = player.keypoints[idx]
        if len(kp) < 3 or kp[2] < 0.30:
            return None
        x1, y1, x2, y2 = player.bbox
        return np.array(
            [x1 + kp[0] * (x2 - x1), y1 + kp[1] * (y2 - y1)],
            dtype=np.float32,
        )

    def _center(self, player: PlayerState) -> np.ndarray:
        b = player.bbox
        return np.array([(b[0] + b[2]) / 2, (b[1] + b[3]) / 2], dtype=np.float32)

    def _distance(self, a: PlayerState, b: PlayerState) -> float:
        return float(np.linalg.norm(self._center(a) - self._center(b)))

    def _bbox_overlap(self, a: list, b: list) -> float:
        ix1 = max(a[0], b[0]); iy1 = max(a[1], b[1])
        ix2 = min(a[2], b[2]); iy2 = min(a[3], b[3])
        inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
        if inter == 0:
            return 0.0
        area_a = (a[2] - a[0]) * (a[3] - a[1])
        area_b = (b[2] - b[0]) * (b[3] - b[1])
        return inter / (area_a + area_b - inter + 1e-6)

    def _speed(self, player: PlayerState) -> float:
        return float(np.linalg.norm(player.velocity))

    def _prev(self, track_id: int, n: int = 5) -> Optional[PlayerState]:
        history_list = list(self.history)
        if len(history_list) <= n:
            return None
        frame = history_list[-(n + 1)]
        return next((p for p in frame if p.track_id == track_id), None)

    def _init_player_state(self, track_id: int) -> None:
        if track_id not in self._player_state:
            self._player_state[track_id] = {
                "dribbling":           False,
                "steps":               0,
                "held_frames":         0,
                "dribbled_before_hold": False,
                "prev_ankle_y":        None,
            }

    # ── Contact fouls ─────────────────────────────────────────────────────────

    def _check_blocking(self, players: list[PlayerState]) -> Optional[FoulEvent]:
        if len(players) < 2:
            return None
        for i, defender in enumerate(players):
            for j, attacker in enumerate(players):
                if i == j:
                    continue
                if self._bbox_overlap(defender.bbox, attacker.bbox) < 0.08:
                    continue
                def_lateral = abs(defender.velocity[0])
                att_forward = self._speed(attacker)
                if def_lateral > 4.0 and att_forward > 2.0:
                    conf = min(0.95, 0.55 + def_lateral * 0.04)
                    return self._make_event("blocking", conf,
                                           [defender.track_id, attacker.track_id])
        return None

    def _check_charging(self, players: list[PlayerState]) -> Optional[FoulEvent]:
        if len(players) < 2:
            return None
        for i, attacker in enumerate(players):
            for j, defender in enumerate(players):
                if i == j:
                    continue
                if self._bbox_overlap(attacker.bbox, defender.bbox) < 0.08:
                    continue
                att_speed = self._speed(attacker)
                def_speed = self._speed(defender)
                if att_speed > 8.0 and def_speed < 2.5:
                    conf = min(0.95, 0.50 + (att_speed - 8.0) * 0.03)
                    return self._make_event("charging", conf,
                                           [attacker.track_id, defender.track_id])
        return None

    def _check_hand_check(self, players: list[PlayerState]) -> Optional[FoulEvent]:
        if len(players) < 2:
            return None
        for i, defender in enumerate(players):
            wrists = [
                self._get_kp(defender, KP_L_WRIST),
                self._get_kp(defender, KP_R_WRIST),
            ]
            for j, ball_handler in enumerate(players):
                if i == j or not ball_handler.is_ball_handler:
                    continue
                if self._distance(defender, ball_handler) > 120:
                    continue
                bx1, by1, bx2, by2 = ball_handler.bbox
                for wrist in wrists:
                    if wrist is None:
                        continue
                    wx, wy = wrist
                    if bx1 < wx < bx2 and by1 < wy < by2:
                        return self._make_event("hand_check", 0.82,
                                               [defender.track_id, ball_handler.track_id])
        return None

    def _check_shooting_foul(self, players: list[PlayerState]) -> Optional[FoulEvent]:
        if len(self.history) < 8:
            return None
        for shooter in players:
            rw_now = self._get_kp(shooter, KP_R_WRIST)
            lw_now = self._get_kp(shooter, KP_L_WRIST)
            prev   = self._prev(shooter.track_id, n=8)
            if prev is None:
                continue
            rw_prev = self._get_kp(prev, KP_R_WRIST)
            lw_prev = self._get_kp(prev, KP_L_WRIST)

            wrist_rise = 0.0
            if rw_now is not None and rw_prev is not None:
                wrist_rise = max(wrist_rise, float(rw_prev[1] - rw_now[1]))
            if lw_now is not None and lw_prev is not None:
                wrist_rise = max(wrist_rise, float(lw_prev[1] - lw_now[1]))

            if wrist_rise < 15:   # pixels
                continue

            for defender in players:
                if defender.track_id == shooter.track_id:
                    continue
                if self._bbox_overlap(defender.bbox, shooter.bbox) > 0.05:
                    conf = min(0.93, 0.60 + wrist_rise * 0.005)
                    return self._make_event("shooting_foul", conf,
                                           [defender.track_id, shooter.track_id])
        return None

    def _check_reach_in(self, players: list[PlayerState]) -> Optional[FoulEvent]:
        if len(self.history) < 5:
            return None
        for defender in players:
            prev = self._prev(defender.track_id, n=5)
            if prev is None:
                continue
            for elbow_idx, wrist_idx in [
                (KP_L_ELBOW, KP_L_WRIST),
                (KP_R_ELBOW, KP_R_WRIST),
            ]:
                elbow_now  = self._get_kp(defender, elbow_idx)
                wrist_now  = self._get_kp(defender, wrist_idx)
                elbow_prev = self._get_kp(prev, elbow_idx)
                wrist_prev = self._get_kp(prev, wrist_idx)

                if any(x is None for x in [elbow_now, wrist_now, elbow_prev, wrist_prev]):
                    continue

                ext_now  = float(np.linalg.norm(wrist_now  - elbow_now))
                ext_prev = float(np.linalg.norm(wrist_prev - elbow_prev))
                delta    = ext_now - ext_prev

                if delta < 8:   # pixels of arm extension increase
                    continue

                for ball_handler in players:
                    if ball_handler.track_id == defender.track_id:
                        continue
                    if ball_handler.is_ball_handler and self._distance(defender, ball_handler) < 150:
                        conf = min(0.90, 0.55 + delta * 0.01)
                        return self._make_event("reach_in", conf,
                                               [defender.track_id, ball_handler.track_id])
        return None

    def _check_illegal_screen(self, players: list[PlayerState]) -> Optional[FoulEvent]:
        if len(self.history) < 10:
            return None
        for screener in players:
            prev = self._prev(screener.track_id, n=10)
            if prev is None:
                continue
            prev_speed = float(np.linalg.norm(prev.velocity))
            curr_speed = self._speed(screener)
            if prev_speed > 3.0 or curr_speed < 3.5:
                continue
            for other in players:
                if other.track_id == screener.track_id:
                    continue
                if self._bbox_overlap(screener.bbox, other.bbox) > 0.06:
                    return self._make_event("illegal_screen", 0.74,
                                           [screener.track_id, other.track_id])
        return None

    # ── Ball-handling violations ──────────────────────────────────────────────

    def _check_traveling(self, players: list[PlayerState]) -> Optional[FoulEvent]:
        """
        Detect traveling: ball-handler takes >2 steps without dribbling.
        Uses ankle keypoints for step counting; falls back to bbox displacement.
        """
        for player in players:
            if not player.is_ball_handler:
                continue
            self._init_player_state(player.track_id)
            state = self._player_state[player.track_id]

            l_ankle = self._get_kp(player, KP_L_ANKLE)
            r_ankle = self._get_kp(player, KP_R_ANKLE)

            if l_ankle is not None and r_ankle is not None:
                prev = self._prev(player.track_id, n=3)
                if prev is not None:
                    pl_ankle = self._get_kp(prev, KP_L_ANKLE)
                    pr_ankle = self._get_kp(prev, KP_R_ANKLE)
                    if pl_ankle is not None and pr_ankle is not None:
                        if abs(float(l_ankle[1] - pl_ankle[1])) > 6:
                            state["steps"] += 1
                        if abs(float(r_ankle[1] - pr_ankle[1])) > 6:
                            state["steps"] += 1
            else:
                # fallback: count by overall player displacement
                if self._speed(player) > 4.0:
                    state["steps"] += 1

            if state["steps"] > 4:   # >2 full steps
                state["steps"] = 0
                return self._make_event("traveling", 0.78, [player.track_id])

        # reset step count for non-ball-handlers
        for player in players:
            if not player.is_ball_handler and player.track_id in self._player_state:
                self._player_state[player.track_id]["steps"] = 0

        return None

    def _check_double_dribble(self, players: list[PlayerState]) -> Optional[FoulEvent]:
        """
        Double dribble state machine:
          DRIBBLING → (picks up ball) → HOLDING → (dribbles again) → DOUBLE DRIBBLE
        """
        for player in players:
            if not player.is_ball_handler:
                continue
            self._init_player_state(player.track_id)
            state = self._player_state[player.track_id]

            # heuristic: vertical velocity + overall speed → dribbling
            vy = abs(player.velocity[1])
            is_dribbling_now = vy > 3.0 or self._speed(player) > 5.0

            if is_dribbling_now:
                if not state["dribbling"]:
                    # just transitioned from holding → dribbling
                    if state["dribbled_before_hold"]:
                        state["dribbling"] = True
                        state["dribbled_before_hold"] = False
                        state["held_frames"] = 0
                        return self._make_event("double_dribble", 0.75, [player.track_id])
                    state["dribbled_before_hold"] = True
                state["dribbling"] = True
                state["held_frames"] = 0
            else:
                if state["dribbling"]:
                    state["held_frames"] = 0
                state["dribbling"] = False
                state["held_frames"] += 1

        return None
