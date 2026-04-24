import time
import numpy as np
from collections import deque
from dataclasses import dataclass
from typing import Optional


FOUL_TYPES = [
    "no_foul", "blocking", "charging", "hand_check",
    "shooting_foul", "reach_in", "illegal_screen",
]

FOUL_EXPLANATIONS = {
    "blocking":        "Defender was NOT in legal guarding position at moment of contact.",
    "charging":        "Offensive player ran into a stationary, legally-positioned defender.",
    "hand_check":      "Defender used hand or forearm to impede the ball-handler's movement.",
    "shooting_foul":   "Contact was made with the shooter's arm/body during the shooting motion.",
    "reach_in":        "Defender reached in and made illegal contact on the ball or ball-handler.",
    "illegal_screen":  "Screener moved, widened stance, or extended arms to create illegal contact.",
}

FOUL_SEVERITY = {
    "blocking": 2, "charging": 2, "hand_check": 1,
    "shooting_foul": 3, "reach_in": 1, "illegal_screen": 2,
}


@dataclass
class PlayerState:
    track_id: int
    bbox: list[int]
    keypoints: list
    velocity: list[float]
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


class RuleBasedFoulDetector:

    def __init__(self, history_len: int = 20):
        self.history: deque = deque(maxlen=history_len)
        self.cooldown: dict = {}
        self.frame_num = 0
        self.COOLDOWN_FRAMES = 45  # ~1.5 sec @ 30fps between same foul type

    def update(self, players: list[PlayerState]) -> list[FoulEvent]:
        self.frame_num += 1
        self.history.append(players)
        events = []

        if len(self.history) < 5:
            return events

        for rule_fn in [
            self._check_blocking,
            self._check_charging,
            self._check_hand_check,
            self._check_shooting_foul,
            self._check_reach_in,
            self._check_illegal_screen,
        ]:
            event = rule_fn(players)
            if event and self._cooldown_ok(event.foul_type):
                events.append(event)
                self.cooldown[event.foul_type] = self.frame_num

        return events

    def _cooldown_ok(self, foul_type: str) -> bool:
        last = self.cooldown.get(foul_type, -999)
        return (self.frame_num - last) > self.COOLDOWN_FRAMES

    def _get_keypoint(self, player: PlayerState, idx: int) -> Optional[np.ndarray]:
        if not player.keypoints or len(player.keypoints) <= idx:
            return None
        kp = player.keypoints[idx]
        if kp[2] < 0.3:
            return None
        return np.array(kp[:2], dtype=np.float32)

    def _distance(self, p1: PlayerState, p2: PlayerState) -> float:
        c1 = [(p1.bbox[0] + p1.bbox[2]) / 2, (p1.bbox[1] + p1.bbox[3]) / 2]
        c2 = [(p2.bbox[0] + p2.bbox[2]) / 2, (p2.bbox[1] + p2.bbox[3]) / 2]
        return float(np.linalg.norm(np.array(c1) - np.array(c2)))

    def _bbox_overlap(self, a: list, b: list) -> float:
        ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
        ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
        inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
        if inter == 0:
            return 0.0
        area_a = (a[2] - a[0]) * (a[3] - a[1])
        area_b = (b[2] - b[0]) * (b[3] - b[1])
        return inter / (area_a + area_b - inter + 1e-6)

    def _player_speed(self, player: PlayerState) -> float:
        return float(np.linalg.norm(player.velocity))

    def _check_blocking(self, players: list[PlayerState]) -> Optional[FoulEvent]:
        if len(players) < 2:
            return None
        for i, def_player in enumerate(players):
            for j, off_player in enumerate(players):
                if i == j:
                    continue
                if self._bbox_overlap(def_player.bbox, off_player.bbox) < 0.08:
                    continue
                def_lateral_speed = abs(def_player.velocity[0])
                off_forward_speed = self._player_speed(off_player)
                if def_lateral_speed > 4.0 and off_forward_speed > 2.0:
                    conf = min(0.95, 0.55 + def_lateral_speed * 0.04)
                    return FoulEvent(
                        foul_type="blocking", confidence=conf,
                        explanation=FOUL_EXPLANATIONS["blocking"],
                        severity=FOUL_SEVERITY["blocking"],
                        timestamp=time.time(),
                        player_ids=[def_player.track_id, off_player.track_id],
                        frame_number=self.frame_num,
                    )
        return None

    def _check_charging(self, players: list[PlayerState]) -> Optional[FoulEvent]:
        if len(players) < 2:
            return None
        for i, off_player in enumerate(players):
            for j, def_player in enumerate(players):
                if i == j:
                    continue
                if self._bbox_overlap(off_player.bbox, def_player.bbox) < 0.08:
                    continue
                off_speed = self._player_speed(off_player)
                def_speed = self._player_speed(def_player)
                if off_speed > 8.0 and def_speed < 2.5:
                    conf = min(0.95, 0.5 + (off_speed - 8.0) * 0.03)
                    return FoulEvent(
                        foul_type="charging", confidence=conf,
                        explanation=FOUL_EXPLANATIONS["charging"],
                        severity=FOUL_SEVERITY["charging"],
                        timestamp=time.time(),
                        player_ids=[off_player.track_id, def_player.track_id],
                        frame_number=self.frame_num,
                    )
        return None

    def _check_hand_check(self, players: list[PlayerState]) -> Optional[FoulEvent]:
        if len(players) < 2:
            return None
        for i, def_player in enumerate(players):
            wrists = [self._get_keypoint(def_player, 9), self._get_keypoint(def_player, 10)]
            for j, off_player in enumerate(players):
                if i == j:
                    continue
                if self._distance(def_player, off_player) > 120:
                    continue
                bbox = off_player.bbox
                for wrist in wrists:
                    if wrist is None:
                        continue
                    bw = def_player.bbox[2] - def_player.bbox[0]
                    bh = def_player.bbox[3] - def_player.bbox[1]
                    wx = def_player.bbox[0] + wrist[0] * bw
                    wy = def_player.bbox[1] + wrist[1] * bh
                    if bbox[0] < wx < bbox[2] and bbox[1] < wy < bbox[3]:
                        if off_player.is_ball_handler:
                            return FoulEvent(
                                foul_type="hand_check", confidence=0.82,
                                explanation=FOUL_EXPLANATIONS["hand_check"],
                                severity=FOUL_SEVERITY["hand_check"],
                                timestamp=time.time(),
                                player_ids=[def_player.track_id, off_player.track_id],
                                frame_number=self.frame_num,
                            )
        return None

    def _check_shooting_foul(self, players: list[PlayerState]) -> Optional[FoulEvent]:
        if len(self.history) < 8:
            return None
        prev_states = list(self.history)[-8]
        for off_player in players:
            rw_now = self._get_keypoint(off_player, 10)
            prev_off = next((p for p in prev_states if p.track_id == off_player.track_id), None)
            if prev_off is None or rw_now is None:
                continue
            rw_prev = self._get_keypoint(prev_off, 10)
            if rw_prev is None:
                continue
            wrist_rise = rw_prev[1] - rw_now[1]
            if wrist_rise < 0.06:
                continue
            for def_player in players:
                if def_player.track_id == off_player.track_id:
                    continue
                if self._bbox_overlap(def_player.bbox, off_player.bbox) > 0.05:
                    conf = min(0.93, 0.60 + wrist_rise * 2.0)
                    return FoulEvent(
                        foul_type="shooting_foul", confidence=conf,
                        explanation=FOUL_EXPLANATIONS["shooting_foul"],
                        severity=FOUL_SEVERITY["shooting_foul"],
                        timestamp=time.time(),
                        player_ids=[def_player.track_id, off_player.track_id],
                        frame_number=self.frame_num,
                    )
        return None

    def _check_reach_in(self, players: list[PlayerState]) -> Optional[FoulEvent]:
        if len(self.history) < 5:
            return None
        prev_states = list(self.history)[-5]
        for def_player in players:
            prev_def = next((p for p in prev_states if p.track_id == def_player.track_id), None)
            if prev_def is None:
                continue
            elbow_now  = self._get_keypoint(def_player, 8)
            wrist_now  = self._get_keypoint(def_player, 10)
            elbow_prev = self._get_keypoint(prev_def, 8)
            wrist_prev = self._get_keypoint(prev_def, 10)
            if any(x is None for x in [elbow_now, wrist_now, elbow_prev, wrist_prev]):
                continue
            arm_extension_delta = (
                np.linalg.norm(wrist_now - elbow_now) - np.linalg.norm(wrist_prev - elbow_prev)
            )
            if arm_extension_delta < 0.06:
                continue
            for off_player in players:
                if off_player.track_id == def_player.track_id:
                    continue
                if off_player.is_ball_handler and self._distance(def_player, off_player) < 150:
                    conf = min(0.90, 0.55 + arm_extension_delta * 3.0)
                    return FoulEvent(
                        foul_type="reach_in", confidence=conf,
                        explanation=FOUL_EXPLANATIONS["reach_in"],
                        severity=FOUL_SEVERITY["reach_in"],
                        timestamp=time.time(),
                        player_ids=[def_player.track_id, off_player.track_id],
                        frame_number=self.frame_num,
                    )
        return None

    def _check_illegal_screen(self, players: list[PlayerState]) -> Optional[FoulEvent]:
        if len(self.history) < 10:
            return None
        prev_states = list(self.history)[-10]
        for screener in players:
            prev_screener = next((p for p in prev_states if p.track_id == screener.track_id), None)
            if prev_screener is None:
                continue
            prev_speed = np.linalg.norm(prev_screener.velocity)
            curr_speed = self._player_speed(screener)
            if prev_speed > 3.0 or curr_speed < 3.5:
                continue
            for off_player in players:
                if off_player.track_id == screener.track_id:
                    continue
                if self._bbox_overlap(screener.bbox, off_player.bbox) > 0.06:
                    return FoulEvent(
                        foul_type="illegal_screen", confidence=0.74,
                        explanation=FOUL_EXPLANATIONS["illegal_screen"],
                        severity=FOUL_SEVERITY["illegal_screen"],
                        timestamp=time.time(),
                        player_ids=[screener.track_id, off_player.track_id],
                        frame_number=self.frame_num,
                    )
        return None
