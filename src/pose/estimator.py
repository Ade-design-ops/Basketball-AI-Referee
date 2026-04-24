import numpy as np
from typing import Optional

# MediaPipe BlazePose → COCO-17 keypoint index mapping
MP_TO_COCO = [0, 2, 5, 7, 8, 11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28]

try:
    import mediapipe as mp
    _mp_pose = mp.solutions.pose
    ML_AVAILABLE = True
except ImportError:
    ML_AVAILABLE = False


class PoseEstimator:
    def __init__(self):
        if not ML_AVAILABLE:
            return
        self.pose = _mp_pose.Pose(
            static_image_mode=True,
            model_complexity=0,  # 0 = fastest for real-time
            min_detection_confidence=0.4,
        )

    def get_keypoints(self, frame: np.ndarray, bbox: list[int]) -> Optional[np.ndarray]:
        """Returns (17, 3) array of [x, y, visibility] in normalized bbox coords, or None."""
        if not ML_AVAILABLE:
            return None
        import cv2
        x1, y1, x2, y2 = [max(0, v) for v in bbox]
        h, w = frame.shape[:2]
        x2, y2 = min(w, x2), min(h, y2)
        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            return None
        rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        result = self.pose.process(rgb)
        if not result.pose_landmarks:
            return None
        kps = np.zeros((17, 3), dtype=np.float32)
        for ci, mi in enumerate(MP_TO_COCO):
            lm = result.pose_landmarks.landmark[mi]
            kps[ci] = [lm.x, lm.y, lm.visibility]
        return kps
