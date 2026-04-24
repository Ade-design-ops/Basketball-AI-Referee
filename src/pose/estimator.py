import numpy as np
from typing import Optional

# MediaPipe BlazePose → COCO-17 keypoint index mapping
MP_TO_COCO = [0, 2, 5, 7, 8, 11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28]

class PoseEstimator:
    """
    Stub estimator for local CPU dev — always returns None (no keypoints).
    Velocity + bbox foul rules still work without keypoints.
    Replace with MMPose on the GPU machine for full keypoint support.
    """

    def get_keypoints(self, frame: np.ndarray, bbox: list[int]) -> Optional[np.ndarray]:
        return None
