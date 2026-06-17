"""
estimator.py — Thin shim. Pose estimation is now handled inside PlayerTracker.

PlayerTracker._run_pose() runs YOLOv8m-pose on the full frame once every
POSE_EVERY_N frames and matches keypoints to tracked players by IoU.
This is far more efficient than per-player crop inference.

This file is kept for pipeline.py import compatibility only.
If you want to swap in a different pose backend (MMPose, ViTPose),
implement it in tracker.py _run_pose() — not here.
"""

import numpy as np
from typing import Optional


class PoseEstimator:
    """
    Shim — keypoints come pre-attached from PlayerTracker via det["keypoints"].
    pipeline.py reads them directly from the detection dict.
    get_keypoints() is no longer called in the hot path.
    """

    def get_keypoints(
        self, frame: np.ndarray, bbox: list[int]
    ) -> Optional[np.ndarray]:
        # Not used — keypoints come from tracker._run_pose()
        return None
