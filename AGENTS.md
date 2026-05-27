# Basketball Ref AI

## Project Overview

**Goal:** Real-time AI basketball referee — detect traveling, double dribbles, carries, and charging fouls from live video.

**Team:** Two developers  
**Current phase:** Phase 0 — environment setup and planning

## Dev Environment

- **Dev machine:** MacBook Air (Apple Silicon) — no CUDA, no TAO/DeepStream locally
- **Deployment target:** Linux GPU machine or cloud GPU (for ML training and inference)

## Tech Stack

- **Language:** Python 3.10
- **Detection/Tracking:** YOLOv8/v11 (Ultralytics), ByteTrack
- **Pose estimation:** MMPose
- **ML training/optimization:** NVIDIA TAO Toolkit, TensorRT
- **Inference pipeline:** NVIDIA DeepStream SDK
- **Backend:** FastAPI
- **Frontend:** Next.js
- **Streaming:** WebRTC
- **Access:** NVIDIA Developer Program, Codex Pro

## Conventions

- Ask before installing any dependency over 1GB
- Commit after each working milestone
- Keep experiments in dated folders under `experiments/` (e.g. `experiments/2026-04-23-pose-baseline/`)

## Open Questions

1. **GPU machine decision** — build a local Linux box or rent cloud GPU (e.g. Lambda, Vast.ai, RunPod)?
2. **Data collection** — what source for pickup game footage to train and validate violation detection?
3. **Camera hardware** — what camera to use for the live demo (webcam, GoPro, fixed court cam)?
