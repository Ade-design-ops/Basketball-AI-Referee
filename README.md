# Basketball AI Referee

Real-time basketball referee prototype that uses computer vision to detect and track players from a live webcam feed, apply rule-based foul logic, and display annotated results in a browser dashboard.

The current version is a working local demo. It runs on a MacBook CPU for development, but the target production/demo setup is a Linux or cloud machine with an NVIDIA GPU for faster inference and future pose-estimation work.

## What It Does

- Streams webcam frames from a Next.js frontend to a FastAPI backend over WebSocket.
- Uses YOLOv8 through Ultralytics to detect players and the basketball.
- Uses ByteTrack-style tracking from Ultralytics to keep player IDs consistent across frames.
- Estimates player velocity from tracked bounding boxes.
- Applies rule-based foul detection for:
  - blocking
  - charging
  - hand check
  - shooting foul
  - reach in
  - illegal screen
- Draws bounding boxes, player IDs, ball markers, and foul banners onto the returned video feed.
- Shows a live foul log in the frontend.

## Current Architecture

```text
Browser webcam
    -> Next.js frontend
    -> WebSocket frame stream
    -> FastAPI backend
    -> FrameProcessor
    -> YOLOv8 player/ball detection
    -> tracking + velocity estimation
    -> rule-based foul detector
    -> annotated JPEG frame
    -> browser display + foul log
```

## Project Structure

```text
.
|-- src/
|   |-- api/
|   |   `-- main.py              # FastAPI app, health route, WebSocket endpoint
|   |-- tracking/
|   |   `-- tracker.py           # YOLOv8 detection and tracking
|   |-- pose/
|   |   `-- estimator.py         # Local pose stub; planned MMPose replacement
|   |-- violations/
|   |   `-- detector.py          # Rule-based foul detection
|   `-- pipeline.py              # Orchestrates frame processing end to end
|-- frontend/
|   |-- app/
|   |   `-- page.tsx             # Live referee UI
|   `-- package.json             # Next.js scripts and dependencies
|-- experiments/
|   `-- 2026-04-23-smoke-test/   # YOLO smoke test experiment
|-- data/
|   |-- raw/
|   |-- processed/
|   `-- annotations/
|-- notebooks/
|-- requirements.txt
`-- README.md
```

## Backend

The backend is a FastAPI app.

Main file:

```text
src/api/main.py
```

Important routes:

- `GET /health` - confirms the backend is running.
- `WS /ws/live` - receives base64-encoded webcam frames and returns annotated frames plus foul data.

The backend pipeline lives in:

```text
src/pipeline.py
```

It processes each frame by:

1. Running player and ball detection.
2. Assigning persistent track IDs.
3. Estimating velocity from movement across frames.
4. Inferring the likely ball handler.
5. Running foul rules.
6. Drawing overlays.
7. Returning the frame and foul log to the frontend.

## Frontend

The frontend is a Next.js app in:

```text
frontend/
```

It:

- requests camera access from the browser
- captures frames from the webcam
- downscales frames before sending them to the backend
- sends frames over WebSocket to `ws://localhost:8000/ws/live`
- displays the annotated video returned by the backend
- shows connection status, FPS, player count, and foul log

## Local Setup

### 1. Create and activate Python environment

```bash
python3.10 -m venv .venv
source .venv/bin/activate
```

### 2. Install Python dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the backend

```bash
uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload
```

Check the backend:

```text
http://localhost:8000/health
```

### 4. Run the frontend

Open a second terminal:

```bash
cd frontend
npm install
npm run dev
```

Open:

```text
http://localhost:3000
```

Then click `Start Referee` and allow camera access.

## Development Notes

- Run the backend and frontend in separate terminal tabs.
- The MacBook version is CPU-only, so it is slower than the target GPU setup.
- The frontend downscales frames and throttles frame sending to keep the live feed smoother on CPU.
- YOLO model weights such as `yolov8n.pt` are not committed. Ultralytics downloads them automatically when needed.
- Do not commit large datasets, model weights, or generated experiment outputs unless intentionally versioned.

## Current Limitations

- Pose estimation is currently a local development stub. It returns no keypoints.
- Keypoint-heavy foul rules, such as hand check, reach in, and shooting foul, need MMPose or another working pose model to become reliable.
- Current foul detection is rule-based, not a trained violation classifier.
- Ball detection uses the generic COCO basketball class, so it may be inconsistent on real court footage.
- Full real-time performance requires a GPU backend.

## Next Steps

1. Run the backend on a Linux/NVIDIA GPU machine.
2. Replace the pose stub with MMPose.
3. Test with real basketball footage instead of only webcam movement.
4. Collect and label basketball-specific data.
5. Train or fine-tune a custom player/ball detector.
6. Tune foul thresholds using recorded clips.
7. Deploy the frontend to Vercel.
8. Deploy the backend to a GPU host such as RunPod, Lambda Labs, or a partner's GPU machine.

## Collaboration Workflow

This project is tracked with Git.

Common commands:

```bash
git status
git add .
git commit -m "Describe what changed"
git push
git pull
```

Use commits as working checkpoints. Push to GitHub when the local code is stable enough for teammates to pull.

## Tech Stack

- Python 3.10
- FastAPI
- OpenCV
- NumPy
- Ultralytics YOLOv8
- ByteTrack tracking through Ultralytics
- Next.js
- TypeScript
- Tailwind CSS
- WebSocket streaming
