"""
Basketball AI Referee — Live Backend
=====================================
WebSocket endpoint: browser streams webcam frames → server runs detection → pushes annotated frames + foul events back.

Run:
    uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload
"""

import json
import base64
import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

from src.pipeline import FrameProcessor
from src.violations.detector import FOUL_TYPES
from src.tracking.tracker import ML_AVAILABLE

app = FastAPI(title="Basketball AI Referee — Live", version="2.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

processor = FrameProcessor()


@app.websocket("/ws/live")
async def websocket_live(ws: WebSocket):
    await ws.accept()
    print("[WS] Client connected")
    try:
        while True:
            data = await ws.receive_text()
            msg = json.loads(data)

            if msg.get("type") == "frame":
                img_bytes = base64.b64decode(msg["data"])
                arr = np.frombuffer(img_bytes, np.uint8)
                import cv2
                frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
                if frame is None:
                    continue
                result = processor.process(frame)
                await ws.send_text(json.dumps(result))

            elif msg.get("type") == "reset_log":
                processor.reset_log()
                await ws.send_text(json.dumps({"type": "log_reset"}))

    except WebSocketDisconnect:
        print("[WS] Client disconnected")
    except Exception as e:
        print(f"[WS ERROR] {e}")


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "ml_available": ML_AVAILABLE,
        "demo_mode": not ML_AVAILABLE,
        "foul_types": FOUL_TYPES,
    }
