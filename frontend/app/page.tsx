"use client";

import { useEffect, useRef, useState, useCallback } from "react";

interface FoulEvent {
  foul_type: string;
  confidence: number;
  explanation: string;
  severity: number;
  timestamp: number;
  player_ids: number[];
  frame_number: number;
}

interface ServerMessage {
  frame: string;
  player_count: number;
  fouls: FoulEvent[];
  foul_log: FoulEvent[];
  frame_number: number;
  demo_mode: boolean;
}

const FOUL_COLORS: Record<string, string> = {
  blocking: "bg-orange-500",
  charging: "bg-yellow-400",
  hand_check: "bg-blue-400",
  shooting_foul: "bg-red-500",
  reach_in: "bg-green-400",
  illegal_screen: "bg-purple-500",
};

export default function Home() {
  const webcamRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const displayRef = useRef<HTMLImageElement>(null);
  const wsRef = useRef<WebSocket | null>(null);
  const animRef = useRef<number>(0);

  const [connected, setConnected] = useState(false);
  const [running, setRunning] = useState(false);
  const [playerCount, setPlayerCount] = useState(0);
  const [foulLog, setFoulLog] = useState<FoulEvent[]>([]);
  const [activeFoul, setActiveFoul] = useState<FoulEvent | null>(null);
  const [demoMode, setDemoMode] = useState(false);
  const [fps, setFps] = useState(0);
  const frameTimesRef = useRef<number[]>([]);

  const connect = useCallback(() => {
    const ws = new WebSocket("ws://localhost:8000/ws/live");
    ws.onopen = () => setConnected(true);
    ws.onclose = () => { setConnected(false); setRunning(false); };
    ws.onmessage = (e) => {
      const data: ServerMessage = JSON.parse(e.data);
      if (displayRef.current) {
        displayRef.current.src = `data:image/jpeg;base64,${data.frame}`;
      }
      setPlayerCount(data.player_count);
      setDemoMode(data.demo_mode);
      setFoulLog(data.foul_log.slice().reverse());
      if (data.fouls.length > 0) {
        setActiveFoul(data.fouls[0]);
        setTimeout(() => setActiveFoul(null), 2500);
      }
      const now = Date.now();
      frameTimesRef.current.push(now);
      frameTimesRef.current = frameTimesRef.current.filter(t => now - t < 1000);
      setFps(frameTimesRef.current.length);
    };
    wsRef.current = ws;
  }, []);

  const startCamera = useCallback(async () => {
    const stream = await navigator.mediaDevices.getUserMedia({ video: true });
    if (webcamRef.current) {
      webcamRef.current.srcObject = stream;
      await webcamRef.current.play();
    }
  }, []);

  const lastSendRef = useRef<number>(0);
  const FRAME_INTERVAL = 150; // ms between sends (~7fps matches CPU inference speed)

  const sendFrames = useCallback(() => {
    const video = webcamRef.current;
    const canvas = canvasRef.current;
    const ws = wsRef.current;
    if (!video || !canvas || !ws || ws.readyState !== WebSocket.OPEN) return;

    const now = Date.now();
    if (now - lastSendRef.current >= FRAME_INTERVAL) {
      lastSendRef.current = now;
      const ctx = canvas.getContext("2d")!;
      canvas.width = 640;
      canvas.height = 480;
      ctx.drawImage(video, 0, 0, 640, 480);
      canvas.toBlob((blob) => {
        if (!blob) return;
        const reader = new FileReader();
        reader.onloadend = () => {
          const b64 = (reader.result as string).split(",")[1];
          ws.send(JSON.stringify({ type: "frame", data: b64 }));
        };
        reader.readAsDataURL(blob);
      }, "image/jpeg", 0.7);
    }

    animRef.current = requestAnimationFrame(sendFrames);
  }, []);

  const handleStart = useCallback(async () => {
    if (!connected) connect();
    await startCamera();
    setRunning(true);
  }, [connected, connect, startCamera]);

  useEffect(() => {
    if (running) {
      animRef.current = requestAnimationFrame(sendFrames);
    } else {
      cancelAnimationFrame(animRef.current);
    }
    return () => cancelAnimationFrame(animRef.current);
  }, [running, sendFrames]);

  const resetLog = () => {
    wsRef.current?.send(JSON.stringify({ type: "reset_log" }));
    setFoulLog([]);
    setActiveFoul(null);
  };

  return (
    <main className="min-h-screen bg-gray-950 text-white flex flex-col">
      {/* Header */}
      <header className="flex items-center justify-between px-6 py-4 border-b border-gray-800">
        <div className="flex items-center gap-3">
          <div className="w-3 h-3 rounded-full bg-orange-500" />
          <h1 className="text-xl font-bold tracking-tight">Basketball AI Referee</h1>
          {demoMode && (
            <span className="text-xs bg-yellow-600 text-yellow-100 px-2 py-0.5 rounded">DEMO MODE</span>
          )}
        </div>
        <div className="flex items-center gap-4 text-sm text-gray-400">
          <span className={`flex items-center gap-1.5 ${connected ? "text-green-400" : "text-red-400"}`}>
            <span className={`w-2 h-2 rounded-full ${connected ? "bg-green-400" : "bg-red-400"}`} />
            {connected ? "Connected" : "Disconnected"}
          </span>
          <span>{fps} fps</span>
          <span>{playerCount} players</span>
        </div>
      </header>

      <div className="flex flex-1 overflow-hidden">
        {/* Video panel */}
        <div className="flex-1 flex flex-col items-center justify-center p-6 gap-4">
          <div className="relative w-full max-w-2xl aspect-video bg-gray-900 rounded-xl overflow-hidden border border-gray-800">
            {running ? (
              // eslint-disable-next-line @next/next/no-img-element
              <img ref={displayRef} className="w-full h-full object-cover" alt="live feed" />
            ) : (
              <div className="absolute inset-0 flex flex-col items-center justify-center gap-3 text-gray-500">
                <div className="text-5xl">🏀</div>
                <p className="text-sm">Camera feed will appear here</p>
              </div>
            )}

            {activeFoul && (
              <div className="absolute top-4 left-4 right-4 bg-red-600/90 backdrop-blur rounded-lg px-4 py-3 animate-pulse">
                <p className="font-bold text-lg uppercase tracking-wide">
                  {activeFoul.foul_type.replace(/_/g, " ")}
                </p>
                <p className="text-sm text-red-200">{activeFoul.explanation}</p>
                <p className="text-xs text-red-300 mt-1">{Math.round(activeFoul.confidence * 100)}% confidence</p>
              </div>
            )}
          </div>

          <video ref={webcamRef} className="hidden" muted playsInline />
          <canvas ref={canvasRef} className="hidden" />

          <div className="flex gap-3">
            {!running ? (
              <button
                onClick={handleStart}
                className="px-6 py-2.5 bg-orange-500 hover:bg-orange-400 text-white font-semibold rounded-lg transition"
              >
                Start Referee
              </button>
            ) : (
              <button
                onClick={() => setRunning(false)}
                className="px-6 py-2.5 bg-gray-700 hover:bg-gray-600 text-white font-semibold rounded-lg transition"
              >
                Stop
              </button>
            )}
            {!connected && (
              <button
                onClick={connect}
                className="px-6 py-2.5 bg-gray-700 hover:bg-gray-600 text-white font-semibold rounded-lg transition"
              >
                Connect to Server
              </button>
            )}
          </div>
        </div>

        {/* Foul log sidebar */}
        <aside className="w-80 border-l border-gray-800 flex flex-col">
          <div className="flex items-center justify-between px-4 py-3 border-b border-gray-800">
            <h2 className="font-semibold text-sm text-gray-300">Foul Log</h2>
            <button onClick={resetLog} className="text-xs text-gray-500 hover:text-gray-300 transition">
              Clear
            </button>
          </div>
          <div className="flex-1 overflow-y-auto p-3 flex flex-col gap-2">
            {foulLog.length === 0 ? (
              <p className="text-gray-600 text-sm text-center mt-8">No fouls detected yet</p>
            ) : (
              foulLog.map((foul, i) => (
                <div key={i} className="bg-gray-900 rounded-lg p-3 border border-gray-800">
                  <div className="flex items-center gap-2 mb-1">
                    <span className={`w-2 h-2 rounded-full ${FOUL_COLORS[foul.foul_type] ?? "bg-gray-400"}`} />
                    <span className="font-semibold text-sm uppercase tracking-wide">
                      {foul.foul_type.replace(/_/g, " ")}
                    </span>
                    <span className="ml-auto text-xs text-gray-500">
                      {Math.round(foul.confidence * 100)}%
                    </span>
                  </div>
                  <p className="text-xs text-gray-400">{foul.explanation}</p>
                  <p className="text-xs text-gray-600 mt-1">
                    Players {foul.player_ids.join(" & ")} · Frame {foul.frame_number}
                  </p>
                </div>
              ))
            )}
          </div>
        </aside>
      </div>
    </main>
  );
}
