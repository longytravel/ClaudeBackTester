"""FastAPI dashboard server — WebSocket + REST for optimization monitoring.

Runs on a background daemon thread inside the optimizer process.
Broadcasts progress events to connected WebSocket clients.
"""

import asyncio
import json
import logging
import os
import threading
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse

logger = logging.getLogger(__name__)


class DashboardServer:
    """Manages the FastAPI server lifecycle and WebSocket broadcasting."""

    def __init__(self, port: int = 8765, static_dir: str | None = None):
        self.port = port
        self.static_dir = static_dir
        self._clients: list[Any] = []  # WebSocket connections
        self._lock = threading.Lock()
        self._last_broadcast_time = 0.0
        self._throttle_interval = 0.1  # 100ms min between broadcasts
        self._pending_msg: dict | None = None  # Holds throttled message
        self._loop: asyncio.AbstractEventLoop | None = None
        self._thread: threading.Thread | None = None
        self._state_snapshot: dict | None = None  # Latest full state for new connections
        self._batch_history: list[dict] = []  # All batch updates for reconnect replay
        self._run_config: dict | None = None
        self._stage_results: list[dict] = []
        self._pipeline_results: list[dict] = []
        self._final_report: dict | None = None
        self._app = self._create_app()

    def _create_app(self):
        app = FastAPI(title="Backtester Dashboard")

        app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_methods=["*"],
            allow_headers=["*"],
        )

        @app.websocket("/ws/optimization")
        async def ws_optimization(websocket: WebSocket):
            await websocket.accept()
            with self._lock:
                self._clients.append(websocket)
            try:
                # Send state snapshot on connect
                snapshot = self._build_snapshot()
                if snapshot:
                    await websocket.send_json(snapshot)

                # Keep connection alive, listen for pings
                while True:
                    try:
                        await asyncio.wait_for(websocket.receive_text(), timeout=30)
                    except asyncio.TimeoutError:
                        # Send ping to keep alive
                        try:
                            await websocket.send_json({"type": "ping"})
                        except Exception:
                            break
            except WebSocketDisconnect:
                pass
            except Exception:
                pass
            finally:
                with self._lock:
                    if websocket in self._clients:
                        self._clients.remove(websocket)

        @app.get("/api/run/status")
        async def run_status():
            if self._final_report:
                status = "complete"
            elif self._run_config:
                status = "running"
            else:
                status = "idle"
            return JSONResponse({"status": status})

        @app.get("/api/runs")
        async def list_runs():
            results_dir = Path("results")
            runs = []
            if results_dir.exists():
                for entry in sorted(results_dir.iterdir()):
                    report_path = entry / "report.json"
                    if entry.is_dir() and report_path.exists():
                        try:
                            with open(report_path) as f:
                                report = json.load(f)
                            best = None
                            for c in report.get("candidates", []):
                                if not c.get("eliminated"):
                                    if best is None or c.get("composite_score", 0) > best.get("composite_score", 0):
                                        best = c
                            runs.append({
                                "id": entry.name,
                                "strategy": report.get("strategy", ""),
                                "pair": report.get("pair", ""),
                                "timeframe": report.get("timeframe", ""),
                                "rating": best.get("rating") if best else None,
                                "composite_score": best.get("composite_score") if best else None,
                                "timestamp": entry.stat().st_mtime,
                            })
                        except Exception:
                            continue
            return JSONResponse(runs)

        @app.get("/api/runs/{run_id}/report")
        async def get_report(run_id: str):
            report_path = Path("results") / run_id / "report.json"
            if not report_path.exists():
                return JSONResponse({"error": "not found"}, status_code=404)
            with open(report_path) as f:
                return JSONResponse(json.load(f))

        # Serve static frontend as a catch-all GET route (NOT app.mount)
        # app.mount("/", StaticFiles(...)) would shadow WebSocket routes
        if self.static_dir and os.path.isdir(self.static_dir):
            @app.get("/{full_path:path}")
            async def serve_spa(full_path: str):
                file_path = os.path.join(self.static_dir, full_path)
                if full_path and os.path.isfile(file_path):
                    return FileResponse(file_path)
                index_path = os.path.join(self.static_dir, "index.html")
                if os.path.isfile(index_path):
                    return FileResponse(index_path)
                return JSONResponse({"error": "not found"}, status_code=404)

        return app

    def _build_snapshot(self) -> dict | None:
        """Build a state snapshot for newly connected clients."""
        if not self._run_config:
            if self._final_report:
                return {"type": "run_complete", "report": self._final_report}
            return None

        return {
            "type": "snapshot",
            "run_config": self._run_config,
            "stage_results": self._stage_results,
            "pipeline_results": self._pipeline_results,
            "last_state": self._state_snapshot,
            "batch_history": self._batch_history,
            "final_report": self._final_report,
        }

    def broadcast(self, msg: dict) -> None:
        """Broadcast a message to all connected WebSocket clients.

        Throttles to max 5 messages/sec. The final message per stage
        is always sent (via flush on stage_complete).
        """
        now = time.time()
        elapsed = now - self._last_broadcast_time

        # Store for snapshot
        msg_type = msg.get("type")
        if msg_type == "run_config":
            self._run_config = msg
            self._batch_history = []
            self._stage_results = []
            self._pipeline_results = []
            self._final_report = None
        elif msg_type == "batch":
            self._state_snapshot = msg
            self._batch_history.append(msg)
        elif msg_type == "stage_complete":
            self._stage_results.append(msg)
        elif msg_type == "pipeline":
            self._pipeline_results.append(msg)
        elif msg_type == "run_complete":
            self._final_report = msg.get("report")

        # Throttle batch updates (but always send non-batch messages)
        if msg_type == "batch" and elapsed < self._throttle_interval:
            self._pending_msg = msg
            return

        self._do_broadcast(msg)
        self._last_broadcast_time = time.time()
        self._pending_msg = None

    def flush(self) -> None:
        """Send any pending throttled message."""
        if self._pending_msg:
            self._do_broadcast(self._pending_msg)
            self._pending_msg = None

    def _do_broadcast(self, msg: dict) -> None:
        """Actually send message to all clients."""
        if not self._clients or not self._loop:
            return

        data = json.dumps(msg, default=str)

        async def _send_all():
            with self._lock:
                clients = list(self._clients)
            dead = []
            for ws in clients:
                try:
                    await ws.send_text(data)
                except Exception:
                    dead.append(ws)
            if dead:
                with self._lock:
                    for ws in dead:
                        if ws in self._clients:
                            self._clients.remove(ws)

        try:
            asyncio.run_coroutine_threadsafe(_send_all(), self._loop)
        except Exception:
            pass

    def start(self) -> None:
        """Start the server on a background daemon thread."""
        import uvicorn

        def _run():
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            self._loop = loop

            config = uvicorn.Config(
                self._app,
                host="0.0.0.0",
                port=self.port,
                log_level="warning",
                loop="asyncio",
            )
            server = uvicorn.Server(config)
            loop.run_until_complete(server.serve())

        self._thread = threading.Thread(target=_run, daemon=True)
        self._thread.start()

        # Give server a moment to start
        time.sleep(0.5)
        logger.info(f"Dashboard server started on http://localhost:{self.port}")

    def stop(self) -> None:
        """Stop the server (best-effort, it's a daemon thread)."""
        self._loop = None
        logger.info("Dashboard server stopped")
