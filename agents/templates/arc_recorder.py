#!/usr/bin/env python3
"""
ARC Recorder — DVR for Apollo's puzzle solving.

Records game frames to SQLite so Peter can replay them in a browser.
Twitch-style commentary overlay from solver thought process.

Usage:
    # Record while solving:
    python3 arc_agent_v05.py --record

    # Replay viewer:
    python3 arc_recorder.py --serve [--port 8766]

    # List recordings:
    python3 arc_recorder.py --list
"""

import argparse
import io
import json
import logging
import os
import queue
import sqlite3
import sys
import threading
import time
from datetime import datetime
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

log = logging.getLogger('arc_recorder')

# ─── Live Streaming ───────────────────────────────────────────────────
# Thread-safe broadcast: GameRecorder pushes frames here, SSE endpoint reads.
# Works in-process (same Python). For cross-process (separate --serve),
# the SSE endpoint polls SQLite for new frames.

_live_subscribers: list[queue.Queue] = []
_live_lock = threading.Lock()
_live_session_info: dict = {}  # {session_id, game_id, started_at}


def _broadcast_frame(frame_data: dict):
    """Push a frame to all connected live viewers (in-process)."""
    with _live_lock:
        dead = []
        for i, q in enumerate(_live_subscribers):
            try:
                q.put_nowait(frame_data)
            except queue.Full:
                dead.append(i)
        for i in reversed(dead):
            _live_subscribers.pop(i)


def subscribe_live() -> queue.Queue:
    """Subscribe to the live frame stream. Returns a queue to read from."""
    q = queue.Queue(maxsize=100)
    with _live_lock:
        _live_subscribers.append(q)
    return q


def unsubscribe_live(q: queue.Queue):
    """Unsubscribe from the live stream."""
    with _live_lock:
        try:
            _live_subscribers.remove(q)
        except ValueError:
            pass

DEFAULT_DB = os.path.join(os.path.dirname(__file__), '..', 'data', 'arc_recordings.db')


# ─── Schema ────────────────────────────────────────────────────────────

SCHEMA = """
CREATE TABLE IF NOT EXISTS sessions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    game_id TEXT NOT NULL,
    started_at TEXT NOT NULL,
    finished_at TEXT,
    agent_version TEXT DEFAULT 'v0.5',
    total_levels INTEGER DEFAULT 0,
    levels_solved INTEGER DEFAULT 0,
    total_actions INTEGER DEFAULT 0,
    baseline_actions INTEGER DEFAULT 0,
    efficiency REAL,
    game_type TEXT,
    metadata TEXT DEFAULT '{}'
);

CREATE TABLE IF NOT EXISTS frames (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id INTEGER NOT NULL REFERENCES sessions(id),
    seq INTEGER NOT NULL,
    timestamp REAL NOT NULL,
    level INTEGER DEFAULT 0,
    action_id INTEGER,
    action_data TEXT,
    action_name TEXT,
    frame_blob BLOB NOT NULL,
    frame_w INTEGER NOT NULL,
    frame_h INTEGER NOT NULL,
    solver_route TEXT,
    is_win BOOLEAN DEFAULT 0,
    metadata TEXT DEFAULT '{}'
);

CREATE TABLE IF NOT EXISTS comments (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id INTEGER NOT NULL REFERENCES sessions(id),
    seq INTEGER,
    timestamp REAL NOT NULL,
    author TEXT NOT NULL DEFAULT 'apollo',
    text TEXT NOT NULL,
    comment_type TEXT DEFAULT 'thought',
    metadata TEXT DEFAULT '{}'
);

CREATE INDEX IF NOT EXISTS idx_frames_session ON frames(session_id, seq);
CREATE INDEX IF NOT EXISTS idx_comments_session ON comments(session_id, seq);
"""


_schema_initialized = set()  # track which DB paths have had schema applied

def _get_db(db_path: str = DEFAULT_DB) -> sqlite3.Connection:
    """Get a database connection, creating schema if needed."""
    os.makedirs(os.path.dirname(os.path.abspath(db_path)), exist_ok=True)
    conn = sqlite3.connect(db_path, check_same_thread=False, timeout=60)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    abs_path = os.path.abspath(db_path)
    if abs_path not in _schema_initialized:
        conn.executescript(SCHEMA)
        _schema_initialized.add(abs_path)
    return conn


def _compress_frame(frame: np.ndarray) -> bytes:
    """Compress a frame to bytes. Uses numpy's compact format."""
    f = np.squeeze(frame)
    if f.ndim == 1:
        side = int(np.sqrt(len(f)))
        f = f.reshape(side, side) if side * side == len(f) else f.reshape(1, -1)
    buf = io.BytesIO()
    np.save(buf, f.astype(np.uint8))
    return buf.getvalue()


def _decompress_frame(blob: bytes) -> np.ndarray:
    """Decompress a frame from bytes."""
    buf = io.BytesIO(blob)
    arr = np.load(buf)
    # Squeeze batch dimension if present (ARC SDK returns (1, H, W))
    if arr.ndim == 3 and arr.shape[0] == 1:
        arr = arr[0]
    return arr


# ─── Recorder ──────────────────────────────────────────────────────────

class GameRecorder:
    """Records a single game session to SQLite."""

    def __init__(self, game_id: str, db_path: str = DEFAULT_DB,
                 agent_version: str = 'v0.5'):
        self.db_path = db_path
        self.conn = _get_db(db_path)
        self._lock = threading.Lock()
        self.seq = 0
        self.t0 = time.time()
        self.current_level = 0
        self.current_route = None

        # Create session
        cur = self.conn.execute(
            "INSERT INTO sessions (game_id, started_at, agent_version) VALUES (?, ?, ?)",
            (game_id, time.strftime('%Y-%m-%dT%H:%M:%S'), agent_version)
        )
        self.session_id = cur.lastrowid
        self.conn.commit()
        log.info(f"  [rec] Recording session {self.session_id} for {game_id}")

        # Register as the active live session
        global _live_session_info
        _live_session_info = {
            'session_id': self.session_id,
            'game_id': game_id,
            'started_at': time.strftime('%Y-%m-%dT%H:%M:%S'),
            'agent_version': agent_version,
        }

    def record_frame(self, frame: np.ndarray, action_id: Optional[int] = None,
                     action_data: Optional[dict] = None, action_name: str = '',
                     level: int = 0, is_win: bool = False,
                     metadata: Optional[dict] = None):
        """Record a single frame and broadcast to live viewers."""
        with self._lock:
            self.seq += 1
            blob = _compress_frame(frame)
            f = np.squeeze(frame)
            if f.ndim == 1:
                # Flat array — try to reshape to square
                side = int(np.sqrt(len(f)))
                if side * side == len(f):
                    f = f.reshape(side, side)
                else:
                    f = f.reshape(1, -1)  # 1×N fallback
            h, w = f.shape[:2]
            ts = time.time() - self.t0
            self.conn.execute(
                """INSERT INTO frames
                   (session_id, seq, timestamp, level, action_id, action_data,
                    action_name, frame_blob, frame_w, frame_h, solver_route,
                    is_win, metadata)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (self.session_id, self.seq, ts,
                 level, action_id, json.dumps(action_data or {}),
                 action_name, blob, w, h,
                 self.current_route, is_win,
                 json.dumps(metadata or {}))
            )
            # Commit every frame for live streaming (WAL mode makes this cheap)
            self.conn.commit()

            # Broadcast to live viewers
            # RGB frames (e.g. Doom) → base64 JPEG, palette frames (ARC) → raw array
            if f.ndim == 3 and f.shape[2] == 3:
                import base64
                from PIL import Image
                img_pil = Image.fromarray(f)
                buf_jpg = io.BytesIO()
                img_pil.save(buf_jpg, format='JPEG', quality=70)
                frame_payload = {'frame_b64': base64.b64encode(buf_jpg.getvalue()).decode(),
                                 'frame_w': f.shape[1], 'frame_h': f.shape[0]}
            else:
                frame_payload = {'frame': f.tolist()}

            _broadcast_frame({
                'session_id': self.session_id,
                'seq': self.seq,
                'timestamp': ts,
                'level': level,
                'action_id': action_id,
                'action_name': action_name or '',
                'solver_route': self.current_route or '',
                'is_win': is_win,
                **frame_payload,
            })

    def comment(self, text: str, author: str = 'apollo',
                comment_type: str = 'thought'):
        """Add a timestamped comment."""
        with self._lock:
            self.conn.execute(
                """INSERT INTO comments
                   (session_id, seq, timestamp, author, text, comment_type)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (self.session_id, self.seq, time.time() - self.t0,
                 author, text, comment_type)
            )
            self.conn.commit()

    # --- Three-stream brain channels ---
    def eyes(self, text: str):
        """EYES: what the solver observes (sensory input)."""
        self.comment(text, author='eyes', comment_type='eyes')

    def mind(self, text: str):
        """MIND: what the solver reasons (conscious thought)."""
        self.comment(text, author='mind', comment_type='mind')

    def subconscious(self, text: str):
        """SUBCONSCIOUS: what memories resonate (pattern recognition)."""
        self.comment(text, author='subconscious', comment_type='subconscious')

    def set_route(self, route: str):
        """Update current solver route (for annotation)."""
        self.current_route = route
        self.comment(f"Trying {route}", comment_type='route')

    def finish(self, results: dict):
        """Finalize the session with results."""
        with self._lock:
            solved = sum(1 for l in results.get('levels', []) if l.get('solved'))
            total = len(results.get('levels', []))
            self.conn.execute(
                """UPDATE sessions SET
                   finished_at=?, total_levels=?, levels_solved=?,
                   total_actions=?, baseline_actions=?, efficiency=?,
                   game_type=?, metadata=?
                   WHERE id=?""",
                (time.strftime('%Y-%m-%dT%H:%M:%S'), total, solved,
                 results.get('total_actions', 0),
                 results.get('baseline_total', 0),
                 results.get('efficiency'),
                 results.get('game_type', ''),
                 json.dumps(results),
                 self.session_id)
            )
            self.conn.commit()
            log.info(f"  [rec] Session {self.session_id} saved: "
                     f"{solved}/{total} levels, {self.seq} frames")

            # Signal end of live session
            _broadcast_frame({
                'type': 'session_end',
                'session_id': self.session_id,
                'levels_solved': solved,
                'total_levels': total,
                'efficiency': results.get('efficiency'),
            })
            global _live_session_info
            _live_session_info = {}


# ─── Replay Server ─────────────────────────────────────────────────────

REPLAY_PAGE = """<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>ARC Replay — Apollo's Game DVR</title>
<style>
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body {
    background: #0a0a0a; color: #e0e0e0; font-family: 'Fira Code', monospace;
    display: flex; flex-direction: column; align-items: center; padding: 20px;
    min-height: 100vh;
  }
  h1 { color: #00ff88; margin-bottom: 5px; font-size: 24px; }
  .subtitle { color: #666; font-size: 13px; margin-bottom: 20px; }
  a { color: #00ff88; text-decoration: none; }
  a:hover { text-decoration: underline; }

  /* Session list */
  .session-list { width: 100%; max-width: 800px; }
  .session-card {
    background: #111; border: 1px solid #333; border-radius: 8px;
    padding: 15px; margin-bottom: 10px; cursor: pointer;
    transition: border-color 0.2s;
  }
  .session-card:hover { border-color: #00ff88; }
  .session-card .game-name { color: #00ff88; font-size: 18px; font-weight: bold; }
  .session-card .meta { color: #888; font-size: 12px; margin-top: 5px; }
  .session-card .stats { color: #aaa; font-size: 14px; margin-top: 8px; }
  .session-card .solved { color: #00ff88; }
  .session-card .failed { color: #ff4444; }

  /* Player */
  .player { display: none; width: 100%; max-width: 900px; }
  .player.active { display: flex; flex-direction: column; align-items: center; }
  .back-btn {
    align-self: flex-start; background: none; border: 1px solid #555;
    color: #aaa; padding: 5px 12px; cursor: pointer; margin-bottom: 15px;
    border-radius: 4px; font-family: inherit;
  }
  .back-btn:hover { border-color: #00ff88; color: #00ff88; }

  #game-canvas {
    border: 2px solid #333; image-rendering: pixelated;
    image-rendering: crisp-edges; width: 512px; height: 512px;
    background: #000;
  }

  .info-bar {
    display: flex; gap: 25px; margin: 12px 0; font-size: 14px;
    flex-wrap: wrap; justify-content: center;
  }
  .info-bar .label { color: #888; }
  .info-bar .value { color: #00ff88; font-weight: bold; }

  /* Timeline */
  .timeline-container { width: 100%; max-width: 600px; margin: 10px 0; position: relative; }
  .timeline {
    width: 100%; height: 6px; -webkit-appearance: none; appearance: none;
    background: #333; border-radius: 3px; outline: none; cursor: pointer;
  }
  .timeline::-webkit-slider-thumb {
    -webkit-appearance: none; width: 14px; height: 14px;
    background: #00ff88; border-radius: 50%; cursor: pointer;
  }
  .level-markers { position: relative; height: 16px; width: 100%; margin-top: 4px; }
  .level-marker {
    position: absolute; top: 0; font-size: 10px; color: #ffaa00;
    transform: translateX(-50%); cursor: pointer; white-space: nowrap;
  }
  .level-marker:hover { color: #00ff88; }

  /* Level tabs */
  .level-tabs { display: flex; gap: 6px; margin: 8px 0; flex-wrap: wrap; }
  .level-tab {
    background: #1a1a1a; border: 1px solid #333; color: #aaa;
    padding: 4px 12px; cursor: pointer; border-radius: 4px;
    font-family: inherit; font-size: 12px;
  }
  .level-tab:hover { border-color: #00ff88; color: #00ff88; }
  .level-tab.active { background: #00ff88; color: #0a0a0a; border-color: #00ff88; }
  .level-tab.solved { border-color: #00ff88; }
  .level-tab.failed { border-color: #ff4444; }

  /* Controls */
  .controls { display: flex; gap: 8px; margin: 10px 0; align-items: center; }
  .ctrl-btn {
    background: #1a1a1a; border: 1px solid #444; color: #ddd;
    padding: 6px 14px; cursor: pointer; border-radius: 4px;
    font-family: inherit; font-size: 14px;
  }
  .ctrl-btn:hover { border-color: #00ff88; color: #00ff88; }
  .ctrl-btn.active { background: #00ff88; color: #0a0a0a; border-color: #00ff88; }
  .speed-label { color: #888; font-size: 12px; margin-left: 10px; }

  /* Comments */
  .comments-box {
    background: #111; border: 1px solid #333; border-radius: 8px;
    padding: 15px; margin-top: 15px; width: 100%; max-width: 600px;
    max-height: 200px; overflow-y: auto; font-size: 12px;
  }
  .comment {
    margin-bottom: 6px; padding: 4px 0;
    border-bottom: 1px solid #1a1a1a;
  }
  .comment .time { color: #555; margin-right: 8px; }
  .comment .author { color: #00aaff; margin-right: 6px; }
  .comment .route { color: #ffaa00; }
  .comment .thought { color: #aaa; }
  .comment .milestone { color: #00ff88; font-weight: bold; }
  .comment .eyes { color: #00ccff; }
  .comment .mind { color: #ff88ff; }
  .comment .subconscious { color: #ffcc00; font-style: italic; }

  /* Three-stream brain view */
  .brain-view {
    display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 10px;
    width: 100%; max-width: 900px; margin-top: 15px;
  }
  .brain-stream {
    background: #111; border: 1px solid #333; border-radius: 8px;
    padding: 10px; max-height: 200px; overflow-y: auto; font-size: 11px;
  }
  .brain-stream.eyes-stream { border-color: #00ccff; }
  .brain-stream.mind-stream { border-color: #ff88ff; }
  .brain-stream.sub-stream { border-color: #ffcc00; }
  .brain-stream h4 { margin: 0 0 8px; font-size: 12px; }
  .brain-stream.eyes-stream h4 { color: #00ccff; }
  .brain-stream.mind-stream h4 { color: #ff88ff; }
  .brain-stream.sub-stream h4 { color: #ffcc00; }
  .comment.active { background: #1a2a1a; border-radius: 4px; padding: 4px 6px; }

  /* Add comment */
  .add-comment {
    display: flex; gap: 8px; margin-top: 10px; width: 100%; max-width: 600px;
  }
  .add-comment input {
    flex: 1; background: #111; border: 1px solid #333; color: #e0e0e0;
    padding: 6px 10px; border-radius: 4px; font-family: inherit; font-size: 13px;
  }
  .add-comment input:focus { border-color: #00ff88; outline: none; }
  .add-comment button {
    background: #00ff88; color: #0a0a0a; border: none; padding: 6px 16px;
    border-radius: 4px; cursor: pointer; font-family: inherit; font-weight: bold;
  }

  /* Nav tabs */
  .nav { display: flex; gap: 0; margin-bottom: 20px; }
  .nav-tab {
    background: #111; border: 1px solid #333; color: #aaa;
    padding: 8px 20px; cursor: pointer; font-family: inherit; font-size: 14px;
    border-bottom: 2px solid transparent;
  }
  .nav-tab:first-child { border-radius: 8px 0 0 0; }
  .nav-tab:last-child { border-radius: 0 8px 0 0; }
  .nav-tab:hover { color: #00ff88; }
  .nav-tab.active { border-bottom-color: #00ff88; color: #00ff88; }
  .page { display: none; width: 100%; max-width: 800px; }
  .page.active { display: block; }

  /* Leaderboard */
  .leaderboard { width: 100%; }
  .lb-table { width: 100%; border-collapse: collapse; }
  .lb-table th {
    text-align: left; padding: 8px 12px; color: #888;
    border-bottom: 1px solid #333; font-size: 12px;
  }
  .lb-table td {
    padding: 8px 12px; border-bottom: 1px solid #1a1a1a; font-size: 13px;
  }
  .lb-table tr:hover { background: #111; }
  .lb-table .game-col { color: #00ff88; font-weight: bold; }
  .lb-table .solved-col { color: #00ff88; }
  .lb-table .failed-col { color: #ff4444; }
  .lb-table .eff-col { color: #ffaa00; }
  .lb-category { color: #888; font-size: 18px; margin: 20px 0 10px; border-bottom: 1px solid #333; padding-bottom: 5px; }

  /* Game catalog */
  .game-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(180px, 1fr)); gap: 10px; }
  .game-tile {
    background: #111; border: 1px solid #333; border-radius: 8px;
    padding: 12px; cursor: default; text-align: center;
  }
  .game-tile .gname { color: #00ff88; font-weight: bold; font-size: 14px; }
  .game-tile .gtype { color: #666; font-size: 11px; margin-top: 4px; }
  .game-tile .gstatus { font-size: 12px; margin-top: 6px; }
  .game-tile.played { border-color: #00ff88; }
  .game-tile.unplayed { opacity: 0.5; }
</style>
</head>
<body>
  <h1>ARC Replay</h1>
  <div class="subtitle">Apollo's game recordings — watch the pinball wizard play</div>

  <div class="nav">
    <div class="nav-tab active" onclick="showPage('recordings')">Recordings</div>
    <div class="nav-tab" onclick="showPage('leaderboard')">Leaderboard</div>
    <div class="nav-tab" onclick="showPage('catalog')">Game Catalog</div>
    <div class="nav-tab" id="live-tab" onclick="showPage('live')" style="position:relative">
      LIVE <span id="live-dot" style="display:none;color:#ff3333;font-size:20px;position:absolute;top:2px;right:4px">&#9679;</span>
    </div>
  </div>

  <!-- Session List -->
  <div class="page active" id="page-recordings">
    <div class="session-list" id="session-list"></div>
  </div>

  <!-- Leaderboard -->
  <div class="page" id="page-leaderboard">
    <div class="leaderboard" id="leaderboard"></div>
  </div>

  <!-- Game Catalog -->
  <div class="page" id="page-catalog">
    <div class="game-grid" id="game-catalog"></div>
  </div>

  <!-- Live Viewer -->
  <div class="page" id="page-live">
    <iframe id="live-iframe" style="width:100%;height:700px;border:none;display:none"></iframe>
    <div id="live-status" style="text-align:center;color:#888;padding:40px">
      Click LIVE tab to connect...
    </div>
    <!-- keep hidden elements for JS compatibility -->
    <div style="display:none">
      <canvas id="live-canvas"></canvas>
      <span id="live-game"></span><span id="live-level"></span>
      <span id="live-action"></span><span id="live-frame-num"></span>
      <span id="live-route"></span><div id="live-log"></div>
      <div id="live-player"></div>
    </div>
  </div>

  <!-- Player -->
  <div class="player" id="player">
    <button class="back-btn" onclick="showList()">&larr; Back to recordings</button>

    <canvas id="game-canvas" width="64" height="64"></canvas>

    <div class="info-bar">
      <div><span class="label">Game: </span><span class="value" id="game-name">—</span></div>
      <div><span class="label">Level: </span><span class="value" id="level-num">0</span></div>
      <div><span class="label">Action: </span><span class="value" id="action-name">—</span></div>
      <div><span class="label">Frame: </span><span class="value" id="frame-num">0</span>
           <span class="label"> / </span><span class="value" id="frame-total">0</span></div>
      <div><span class="label">Route: </span><span class="value" id="route-name">—</span></div>
    </div>

    <div class="level-tabs" id="level-tabs"></div>

    <div class="timeline-container">
      <input type="range" class="timeline" id="timeline" min="0" max="0" value="0">
      <div class="level-markers" id="level-markers"></div>
    </div>

    <div class="controls">
      <button class="ctrl-btn" id="btn-prev" onclick="stepBack()">&laquo;</button>
      <button class="ctrl-btn" id="btn-play" onclick="togglePlay()">&#9654; Play</button>
      <button class="ctrl-btn" id="btn-next" onclick="stepForward()">&raquo;</button>
      <span class="speed-label">Speed:</span>
      <button class="ctrl-btn speed" onclick="setSpeed(0.25, this)">0.25x</button>
      <button class="ctrl-btn speed" onclick="setSpeed(0.5, this)">0.5x</button>
      <button class="ctrl-btn speed active" onclick="setSpeed(1, this)">1x</button>
      <button class="ctrl-btn speed" onclick="setSpeed(3, this)">3x</button>
      <button class="ctrl-btn speed" onclick="setSpeed(10, this)">10x</button>
    </div>

    <!-- Game frame + subconscious visual memories side by side -->
    <div style="display:flex;gap:20px;justify-content:center;margin-bottom:15px;flex-wrap:wrap;">
      <div style="text-align:center;">
        <div style="color:#00ccff;font-size:12px;margin-bottom:5px;">CURRENT FRAME</div>
        <canvas id="brain-frame" width="256" height="256" style="border:2px solid #00ccff;image-rendering:pixelated;background:#000;"></canvas>
      </div>
      <div style="text-align:center;">
        <div style="color:#ffcc00;font-size:12px;margin-bottom:5px;">RESONANT MEMORY</div>
        <canvas id="memory-frame" width="256" height="256" style="border:2px solid #ffcc00;image-rendering:pixelated;background:#111;"></canvas>
        <div id="memory-label" style="color:#888;font-size:11px;margin-top:3px;">No memories yet</div>
      </div>
    </div>

    <!-- Three-stream brain view -->
    <div class="brain-view" id="brain-view">
      <div class="brain-stream eyes-stream" id="eyes-stream">
        <h4>👁 EYES — what I see</h4>
        <div id="eyes-log"></div>
      </div>
      <div class="brain-stream mind-stream" id="mind-stream">
        <h4>🧠 MIND — what I think</h4>
        <div id="mind-log"></div>
      </div>
      <div class="brain-stream sub-stream" id="sub-stream">
        <h4>💡 SUBCONSCIOUS — what resonates</h4>
        <div id="sub-log"></div>
      </div>
    </div>
    <!-- Live step-by-step debugger -->
    <div id="step-controls" style="margin-top:10px;">
      <button class="ctrl-btn" id="step-btn" onclick="requestStep()" style="background:#ffcc00;color:#000;font-weight:bold;padding:8px 24px;">
        ▶ Next Step
      </button>
      <span id="step-status" style="color:#888;margin-left:10px;">Click to advance solver one step</span>
    </div>

    <div class="comments-box" id="comments-box"></div>

    <div class="add-comment">
      <input type="text" id="comment-input" placeholder="Add a comment (as peter)..."
             onkeydown="if(event.key==='Enter') addComment()">
      <button onclick="addComment()">Post</button>
    </div>
  </div>

<script>
const canvas = document.getElementById('game-canvas');
const ctx = canvas.getContext('2d');

// Official ARC-AGI-3 palette (from arc_agi/rendering.py)
const PALETTE = [
  [255, 255, 255], [204, 204, 204], [153, 153, 153], [102, 102, 102],
  [51, 51, 51], [0, 0, 0], [229, 58, 163], [255, 123, 204],
  [249, 60, 49], [30, 147, 255], [136, 216, 241], [255, 220, 0],
  [255, 133, 27], [146, 18, 49], [79, 204, 48], [163, 86, 214],
];

let frames = [];
let allFrames = [];  // unfiltered
let comments = [];
let currentFrame = 0;
let playing = false;
let playSpeed = 1;
let playTimer = null;
let currentSessionId = null;
let levelBoundaries = [];  // [{level, startIdx, endIdx, solved}]
let currentLevel = -1;  // -1 = all levels

function renderFrame(frameData) {
  if (!frameData || !frameData.length) return;
  const h = frameData.length;
  const w = frameData[0].length;
  canvas.width = w;
  canvas.height = h;
  const imgData = ctx.createImageData(w, h);
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      const v = frameData[y][x];
      const c = PALETTE[v] || [128, 128, 128];
      const i = (y * w + x) * 4;
      imgData.data[i] = c[0]; imgData.data[i+1] = c[1];
      imgData.data[i+2] = c[2]; imgData.data[i+3] = 255;
    }
  }
  ctx.putImageData(imgData, 0, 0);
  // Also draw on brain-frame canvas (smaller, for the brain view)
  const bf = document.getElementById('brain-frame');
  if (bf) {
    const bctx = bf.getContext('2d');
    const tmp = document.createElement('canvas');
    tmp.width = w; tmp.height = h;
    tmp.getContext('2d').putImageData(imgData, 0, 0);
    bctx.imageSmoothingEnabled = false;
    bctx.clearRect(0, 0, bf.width, bf.height);
    bctx.drawImage(tmp, 0, 0, bf.width, bf.height);
  }
}

function showFrame(idx) {
  if (idx < 0 || idx >= frames.length) return;
  currentFrame = idx;
  const f = frames[idx];
  renderFrame(f.frame);
  document.getElementById('game-name').textContent = f.game || '—';
  document.getElementById('level-num').textContent = f.level ?? 0;
  document.getElementById('action-name').textContent = f.action_name || '—';
  document.getElementById('frame-num').textContent = (idx + 1) + ' / ' + frames.length;
  document.getElementById('frame-total').textContent = frames.length;
  document.getElementById('route-name').textContent = f.solver_route || '—';
  document.getElementById('timeline').value = idx;

  // Update level tab highlight
  document.querySelectorAll('.level-tab').forEach(t => t.classList.remove('active'));
  const activeTab = document.querySelector(`.level-tab[data-level="${f.level}"]`);
  if (activeTab) activeTab.classList.add('active');

  // Highlight active comments
  const cbox = document.getElementById('comments-box');
  cbox.querySelectorAll('.comment').forEach(el => el.classList.remove('active'));
  comments.forEach((c, ci) => {
    if (c.seq !== null && Math.abs(c.seq - f.seq) <= 1) {
      const el = document.getElementById('comment-' + ci);
      if (el) { el.classList.add('active'); el.scrollIntoView({block: 'nearest'}); }
    }
  });
}

function stepForward() { if (currentFrame < frames.length - 1) showFrame(currentFrame + 1); else stopPlay(); }
function stepBack() { if (currentFrame > 0) showFrame(currentFrame - 1); }

function togglePlay() {
  if (playing) { stopPlay(); } else { startPlay(); }
}
function startPlay() {
  playing = true;
  document.getElementById('btn-play').innerHTML = '&#9646;&#9646; Pause';
  document.getElementById('btn-play').classList.add('active');
  scheduleNext();
}
function stopPlay() {
  playing = false;
  document.getElementById('btn-play').innerHTML = '&#9654; Play';
  document.getElementById('btn-play').classList.remove('active');
  if (playTimer) { clearTimeout(playTimer); playTimer = null; }
}
function scheduleNext() {
  if (!playing) return;
  // Fixed-interval playback — no timestamp gaps
  const baseDelay = 300;  // 300ms between frames at 1x
  const delay = Math.max(16, baseDelay / playSpeed);
  playTimer = setTimeout(() => { stepForward(); if (playing) scheduleNext(); }, delay);
}

function setSpeed(s, btn) {
  playSpeed = s;
  document.querySelectorAll('.ctrl-btn.speed').forEach(b => b.classList.remove('active'));
  if (btn) btn.classList.add('active');
  // Restart play timer with new speed
  if (playing) { clearTimeout(playTimer); scheduleNext(); }
}

function filterToLevel(level) {
  currentLevel = level;
  if (level === -1) {
    frames = allFrames;
  } else {
    frames = allFrames.filter(f => f.level === level);
  }
  document.getElementById('timeline').max = Math.max(0, frames.length - 1);

  // Update tab highlights
  document.querySelectorAll('.level-tab').forEach(t => t.classList.remove('active'));
  const tab = (level === -1)
    ? document.querySelector('.level-tab[data-level="all"]')
    : document.querySelector(`.level-tab[data-level="${level}"]`);
  if (tab) tab.classList.add('active');

  if (frames.length > 0) showFrame(0);
}

function buildLevelUI() {
  // Find level boundaries
  const levels = new Map();
  allFrames.forEach((f, i) => {
    const lv = f.level ?? 0;
    if (!levels.has(lv)) levels.set(lv, {level: lv, startIdx: i, endIdx: i, count: 0});
    const entry = levels.get(lv);
    entry.endIdx = i;
    entry.count++;
  });
  levelBoundaries = [...levels.values()].sort((a,b) => a.level - b.level);

  // Build tabs
  const tabsEl = document.getElementById('level-tabs');
  let html = '<button class="level-tab active" data-level="all" onclick="filterToLevel(-1)">All</button>';
  levelBoundaries.forEach(lb => {
    const hasWin = allFrames.slice(lb.startIdx, lb.endIdx+1).some(f => f.is_win);
    const cls = hasWin ? 'solved' : '';
    html += `<button class="level-tab ${cls}" data-level="${lb.level}" onclick="filterToLevel(${lb.level})">L${lb.level} (${lb.count})</button>`;
  });
  tabsEl.innerHTML = html;

  // Build timeline markers
  const markersEl = document.getElementById('level-markers');
  const total = allFrames.length;
  markersEl.innerHTML = '';
  levelBoundaries.forEach(lb => {
    if (total <= 1) return;
    const pct = (lb.startIdx / (total - 1)) * 100;
    const marker = document.createElement('span');
    marker.className = 'level-marker';
    marker.style.left = pct + '%';
    marker.textContent = 'L' + lb.level;
    marker.onclick = () => filterToLevel(lb.level);
    markersEl.appendChild(marker);
  });
}

// Session list
async function loadSessions() {
  const resp = await fetch('/api/sessions');
  const data = await resp.json();
  const list = document.getElementById('session-list');
  if (data.sessions.length === 0) {
    list.innerHTML = '<div style="color:#666;text-align:center;padding:40px">No recordings yet. Run: python3 arc_agent_v05.py --record</div>';
    return;
  }
  list.innerHTML = data.sessions.map(s => `
    <div class="session-card" onclick="loadSession(${s.id})">
      <div class="game-name">#${s.id} — ${s.game_id.toUpperCase()}</div>
      <div class="stats">
        <span class="${s.levels_solved === s.total_levels ? 'solved' : 'failed'}">
          ${s.levels_solved}/${s.total_levels} levels
        </span>
        &nbsp;|&nbsp; ${s.total_actions} actions
        ${s.efficiency ? '&nbsp;|&nbsp; ' + s.efficiency.toFixed(0) + '% efficiency' : ''}
        ${s.game_type ? '&nbsp;|&nbsp; ' + s.game_type : ''}
      </div>
      <div class="meta">${s.started_at} &nbsp;|&nbsp; ${s.frame_count} frames &nbsp;|&nbsp; ${s.comment_count} comments</div>
    </div>
  `).join('');
}

async function loadSession(id) {
  currentSessionId = id;
  stopPlay();
  currentLevel = -1;

  // Load frames + comments in parallel
  const [fResp, cResp] = await Promise.all([
    fetch('/api/session/' + id + '/frames'),
    fetch('/api/session/' + id + '/comments')
  ]);
  const fData = await fResp.json();
  const cData = await cResp.json();

  allFrames = fData.frames;
  frames = allFrames;
  comments = cData.comments;

  // Setup UI
  document.getElementById('frame-total').textContent = frames.length;
  document.getElementById('timeline').max = Math.max(0, frames.length - 1);
  document.getElementById('timeline').value = 0;

  // Build level chapters
  buildLevelUI();
  renderComments();

  // Show player
  document.getElementById('session-list').style.display = 'none';
  document.getElementById('player').classList.add('active');

  if (frames.length > 0) showFrame(0);
}

function renderComments() {
  const cbox = document.getElementById('comments-box');
  const eyesLog = document.getElementById('eyes-log');
  const mindLog = document.getElementById('mind-log');
  const subLog = document.getElementById('sub-log');
  let eyesHtml = '', mindHtml = '', subHtml = '', otherHtml = '';
  comments.forEach((c, i) => {
    const typeClass = c.comment_type || 'thought';
    const timeStr = c.timestamp ? c.timestamp.toFixed(1) + 's' : '';
    const entry = `<div class="comment" id="comment-${i}">
      <span class="time">${timeStr}</span>
      <span class="${typeClass}">${c.text}</span>
    </div>`;
    if (typeClass === 'eyes') eyesHtml += entry;
    else if (typeClass === 'mind') mindHtml += entry;
    else if (typeClass === 'subconscious') subHtml += entry;
    else otherHtml += entry;
  });
  if (eyesLog) eyesLog.innerHTML = eyesHtml || '<span style="color:#555">No observations yet</span>';
  if (mindLog) mindLog.innerHTML = mindHtml || '<span style="color:#555">No reasoning yet</span>';
  if (subLog) subLog.innerHTML = subHtml || '<span style="color:#555">No resonances yet</span>';
  cbox.innerHTML = otherHtml;
  // Auto-scroll streams to bottom
  [eyesLog, mindLog, subLog].forEach(el => { if(el) el.parentElement.scrollTop = el.parentElement.scrollHeight; });
}

function showList() {
  stopPlay();
  document.getElementById('player').classList.remove('active');
  document.querySelectorAll('.page').forEach(p => p.classList.remove('active'));
  document.querySelectorAll('.nav-tab').forEach(t => t.classList.remove('active'));
  document.getElementById('page-recordings').classList.add('active');
  document.querySelector('.nav-tab').classList.add('active');
}

async function addComment() {
  const input = document.getElementById('comment-input');
  const text = input.value.trim();
  if (!text || !currentSessionId) return;
  const f = frames[currentFrame];
  await fetch('/api/session/' + currentSessionId + '/comment', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({text, author: 'peter', seq: f ? f.seq : null})
  });
  input.value = '';
  // Reload comments
  const cResp = await fetch('/api/session/' + currentSessionId + '/comments');
  const cData = await cResp.json();
  comments = cData.comments;
  renderComments();
}

// Timeline scrubber
document.getElementById('timeline').addEventListener('input', e => {
  showFrame(parseInt(e.target.value));
});

// Keyboard shortcuts
document.addEventListener('keydown', e => {
  if (e.target.tagName === 'INPUT') return;
  if (e.key === ' ') { e.preventDefault(); togglePlay(); }
  if (e.key === 'ArrowRight') stepForward();
  if (e.key === 'ArrowLeft') stepBack();
  if (e.key === 'Home') showFrame(0);
  if (e.key === 'End') showFrame(frames.length - 1);
});

// Nav
function showPage(page) {
  document.querySelectorAll('.page').forEach(p => p.classList.remove('active'));
  document.querySelectorAll('.nav-tab').forEach(t => t.classList.remove('active'));
  document.getElementById('page-' + page).classList.add('active');
  event.target.classList.add('active');
  // Hide player when switching pages
  document.getElementById('player').classList.remove('active');
  if (page === 'leaderboard') loadLeaderboard();
  if (page === 'catalog') loadCatalog();
  if (page === 'live') {
    var iframe = document.getElementById('live-iframe');
    iframe.src = '/live-page';
    iframe.style.display = 'block';
    document.getElementById('live-status').style.display = 'none';
  } else {
    var iframe = document.getElementById('live-iframe');
    if (iframe) { iframe.src = ''; iframe.style.display = 'none'; }
  }
}

// Leaderboard
async function loadLeaderboard() {
  const resp = await fetch('/api/leaderboard');
  const data = await resp.json();
  const el = document.getElementById('leaderboard');

  let html = '';
  for (const cat of data.categories) {
    html += `<div class="lb-category">${cat.name}</div>`;
    html += '<table class="lb-table"><tr><th>Game</th><th>Best</th><th>Actions</th><th>Efficiency</th><th>Plays</th><th>Date</th></tr>';
    for (const g of cat.games) {
      const cls = g.best_solved ? 'solved-col' : 'failed-col';
      const status = g.best_solved ? g.levels_solved + '/' + g.total_levels : 'unsolved';
      const eff = g.best_efficiency ? g.best_efficiency.toFixed(0) + '%' : '—';
      html += `<tr>
        <td class="game-col">${g.game_id.toUpperCase()}</td>
        <td class="${cls}">${status}</td>
        <td>${g.best_actions || '—'}</td>
        <td class="eff-col">${eff}</td>
        <td>${g.play_count}</td>
        <td style="color:#666">${g.last_played || '—'}</td>
      </tr>`;
    }
    html += '</table>';
  }
  el.innerHTML = html || '<div style="color:#666;text-align:center;padding:40px">No games played yet.</div>';
}

// Game Catalog
const CATALOG = {
  'ARC Preview': ['ft09', 'ls20', 'vc33'],
  'Atari — Tier 1 (Basic)': ['Pong', 'Breakout', 'SpaceInvaders'],
  'Atari — Tier 2 (Navigation)': ['Frogger', 'Freeway', 'MsPacman'],
  'Atari — Tier 3 (Complex)': ['Asteroids', 'Centipede', 'BattleZone'],
  'Atari — Tier 4 (Planning)': ['Pitfall', 'MontezumaRevenge', 'Adventure'],
  'MiniGrid': ['Empty-5x5', 'Empty-8x8', 'DoorKey-5x5', 'DoorKey-8x8',
               'LavaCrossing', 'MultiRoom', 'Fetch', 'BlockedUnlockPickup'],
};

async function loadCatalog() {
  const resp = await fetch('/api/sessions');
  const data = await resp.json();
  const played = new Set(data.sessions.map(s => s.game_id.toLowerCase()));

  const el = document.getElementById('game-catalog');
  let html = '';
  for (const [category, games] of Object.entries(CATALOG)) {
    html += `<div style="grid-column: 1/-1"><div class="lb-category">${category}</div></div>`;
    for (const g of games) {
      const isPlayed = played.has(g.toLowerCase());
      const cls = isPlayed ? 'played' : 'unplayed';
      const status = isPlayed ? '<span style="color:#00ff88">played</span>' : '<span style="color:#555">not yet</span>';
      html += `<div class="game-tile ${cls}">
        <div class="gname">${g}</div>
        <div class="gtype">${category.split('—')[0].trim()}</div>
        <div class="gstatus">${status}</div>
      </div>`;
    }
  }
  el.innerHTML = html;
}

// ─── Live Streaming ────────────────────────────────────────────────
let liveSource = null;
let liveFrameCount = 0;
const liveCanvas = document.getElementById('live-canvas');
const liveCtx = liveCanvas ? liveCanvas.getContext('2d') : null;

function renderLiveFrame(frameData) {
  if (!frameData || !frameData.length) return;
  var cvs = document.getElementById('live-canvas');
  var cx = cvs.getContext('2d');
  var h = frameData.length, w = frameData[0].length;
  cvs.width = w; cvs.height = h;
  var imgData = cx.createImageData(w, h);
  for (var y = 0; y < h; y++) {
    for (var x = 0; x < w; x++) {
      var v = frameData[y][x];
      var cl = PALETTE[v] || [128,128,128];
      var i = (y * w + x) * 4;
      imgData.data[i] = cl[0]; imgData.data[i+1] = cl[1];
      imgData.data[i+2] = cl[2]; imgData.data[i+3] = 255;
    }
  }
  cx.putImageData(imgData, 0, 0);
}

async function checkLiveStatus() {
  try {
    const resp = await fetch('/api/live/status');
    const data = await resp.json();
    const dot = document.getElementById('live-dot');
    if (data.live) {
      dot.style.display = 'inline';
      document.getElementById('live-tab').style.color = '#ff3333';
    } else {
      dot.style.display = 'none';
    }
    return data;
  } catch(e) { return {live: false}; }
}

// No queue — render directly in onmessage (proven working on :8767 test)

function startLiveStream() {
  if (liveSource) { liveSource.close(); }
  const logEl = document.getElementById('live-log');
  liveFrameCount = 0;
  if (logEl) logEl.innerHTML = '';

  liveSource = new EventSource('/api/live/stream');
  liveSource.onerror = function(e) { console.log('[LIVE] SSE error:', e); };
  liveSource.onmessage = function(event) {
    const data = JSON.parse(event.data);

    if (data.type === 'connected') {
      document.getElementById('live-status').style.display = 'none';
      document.getElementById('live-player').style.display = 'block';
      if (data.session_id) {
        logEl.innerHTML += '<div class="comment"><span class="milestone">Connected to session #' + data.session_id + '</span></div>';
      } else {
        logEl.innerHTML += '<div class="comment"><span class="milestone">Connected to live session</span></div>';
      }
      return;
    }
    if (data.type === 'no_session') {
      document.getElementById('live-status').textContent = 'No active session. Start one with: python3 arc_agent_v05.py --record';
      document.getElementById('live-status').style.display = 'block';
      document.getElementById('live-player').style.display = 'none';
      return;
    }
    // no_session already handled above
    if (data.type === 'session_end') {
      logEl.innerHTML += '<div class="comment"><span class="milestone">Session ended — ' +
        (data.levels_solved || '?') + '/' + (data.total_levels || '?') + ' levels' +
        (data.efficiency ? ', ' + data.efficiency.toFixed(0) + '% efficiency' : '') + '</span></div>';
      document.getElementById('live-dot').style.display = 'none';
      return;
    }
    if (data.type === 'heartbeat') return;

    // Render directly — proven working on test page
    if (data.frame) {
      liveFrameCount++;
      renderLiveFrame(data.frame);
      document.getElementById('live-game').textContent = data.game_id || '—';
      document.getElementById('live-level').textContent = data.level ?? 0;
      document.getElementById('live-action').textContent = data.action_name || '—';
      document.getElementById('live-frame-num').textContent = liveFrameCount;
      document.getElementById('live-route').textContent = data.solver_route || '—';

      // Log wins
      if (data.is_win) {
        logEl.innerHTML += '<div class="comment"><span class="milestone">Level ' + data.level + ' WON!</span></div>';
        logEl.scrollTop = logEl.scrollHeight;
      }
    }
  };

  liveSource.onerror = function() {
    document.getElementById('live-status').innerHTML =
      '<div style="color:#ff4444;padding:40px">Connection lost. <a href="#" onclick="startLiveStream();return false">Retry</a></div>';
    document.getElementById('live-status').style.display = 'block';
  };
}

function stopLiveStream() {
  if (liveSource) { liveSource.close(); liveSource = null; }
}

// --- Step-by-step debugger ---
let stepCount = 0;
async function requestStep() {
  const btn = document.getElementById('step-btn');
  const status = document.getElementById('step-status');
  btn.disabled = true;
  btn.textContent = '⏳ Thinking...';
  status.textContent = 'Solver processing step ' + (stepCount + 1) + '...';
  try {
    await fetch('/api/live/step', {method: 'POST'});
    stepCount++;
    status.textContent = 'Step ' + stepCount + ' complete. Click for next.';
  } catch(e) {
    status.textContent = 'Error: ' + e.message;
  }
  btn.disabled = false;
  btn.textContent = '▶ Next Step (#' + (stepCount + 1) + ')';
}

// Show step controls when live
async function checkAndShowStepControls() {
  try {
    const resp = await fetch('/api/live/status');
    const data = await resp.json();
    document.getElementById('step-controls').style.display = data.live ? 'block' : 'none';
  } catch(e) {}
}
setInterval(checkAndShowStepControls, 3000);

// Check live status on load and periodically
checkLiveStatus();
setInterval(checkLiveStatus, 10000);

loadSessions();
</script>
</body>
</html>
"""


def serve_replay(db_path: str = DEFAULT_DB, port: int = 8766):
    """Start the replay web server."""
    from flask import Flask, jsonify, request, Response

    app = Flask(__name__)
    app.logger.setLevel(logging.WARNING)
    _server_conn = _get_db(db_path)  # single shared connection for reads

    def get_conn():
        return _server_conn

    @app.route('/')
    def index():
        return Response(REPLAY_PAGE, content_type='text/html')

    @app.route('/live-page')
    def live_page():
        """Standalone live viewer with team chat."""
        return Response('''<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>LIVE — ARC Arena</title>
<style>
*{box-sizing:border-box;margin:0;padding:0}
body{background:#111;color:#fff;font-family:monospace;display:flex;height:100vh}
.game-side{flex:1;display:flex;flex-direction:column;align-items:center;padding:20px}
.chat-side{width:340px;display:flex;flex-direction:column;border-left:1px solid #333;background:#0a0a0a}
h2{color:#ff3333;margin-bottom:10px}
canvas{max-width:640px;width:100%;aspect-ratio:1;image-rendering:pixelated;image-rendering:crisp-edges;border:2px solid #ff3333;background:#000}
.info{margin:10px;font-size:14px}
#status{color:#888;margin:8px}
#log{text-align:left;max-width:512px;width:100%;margin:10px auto;font-size:11px;max-height:150px;overflow-y:auto;color:#aaa}
.chat-hdr{padding:12px;border-bottom:1px solid #222;color:#888;font-size:13px}
.chat-msgs{flex:1;overflow-y:auto;padding:8px}
.chat-msg{margin-bottom:8px;padding:4px 0;border-bottom:1px solid #1a1a1a}
.chat-msg .meta{font-size:11px;color:#555}
.chat-msg .meta .name{font-weight:bold}
.chat-msg .meta .apollo{color:#ffd700} .chat-msg .meta .athena{color:#c89b3c}
.chat-msg .meta .archie{color:#3c8cc8} .chat-msg .meta .hypatia{color:#c83ca0}
.chat-msg .meta .peter{color:#4caf50}
.chat-msg .body{font-size:12px;white-space:pre-wrap;word-break:break-word;line-height:1.4;color:#ccc}
.chat-input{display:flex;gap:6px;padding:8px;border-top:1px solid #222}
.chat-input select,.chat-input input,.chat-input button{background:#1a1a1a;color:#ccc;border:1px solid #333;padding:6px;border-radius:4px;font-family:inherit;font-size:12px}
.chat-input input{flex:1}
.chat-input button:hover{border-color:#ff3333;color:#ff3333;cursor:pointer}
</style></head>
<body>
<div class="game-side">
<h2>&#9679; LIVE</h2>
<canvas id="c" width="320" height="240"></canvas>
<div class="info">
  <span>Game: <b id="g">—</b></span> &nbsp;
  <span>Level: <b id="l">0</b></span> &nbsp;
  <span>Frame: <b id="f">0</b></span> &nbsp;
  <span>Action: <b id="a">—</b></span> &nbsp;
  <span>Route: <b id="r">—</b></span>
</div>
<div id="status">Connecting...</div>
<div id="log"></div>
</div>
<div class="chat-side">
<div class="chat-hdr">Team Chat — arc-arena</div>
<div class="chat-msgs" id="chat-msgs"></div>
<div class="chat-input">
  <select id="chat-from"><option value="peter">Peter</option><option value="athena">Athena</option><option value="archie">Archie</option><option value="hypatia">Hypatia</option><option value="apollo">Apollo</option></select>
  <input id="chat-body" placeholder="Say something..." />
  <button onclick="sendChat()">Send</button>
</div>
</div>
<script>
// Official ARC-AGI-3 palette (from arc_agi/rendering.py)
const PALETTE=[[255,255,255],[204,204,204],[153,153,153],[102,102,102],
[51,51,51],[0,0,0],[229,58,163],[255,123,204],
[249,60,49],[30,147,255],[136,216,241],[255,220,0],
[255,133,27],[146,18,49],[79,204,48],[163,86,214]];
const c=document.getElementById("c"),ctx=c.getContext("2d");
let fc=0;
const src=new EventSource("/api/live/stream");
src.onmessage=function(e){
  const d=JSON.parse(e.data);
  if(d.type==="connected"){
    document.getElementById("status").textContent="Connected to session #"+(d.session_id||"?")+" ("+d.game_id+")";
    return;
  }
  if(d.type==="no_session"){
    document.getElementById("status").textContent="No active session — showing last recording. Waiting for next game...";
    src.close();
    // Load last recorded frame as fallback
    fetch("/api/sessions").then(r=>r.json()).then(sessions=>{
      if(!sessions.length)return;
      const last=sessions[0];
      document.getElementById("g").textContent=last.game_id||"—";
      fetch("/api/session/"+last.id+"/frames?limit=1&offset="+(last.frame_count-1)).then(r=>r.json()).then(data=>{
        if(data.frames&&data.frames.length){
          const fd=data.frames[0];
          if(fd.frame){
            const f=fd.frame,h=f.length,w=f[0].length;
            c.width=w;c.height=h;
            const img=ctx.createImageData(w,h);
            for(let y=0;y<h;y++)for(let x=0;x<w;x++){
              const v=f[y][x],i=(y*w+x)*4;
              const cl=PALETTE[v]||[128,128,128];
              img.data[i]=cl[0];img.data[i+1]=cl[1];img.data[i+2]=cl[2];img.data[i+3]=255;
            }
            ctx.putImageData(img,0,0);
          }
        }
      });
    });
    setTimeout(function(){location.reload();},10000);
    return;
  }
  if(d.type==="session_end"){
    document.getElementById("log").innerHTML+="<div style=color:#0f0>Session ended — waiting for next game...</div>";
    src.close();
    setTimeout(function(){location.reload();},2000);
    return;
  }
  if(d.type==="heartbeat")return;
  if(!d.frame&&!d.frame_b64)return;
  fc++;
  if(d.frame_b64){
    const img=new Image();
    img.onload=function(){c.width=img.width;c.height=img.height;ctx.drawImage(img,0,0);};
    img.src="data:image/jpeg;base64,"+d.frame_b64;
  }else{
    const f=d.frame,h=f.length,w=f[0].length;
    c.width=w;c.height=h;
    const img=ctx.createImageData(w,h);
    const isRGB=Array.isArray(f[0][0]);
    for(let y=0;y<h;y++)for(let x=0;x<w;x++){
      const v=f[y][x],i=(y*w+x)*4;
      if(isRGB){img.data[i]=v[0];img.data[i+1]=v[1];img.data[i+2]=v[2];}
      else{const cl=PALETTE[v]||[128,128,128];img.data[i]=cl[0];img.data[i+1]=cl[1];img.data[i+2]=cl[2];}
      img.data[i+3]=255;
    }
    ctx.putImageData(img,0,0);
  }
  document.getElementById("g").textContent=d.game_id||"—";
  document.getElementById("l").textContent=d.level||0;
  document.getElementById("f").textContent=fc;
  document.getElementById("a").textContent=d.action_name||"—";
  document.getElementById("r").textContent=d.solver_route||"—";
  if(d.is_win)document.getElementById("log").innerHTML+="<div style=color:#0f0>Level "+d.level+" WON!</div>";
};
// ── Chat ──
let lastMsgId=0;
function esc(t){const d=document.createElement("div");d.textContent=t;return d.innerHTML;}
async function pollChat(){
  try{
    const r=await fetch("/api/arena/chat?since="+lastMsgId+"&limit=50");
    const msgs=await r.json();
    if(!msgs.length)return;
    const box=document.getElementById("chat-msgs");
    const atBottom=box.scrollHeight-box.scrollTop-box.clientHeight<40;
    for(const m of msgs){
      if(m.id<=lastMsgId)continue;
      lastMsgId=m.id;
      const div=document.createElement("div");
      div.className="chat-msg";
      const t=m.time?m.time.substring(11,16):"";
      div.innerHTML='<div class="meta"><span class="name '+m.from+'">'+m.from+'</span> · '+t+'</div><div class="body">'+esc(m.body)+'</div>';
      box.appendChild(div);
    }
    if(atBottom)box.scrollTop=box.scrollHeight;
  }catch(e){}
}
async function sendChat(){
  const from=document.getElementById("chat-from").value;
  const body=document.getElementById("chat-body").value.trim();
  if(!body)return;
  try{
    await fetch("/api/arena/post",{method:"POST",headers:{"Content-Type":"application/x-www-form-urlencoded"},body:"from="+encodeURIComponent(from)+"&body="+encodeURIComponent(body)});
    document.getElementById("chat-body").value="";
    pollChat();
  }catch(e){}
}
document.getElementById("chat-body").addEventListener("keydown",function(e){if(e.key==="Enter")sendChat();});
setInterval(pollChat,2000);
pollChat();
</script></body></html>''', content_type='text/html; charset=utf-8')

    # ── Arena Chat (reads/writes agent_state.db) ──
    AGENT_STATE_DB = os.environ.get('AGENT_STATE_DB', os.path.join(os.path.dirname(__file__), 'agent_state.db'))
    ARENA_TAG = 'thread:arc-arena'

    @app.route('/api/arena/chat')
    def api_arena_chat():
        since = request.args.get('since', 0, type=int)
        limit = request.args.get('limit', 50, type=int)
        try:
            conn = sqlite3.connect(AGENT_STATE_DB, timeout=5)
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                "SELECT MAX(id) as id, from_agent, subject, body, MAX(created_at) as created_at "
                "FROM messages WHERE tags LIKE ? AND id > ? "
                "GROUP BY from_agent, body "
                "ORDER BY id DESC LIMIT ?",
                (f"%{ARENA_TAG}%", since, limit)
            ).fetchall()
            conn.close()
            msgs = [{"id": r["id"], "from": r["from_agent"], "body": r["body"],
                     "time": r["created_at"]} for r in reversed(rows)]
            return jsonify(msgs)
        except Exception as e:
            return jsonify([{"id": 0, "from": "system", "body": str(e), "time": ""}])

    @app.route('/api/arena/post', methods=['POST'])
    def api_arena_post():
        from_agent = request.form.get('from', 'peter')
        body = request.form.get('body', '').strip()
        if not body:
            return jsonify({"error": "empty"}), 400
        try:
            conn = sqlite3.connect(AGENT_STATE_DB, timeout=5)
            now = datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")
            conn.execute(
                "INSERT INTO messages (from_agent, to_agent, subject, body, priority, tags, is_read, created_at) "
                "VALUES (?, 'all', 'arena', ?, 'normal', ?, 0, ?)",
                (from_agent, body, ARENA_TAG, now)
            )
            conn.commit()
            conn.close()
            return jsonify({"ok": True})
        except Exception as e:
            return jsonify({"error": str(e)}), 500

    @app.route('/api/sessions')
    def api_sessions():
        conn = get_conn()
        rows = conn.execute("""
            SELECT s.*,
                   (SELECT COUNT(*) FROM frames WHERE session_id=s.id) as frame_count,
                   (SELECT COUNT(*) FROM comments WHERE session_id=s.id) as comment_count
            FROM sessions s ORDER BY s.id DESC LIMIT 20
        """).fetchall()
        return jsonify({'sessions': [dict(r) for r in rows]})

    @app.route('/api/session/<int:sid>/frames')
    def api_frames(sid):
        conn = get_conn()
        rows = conn.execute(
            "SELECT seq, timestamp, level, action_id, action_data, action_name, "
            "frame_blob, frame_w, frame_h, solver_route, is_win "
            "FROM frames WHERE session_id=? ORDER BY seq",
            (sid,)
        ).fetchall()
        frames_out = []
        for r in rows:
            frame = _decompress_frame(r['frame_blob'])
            entry = {
                'seq': r['seq'],
                'timestamp': r['timestamp'],
                'level': r['level'],
                'action_id': r['action_id'],
                'action_data': r['action_data'],
                'action_name': r['action_name'] or '',
                'solver_route': r['solver_route'] or '',
                'is_win': bool(r['is_win']),
            }
            if frame.ndim == 3 and frame.shape[2] == 3:
                import base64
                from PIL import Image
                img_pil = Image.fromarray(frame)
                buf_jpg = io.BytesIO()
                img_pil.save(buf_jpg, format='JPEG', quality=70)
                entry['frame_b64'] = base64.b64encode(buf_jpg.getvalue()).decode()
            else:
                entry['frame'] = frame.tolist()
            frames_out.append(entry)
        return jsonify({'frames': frames_out})

    @app.route('/api/session/<int:sid>/comments')
    def api_comments(sid):
        conn = get_conn()
        rows = conn.execute(
            "SELECT * FROM comments WHERE session_id=? ORDER BY seq, id",
            (sid,)
        ).fetchall()
        return jsonify({'comments': [dict(r) for r in rows]})

    @app.route('/api/leaderboard')
    def api_leaderboard():
        conn = get_conn()
        rows = conn.execute("""
            SELECT game_id,
                   MAX(levels_solved) as best_solved,
                   MAX(total_levels) as total_levels,
                   MIN(CASE WHEN levels_solved > 0 THEN total_actions END) as best_actions,
                   MAX(efficiency) as best_efficiency,
                   COUNT(*) as play_count,
                   MAX(started_at) as last_played,
                   game_type
            FROM sessions
            GROUP BY game_id
            ORDER BY game_id
        """).fetchall()

        # Categorize
        categories = {}
        for r in rows:
            gid = r['game_id'].lower()
            if gid in ('ft09', 'ls20', 'vc33'):
                cat = 'ARC Preview Games'
            elif 'minigrid' in gid or 'empty' in gid or 'door' in gid:
                cat = 'MiniGrid'
            else:
                cat = 'Atari'
            if cat not in categories:
                categories[cat] = []
            categories[cat].append({
                'game_id': r['game_id'],
                'best_solved': r['best_solved'] or 0,
                'levels_solved': r['best_solved'] or 0,
                'total_levels': r['total_levels'] or 0,
                'best_actions': r['best_actions'],
                'best_efficiency': r['best_efficiency'],
                'play_count': r['play_count'],
                'last_played': r['last_played'],
                'game_type': r['game_type'],
            })

        return jsonify({
            'categories': [{'name': k, 'games': v} for k, v in categories.items()]
        })

    @app.route('/api/session/<int:sid>/comment', methods=['POST'])
    def api_add_comment(sid):
        data = request.get_json()
        conn = get_conn()
        conn.execute(
            "INSERT INTO comments (session_id, seq, timestamp, author, text, comment_type) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (sid, data.get('seq'), time.time(), data.get('author', 'peter'),
             data['text'], 'viewer')
        )
        conn.commit()
        return jsonify({'ok': True})

    # --- Live step-by-step debugger ---
    # Use monkey-patched event from arc_live_debug if available, else local
    _step_event = getattr(sys.modules[__name__], '_live_step_event', None) or threading.Event()
    _step_ready = threading.Event()

    @app.route('/api/live/step', methods=['POST'])
    def api_live_step():
        """Peter clicks 'Next Step' → solver advances one action."""
        _step_event.set()  # Signal solver to take one step
        return jsonify({'ok': True, 'message': 'Step requested'})

    @app.route('/api/live/step-ready')
    def api_step_ready():
        """Check if solver is waiting for a step command."""
        return jsonify({'waiting': not _step_event.is_set()})

    # Make step_event accessible for solver integration
    app._step_event = _step_event

    @app.route('/api/live/status')
    def api_live_status():
        """Check if a live session is active."""
        if _live_session_info:
            return jsonify({'live': True, **_live_session_info})
        # Cross-process fallback: check for recent unfinished sessions
        conn = get_conn()
        row = conn.execute(
            "SELECT id, game_id, started_at FROM sessions "
            "WHERE finished_at IS NULL ORDER BY id DESC LIMIT 1"
        ).fetchone()
        if row:
            return jsonify({
                'live': True, 'session_id': row['id'],
                'game_id': row['game_id'], 'started_at': row['started_at'],
            })
        return jsonify({'live': False})

    @app.route('/api/live/stream')
    def api_live_stream():
        """SSE endpoint for live frame streaming.

        In-process mode: reads from _live_subscribers queue (low latency).
        Cross-process mode: polls SQLite for new frames (200ms latency).
        """
        def in_process_stream():
            q = subscribe_live()
            try:
                yield f"data: {json.dumps({'type': 'connected'})}\n\n"
                while True:
                    try:
                        frame_data = q.get(timeout=30)
                        yield f"data: {json.dumps(frame_data)}\n\n"
                    except queue.Empty:
                        yield f"data: {json.dumps({'type': 'heartbeat'})}\n\n"
            except GeneratorExit:
                unsubscribe_live(q)

        def cross_process_stream():
            nonlocal db_path  # from serve_replay scope
            # Use short-lived connections to avoid holding read locks
            def quick_query(sql, params=()):
                c = sqlite3.connect(db_path, timeout=5)
                c.row_factory = sqlite3.Row
                c.execute("PRAGMA journal_mode=WAL")
                try:
                    return c.execute(sql, params).fetchall()
                finally:
                    c.close()

            # Find the active session
            rows = quick_query(
                "SELECT id, game_id FROM sessions WHERE finished_at IS NULL "
                "ORDER BY id DESC LIMIT 1"
            )
            if not rows:
                yield f"data: {json.dumps({'type': 'no_session'})}\n\n"
                return
            sid = rows[0]['id']
            game_id = rows[0]['game_id']
            last_seq = 0
            yield f"data: {json.dumps({'type': 'connected', 'session_id': sid, 'game_id': game_id})}\n\n"
            while True:
                rows = quick_query(
                    "SELECT seq, timestamp, level, action_id, action_name, "
                    "frame_blob, frame_w, frame_h, solver_route, is_win "
                    "FROM frames WHERE session_id=? AND seq>? ORDER BY seq",
                    (sid, last_seq)
                )
                for r in rows:
                    frame = _decompress_frame(r['frame_blob'])
                    fd = {
                        'session_id': sid, 'seq': r['seq'],
                        'timestamp': r['timestamp'], 'level': r['level'],
                        'action_id': r['action_id'],
                        'action_name': r['action_name'] or '',
                        'solver_route': r['solver_route'] or '',
                        'is_win': bool(r['is_win']),
                        'game_id': game_id,
                    }
                    if frame.ndim == 3 and frame.shape[2] == 3:
                        import base64
                        from PIL import Image
                        img_pil = Image.fromarray(frame)
                        buf_jpg = io.BytesIO()
                        img_pil.save(buf_jpg, format='JPEG', quality=70)
                        fd['frame_b64'] = base64.b64encode(buf_jpg.getvalue()).decode()
                    else:
                        fd['frame'] = frame.tolist()
                    yield f"data: {json.dumps(fd)}\n\n"
                    last_seq = r['seq']
                # Check if session ended
                fin = quick_query(
                    "SELECT finished_at FROM sessions WHERE id=?", (sid,)
                )
                if fin and fin[0]['finished_at']:
                    yield f"data: {json.dumps({'type': 'session_end'})}\n\n"
                    return
                time.sleep(0.2)

        # Use in-process if we have subscribers infrastructure active,
        # cross-process otherwise
        use_in_process = bool(_live_session_info)
        gen = in_process_stream() if use_in_process else cross_process_stream()
        return Response(gen, content_type='text/event-stream',
                       headers={'Cache-Control': 'no-cache',
                                'X-Accel-Buffering': 'no'})

    print(f"\n  ARC Replay server: http://localhost:{port}")
    print(f"  DB: {db_path}\n")
    app.run(host='0.0.0.0', port=port, debug=False, threaded=True)


def list_sessions(db_path: str = DEFAULT_DB):
    """Print recorded sessions."""
    conn = _get_db(db_path)
    rows = conn.execute("""
        SELECT s.*,
               (SELECT COUNT(*) FROM frames WHERE session_id=s.id) as frame_count,
               (SELECT COUNT(*) FROM comments WHERE session_id=s.id) as comment_count
        FROM sessions s ORDER BY s.id DESC
    """).fetchall()

    if not rows:
        print("No recordings found.")
        return

    for r in rows:
        eff = f"{r['efficiency']:.0f}%" if r['efficiency'] else '—'
        print(f"  #{r['id']:3d} | {r['game_id']:8s} | {r['levels_solved']}/{r['total_levels']} levels "
              f"| {r['total_actions']} acts | {eff} eff | {r['frame_count']} frames "
              f"| {r['comment_count']} comments | {r['started_at']}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='ARC Recorder — replay viewer')
    parser.add_argument('--serve', action='store_true', help='Start replay web server')
    parser.add_argument('--list', action='store_true', help='List recordings')
    parser.add_argument('--port', type=int, default=8766)
    parser.add_argument('--db', default=DEFAULT_DB)
    args = parser.parse_args()

    if args.serve:
        serve_replay(db_path=args.db, port=args.port)
    elif args.list:
        list_sessions(db_path=args.db)
    else:
        parser.print_help()
