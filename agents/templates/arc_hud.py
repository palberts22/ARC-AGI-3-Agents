"""ARC HUD — Heads-Up Display for the solver's visual field.

The solver sees the game. The HUD shows it its own heartbeat.
Overlaid at the edges of the frame — game in center, instruments around it.

Components:
  Top bar:    System vitals (CPU, RAM, time)
  Left edge:  Route cascade status + failure patterns
  Right edge: Game progress (level, actions, efficiency)
  Bottom bar: Pilot status + last directive

The HUD serves two purposes:
  1. Visual: when recording (arc_recorder), humans see what the solver sees
  2. Cognitive: when the pilot engages with vision, it sees game AND instruments
     in one image — one API call, both streams

Design: Peter's idea — "build a little HUD into a corner of your eyes"
Implementation: Apollo, 2026-03-12.
"""

import time
import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

try:
    from PIL import Image, ImageDraw, ImageFont
    HAS_PIL = True
except ImportError:
    HAS_PIL = False

log = logging.getLogger("arc_hud")

# HUD colors (RGB)
COLOR_BG = (20, 20, 30)          # Dark blue-black
COLOR_TEXT = (200, 220, 240)      # Soft blue-white
COLOR_GOOD = (80, 220, 120)      # Green
COLOR_WARN = (255, 200, 60)      # Amber
COLOR_CRIT = (255, 80, 80)       # Red
COLOR_PILOT = (180, 120, 255)    # Purple — the mind
COLOR_ROUTE = (100, 180, 255)    # Light blue — routes
COLOR_DIM = (100, 110, 130)      # Dimmed text
COLOR_BAR_BG = (40, 42, 54)      # Progress bar background
COLOR_SEPARATOR = (60, 65, 80)   # Subtle line


@dataclass
class HUDState:
    """Everything the HUD needs to render one frame."""
    # System
    cpu_percent: float = 0.0
    memory_percent: float = 0.0
    memory_mb: float = 0.0
    uptime_s: float = 0.0

    # Game progress
    game_id: str = ""
    game_type: str = ""
    level: int = 0
    levels_total: int = 0
    levels_solved: int = 0
    actions_this_level: int = 0
    baseline_this_level: int = 0
    total_actions: int = 0
    total_baseline: int = 0
    efficiency: float = 0.0  # percentage

    # Time
    level_time_spent: float = 0.0
    level_time_budget: float = 0.0
    game_time_spent: float = 0.0

    # Active route
    active_route: str = ""
    route_cascade_position: int = 0  # which route in the cascade (0-15)

    # Route performance (from FlightRecorder)
    route_stats: dict = field(default_factory=dict)

    # Failure patterns (from FailureDetector)
    failure_warnings: list = field(default_factory=list)  # list of warning strings

    # Pilot
    pilot_active: bool = False
    pilot_budget: int = 0
    pilot_last_directive: str = ""
    pilot_success_rate: float = 0.0
    pilot_reasoning: str = ""


class FailureDetector:
    """Watches for patterns in failure. The self's pattern recognition.

    Detects:
    - Route futility: same route failing repeatedly on same game type
    - Type misclassification: routes for type X failing, maybe it's type Y
    - Time waste: routes consuming budget with no progress
    - Regression: a route that worked before stops working

    Optionally loads historical data from FlightRecorder's DB on startup.
    """

    def __init__(self, telemetry_db: str = ""):
        # Track failures: (game_type, route) -> [timestamps]
        self._failures: dict[tuple, list] = {}
        # Track successes for contrast
        self._successes: dict[tuple, list] = {}
        # Active warnings
        self.warnings: list[str] = []

        # Load historical route performance if DB available
        if telemetry_db:
            self._load_history(telemetry_db)

    def _load_history(self, db_path: str):
        """Bootstrap from FlightRecorder's pilot_telemetry.db."""
        try:
            import sqlite3
            db = sqlite3.connect(db_path)
            db.row_factory = sqlite3.Row
            rows = db.execute("""
                SELECT game_type, route, success, timestamp
                FROM route_performance
                ORDER BY timestamp
            """).fetchall()
            db.close()

            for row in rows:
                key = (row['game_type'] or 'UNKNOWN', row['route'] or 'unknown')
                ts = row['timestamp']
                if row['success']:
                    self._successes.setdefault(key, []).append(ts)
                else:
                    self._failures.setdefault(key, []).append(ts)

            total = len(rows)
            if total > 0:
                log.info(f"  [self] Loaded {total} historical route outcomes "
                         f"({len(self._failures)} failure keys, "
                         f"{len(self._successes)} success keys)")
                self._detect_patterns()
        except Exception as e:
            log.debug(f"  [self] Failed to load history: {e}")

    def record(self, game_type: str, route: str, success: bool,
               time_spent: float = 0.0):
        """Record a route outcome."""
        key = (game_type, route)
        if success:
            self._successes.setdefault(key, []).append(time.time())
        else:
            self._failures.setdefault(key, []).append(time.time())

        # Re-evaluate warnings
        self._detect_patterns()

    def _detect_patterns(self):
        """Scan for failure patterns."""
        self.warnings.clear()

        for (gtype, route), failures in self._failures.items():
            successes = self._successes.get((gtype, route), [])
            n_fail = len(failures)
            n_succ = len(successes)
            total = n_fail + n_succ

            if total < 3:
                continue  # Not enough data

            fail_rate = n_fail / total

            # Pattern 1: Route futility — high failure rate
            if fail_rate > 0.8 and n_fail >= 3:
                self.warnings.append(
                    f"FUTILE: {route} on {gtype}: {n_fail}/{total} failures "
                    f"({fail_rate:.0%}) — stop trying")

            # Pattern 2: Regression — had successes, now failing
            if n_succ > 0 and n_fail >= 2:
                last_success = max(successes)
                recent_failures = [t for t in failures if t > last_success]
                if len(recent_failures) >= 2:
                    self.warnings.append(
                        f"REGRESSION: {route} on {gtype}: worked before, "
                        f"now {len(recent_failures)} consecutive failures")

            # Pattern 3: Time sink — route takes long and fails
            # (would need time_spent tracking — future enhancement)

    def get_warnings(self, game_type: str = "") -> list[str]:
        """Get active warnings, optionally filtered by game type."""
        if not game_type:
            return self.warnings
        return [w for w in self.warnings if game_type in w]

    def should_skip_route(self, game_type: str, route: str) -> bool:
        """Should the solver skip this route based on failure patterns?"""
        key = (game_type, route)
        failures = self._failures.get(key, [])
        successes = self._successes.get(key, [])
        total = len(failures) + len(successes)

        if total < 3:
            return False  # Not enough data to judge

        fail_rate = len(failures) / total
        return fail_rate > 0.85 and len(failures) >= 4


class HUD:
    """Heads-Up Display renderer.

    Composites telemetry onto game frames. The solver's peripheral vision.
    """

    # HUD dimensions
    MARGIN = 4          # pixels between elements
    BAR_HEIGHT = 16     # top/bottom bars
    SIDE_WIDTH = 120    # left/right panels
    FONT_SIZE = 10

    def __init__(self, enabled: bool = True, show_sides: bool = True):
        self.enabled = enabled
        self.show_sides = show_sides
        self.state = HUDState()
        self.failure_detector = FailureDetector()
        self._font = None

        if HAS_PIL:
            try:
                self._font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
                                                self.FONT_SIZE)
            except (OSError, IOError):
                try:
                    self._font = ImageFont.load_default()
                except Exception:
                    pass

    def update(self, **kwargs):
        """Update HUD state. Pass any HUDState field as keyword arg."""
        for k, v in kwargs.items():
            if hasattr(self.state, k):
                setattr(self.state, k, v)

    def render(self, game_frame: np.ndarray) -> np.ndarray:
        """Composite HUD onto a game frame. Returns new frame with HUD overlay.

        The game frame is untouched in the center. HUD is added as borders.
        This means the HUD never obscures game content.
        """
        if not self.enabled or not HAS_PIL:
            return game_frame

        # Normalize frame to HxWxC
        frame = game_frame
        if frame.ndim == 3 and frame.shape[0] in (1, 3, 4):
            frame = np.transpose(frame, (1, 2, 0))
        if frame.ndim == 2:
            frame = np.stack([frame] * 3, axis=-1)

        gh, gw = frame.shape[:2]

        # Calculate HUD dimensions
        top_h = self.BAR_HEIGHT
        bot_h = self.BAR_HEIGHT
        left_w = self.SIDE_WIDTH if self.show_sides else 0
        right_w = self.SIDE_WIDTH if self.show_sides else 0

        total_w = left_w + gw + right_w
        total_h = top_h + gh + bot_h

        # Create HUD canvas
        canvas = np.full((total_h, total_w, 3), COLOR_BG, dtype=np.uint8)

        # Place game frame in center
        canvas[top_h:top_h + gh, left_w:left_w + gw, :3] = frame[:, :, :3]

        # Render HUD elements using PIL
        img = Image.fromarray(canvas)
        draw = ImageDraw.Draw(img)
        font = self._font

        # ─── TOP BAR: System vitals + time ───
        s = self.state
        vitals_text = f"CPU:{s.cpu_percent:.0f}%  RAM:{s.memory_percent:.0f}%  MEM:{s.memory_mb:.0f}MB"
        time_text = f"T:{s.level_time_spent:.1f}/{s.level_time_budget:.0f}s"
        game_text = f"{s.game_id.upper()} L{s.level}"

        # Vitals color based on pressure
        v_color = COLOR_GOOD if s.memory_percent < 70 else (
            COLOR_WARN if s.memory_percent < 85 else COLOR_CRIT)

        draw.text((self.MARGIN, 2), vitals_text, fill=v_color, font=font)
        draw.text((total_w // 2 - 30, 2), game_text, fill=COLOR_TEXT, font=font)
        draw.text((total_w - 130, 2), time_text, fill=COLOR_TEXT, font=font)

        if self.show_sides:
            # ─── LEFT PANEL: Route status + failure warnings ───
            y = top_h + self.MARGIN
            draw.text((self.MARGIN, y), "ROUTES", fill=COLOR_DIM, font=font)
            y += 14

            # Active route indicator
            if s.active_route:
                draw.text((self.MARGIN, y), f"► {s.active_route}", fill=COLOR_ROUTE, font=font)
                y += 12

            # Route stats (top 5)
            for route, stats in list(s.route_stats.items())[:5]:
                rate = stats.get('success_rate', 0)
                n = stats.get('attempts', 0)
                color = COLOR_GOOD if rate > 0.6 else (COLOR_WARN if rate > 0.3 else COLOR_CRIT)
                txt = f"{route[:10]:10s} {rate:.0%} ({n})"
                draw.text((self.MARGIN, y), txt, fill=color, font=font)
                y += 11

            # Failure warnings
            if s.failure_warnings:
                y += 4
                draw.text((self.MARGIN, y), "⚠ PATTERNS", fill=COLOR_WARN, font=font)
                y += 14
                for warn in s.failure_warnings[:3]:
                    # Truncate to fit
                    short = warn[:18]
                    draw.text((self.MARGIN, y), short, fill=COLOR_CRIT, font=font)
                    y += 11

            # ─── RIGHT PANEL: Game progress + pilot status ───
            x = left_w + gw + self.MARGIN
            y = top_h + self.MARGIN

            draw.text((x, y), "PROGRESS", fill=COLOR_DIM, font=font)
            y += 14

            # Level progress
            draw.text((x, y), f"Level: {s.level}/{s.levels_total or '?'}", fill=COLOR_TEXT, font=font)
            y += 12
            draw.text((x, y), f"Solved: {s.levels_solved}", fill=COLOR_GOOD, font=font)
            y += 12

            # Actions
            eff_color = COLOR_GOOD if s.efficiency > 100 else (
                COLOR_WARN if s.efficiency > 50 else COLOR_TEXT)
            draw.text((x, y), f"Acts: {s.actions_this_level}", fill=COLOR_TEXT, font=font)
            y += 12
            if s.baseline_this_level:
                draw.text((x, y), f"Base: {s.baseline_this_level}", fill=COLOR_DIM, font=font)
                y += 12
            if s.efficiency:
                draw.text((x, y), f"Eff: {s.efficiency:.0f}%", fill=eff_color, font=font)
                y += 12

            # Pilot status
            y += 8
            draw.text((x, y), "PILOT", fill=COLOR_DIM, font=font)
            y += 14

            if s.pilot_active:
                draw.text((x, y), f"ENGAGED", fill=COLOR_PILOT, font=font)
                y += 12
                if s.pilot_last_directive:
                    draw.text((x, y), s.pilot_last_directive[:15], fill=COLOR_PILOT, font=font)
                    y += 12
            else:
                draw.text((x, y), f"Standby ({s.pilot_budget})", fill=COLOR_DIM, font=font)
                y += 12

            if s.pilot_success_rate > 0:
                draw.text((x, y), f"Rate: {s.pilot_success_rate:.0%}",
                         fill=COLOR_GOOD if s.pilot_success_rate > 0.5 else COLOR_WARN, font=font)

        # ─── BOTTOM BAR: Pilot reasoning / status message ───
        bot_y = top_h + gh + 2
        if s.pilot_reasoning:
            draw.text((self.MARGIN, bot_y), f"PILOT: {s.pilot_reasoning[:80]}",
                     fill=COLOR_PILOT, font=font)
        elif s.active_route:
            draw.text((self.MARGIN, bot_y), f"Route: {s.active_route}  |  "
                     f"Game: {s.game_time_spent:.1f}s  |  "
                     f"Total: {s.total_actions} acts",
                     fill=COLOR_DIM, font=font)

        return np.array(img)

    def render_compact(self, game_frame: np.ndarray) -> np.ndarray:
        """Minimal HUD — just top bar, no side panels.

        For constrained displays or when side panels would distort
        the frame for vision LLMs.
        """
        old_show = self.show_sides
        self.show_sides = False
        result = self.render(game_frame)
        self.show_sides = old_show
        return result

    def text_hud(self) -> str:
        """Text-only HUD for non-vision contexts (logs, text LLMs).

        Returns a compact multi-line string.
        """
        s = self.state
        lines = []
        lines.append(f"═══ {s.game_id.upper()} L{s.level}/{s.levels_total or '?'} "
                     f"({s.levels_solved} solved) ═══")
        lines.append(f"CPU:{s.cpu_percent:.0f}% RAM:{s.memory_percent:.0f}% "
                     f"T:{s.level_time_spent:.1f}/{s.level_time_budget:.0f}s")

        if s.active_route:
            lines.append(f"Route: {s.active_route}")

        if s.route_stats:
            top = sorted(s.route_stats.items(),
                        key=lambda x: x[1].get('successes', 0), reverse=True)[:3]
            route_str = " | ".join(f"{r}:{st.get('success_rate',0):.0%}" for r, st in top)
            lines.append(f"Routes: {route_str}")

        if s.failure_warnings:
            for w in s.failure_warnings[:2]:
                lines.append(f"⚠ {w}")

        if s.pilot_active:
            lines.append(f"PILOT: {s.pilot_last_directive} ({s.pilot_success_rate:.0%})")
        else:
            lines.append(f"Pilot: standby ({s.pilot_budget} calls left)")

        return "\n".join(lines)
