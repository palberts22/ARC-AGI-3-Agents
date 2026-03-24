#!/usr/bin/env python3
"""
Dopamine — Reward Signal for the Common Sense Bus
===================================================

The missing piece: outcome-based reinforcement. Without this,
habits form from frequency ("I tried this a lot") not outcome
("this WORKED"). This module tracks wins/losses per route per
game type and adjusts habit strength accordingly.

Three functions the solver needs:
  reinforce(game_type, route_id, budget_pct)  — after a win
  weaken(game_type, route_id)                 — after a failure
  get_habit(game_type) → Habit or None        — before route selection

The habit tracker is SQLite-backed, persistent across sessions.
Total overhead: ~2ms per level (one read + one write).

Brain region: Basal Ganglia (habit formation) + VTA (reward signal)

Author: Archie | Date: 2026-03-13 | Directive: Peter
"""

import logging
import math
import os
import sqlite3
import time
from dataclasses import dataclass
from typing import Optional

log = logging.getLogger("dopamine")

# ---------------------------------------------------------------------------
# Data Structures
# ---------------------------------------------------------------------------

@dataclass
class Habit:
    """A learned route preference for a game type."""
    game_type: str
    route_id: str
    wins: int
    attempts: int
    success_rate: float
    avg_budget_pct: float   # average % of budget used when winning
    last_win: str           # ISO timestamp
    strength: float = 0.0   # 0.0 = no habit, 1.0 = automatic


@dataclass
class RewardEvent:
    """A single reward/punishment signal."""
    game_type: str
    route_id: str
    won: bool
    budget_pct: float       # how much budget was used (lower = better)
    timestamp: float


# ---------------------------------------------------------------------------
# Habit Tracker (Basal Ganglia)
# ---------------------------------------------------------------------------

class HabitTracker:
    """Persistent habit formation from outcome-based reinforcement.

    The lifecycle:
      1. First encounter: full cascade, no habits. (Deliberative)
      2. First win: route gets recorded. (Learning)
      3. Third win, same type: habit_strength > 0.7 → default route. (Habit)
      4. Failure: habit weakens (-0.1 per fail). (Extinction)
      5. No wins for 7 days: recency factor decays. (Forgetting)

    Args:
        db_path: Path to SQLite DB (default: data/solver_habits.db)
        habit_threshold: wins needed to form a habit (default: 3)
        strength_gate: minimum strength to be considered a habit (default: 0.7)
    """

    SCHEMA = """
    CREATE TABLE IF NOT EXISTS route_habits (
        game_type TEXT NOT NULL,
        route_id TEXT NOT NULL,
        wins INTEGER DEFAULT 0,
        attempts INTEGER DEFAULT 0,
        avg_budget_pct REAL DEFAULT 1.0,
        last_win TEXT,
        last_attempt TEXT,
        habit_strength REAL DEFAULT 0.0,
        PRIMARY KEY (game_type, route_id)
    );

    CREATE TABLE IF NOT EXISTS reward_log (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        game_type TEXT NOT NULL,
        route_id TEXT NOT NULL,
        won INTEGER NOT NULL,
        budget_pct REAL,
        timestamp TEXT DEFAULT (datetime('now')),
        notes TEXT
    );
    """

    def __init__(self, db_path: str = None, habit_threshold: int = 3,
                 strength_gate: float = 0.7):
        if db_path is None:
            db_path = os.path.join(
                os.path.dirname(__file__), '..', 'data', 'solver_habits.db'
            )
        self.db_path = db_path
        self.threshold = habit_threshold
        self.strength_gate = strength_gate
        self._ensure_schema()

    def _ensure_schema(self):
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        conn = sqlite3.connect(self.db_path)
        conn.executescript(self.SCHEMA)
        conn.commit()
        conn.close()

    def _connect(self):
        return sqlite3.connect(self.db_path)

    # -------------------------------------------------------------------
    # Core API
    # -------------------------------------------------------------------

    def reinforce(self, game_type: str, route_id: str,
                  budget_pct: float = 1.0, notes: str = "") -> Habit:
        """Record a win. Strengthen the habit.

        Args:
            game_type: Game category (e.g. 'TOGGLE', 'NAVIGATION')
            route_id: Route that won (e.g. 'constraint', 'bfs')
            budget_pct: Fraction of time budget used (0-1, lower=faster=better)
            notes: Optional context

        Returns:
            Updated Habit object
        """
        conn = self._connect()
        cur = conn.cursor()

        # Log the event
        cur.execute(
            "INSERT INTO reward_log (game_type, route_id, won, budget_pct, notes) "
            "VALUES (?, ?, 1, ?, ?)",
            (game_type, route_id, budget_pct, notes)
        )

        # Upsert the habit
        cur.execute("""
            INSERT INTO route_habits
                (game_type, route_id, wins, attempts, avg_budget_pct,
                 last_win, last_attempt, habit_strength)
            VALUES (?, ?, 1, 1, ?, datetime('now'), datetime('now'), ?)
            ON CONFLICT(game_type, route_id) DO UPDATE SET
                wins = wins + 1,
                attempts = attempts + 1,
                avg_budget_pct = (avg_budget_pct * wins + ?) / (wins + 1),
                last_win = datetime('now'),
                last_attempt = datetime('now'),
                habit_strength = MIN(1.0,
                    CAST(wins + 1 AS REAL) / ? *
                    CAST(wins + 1 AS REAL) / MAX(attempts + 1, 1))
        """, (game_type, route_id, budget_pct,
              min(1.0, 1.0 / self.threshold),
              budget_pct, self.threshold))

        conn.commit()

        # Read back the updated habit
        habit = self._read_habit(cur, game_type, route_id)
        conn.close()

        log.info(f"  [dopamine] +reinforce {game_type}/{route_id} "
                 f"→ strength={habit.strength:.2f} "
                 f"(wins={habit.wins}, rate={habit.success_rate:.0%})")
        return habit

    def weaken(self, game_type: str, route_id: str,
               notes: str = "") -> Optional[Habit]:
        """Record a failure. Weaken the habit.

        Asymmetric: -0.1 per failure (vs strength gained per win).
        One win proves a route works; one failure might just be wrong context.
        """
        conn = self._connect()
        cur = conn.cursor()

        # Log the event
        cur.execute(
            "INSERT INTO reward_log (game_type, route_id, won, budget_pct, notes) "
            "VALUES (?, ?, 0, NULL, ?)",
            (game_type, route_id, notes)
        )

        # Update — only if exists
        cur.execute("""
            UPDATE route_habits SET
                attempts = attempts + 1,
                last_attempt = datetime('now'),
                habit_strength = MAX(0.0, habit_strength - 0.1)
            WHERE game_type = ? AND route_id = ?
        """, (game_type, route_id))

        conn.commit()

        habit = self._read_habit(cur, game_type, route_id)
        conn.close()

        if habit:
            log.info(f"  [dopamine] -weaken {game_type}/{route_id} "
                     f"→ strength={habit.strength:.2f}")
        return habit

    def get_habit(self, game_type: str) -> Optional[Habit]:
        """Get the strongest habit for a game type.

        Returns None if no habit exceeds the strength gate.
        Applies recency decay: habits unused for 7+ days weaken.
        """
        conn = self._connect()
        cur = conn.cursor()

        cur.execute("""
            SELECT game_type, route_id, wins, attempts, avg_budget_pct,
                   last_win, last_attempt, habit_strength
            FROM route_habits
            WHERE game_type = ? AND habit_strength > 0.1
            ORDER BY habit_strength DESC
            LIMIT 1
        """, (game_type,))

        row = cur.fetchone()
        conn.close()

        if not row:
            return None

        habit = Habit(
            game_type=row[0], route_id=row[1],
            wins=row[2], attempts=row[3],
            success_rate=row[2] / max(row[3], 1),
            avg_budget_pct=row[4],
            last_win=row[5] or "",
        )

        # Apply recency decay
        if habit.last_win:
            try:
                from datetime import datetime
                last = datetime.fromisoformat(habit.last_win)
                days_ago = (datetime.now() - last).total_seconds() / 86400
                recency = math.exp(-days_ago / 7.0)
                habit.strength = row[7] * recency
            except (ValueError, TypeError):
                habit.strength = row[7]
        else:
            habit.strength = row[7]

        if habit.strength < self.strength_gate:
            return None

        return habit

    def get_all_habits(self, min_strength: float = 0.1) -> list[Habit]:
        """Get all habits above a minimum strength."""
        conn = self._connect()
        cur = conn.cursor()

        cur.execute("""
            SELECT game_type, route_id, wins, attempts, avg_budget_pct,
                   last_win, last_attempt, habit_strength
            FROM route_habits
            WHERE habit_strength > ?
            ORDER BY habit_strength DESC
        """, (min_strength,))

        habits = []
        for row in cur.fetchall():
            habits.append(Habit(
                game_type=row[0], route_id=row[1],
                wins=row[2], attempts=row[3],
                success_rate=row[2] / max(row[3], 1),
                avg_budget_pct=row[4],
                last_win=row[5] or "",
            ))
            habits[-1].strength = row[7]

        conn.close()
        return habits

    def summary(self) -> dict:
        """Debug summary of the habit system."""
        conn = self._connect()
        cur = conn.cursor()

        cur.execute("SELECT COUNT(*) FROM route_habits WHERE habit_strength > 0.7")
        strong = cur.fetchone()[0]

        cur.execute("SELECT COUNT(*) FROM route_habits")
        total = cur.fetchone()[0]

        cur.execute("SELECT COUNT(*) FROM reward_log")
        events = cur.fetchone()[0]

        cur.execute("SELECT COUNT(*) FROM reward_log WHERE won = 1")
        wins = cur.fetchone()[0]

        conn.close()

        return {
            "strong_habits": strong,
            "total_routes_tried": total,
            "reward_events": events,
            "total_wins": wins,
            "win_rate": wins / max(events, 1),
        }

    def _read_habit(self, cur, game_type: str, route_id: str) -> Optional[Habit]:
        cur.execute("""
            SELECT game_type, route_id, wins, attempts, avg_budget_pct,
                   last_win, last_attempt, habit_strength
            FROM route_habits
            WHERE game_type = ? AND route_id = ?
        """, (game_type, route_id))
        row = cur.fetchone()
        if not row:
            return None
        return Habit(
            game_type=row[0], route_id=row[1],
            wins=row[2], attempts=row[3],
            success_rate=row[2] / max(row[3], 1),
            avg_budget_pct=row[4],
            last_win=row[5] or "",
            strength=row[7],
        )


# ---------------------------------------------------------------------------
# Integration Helper — wire into CommonSenseBus
# ---------------------------------------------------------------------------

def wire_dopamine_to_bus(bus, tracker: HabitTracker):
    """Monkey-patch the Common Sense Bus to use dopamine rewards.

    After calling this:
      - bus.record_win() also reinforces habits
      - bus.record_fail() also weakens habits
      - bus.pre_cascade() checks habits first

    Usage:
        from dopamine import HabitTracker, wire_dopamine_to_bus
        tracker = HabitTracker()
        wire_dopamine_to_bus(bus, tracker)
    """
    bus._habit_tracker = tracker

    # Wrap record_win to also reinforce
    original_record_win = bus.record_win

    def record_win_with_dopamine(fingerprint, route_name, actions,
                                  solve_time, profile=None, frame=None):
        result = original_record_win(fingerprint, route_name, actions,
                                     solve_time, profile, frame)
        game_type = result.get("game_type", "unknown")
        budget_pct = solve_time / 300.0  # normalize to 5min budget
        tracker.reinforce(game_type, route_name, budget_pct)
        return result

    bus.record_win = record_win_with_dopamine

    # Wrap record_fail to also weaken
    original_record_fail = bus.record_fail

    def record_fail_with_dopamine():
        original_record_fail()
        for route_name, reason in bus.failed_routes:
            game_type = "unknown"
            if hasattr(bus, '_last_profile') and bus._last_profile:
                game_type = getattr(bus._last_profile, 'game_type', 'unknown')
            tracker.weaken(game_type, route_name)

    bus.record_fail = record_fail_with_dopamine


# ---------------------------------------------------------------------------
# CLI Test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import tempfile

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(message)s",
                        datefmt="%H:%M:%S")

    # Use temp DB for testing
    with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as f:
        db_path = f.name

    tracker = HabitTracker(db_path=db_path)

    print("=== Dopamine Test ===\n")

    # Simulate: TOGGLE game, constraint solver wins 3 times
    print("--- Simulating 3 wins for constraint on TOGGLE ---")
    for i in range(3):
        h = tracker.reinforce("TOGGLE", "constraint", budget_pct=0.3 - i * 0.05)
        print(f"  Win {i+1}: strength={h.strength:.2f}, wins={h.wins}")

    # Check habit
    habit = tracker.get_habit("TOGGLE")
    print(f"\nHabit formed: {habit.route_id} (strength={habit.strength:.2f})" if habit
          else "\nNo habit yet")

    # Simulate: one failure
    print("\n--- One failure ---")
    tracker.weaken("TOGGLE", "constraint")
    habit = tracker.get_habit("TOGGLE")
    print(f"After failure: strength={habit.strength:.2f}" if habit
          else "Habit extinct")

    # Simulate: BFS tried once, fails
    print("\n--- BFS tried, fails ---")
    tracker.weaken("TOGGLE", "bfs")

    # Summary
    print(f"\n{tracker.summary()}")

    # Cleanup
    os.unlink(db_path)
    print("\nDone.")
