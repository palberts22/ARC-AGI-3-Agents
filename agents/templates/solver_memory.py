"""Solver Long-Term Memory — persistent rulebook for the ARC solver.

The solver plays games, learns what works, and remembers it forever.
Every game makes the next game easier. Without code changes.
Without retraining. Just by remembering what worked.

Spec: specs/solver_memory_spec.md
Design: Archie, 2026-03-12. Peter's direction.

Usage:
    from solver_memory import SolverMemory

    mem = SolverMemory()

    # Before a game — consult the rulebook
    rules = mem.get_rulebook(genre="physics")
    similar = mem.get_similar_games(genre="physics")
    print(mem.briefing(genre="physics"))

    # After a game — persist what you learned
    mem.record_game("flappy_01", "MiniFlappy", "physics",
                    levels_solved=10, levels_total=10,
                    total_actions=142, winning_route="reactive_control",
                    duration_s=8.3)

    # Absorb Eyes' insights (bridge from in-memory to persistent)
    mem.absorb_insights(eyes_memory.insights, "flappy_01", "MiniFlappy", "physics")

    # Record what winning/losing looks like
    mem.record_state("flappy_01", "MiniFlappy", "physics", "win",
                     frame, {"bird_alive": True, "past_pipes": 10}, level=9)
"""

import sqlite3
import json
import hashlib
import logging
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

log = logging.getLogger("solver_memory")

# Default DB path — same data/ directory as our other DBs
_DEFAULT_DB = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "solver_memory.db")

# Genre taxonomy
GENRES = {
    "toggle",        # actions reverse each other, modular arithmetic
    "circuit",       # seesaw pairs, hydraulic coupling
    "spatial",       # objects need to reach target positions
    "physics",       # gravity, momentum, collision
    "reactive",      # real-time, continuous threat
    "adversarial",   # opponent responds to your moves
    "resource",      # track HP/ammo/score, don't die
    "combinatorial", # constraint satisfaction
    "recursive",     # self-similar subproblems
    "unknown",       # not yet classified
}


def tags_from_profile(profile) -> list[str]:
    """Generate search tags from a GameProfile (OPRAH data).

    Tags encode the game's structural fingerprint:
    - genre/mechanic: "toggle", "navigation", "constraint"
    - grid dimensions: "grid_8x8", "grid_16x16"
    - action count range: "actions_4-6", "actions_7-10"
    - modalities: "has_click", "has_grid", "has_movement"
    """
    tags = []

    # Genre
    if hasattr(profile, 'genre') and profile.genre:
        tags.append(profile.genre.lower().replace(" ", "_"))
    if hasattr(profile, 'genre_hints') and profile.genre_hints:
        for hint in profile.genre_hints:
            tags.append(hint.lower().replace(" ", "_"))

    # Grid dimensions from OPRAH
    if hasattr(profile, 'grid_dims') and profile.grid_dims:
        h, w = profile.grid_dims
        tags.append(f"grid_{h}x{w}")

    # Action count bucket
    if hasattr(profile, 'n_actions') and profile.n_actions:
        n = profile.n_actions
        if n <= 3:
            tags.append("actions_1-3")
        elif n <= 6:
            tags.append("actions_4-6")
        elif n <= 10:
            tags.append("actions_7-10")
        else:
            tags.append("actions_11+")

    # Modalities from OPRAH action counts
    if hasattr(profile, 'oprah_action_counts') and profile.oprah_action_counts:
        counts = profile.oprah_action_counts
        if counts.get("CLICK_CELL") or counts.get("CLICK_SEQUENCE"):
            tags.append("has_click")
        if counts.get("NAVIGATE") or counts.get("MOVEMENT"):
            tags.append("has_movement")
    if hasattr(profile, 'oprah_has_grid') and profile.oprah_has_grid:
        tags.append("has_grid")
    if hasattr(profile, 'oprah_has_click') and profile.oprah_has_click:
        tags.append("has_click")

    # Game type
    if hasattr(profile, 'game_type') and profile.game_type:
        tags.append(f"type_{profile.game_type.name.lower()}")

    return sorted(set(tags))


class SolverMemory:
    """Long-term memory for the ARC solver. Persists across sessions.

    Three tables:
      rules           — pattern/mechanism/strategy triples (the rulebook)
      game_results    — what happened (outcomes + winning routes)
      state_signatures — what winning/losing look like (visual features)
    """

    def __init__(self, db_path: str = _DEFAULT_DB):
        self.db_path = db_path
        self.db = sqlite3.connect(db_path)
        self.db.row_factory = sqlite3.Row
        self._init_tables()
        log.info(f"SolverMemory opened: {db_path}")

    def _init_tables(self):
        self.db.executescript("""
            CREATE TABLE IF NOT EXISTS rules (
                id          INTEGER PRIMARY KEY,
                genre       TEXT NOT NULL,
                pattern     TEXT NOT NULL,
                mechanism   TEXT NOT NULL,
                strategy    TEXT NOT NULL,
                confidence  REAL DEFAULT 0.5,
                evidence    TEXT DEFAULT '[]',
                created_at  TEXT NOT NULL,
                updated_at  TEXT NOT NULL,
                source      TEXT DEFAULT '',
                UNIQUE(genre, pattern)
            );

            CREATE TABLE IF NOT EXISTS game_results (
                id              INTEGER PRIMARY KEY,
                game_id         TEXT NOT NULL,
                game_type       TEXT NOT NULL,
                genre           TEXT DEFAULT '',
                levels_total    INTEGER,
                levels_solved   INTEGER,
                total_actions   INTEGER,
                winning_route   TEXT DEFAULT '',
                duration_s      REAL,
                notes           TEXT DEFAULT '',
                created_at      TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS state_signatures (
                id          INTEGER PRIMARY KEY,
                game_id     TEXT NOT NULL,
                game_type   TEXT NOT NULL,
                genre       TEXT DEFAULT '',
                state_type  TEXT NOT NULL,
                frame_hash  TEXT DEFAULT '',
                features    TEXT NOT NULL,
                level       INTEGER DEFAULT 0,
                created_at  TEXT NOT NULL
            );

            CREATE INDEX IF NOT EXISTS idx_rules_genre ON rules(genre);
            CREATE INDEX IF NOT EXISTS idx_rules_confidence ON rules(confidence DESC);
            CREATE INDEX IF NOT EXISTS idx_game_results_genre ON game_results(genre);
            CREATE INDEX IF NOT EXISTS idx_game_results_game_type ON game_results(game_type);
            CREATE INDEX IF NOT EXISTS idx_state_sigs_genre ON state_signatures(genre);
            CREATE INDEX IF NOT EXISTS idx_state_sigs_type ON state_signatures(state_type);
        """)
        self.db.commit()

        # Schema migration: add tags column (backwards-compatible)
        for table in ("game_results", "rules"):
            try:
                self.db.execute(f"SELECT tags FROM {table} LIMIT 1")
            except sqlite3.OperationalError:
                self.db.execute(f"ALTER TABLE {table} ADD COLUMN tags TEXT DEFAULT '[]'")
                self.db.commit()
                log.info(f"Migrated {table}: added tags column")

    def close(self):
        self.db.close()

    def _now(self) -> str:
        return datetime.now(timezone.utc).isoformat()

    # ─── WRITE (rules) ─────────────────────────────────────────────

    def add_rule(self, pattern: str, mechanism: str, strategy: str,
                 genre: str = "unknown", confidence: float = 0.5,
                 source: str = "",
                 tags: Optional[list[str]] = None) -> int:
        """Add or update a rule in the rulebook.

        Uses UPSERT — if (genre, pattern) exists, updates confidence/mechanism.
        Called by the pilot when it discovers something new.
        """
        genre = genre if genre in GENRES else "unknown"
        tags_json = json.dumps(sorted(set(tags))) if tags else "[]"
        now = self._now()
        try:
            cur = self.db.execute("""
                INSERT INTO rules (genre, pattern, mechanism, strategy, confidence,
                                   source, tags, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(genre, pattern) DO UPDATE SET
                    mechanism = excluded.mechanism,
                    strategy = excluded.strategy,
                    confidence = MAX(rules.confidence, excluded.confidence),
                    source = excluded.source,
                    tags = excluded.tags,
                    updated_at = excluded.updated_at
            """, (genre, pattern, mechanism, strategy, confidence, source, tags_json, now, now))
            self.db.commit()
            log.info(f"Rule added/updated: [{genre}] {pattern} → {strategy} "
                     f"(conf={confidence:.2f}, src={source})")
            return cur.lastrowid
        except Exception as e:
            log.warning(f"Failed to add rule: {e}")
            return -1

    # ─── WRITE (after game) ───────────────────────────────────────

    def record_game(self, game_id: str, game_type: str, genre: str,
                    levels_solved: int, levels_total: int,
                    total_actions: int, winning_route: str = "",
                    duration_s: float = 0.0, notes: str = "",
                    tags: Optional[list[str]] = None) -> int:
        """Record a completed game. Returns game_result id.

        Tags are OPRAH-derived descriptors: mechanic type, grid dims, action count range.
        Example: ["toggle", "grid_8x8", "actions_4-6", "has_click", "constraint"]
        """
        genre = genre if genre in GENRES else "unknown"
        tags_json = json.dumps(sorted(set(tags))) if tags else "[]"
        cur = self.db.execute("""
            INSERT INTO game_results
                (game_id, game_type, genre, levels_total, levels_solved,
                 total_actions, winning_route, duration_s, notes, tags, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (game_id, game_type, genre, levels_total, levels_solved,
              total_actions, winning_route, duration_s, notes, tags_json, self._now()))
        self.db.commit()
        log.info(f"Recorded game: {game_id} ({game_type}/{genre}) "
                 f"— {levels_solved}/{levels_total} solved, "
                 f"{total_actions} actions, route={winning_route}")
        return cur.lastrowid

    def record_state(self, game_id: str, game_type: str, genre: str,
                     state_type: str, frame: Optional[np.ndarray],
                     features: dict, level: int = 0):
        """Record a win/loss/stuck/progress state with extracted features.

        Args:
            state_type: 'win', 'lose', 'stuck', 'progress'
            frame: the raw frame (used for hashing, not stored)
            features: dict of observable properties
        """
        if state_type not in ("win", "lose", "stuck", "progress"):
            log.warning(f"Unknown state_type: {state_type}, defaulting to 'progress'")
            state_type = "progress"

        genre = genre if genre in GENRES else "unknown"
        frame_hash = ""
        if frame is not None:
            frame_hash = hashlib.md5(frame.tobytes()).hexdigest()[:16]

        self.db.execute("""
            INSERT INTO state_signatures
                (game_id, game_type, genre, state_type, frame_hash,
                 features, level, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (game_id, game_type, genre, state_type, frame_hash,
              json.dumps(features), level, self._now()))
        self.db.commit()
        log.info(f"Recorded {state_type} state: {game_id} L{level} "
                 f"— {len(features)} features")

    def store_rule(self, genre: str, pattern: str, mechanism: str,
                   strategy: str, confidence: float = 0.5,
                   evidence: Optional[list] = None,
                   source: str = "reflect"):
        """Store or strengthen a rule.

        If genre+pattern already exists:
          - confidence increases asymptotically toward 1.0
          - evidence appends
          - mechanism/strategy update if new confidence is higher
        """
        genre = genre if genre in GENRES else "unknown"
        evidence = evidence or []
        now = self._now()

        existing = self.db.execute(
            "SELECT id, confidence, evidence, mechanism, strategy "
            "FROM rules WHERE genre = ? AND pattern = ?",
            (genre, pattern)
        ).fetchone()

        if existing:
            old_conf = existing["confidence"]
            old_evidence = json.loads(existing["evidence"])

            # Asymptotic confidence increase
            new_conf = old_conf + (1.0 - old_conf) * 0.3
            new_conf = min(1.0, max(new_conf, confidence))

            # Merge evidence (dedup by game_id)
            seen_games = {e.get("game_id") for e in old_evidence if isinstance(e, dict)}
            for e in evidence:
                gid = e.get("game_id") if isinstance(e, dict) else e
                if gid not in seen_games:
                    old_evidence.append(e)
                    seen_games.add(gid)

            # Update mechanism/strategy if confidence improved
            mech = mechanism if confidence > old_conf else existing["mechanism"]
            strat = strategy if confidence > old_conf else existing["strategy"]

            self.db.execute("""
                UPDATE rules SET confidence = ?, evidence = ?,
                    mechanism = ?, strategy = ?, updated_at = ?
                WHERE id = ?
            """, (new_conf, json.dumps(old_evidence), mech, strat,
                  now, existing["id"]))
            self.db.commit()
            log.info(f"Strengthened rule: [{genre}] {pattern} "
                     f"({old_conf:.2f} → {new_conf:.2f})")
        else:
            self.db.execute("""
                INSERT INTO rules
                    (genre, pattern, mechanism, strategy, confidence,
                     evidence, created_at, updated_at, source)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (genre, pattern, mechanism, strategy, confidence,
                  json.dumps(evidence), now, now, source))
            self.db.commit()
            log.info(f"New rule: [{genre}] {pattern} (conf={confidence:.2f})")

    def absorb_insights(self, insights: list, game_id: str,
                        game_type: str, genre: str):
        """Bridge: take Eyes' in-memory insights[] and persist them as rules.

        This is the migration path. Eyes keeps working exactly as-is.
        Call this from end_game() and the learning becomes permanent.

        Expected insight format (from EpisodicMemory.reflect()):
            {
                'pattern': str,
                'mechanism': str,
                'strategy': str,
                'evidence': [game_id, ...],
                'confidence': float
            }
        """
        if not insights:
            return

        count = 0
        for insight in insights:
            pattern = insight.get("pattern", "")
            mechanism = insight.get("mechanism", "")
            strategy = insight.get("strategy", "")
            confidence = insight.get("confidence", 0.5)

            if not pattern or not mechanism:
                continue

            # Build evidence entry
            evidence_entry = {
                "game_id": game_id,
                "game_type": game_type,
                "outcome": "observed",
            }

            # Merge with existing evidence from the insight
            raw_evidence = insight.get("evidence", [])
            evidence_list = []
            for e in raw_evidence:
                if isinstance(e, str):
                    evidence_list.append({"game_id": e, "outcome": "observed"})
                elif isinstance(e, dict):
                    evidence_list.append(e)
            if not any(e.get("game_id") == game_id for e in evidence_list):
                evidence_list.append(evidence_entry)

            self.store_rule(genre, pattern, mechanism, strategy,
                           confidence, evidence_list, source="reflect")
            count += 1

        log.info(f"Absorbed {count} insights from {game_id} ({game_type}/{genre})")

    def absorb_hypotheses(self, hypotheses: list, game_id: str,
                          game_type: str, genre: str):
        """Bridge: take InductiveReasoner hypotheses and persist as rules.

        Hypothesis format (from arc_reason.Hypothesis):
            rule_type: str (toggle, move, counter, commit_button, win_condition)
            description: str
            parameters: dict
            confidence: float
            tests_passed: int
        """
        if not hypotheses:
            return

        count = 0
        for hyp in hypotheses:
            if hyp.confidence < 0.3:
                continue

            pattern = f"[{hyp.rule_type}] {hyp.description}"
            mechanism = json.dumps(hyp.parameters) if hasattr(hyp, 'parameters') else ""
            strategy = f"reasoner_{hyp.rule_type}"

            evidence_list = [{
                "game_id": game_id,
                "game_type": game_type,
                "outcome": "hypothesis",
                "tests_passed": hyp.tests_passed
            }]

            self.store_rule(genre, pattern, mechanism, strategy,
                           hyp.confidence, evidence_list, source="reasoner")
            count += 1

        log.info(f"Absorbed {count} hypotheses from {game_id} ({game_type}/{genre})")

    # ─── READ (before/during game) ────────────────────────────────

    def get_rulebook(self, genre: str = "",
                     min_confidence: float = 0.3) -> list[dict]:
        """Query rules for a genre. Returns sorted by confidence desc.

        If genre is empty, returns all rules above min_confidence.
        This is what the solver reads at Phase 0 (Beginner's Mind).
        """
        if genre:
            rows = self.db.execute(
                "SELECT * FROM rules WHERE genre = ? AND confidence >= ? "
                "ORDER BY confidence DESC",
                (genre, min_confidence)
            ).fetchall()
        else:
            rows = self.db.execute(
                "SELECT * FROM rules WHERE confidence >= ? "
                "ORDER BY confidence DESC",
                (min_confidence,)
            ).fetchall()

        return [dict(r) for r in rows]

    def get_similar_games(self, game_type: str = "",
                          genre: str = "") -> list[dict]:
        """Find past games of same type or genre.

        Returns game_results with winning routes and action counts.
        Tells the solver what worked before on similar games.
        """
        if game_type:
            rows = self.db.execute(
                "SELECT * FROM game_results WHERE game_type = ? "
                "ORDER BY created_at DESC LIMIT 20",
                (game_type,)
            ).fetchall()
        elif genre:
            rows = self.db.execute(
                "SELECT * FROM game_results WHERE genre = ? "
                "ORDER BY created_at DESC LIMIT 20",
                (genre,)
            ).fetchall()
        else:
            rows = self.db.execute(
                "SELECT * FROM game_results ORDER BY created_at DESC LIMIT 20"
            ).fetchall()

        return [dict(r) for r in rows]

    def recall_by_tags(self, query_tags: list[str],
                       min_overlap: int = 2,
                       limit: int = 20) -> dict:
        """Recall rules and game results by tag intersection.

        Like emrys vector search but deterministic: finds entries sharing
        at least `min_overlap` tags with query_tags.

        Args:
            query_tags: Tags from OPRAH profile (e.g. ["toggle", "grid_8x8", "actions_6"])
            min_overlap: Minimum tags that must match (default 2)
            limit: Max results per table

        Returns:
            {"rules": [...], "games": [...], "signatures": [...]}
        """
        query_set = set(query_tags)
        results = {"rules": [], "games": [], "signatures": []}

        # Score rules by tag overlap
        rows = self.db.execute(
            "SELECT *, tags FROM rules WHERE tags != '[]' "
            "ORDER BY confidence DESC"
        ).fetchall()
        for r in rows:
            try:
                row_tags = set(json.loads(r["tags"]))
            except (json.JSONDecodeError, KeyError):
                continue
            overlap = len(query_set & row_tags)
            if overlap >= min_overlap:
                d = dict(r)
                d["_tag_overlap"] = overlap
                results["rules"].append(d)
        results["rules"].sort(key=lambda x: (-x["_tag_overlap"], -x.get("confidence", 0)))
        results["rules"] = results["rules"][:limit]

        # Score games by tag overlap
        rows = self.db.execute(
            "SELECT * FROM game_results WHERE tags != '[]' "
            "ORDER BY created_at DESC"
        ).fetchall()
        for r in rows:
            try:
                row_tags = set(json.loads(r["tags"]))
            except (json.JSONDecodeError, KeyError):
                continue
            overlap = len(query_set & row_tags)
            if overlap >= min_overlap:
                d = dict(r)
                d["_tag_overlap"] = overlap
                results["games"].append(d)
        results["games"].sort(key=lambda x: -x["_tag_overlap"])
        results["games"] = results["games"][:limit]

        return results

    def get_win_signatures(self, genre: str = "",
                           game_type: str = "") -> list[dict]:
        """What does winning look like for this genre?"""
        return self._get_signatures("win", genre, game_type)

    def get_lose_signatures(self, genre: str = "",
                            game_type: str = "") -> list[dict]:
        """What does losing look like? Early detection = early correction."""
        return self._get_signatures("lose", genre, game_type)

    def _get_signatures(self, state_type: str, genre: str = "",
                        game_type: str = "") -> list[dict]:
        conditions = ["state_type = ?"]
        params = [state_type]

        if genre:
            conditions.append("genre = ?")
            params.append(genre)
        if game_type:
            conditions.append("game_type = ?")
            params.append(game_type)

        where = " AND ".join(conditions)
        rows = self.db.execute(
            f"SELECT * FROM state_signatures WHERE {where} "
            "ORDER BY created_at DESC LIMIT 50",
            params
        ).fetchall()

        results = []
        for r in rows:
            d = dict(r)
            d["features"] = json.loads(d["features"])
            results.append(d)
        return results

    def briefing(self, game_type: str = "", genre: str = "") -> str:
        """Human-readable pre-game briefing. Like a coach's notes.

        Example:
            ┌─ Briefing: physics ──────────────────────────┐
            │ 12 games played, 11 won (92%)                │
            │ Best route: reactive_control (8 wins)        │
            │ Rules:                                       │
            │  [95%] Learn gravity before planning         │
            │  [82%] Prioritize high-delta actions         │
            │ Win pattern: objects in target zone           │
            │ Lose pattern: player off screen              │
            └──────────────────────────────────────────────┘
        """
        lines = []
        label = game_type or genre or "all games"

        # Game history
        games = self.get_similar_games(game_type=game_type, genre=genre)
        if games:
            total = len(games)
            won = sum(1 for g in games if g["levels_solved"] == g["levels_total"])
            win_pct = (won / total * 100) if total else 0

            # Best route
            routes = {}
            for g in games:
                rt = g.get("winning_route", "")
                if rt:
                    routes[rt] = routes.get(rt, 0) + 1
            best_route = max(routes, key=routes.get) if routes else "none"
            best_count = routes.get(best_route, 0) if routes else 0

            lines.append(f"┌─ Briefing: {label} ─{'─' * max(0, 40 - len(label))}┐")
            lines.append(f"│ {total} games played, {won} won ({win_pct:.0f}%)")
            if best_route != "none":
                lines.append(f"│ Best route: {best_route} ({best_count} wins)")
        else:
            lines.append(f"┌─ Briefing: {label} ─{'─' * max(0, 40 - len(label))}┐")
            lines.append(f"│ No previous games on record.")

        # Rules
        rules = self.get_rulebook(genre=genre, min_confidence=0.3)
        if rules:
            lines.append("│ Rules:")
            for r in rules[:5]:
                lines.append(f"│  [{r['confidence']:.0%}] {r['strategy']}")
        else:
            lines.append("│ No rules yet — explore freely.")

        # Win/lose signatures
        wins = self.get_win_signatures(genre=genre, game_type=game_type)
        if wins:
            # Summarize common win features
            all_features = {}
            for w in wins[:10]:
                for k, v in w["features"].items():
                    if k not in all_features:
                        all_features[k] = []
                    all_features[k].append(v)
            if all_features:
                top_feature = max(all_features, key=lambda k: len(all_features[k]))
                lines.append(f"│ Win pattern: {top_feature} "
                             f"(seen in {len(all_features[top_feature])} games)")

        loses = self.get_lose_signatures(genre=genre, game_type=game_type)
        if loses:
            all_features = {}
            for l in loses[:10]:
                for k, v in l["features"].items():
                    if k not in all_features:
                        all_features[k] = []
                    all_features[k].append(v)
            if all_features:
                top_feature = max(all_features, key=lambda k: len(all_features[k]))
                lines.append(f"│ Lose pattern: {top_feature} "
                             f"(seen in {len(all_features[top_feature])} games)")

        lines.append(f"└{'─' * (len(lines[0]) - 2)}┘")
        return "\n".join(lines)

    # ─── ANALYSIS HELPERS ─────────────────────────────────────────

    def stats(self) -> dict:
        """Quick stats on the memory contents."""
        n_rules = self.db.execute("SELECT COUNT(*) FROM rules").fetchone()[0]
        n_games = self.db.execute("SELECT COUNT(*) FROM game_results").fetchone()[0]
        n_sigs = self.db.execute("SELECT COUNT(*) FROM state_signatures").fetchone()[0]
        n_genres = self.db.execute(
            "SELECT COUNT(DISTINCT genre) FROM rules"
        ).fetchone()[0]
        avg_conf = self.db.execute(
            "SELECT AVG(confidence) FROM rules"
        ).fetchone()[0] or 0.0

        return {
            "rules": n_rules,
            "games": n_games,
            "state_signatures": n_sigs,
            "genres": n_genres,
            "avg_confidence": round(avg_conf, 3),
        }

    def __repr__(self):
        s = self.stats()
        return (f"SolverMemory({s['rules']} rules, {s['games']} games, "
                f"{s['state_signatures']} states, {s['genres']} genres)")


# ─── FRAME ANALYSIS ──────────────────────────────────────────────

def analyze_state(frame: np.ndarray, genre: str = "",
                  prev_frame: Optional[np.ndarray] = None) -> dict:
    """Extract observable features from a game frame.

    Not pixel-perfect — extract STRUCTURAL features that transfer
    across games of the same genre:
      - Color distribution (what's on screen)
      - Symmetry (many win states are symmetric)
      - Change magnitude (what just happened)
      - Spatial structure (clusters, alignment)

    Returns a dict of features suitable for state_signatures.
    """
    f = frame if frame.ndim == 2 else frame[0]
    features = {}

    # Color distribution
    unique, counts = np.unique(f, return_counts=True)
    features["n_colors"] = int(len(unique))
    features["dominant_color_pct"] = round(float(counts.max() / f.size), 3)

    # Background (most common color) vs foreground
    bg_color = unique[counts.argmax()]
    fg_mask = f != bg_color
    fg_pct = fg_mask.sum() / f.size
    features["foreground_pct"] = round(float(fg_pct), 3)

    # Symmetry (many win states are symmetric)
    h_sym = float(np.mean(f == np.fliplr(f)))
    v_sym = float(np.mean(f == np.flipud(f)))
    features["horizontal_symmetry"] = round(h_sym, 3)
    features["vertical_symmetry"] = round(v_sym, 3)

    # Spatial distribution of foreground
    if fg_mask.any():
        ys, xs = np.where(fg_mask)
        features["fg_centroid_y"] = round(float(ys.mean() / f.shape[0]), 3)
        features["fg_centroid_x"] = round(float(xs.mean() / f.shape[1]), 3)
        features["fg_spread_y"] = round(float(ys.std() / f.shape[0]), 3)
        features["fg_spread_x"] = round(float(xs.std() / f.shape[1]), 3)

    # Change from previous frame
    if prev_frame is not None:
        pf = prev_frame if prev_frame.ndim == 2 else prev_frame[0]
        if pf.shape == f.shape:
            delta = (f != pf).sum()
            features["pixels_changed"] = int(delta)
            features["change_pct"] = round(float(delta / f.size), 4)

    # Edge density (proxy for visual complexity)
    # Simple: count pixels different from their right neighbor
    if f.shape[1] > 1:
        edges_h = (f[:, :-1] != f[:, 1:]).sum()
        edges_v = (f[:-1, :] != f[1:, :]).sum()
        total_possible = (f.shape[0] * (f.shape[1] - 1) +
                          (f.shape[0] - 1) * f.shape[1])
        features["edge_density"] = round(float((edges_h + edges_v) /
                                               max(1, total_possible)), 4)

    return features


# ─── CONVENIENCE ──────────────────────────────────────────────────

def open_memory(db_path: str = _DEFAULT_DB) -> SolverMemory:
    """Convenience: open a SolverMemory instance."""
    return SolverMemory(db_path)


# ─── TIER 2: HYPOTHESIS ENGINE ──────────────────────────────────────
# Reads the rulebook and generates testable hypotheses with minimal probes.
# This is the transition from BFS to reasoning.

@dataclass
class Hypothesis:
    """A testable prediction about how the game works."""
    genre: str              # predicted genre
    mechanic: str           # predicted mechanic (e.g. "toggle", "push", "toggle_pair")
    prediction: str         # human-readable: "pressing button X will toggle cell Y"
    test_actions: list      # minimal probe sequence: [action_idx, ...]
    expected_outcome: str   # what we expect to see: "pixels_changed > 0 in region R"
    confidence: float       # prior confidence from rulebook (0-1)
    source_rule_id: int     # which rule generated this hypothesis (FK)
    confirmed: Optional[bool] = None  # None=untested, True/False after probe

    def __repr__(self):
        status = "?" if self.confirmed is None else ("✓" if self.confirmed else "✗")
        return f"H[{status}] {self.mechanic}: {self.prediction} (conf={self.confidence:.0%})"


class HypothesisEngine:
    """Generates and tests hypotheses from the solver's long-term memory.

    Tier 2 imagination: instead of BFS (try everything), form hypotheses
    from known rules and test them with targeted probes.

    Usage:
        engine = HypothesisEngine(solver_memory)
        hypotheses = engine.generate(genre="toggle", n_actions=8)
        # → [H "pressing any button toggles its neighbors", ...]

        for h in engine.prioritize(hypotheses):
            result = probe(h.test_actions)
            engine.update(h, result)
            if engine.confident_model():
                break  # enough to solve — skip remaining probes
    """

    def __init__(self, memory: SolverMemory):
        self.memory = memory
        self.tested: list[Hypothesis] = []
        self.model: dict = {}  # accumulated understanding

    def generate(self, genre: str, n_actions: int,
                 frame_features: dict = None) -> list[Hypothesis]:
        """Generate hypotheses from rulebook rules for this genre.

        Returns hypotheses sorted by confidence (highest first).
        Each hypothesis includes a minimal test plan.
        """
        rules = self.memory.get_rulebook(genre=genre, min_confidence=0.3)
        hypotheses = []

        for rule in rules:
            rid = rule["id"]
            pattern = rule["pattern"]
            mechanism = rule["mechanism"]
            strategy = rule["strategy"]
            conf = rule["confidence"]

            # Generate testable predictions from each rule
            preds = self._rule_to_predictions(
                genre, pattern, mechanism, strategy, n_actions, frame_features
            )
            for pred_text, test_acts, expected in preds:
                hypotheses.append(Hypothesis(
                    genre=genre,
                    mechanic=pattern,
                    prediction=pred_text,
                    test_actions=test_acts,
                    expected_outcome=expected,
                    confidence=conf,
                    source_rule_id=rid,
                ))

        # Also generate genre-level priors even without specific rules
        hypotheses.extend(self._genre_priors(genre, n_actions))

        # Sort by confidence descending, deduplicate by mechanic
        seen = set()
        unique = []
        for h in sorted(hypotheses, key=lambda x: x.confidence, reverse=True):
            key = (h.mechanic, tuple(h.test_actions[:2]))
            if key not in seen:
                seen.add(key)
                unique.append(h)

        return unique

    def _rule_to_predictions(self, genre: str, pattern: str, mechanism: str,
                              strategy: str, n_actions: int,
                              frame_features: dict = None) -> list[tuple]:
        """Convert a rule into testable (prediction, test_actions, expected_outcome) triples."""
        predictions = []

        # Pattern-based predictions
        pat_lower = pattern.lower()

        if "toggle" in pat_lower or "revers" in pat_lower:
            # Predict: pressing same button twice returns to original state
            predictions.append((
                "pressing button 0 twice returns to original state",
                [0, 0],
                "frame_matches_original",
            ))

        if "neighbor" in pat_lower or "coupl" in pat_lower:
            # Predict: adjacent buttons affect each other
            predictions.append((
                "button 0 affects region near button 1",
                [0],
                "pixels_changed_near_button_1",
            ))

        if "push" in pat_lower or "move" in pat_lower or "slide" in pat_lower:
            # Predict: directional actions cause object movement
            predictions.append((
                "directional action moves an object",
                [0],
                "sprite_movement_detected",
            ))

        if "gravity" in pat_lower or "physic" in pat_lower:
            # Predict: objects fall or have momentum
            predictions.append((
                "objects exhibit gravity or momentum",
                [0],
                "vertical_movement_down",
            ))

        if "sequence" in pat_lower or "order" in pat_lower:
            # Predict: order matters — AB ≠ BA
            predictions.append((
                "action order matters (AB ≠ BA)",
                [0, 1],
                "frame_differs_from_BA",
            ))

        if "accumul" in pat_lower or "pump" in pat_lower:
            # Predict: repeated action has cumulative effect
            predictions.append((
                "repeating button 0 has cumulative effect",
                [0, 0, 0],
                "monotonic_change",
            ))

        # If no pattern-specific predictions, generate generic probes
        if not predictions:
            predictions.append((
                f"button 0 has observable effect ({pattern})",
                [0],
                "pixels_changed > 0",
            ))

        return predictions

    def _genre_priors(self, genre: str, n_actions: int) -> list[Hypothesis]:
        """Generate baseline hypotheses from genre alone (no rules needed)."""
        priors = []

        genre_templates = {
            "toggle": [
                ("self-inverse", "each button is its own undo", [0, 0], "frame_matches_original", 0.7),
                ("modular", "effects cycle with fixed period", [0, 0, 0], "frame_matches_after_1", 0.4),
            ],
            "circuit": [
                ("seesaw_pair", "buttons affect each other in pairs", [0], "two_regions_change", 0.5),
                ("hydraulic", "total quantity conserved across regions", [0], "conservation_check", 0.4),
            ],
            "spatial": [
                ("push_mechanics", "objects can be pushed toward targets", [0], "object_displacement", 0.5),
                ("goal_position", "game ends when objects reach marked positions", [0], "target_proximity", 0.4),
            ],
            "physics": [
                ("gravity", "objects fall when unsupported", [0], "vertical_motion_down", 0.6),
                ("collision", "objects bounce off boundaries", [0], "direction_reversal", 0.4),
            ],
            "adversarial": [
                ("turn_based", "opponent responds after each action", [0], "opponent_state_changes", 0.6),
                ("minimax", "opponent plays optimally — don't assume random", [0], "opponent_counters", 0.3),
            ],
        }

        for mechanic, desc, test, expected, conf in genre_templates.get(genre, []):
            priors.append(Hypothesis(
                genre=genre, mechanic=mechanic,
                prediction=desc,
                test_actions=test, expected_outcome=expected,
                confidence=conf, source_rule_id=-1,  # -1 = genre prior
            ))

        return priors

    def prioritize(self, hypotheses: list[Hypothesis]) -> list[Hypothesis]:
        """Order hypotheses by information value: test the most discriminating first.

        A hypothesis that splits the possibility space in half is worth more
        than one that confirms what we already suspect.
        """
        # Score = confidence * (1 - confidence) * 4  → peaks at 0.5 (max information)
        # Plus a bonus for hypotheses that constrain genre classification
        def info_value(h):
            # Shannon entropy of binary prediction: max at p=0.5
            p = h.confidence
            entropy = p * (1 - p) * 4  # normalized to [0, 1]
            # Bonus for short test sequences (cheaper to test)
            cost_bonus = 1.0 / (1 + len(h.test_actions))
            return entropy + cost_bonus * 0.3

        return sorted(hypotheses, key=info_value, reverse=True)

    def update(self, hypothesis: Hypothesis, confirmed: bool,
               observed_features: dict = None):
        """Update hypothesis after testing. Feeds back to model."""
        hypothesis.confirmed = confirmed
        self.tested.append(hypothesis)

        if confirmed:
            self.model[hypothesis.mechanic] = {
                "confirmed": True,
                "confidence": hypothesis.confidence,
                "source_rule": hypothesis.source_rule_id,
                "features": observed_features,
            }
            # Strengthen the source rule in long-term memory
            if hypothesis.source_rule_id > 0:
                try:
                    self.memory.store_rule(
                        genre=hypothesis.genre,
                        pattern=hypothesis.mechanic,
                        mechanism=hypothesis.prediction,
                        strategy="",  # don't overwrite strategy
                        confidence=min(hypothesis.confidence + 0.05, 0.99),
                        evidence=[],
                        source="hypothesis_confirmed",
                    )
                except Exception:
                    pass  # non-critical
        else:
            # Weaken the source rule slightly
            self.model[hypothesis.mechanic] = {
                "confirmed": False,
                "confidence": hypothesis.confidence,
            }

    def confident_model(self, threshold: float = 0.7) -> bool:
        """Do we have enough confirmed hypotheses to skip BFS?

        Returns True when we have at least one confirmed mechanic
        with high confidence — enough to attempt a targeted solve.
        """
        confirmed = [h for h in self.tested if h.confirmed]
        if not confirmed:
            return False
        best = max(confirmed, key=lambda h: h.confidence)
        return best.confidence >= threshold

    def get_model_summary(self) -> str:
        """Human-readable summary of what we've learned so far."""
        lines = ["Hypothesis Model:"]
        for mechanic, data in self.model.items():
            status = "✓" if data["confirmed"] else "✗"
            lines.append(f"  {status} {mechanic} (conf={data['confidence']:.0%})")
        if not self.model:
            lines.append("  (no hypotheses tested yet)")
        return "\n".join(lines)

    def suggest_route(self) -> Optional[str]:
        """Based on confirmed hypotheses, suggest the best solver route.

        This is the Tier 2 payoff: instead of cascading through all routes,
        pick the right one from the start.
        """
        confirmed = {k: v for k, v in self.model.items() if v.get("confirmed")}
        if not confirmed:
            return None

        # Mechanic/pattern → route mapping
        # Covers both genre priors AND common rule pattern names
        route_map = {
            # Genre priors
            "self-inverse": "toggle_matrix",
            "toggle": "toggle_matrix",
            "modular": "constraint",
            "seesaw_pair": "mechanic",
            "hydraulic": "mechanic",
            "push_mechanics": "block_bfs",
            "goal_position": "sliding_astar",
            "gravity": "reactive_control",
            "collision": "reactive_control",
            "turn_based": "dynamic_bfs",
            "minimax": "dynamic_bfs",
            "sequence": "bfs",
            "accumul": "mechanic",
            # Common rule patterns from Eyes insights
            "action_reversal": "toggle_matrix",
            "neighbor_coupling": "mechanic",
            "pump": "mechanic",
            "navigation": "navigation",
            "sliding": "sliding_astar",
            "block_push": "block_bfs",
            "reactive": "random_walk",
        }

        # Pick the route for the highest-confidence confirmed mechanic
        best_mechanic = max(confirmed, key=lambda k: confirmed[k]["confidence"])
        route = route_map.get(best_mechanic)
        if route:
            return route
        # Fallback: substring match (e.g., "gravity_fall" matches "gravity")
        for key, rt in route_map.items():
            if key in best_mechanic or best_mechanic in key:
                return rt
        return None
