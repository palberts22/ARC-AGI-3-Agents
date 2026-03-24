#!/usr/bin/env python3
"""
Common Sense Bus — Weighted Memory Push for ARC Solver
=======================================================

The glue between perception, memory, and action. Pushes weighted
"hunches" to the solver at decision junctures — involuntary recall,
like a human's pattern recognition firing before conscious thought.

The solver has memory (SolverMemory, EpisodicMemory, visual search,
subconscious) but no PUSH. This module provides the push.

Five injection points:
  IP-1: Pre-cascade — "this looks like game X, try route Y"
  IP-2: Route failure — "BFS always fails on toggle puzzles"
  IP-3: Stall breaker — "you're looping, try action 7"
  IP-4: Pilot enrichment — top-3 memories for LLM context
  IP-5: Post-solve recording — store WHY it worked

Usage:
  from common_sense_bus import CommonSenseBus, Hunch
  bus = CommonSenseBus(solver_memory=sm, eyes_memory=em)
  hunch = bus.pre_cascade(fingerprint, profile)
  if hunch:
      preferred_route = hunch.suggested_route

Author: Archie | Date: 2026-03-13 | Directive: Peter
"""

import logging
import math
import time
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

log = logging.getLogger("common_sense")


# ---------------------------------------------------------------------------
# Data Structures
# ---------------------------------------------------------------------------

@dataclass
class Hunch:
    """A memory-driven suggestion pushed to the solver."""
    type: str           # "route_suggestion", "skip_route", "try_action",
                        # "escalate", "concept_hint", "past_win"
    content: str        # Human-readable explanation
    salience: float     # 0.0-1.0 (weighted score)
    source: str         # "procedural", "failure_memory", "visual",
                        # "subconscious", "episodic", "cingulate"
    suggested_route: str = ""      # Optional: route to try
    suggested_action: int = -1     # Optional: specific action to try
    timestamp: float = field(default_factory=time.time)
    metadata: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Weighting Functions
# ---------------------------------------------------------------------------

def recency_weight(steps_ago: int, decay_rate: float = 0.01) -> float:
    """Exponential decay: recent memories matter more.

    λ=0.01 → half-life ~70 steps.
    steps_ago=0 → 1.0, steps_ago=70 → 0.5, steps_ago=200 → 0.14
    """
    return math.exp(-decay_rate * steps_ago)


def impact_weight(pixel_delta: float, max_delta: float = 1000.0) -> float:
    """Actions with large visual impact are more memorable.

    pixel_delta: number of pixels that changed
    max_delta: normalization cap
    """
    if max_delta <= 0:
        return 0.5
    return min(1.0, pixel_delta / max_delta)


def novelty_weight(times_surfaced: int) -> float:
    """First time seeing a pattern = most informative. Diminishing returns.

    times_surfaced=0 → 1.0, times_surfaced=1 → 0.5, times_surfaced=9 → 0.1
    """
    return 1.0 / (1.0 + times_surfaced)


def arousal_weight(tags: str = "", source: str = "", was_win: bool = False,
                   was_failure: bool = False, was_taught: bool = False) -> float:
    """Emotional arousal proxy — significant memories surface first.

    Maps to amygdala function: memories tagged by felt significance,
    not just topic relevance. Based on Hypatia's circumplex model.

    High arousal: crashes, wins, failures, taught concepts, corrections.
    Low arousal: routine observations, status updates.
    """
    arousal = 0.5  # baseline
    tag_set = set(t.strip().lower() for t in tags.split(",") if t.strip())

    # Context-based arousal signals
    if was_win:
        arousal += 0.3
    if was_failure:
        arousal += 0.25
    if was_taught:
        arousal += 0.2

    # Tag-based arousal signals
    high_arousal_tags = {"crash", "bug", "fix", "milestone", "correction",
                         "lesson", "breakthrough", "peter", "decision"}
    low_arousal_tags = {"routine", "status", "checkpoint"}

    if tag_set & high_arousal_tags:
        arousal += 0.2
    if tag_set & low_arousal_tags:
        arousal -= 0.15

    # Source-based: taught concepts are higher arousal
    if "teach" in source.lower() or "peter" in source.lower():
        arousal += 0.15

    return max(0.1, min(1.0, arousal))


def salience(relevance: float, steps_ago: int = 0,
             pixel_delta: float = 500.0, times_surfaced: int = 0,
             tags: str = "", source: str = "",
             was_win: bool = False, was_failure: bool = False,
             was_taught: bool = False) -> float:
    """Combined salience score: relevance × recency × impact × novelty × arousal.

    Arousal (amygdala) acts as a multiplier — emotionally significant
    memories surface before routine ones at equal relevance.
    """
    return (
        relevance
        * recency_weight(steps_ago)
        * impact_weight(pixel_delta)
        * novelty_weight(times_surfaced)
        * arousal_weight(tags, source, was_win, was_failure, was_taught)
    )


# ---------------------------------------------------------------------------
# Common Sense Bus
# ---------------------------------------------------------------------------

class CommonSenseBus:
    """Pushes weighted memories to the solver at decision junctures.

    Initialize with available memory systems. Missing systems are gracefully
    skipped — the bus works with whatever's available.

    Args:
        solver_memory: GameMemoryBank or compatible (recall, best_strategy_for)
        eyes_memory: EpisodicMemory or compatible (get_effective_actions)
        visual_search: VisualSearchSource (query_by_frame, search_by_frame)
        subconscious: Subconscious (resonate) — optional, heavier
        cingulate: CingulateMonitor — optional, not yet built
    """

    # Salience threshold — only push hunches above this
    MIN_SALIENCE = 0.3

    # Refractory period — suppress recently pushed hunches
    SUPPRESS_STEPS = 50

    def __init__(self, solver_memory=None, eyes_memory=None,
                 visual_search=None, subconscious=None, cingulate=None):
        self.solver_memory = solver_memory
        self.eyes_memory = eyes_memory
        self.visual_search = visual_search
        self.subconscious = subconscious
        self.cingulate = cingulate

        # State tracking
        self.step_count = 0
        self.hunches_pushed: list[Hunch] = []
        self.hunches_acted_on: list[Hunch] = []
        self.suppressed: dict[str, int] = {}  # content_key → steps_until_unsuppressed
        self.failed_routes: list[tuple[str, str]] = []  # (route_name, reason)
        self.available_actions: list = []
        self.fingerprint = None
        self.current_frame: Optional[np.ndarray] = None
        self._surfaced_count: dict[str, int] = {}  # memory_id → times surfaced

    def set_context(self, fingerprint=None, available_actions=None, frame=None):
        """Update solver context. Call at the start of each level."""
        if fingerprint is not None:
            self.fingerprint = fingerprint
        if available_actions is not None:
            self.available_actions = available_actions
        if frame is not None:
            self.current_frame = frame
        self.failed_routes = []

    def step(self):
        """Advance one step. Decays suppressions."""
        self.step_count += 1
        expired = [k for k, v in self.suppressed.items() if v <= 0]
        for k in expired:
            del self.suppressed[k]
        for k in self.suppressed:
            self.suppressed[k] -= 1

    def acknowledge(self, hunch: Hunch):
        """Solver acted on this hunch. Track for learning + suppress repeats."""
        self.hunches_acted_on.append(hunch)
        key = hunch.content[:50]
        self.suppressed[key] = self.SUPPRESS_STEPS

    def _is_suppressed(self, content: str) -> bool:
        return content[:50] in self.suppressed

    def _track_surfaced(self, memory_id: str) -> int:
        """Track how many times a memory has been surfaced. Returns count."""
        self._surfaced_count[memory_id] = self._surfaced_count.get(memory_id, 0) + 1
        return self._surfaced_count[memory_id]

    def _push(self, hunch: Hunch) -> Optional[Hunch]:
        """Filter and push a hunch. Returns None if filtered out."""
        if hunch.salience < self.MIN_SALIENCE:
            return None
        if self._is_suppressed(hunch.content):
            return None
        self.hunches_pushed.append(hunch)
        log.info(f"  [hunch] {hunch.type}: {hunch.content[:80]} (salience={hunch.salience:.2f})")
        return hunch

    # -------------------------------------------------------------------
    # IP-1: Pre-Cascade — before route selection
    # -------------------------------------------------------------------

    def pre_cascade(self, fingerprint=None, profile=None) -> Optional[Hunch]:
        """Query memory for route suggestion based on game similarity.

        Fast path: procedural memory only (<1ms). No subconscious needed.
        """
        fp = fingerprint or self.fingerprint
        hunches = []

        # A. Procedural memory — what route won on similar games?
        if self.solver_memory and fp:
            try:
                similar = self.solver_memory.recall(fp, top_k=5)
                if similar:
                    best = max(similar, key=lambda m: (
                        getattr(m, 'similarity', 0.5) *
                        recency_weight(getattr(m, 'steps_ago', 100))
                    ))
                    sim = getattr(best, 'similarity', 0.5)
                    strategy = getattr(best, 'strategy_name', '') or getattr(best, 'strategy', '')
                    game_id = getattr(best, 'game_id', 'unknown')
                    if strategy and sim > 0.4:
                        times = self._track_surfaced(f"proc:{game_id}")
                        s = salience(sim, times_surfaced=times)
                        hunches.append(Hunch(
                            type="route_suggestion",
                            content=f"Similar to {game_id} ({sim:.0%} match). Won with {strategy}.",
                            salience=s,
                            source="procedural",
                            suggested_route=strategy,
                            metadata={"game_id": game_id, "similarity": sim},
                        ))
            except Exception as e:
                log.debug(f"Pre-cascade procedural lookup failed: {e}")

        # B. Visual search — does this frame look like something we've seen?
        if self.visual_search and self.current_frame is not None:
            try:
                visual_memories = self.visual_search.query_by_frame(
                    self.current_frame, limit=3
                )
                for vm in visual_memories:
                    if vm.relevance > 0.5:
                        times = self._track_surfaced(vm.memory_id)
                        s = salience(vm.relevance, times_surfaced=times)
                        hunches.append(Hunch(
                            type="visual_match",
                            content=vm.content,
                            salience=s,
                            source="visual",
                            metadata={"memory_id": vm.memory_id},
                        ))
            except Exception as e:
                log.debug(f"Pre-cascade visual search failed: {e}")

        if not hunches:
            return None

        best = max(hunches, key=lambda h: h.salience)
        return self._push(best)

    # -------------------------------------------------------------------
    # IP-2: Route Failure — pivot suggestion after a route fails
    # -------------------------------------------------------------------

    def on_route_failure(self, route_name: str, reason: str = "",
                         profile=None, time_left: float = 0) -> Optional[Hunch]:
        """Suggest what to try next after a route fails.

        Checks procedural memory for failure patterns, then optionally
        resonates via subconscious for concept-level hints.
        """
        self.failed_routes.append((route_name, reason))

        # A. Check failure patterns — has this route failed on this game type before?
        if self.solver_memory and profile:
            try:
                game_type = getattr(profile, 'game_type', 'unknown')
                # Try to get suggested strategies based on what worked
                suggested = self.solver_memory.suggested_strategies(self.fingerprint)
                if suggested:
                    # Filter out routes we already tried
                    tried_routes = {r[0] for r in self.failed_routes}
                    untried = [s for s in suggested if s not in tried_routes]
                    if untried:
                        next_route = untried[0]
                        s = salience(0.7, times_surfaced=self._track_surfaced(f"fail:{route_name}"))
                        return self._push(Hunch(
                            type="skip_route",
                            content=f"{route_name} failed on {game_type}. Try {next_route} (memory suggests it works on similar games).",
                            salience=s,
                            source="failure_memory",
                            suggested_route=next_route,
                        ))
            except Exception as e:
                log.debug(f"Route failure pattern check failed: {e}")

        # B. Subconscious resonance — concept-level hints (heavier, ~50ms)
        if self.subconscious and time_left > 5.0:
            try:
                stimulus = f"{route_name} failed"
                if profile:
                    stimulus = f"{getattr(profile, 'game_type', 'unknown')} game. {stimulus}"
                if reason:
                    stimulus += f" ({reason})"

                memories = self.subconscious.resonate(stimulus)
                if memories and memories[0].relevance > 0.5:
                    mem = memories[0]
                    times = self._track_surfaced(mem.memory_id)
                    s = salience(mem.relevance, times_surfaced=times)
                    return self._push(Hunch(
                        type="concept_hint",
                        content=f"{mem.title}: {mem.content[:100]}",
                        salience=s,
                        source="subconscious",
                        metadata={"memory_id": mem.memory_id},
                    ))
            except Exception as e:
                log.debug(f"Route failure subconscious resonance failed: {e}")

        return None

    # -------------------------------------------------------------------
    # IP-3: Stall Breaker — emergency recall when solver is stuck
    # -------------------------------------------------------------------

    def on_stall(self, actions_tried: list = None,
                 frame: np.ndarray = None) -> Optional[Hunch]:
        """Emergency recall when solver is stuck in a loop.

        Checks: what high-impact actions haven't been tried recently?
        If nothing untried, suggests escalation to pilot.
        """
        actions_tried = actions_tried or []

        # What actions have we NOT tried recently?
        if self.eyes_memory and self.available_actions:
            try:
                recent_ids = set()
                for a in actions_tried[-20:]:
                    aid = getattr(a, 'action_id', getattr(a, 'game_action', None))
                    if aid is not None:
                        recent_ids.add(aid)

                effective = self.eyes_memory.get_effective_actions(min_observations=1)
                if effective:
                    # Find effective actions not recently tried
                    untried_effective = [
                        a for a in effective
                        if getattr(a, 'action_id', getattr(a, 'game_action', None)) not in recent_ids
                    ]
                    if untried_effective:
                        best = max(untried_effective,
                                   key=lambda a: getattr(a, 'pixel_delta', 0))
                        aid = getattr(best, 'action_id', getattr(best, 'game_action', 0))
                        delta = getattr(best, 'pixel_delta', 0)
                        return self._push(Hunch(
                            type="try_action",
                            content=f"Stall detected. Action {aid} had high impact ({delta}px change) but hasn't been tried recently.",
                            salience=0.9,  # High — we're stuck
                            source="episodic",
                            suggested_action=aid,
                        ))
            except Exception as e:
                log.debug(f"Stall breaker episodic check failed: {e}")

        # Visual search — does the current (stuck) frame match a previously solved frame?
        if self.visual_search and frame is not None:
            try:
                matches = self.visual_search.search_by_frame(frame, limit=1, min_similarity=0.6)
                if matches and matches[0].get("outcome") in ("win", "solved"):
                    m = matches[0]
                    return self._push(Hunch(
                        type="visual_match",
                        content=f"This frame looks like {m['game_id']} L{m['level']} which was solved. Check that game's winning route.",
                        salience=0.85,
                        source="visual",
                        metadata=m,
                    ))
            except Exception as e:
                log.debug(f"Stall breaker visual search failed: {e}")

        # Fallback: escalate
        return self._push(Hunch(
            type="escalate",
            content=f"Stall after {len(actions_tried)} actions. No untried high-impact actions. Escalate to pilot.",
            salience=0.95,
            source="cingulate",
        ))

    # -------------------------------------------------------------------
    # IP-4: Pilot Context Enrichment
    # -------------------------------------------------------------------

    def enrich_pilot_context(self, profile=None) -> list[Hunch]:
        """Pack top-3 weighted memories into pilot's context.

        Called when all routes failed and the LLM pilot is being consulted.
        Returns list of hunches to include in pilot system prompt.
        """
        hunches = []

        # Procedural: what won on similar games?
        if self.solver_memory and self.fingerprint:
            try:
                similar = self.solver_memory.recall(self.fingerprint, top_k=3)
                for m in similar:
                    sim = getattr(m, 'similarity', 0.5)
                    strategy = getattr(m, 'strategy_name', '') or getattr(m, 'strategy', '')
                    game_id = getattr(m, 'game_id', 'unknown')
                    solve_time = getattr(m, 'solve_time', 0)
                    times = self._track_surfaced(f"pilot:proc:{game_id}")
                    hunches.append(Hunch(
                        type="past_win",
                        content=f"Similar game {game_id} solved with {strategy} in {solve_time:.0f}s",
                        salience=salience(sim, times_surfaced=times),
                        source="procedural",
                    ))
            except Exception as e:
                log.debug(f"Pilot enrichment procedural failed: {e}")

        # Subconscious: any relevant concepts?
        if self.subconscious:
            try:
                stimulus = f"All routes failed"
                if profile:
                    game_type = getattr(profile, 'game_type', 'unknown')
                    stimulus = f"All routes failed on {game_type}. "
                    stimulus += f"Tried: {', '.join(r[0] for r in self.failed_routes[:5])}"

                resonances = self.subconscious.resonate(stimulus)
                for r in resonances[:2]:
                    times = self._track_surfaced(r.memory_id)
                    hunches.append(Hunch(
                        type="concept_hint",
                        content=f"{r.title}: {r.content[:100]}",
                        salience=salience(r.relevance, times_surfaced=times),
                        source="subconscious",
                    ))
            except Exception as e:
                log.debug(f"Pilot enrichment subconscious failed: {e}")

        # Visual: does the current frame match anything?
        if self.visual_search and self.current_frame is not None:
            try:
                matches = self.visual_search.query_by_frame(self.current_frame, limit=1)
                for m in matches:
                    hunches.append(Hunch(
                        type="visual_match",
                        content=m.content,
                        salience=m.relevance,
                        source="visual",
                    ))
            except Exception as e:
                log.debug(f"Pilot enrichment visual failed: {e}")

        # Sort by salience, return top 3
        hunches.sort(key=lambda h: h.salience, reverse=True)
        return hunches[:3]

    # -------------------------------------------------------------------
    # IP-4b: Anti-BFS Guard — discovery > search
    # -------------------------------------------------------------------

    def check_discovery_mode(self, action_count: int,
                              hypothesis_count: int = 0) -> Optional[Hunch]:
        """Cingulate trigger: if action count is high with no hypotheses,
        inject a pause signal. BFS is not learning.

        Call this periodically from the solver (e.g., every 10 actions).
        Returns a Hunch if the solver should pause and reflect, else None.
        """
        if action_count < self.BFS_ACTION_THRESHOLD:
            return None
        if hypothesis_count > 0:
            return None  # solver is hypothesizing — that's discovery

        hunch = Hunch(
            type="discovery_guard",
            content=(
                "PAUSE — you've taken {n} actions without stating a hypothesis. "
                "BFS is not learning. Before the next action, answer: "
                "What do I expect to happen? Why? What would surprise me?"
            ).format(n=action_count),
            salience=0.95,  # high salience — this should override
            source="cingulate",
            metadata={"guard": "anti_bfs", "action_count": action_count},
        )
        return self._push(hunch)

    # -------------------------------------------------------------------
    # IP-5: Post-Solve Recording
    # -------------------------------------------------------------------

    # Brute-force detection threshold — if win took more actions than this
    # without a hypothesis, the reward is halved (brute-force tax)
    BFS_ACTION_THRESHOLD = 50

    def record_win(self, fingerprint, route_name: str, actions: list,
                   solve_time: float, profile=None, frame: np.ndarray = None,
                   hypothesis_logged: bool = False):
        """Record a win with causal context — WHY it worked.

        Stores to solver_memory and optionally indexes the winning frame
        in visual search for future recall.

        Args:
            hypothesis_logged: Whether the solver stated a hypothesis before
                acting. If False and action count is high, reward is reduced
                (brute-force tax — discovery is worth more than search).
        """
        is_brute_force = (len(actions) > self.BFS_ACTION_THRESHOLD
                          and not hypothesis_logged)
        context = {
            "game_type": getattr(profile, 'game_type', 'unknown') if profile else 'unknown',
            "total_actions": len(actions),
            "n_available_actions": len(self.available_actions),
            "failed_routes": [r[0] for r in self.failed_routes],
            "hunches_used": [h.source for h in self.hunches_acted_on],
            "solve_time": solve_time,
            "brute_force": is_brute_force,
            "hypothesis_logged": hypothesis_logged,
        }

        # Store to solver memory with context
        if self.solver_memory:
            try:
                self.solver_memory.remember_win(
                    fingerprint=fingerprint,
                    strategy_name=route_name,
                    actions=actions,
                    solve_time=solve_time,
                )
                log.info(f"  [win] Recorded: {route_name} in {solve_time:.1f}s, "
                         f"{len(actions)} actions, context: {len(context)} keys")
            except Exception as e:
                log.debug(f"Win recording to solver_memory failed: {e}")

        # Index the winning frame for visual recall
        if self.visual_search and frame is not None:
            try:
                game_id = context["game_type"]
                if profile and hasattr(profile, 'game_id'):
                    game_id = profile.game_id
                self.visual_search.store_frame(
                    visual_memory_id=hash(f"win:{game_id}:{route_name}:{self.step_count}") & 0x7FFFFFFF,
                    frame=frame,
                    game_id=game_id,
                    level=getattr(profile, 'level', -1) if profile else -1,
                    outcome="win",
                )
            except Exception as e:
                log.debug(f"Win frame visual indexing failed: {e}")

        # Dopamine: reward hunches that contributed to this win
        # Brute-force tax: half reward if won by exhaustive search
        self._reward_hunches(won=True, brute_force=is_brute_force)

        return context

    def record_fail(self):
        """Record a failed attempt. Decay hunches that didn't help."""
        self._reward_hunches(won=False)

    def _reward_hunches(self, won: bool, brute_force: bool = False):
        """Dopamine signal — adjust KB quality for hunches that were acted on.

        On win: boost source KB entries (+5 quality, capped at 100).
        On fail: decay source KB entries (-2 quality, floored at 10).
        Brute-force tax: wins by exhaustive search get half reward (+2).
        Discovery is worth more than search.
        """
        if not self.hunches_acted_on:
            return

        if won and brute_force:
            delta = 2  # brute-force tax: half reward
            log.info("  [dopamine] Brute-force tax applied — discovery > search")
        else:
            delta = 5 if won else -2
        outcome = "win" if won else "fail"

        for hunch in self.hunches_acted_on:
            kb_id = hunch.metadata.get("kb_id")
            if not kb_id:
                # Try extracting from memory_id format "kb#123"
                mid = hunch.metadata.get("memory_id", "")
                if mid.startswith("kb#"):
                    try:
                        kb_id = int(mid[3:])
                    except ValueError:
                        pass
            if not kb_id:
                continue
            try:
                import sqlite3
                db_path = os.path.join(os.path.dirname(__file__),
                                       '..', 'data', 'knowledge.db')
                conn = sqlite3.connect(db_path)
                cur = conn.cursor()
                cur.execute(
                    "UPDATE knowledge SET quality = MIN(100, MAX(10, quality + ?)) "
                    "WHERE id = ?", (delta, kb_id))
                if cur.rowcount > 0:
                    log.info(f"  [dopamine] KB#{kb_id} quality {'+' if delta > 0 else ''}{delta} "
                             f"({outcome}, hunch: {hunch.content[:40]})")
                conn.commit()
                conn.close()
            except Exception as e:
                log.debug(f"Dopamine signal failed for KB#{kb_id}: {e}")

    # -------------------------------------------------------------------
    # Summary / Debug
    # -------------------------------------------------------------------

    def summary(self) -> dict:
        """Return bus state for debugging / flight recorder."""
        return {
            "step_count": self.step_count,
            "hunches_pushed": len(self.hunches_pushed),
            "hunches_acted_on": len(self.hunches_acted_on),
            "failed_routes": self.failed_routes,
            "suppressed_count": len(self.suppressed),
            "surfaced_memories": len(self._surfaced_count),
        }

    def format_for_pilot(self, hunches: list[Hunch]) -> str:
        """Format hunches as text for inclusion in LLM pilot prompt."""
        if not hunches:
            return ""

        lines = ["## Memory Hunches (from past experience):"]
        for i, h in enumerate(hunches, 1):
            lines.append(f"{i}. [{h.source}] {h.content} (confidence: {h.salience:.0%})")
        return "\n".join(lines)
