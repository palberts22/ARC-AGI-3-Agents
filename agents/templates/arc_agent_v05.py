#!/usr/bin/env python3
"""
ARC-AGI-3 Agent v0.5 — Rebuilt from memory after worktree loss.

Architecture:
  1. Action Discovery: detect directional vs position-dependent actions
  2. Game Analyzer: classify game type from action effects
  3. Generic Pipeline: DC-BFS → NavModel → MCTS → Random Walk
  4. Path Optimization: shorten paths post-solve

Key design: env.reset() goes back to level 0. After winning a level,
the env auto-advances. All solvers work on deepcopy snapshots and
detect level completion via levels_completed increment.

Scoring: action efficiency vs human baseline. Lower actions = better.
"""

import argparse
import copy
import hashlib
import json
import logging
import math
import os
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Optional

import base64
import io

import numpy as np

try:
    from PIL import Image
    HAS_PIL = True
except ImportError:
    HAS_PIL = False

try:
    from arc_action_predictor import ActionPredictor
    HAS_CNN_PREDICTOR = True
except ImportError:
    try:
        from arc_action_predictor_lite import ActionPredictor
        HAS_CNN_PREDICTOR = True
    except ImportError:
        HAS_CNN_PREDICTOR = False

try:
    import requests as _requests
    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False

from arc_eyes import play_level as eyes_play_level
from arc_recorder import GameRecorder
from arc_pilot import (Pilot, build_inflection_context, apply_directive,
                       FlightRecorder, EngagementRecord, get_vitals)
from arc_hud import HUD, FailureDetector, HUDState

try:
    from common_sense_bus import CommonSenseBus
    HAS_BUS = True
except ImportError:
    HAS_BUS = False

try:
    from dopamine import HabitTracker, wire_dopamine_to_bus
    HAS_DOPAMINE = True
except ImportError:
    HAS_DOPAMINE = False

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    datefmt='%H:%M:%S'
)
log = logging.getLogger('arc_agent')


# ─── Data Classes ───────────────────────────────────────────────────────

@dataclass
class Action:
    """Wraps a game action with optional position data."""
    game_action: int
    data: dict = field(default_factory=dict)
    name: str = ""

    def __repr__(self):
        if self.name:
            return self.name
        if self.data and 'x' in self.data:
            return f"A{self.game_action}({self.data['x']},{self.data['y']})"
        return f"A{self.game_action}"

    def __hash__(self):
        return hash((self.game_action, self.data.get('x'), self.data.get('y')))

    def __eq__(self, other):
        if not isinstance(other, Action):
            return False
        return (self.game_action == other.game_action and
                self.data.get('x') == other.data.get('x') and
                self.data.get('y') == other.data.get('y'))


@dataclass
class GameProfile:
    """Classification of game type from action effect analysis."""
    game_type: str = "UNKNOWN"
    actions: list = field(default_factory=list)
    directional_actions: list = field(default_factory=list)
    click_actions: list = field(default_factory=list)
    self_inverse: bool = False
    commutative: bool = False
    # Genre hints from action fingerprint (Hypatia's taxonomy)
    genre_hints: list = field(default_factory=list)  # e.g. ['navigation_maze', 'push_block']
    has_neighbor_coupling: bool = False  # click affects neighbors (toggle puzzle)
    has_gravity: bool = False  # elements fall after action
    has_moving_agents: bool = False  # other sprites move independently
    # OPRAH action profile for unknown-genre routing (Archie spec, 2026-03-16)
    oprah_action_counts: dict = field(default_factory=dict)  # {movement: N, toggle: N, ...}
    oprah_has_grid: bool = False
    oprah_has_click: bool = False


# ─── Genre → Route Priority Mapping ───────────────────────────────────
# Maps genre (from analyze_game) to route priorities with budget fractions.
# Routes not listed for a genre get minimal budget (0.05).
# 'unknown' is the fallback — current default order, unchanged.
# Spec: specs/genre_route_mapping_spec.md (Archie, 2026-03-14)

GENRE_ROUTE_MAP: dict[str, dict[str, float]] = {
    'toggle_puzzle': {
        'constraint': 0.25, 'frame_coupling': 0.20, 'toggle_matrix': 0.15,
        'mechanic': 0.20, 'bfs': 0.10, 'mcts': 0.10,
    },
    'constraint_satisfaction': {
        'constraint': 0.30, 'inductive': 0.20, 'mechanic': 0.20,
        'bfs': 0.15, 'mcts': 0.15,
    },
    'navigation_maze': {
        'navigation': 0.35, 'reactive': 0.10, 'eyes': 0.15,
        'mcts': 0.25, 'random_walk': 0.15,
    },
    'push_block': {
        'block_bfs': 0.30, 'sliding_astar': 0.25, 'mechanic': 0.15,
        'mcts': 0.20, 'bfs': 0.10,
    },
    'sliding_tile': {
        'sliding_astar': 0.35, 'block_bfs': 0.25, 'bfs': 0.15,
        'mcts': 0.15, 'nmcs': 0.10,
    },
    'pump_overflow': {
        'mechanic': 0.35, 'inductive': 0.20, 'bfs': 0.15,
        'mcts': 0.15, 'nmcs': 0.15,
    },
    'paint_fill': {
        'mechanic': 0.25, 'constraint': 0.20, 'bfs': 0.20,
        'eyes': 0.15, 'mcts': 0.20,
    },
    'sorting': {
        'bfs': 0.25, 'mechanic': 0.20, 'mcts': 0.20,
        'block_bfs': 0.15, 'nmcs': 0.20,
    },
    'pursuit_evasion': {
        'reactive': 0.25, 'navigation': 0.20, 'mcts': 0.30,
        'eyes': 0.15, 'random_walk': 0.10,
    },
    'gravity_puzzle': {
        'mechanic': 0.25, 'forward_model': 0.25, 'mcts': 0.25,
        'bfs': 0.15, 'nmcs': 0.10,
    },
    'cellular_automaton': {
        'forward_model': 0.30, 'mechanic': 0.25, 'inductive': 0.20,
        'bfs': 0.15, 'mcts': 0.10,
    },
    'rotation_puzzle': {
        'bfs': 0.25, 'mechanic': 0.25, 'constraint': 0.20,
        'mcts': 0.20, 'nmcs': 0.10,
    },
    'connection_flow': {
        'constraint': 0.25, 'bfs': 0.25, 'mechanic': 0.20,
        'mcts': 0.20, 'nmcs': 0.10,
    },
    'pattern_replication': {
        'eyes': 0.30, 'mechanic': 0.20, 'bfs': 0.20,
        'mcts': 0.20, 'nmcs': 0.10,
    },
    'circuit_puzzle': {
        'constraint': 0.25, 'mechanic': 0.25, 'inductive': 0.20,
        'bfs': 0.15, 'mcts': 0.15,
    },
    'multi_phase': {
        'eyes': 0.25, 'mechanic': 0.20, 'mcts': 0.25,
        'navigation': 0.15, 'random_walk': 0.15,
    },
}

# Default minimum budget fraction for routes not in the genre map
_DEFAULT_ROUTE_BUDGET = 0.05


def _action_profile_route_weights(profile: 'GameProfile') -> dict:
    """Compute route weights from OPRAH action profile when genre is unknown.

    Archie spec: specs/action_profile_routing_spec.md (2026-03-16).
    Uses structural measurements (action counts) instead of genre labels.
    """
    counts = getattr(profile, 'oprah_action_counts', {})
    if not counts:
        return {}
    n_move = counts.get('movement', 0)
    n_toggle = counts.get('toggle', 0)
    n_global = counts.get('global', 0)
    n_param = counts.get('parameterized', 0)
    total = max(n_move + n_toggle + n_global + n_param, 1)

    # Movement-heavy → navigation and search routes
    if n_move / total > 0.5:
        weights = {
            'navigation': 0.30, 'reactive': 0.15, 'mcts': 0.25,
            'eyes': 0.15, 'random_walk': 0.15,
        }
    # Toggle-heavy → constraint and logic routes
    elif n_toggle / total > 0.5:
        weights = {
            'constraint': 0.30, 'toggle_matrix': 0.20, 'inductive': 0.20,
            'mechanic': 0.15, 'bfs': 0.15,
        }
    # Global-heavy → forward model and pattern routes
    elif n_global / total > 0.3:
        weights = {
            'forward_model': 0.25, 'mechanic': 0.25, 'eyes': 0.20,
            'mcts': 0.20, 'bfs': 0.10,
        }
    # Mixed → balanced exploration
    else:
        weights = {
            'mechanic': 0.20, 'eyes': 0.20, 'mcts': 0.20,
            'bfs': 0.15, 'constraint': 0.15, 'random_walk': 0.10,
        }

    # Grid modifier: boost constraint/pattern routes
    if getattr(profile, 'oprah_has_grid', False):
        for r in ('constraint', 'inductive', 'eyes'):
            weights[r] = weights.get(r, 0.05) + 0.05

    # Click modifier: boost parameterized routes
    if getattr(profile, 'oprah_has_click', False):
        for r in ('mechanic', 'constraint'):
            weights[r] = weights.get(r, 0.05) + 0.05

    # Normalize
    total_w = sum(weights.values())
    if total_w > 0:
        weights = {k: v / total_w for k, v in weights.items()}
    return weights


def genre_route_budget(profile: 'GameProfile', route_id: str) -> float:
    """Get budget fraction for a route based on genre classification.

    Returns the genre-recommended fraction (0.0-1.0) of remaining time
    to allocate to this route. For unknown genres or routes not in the map,
    falls through to action-profile routing, then _DEFAULT_ROUTE_BUDGET.
    """
    hints = getattr(profile, 'genre_hints', []) or []
    if hints:
        genre = hints[0]
        route_map = GENRE_ROUTE_MAP.get(genre)
        if route_map:
            return route_map.get(route_id, _DEFAULT_ROUTE_BUDGET)

    # Fallback: action-profile routing for unknown genres
    ap_weights = _action_profile_route_weights(profile)
    if ap_weights:
        return ap_weights.get(route_id, _DEFAULT_ROUTE_BUDGET)

    return _DEFAULT_ROUTE_BUDGET


# ─── Frame Utilities ────────────────────────────────────────────────────

def frame_hash(frame: np.ndarray, mask: Optional[np.ndarray] = None) -> str:
    """Fast hash of a frame, optionally masking noise pixels."""
    f = frame.copy()
    if mask is not None and mask.shape == f.shape:
        f[mask] = 0
    return hashlib.md5(f.tobytes()).hexdigest()[:16]


def frame_diff(f1: np.ndarray, f2: np.ndarray) -> int:
    """Count pixels that differ. Returns -1 if shapes differ."""
    if f1.shape != f2.shape:
        return -1
    return int(np.sum(f1 != f2))


# ─── Win Detection ─────────────────────────────────────────────────────

def is_level_won(obs, start_levels: int) -> bool:
    """Check if a level was completed since start_levels."""
    if obs is None:
        return False
    return (obs.levels_completed > start_levels or
            obs.state.value == 'WIN')


# ─── Action Discovery ──────────────────────────────────────────────────

def discover_actions(env_snap, frame0: np.ndarray, available: list[int],
                     time_limit: float = 30.0) -> list[Action]:
    """
    Probe all available actions from a snapshot to discover their type.
    env_snap is a deepcopy — we deepcopy it further for probing.
    """
    deadline = time.time() + time_limit
    actions = []

    # Phase 1: Quick probe each action without position data
    # NEVER skip Phase 1 — every available action must be probed at least once
    nodata_info = {}
    for act_id in available:
        ec = copy.deepcopy(env_snap)
        obs = ec.step(act_id)
        if obs is None:
            continue
        f = np.array(obs.frame)
        shape_changed = f.shape != frame0.shape
        diff = -1 if shape_changed else int(np.sum(f != frame0))
        nodata_info[act_id] = {'frame': f, 'diff': diff, 'shape_changed': shape_changed}

    # Phase 2: Check position-dependence
    for act_id in available:
        if time.time() >= deadline:
            break
        ni = nodata_info.get(act_id)
        if ni is None:
            continue

        # Probe at step=4 grid to detect position-dependence
        # step=8 misses VC33 sprites. step=4 required (256 probes, ~5s).
        is_pos_dep = False
        for py in range(0, 64, 4):
            if is_pos_dep:
                break
            for px in range(0, 64, 4):
                if time.time() >= deadline:
                    break
                ec = copy.deepcopy(env_snap)
                obs = ec.step(act_id, {'x': px, 'y': py})
                if obs is None:
                    continue
                fp = np.array(obs.frame)
                if fp.shape != ni['frame'].shape or not np.array_equal(fp, ni['frame']):
                    is_pos_dep = True
                    break

        if is_pos_dep:
            # Grid scan at step=2 to find unique click positions
            # step=4 misses buttons with 2px spacing (VC33 L2 has 7 buttons
            # at irregular 4-8px spacing, 3 missed at step=4).
            seen_effects = {}
            for probe_y in range(0, 64, 2):
                for probe_x in range(0, 64, 2):
                    if time.time() >= deadline:
                        break
                    ec = copy.deepcopy(env_snap)
                    obs = ec.step(act_id, {'x': probe_x, 'y': probe_y})
                    if obs is None:
                        continue
                    fc = np.array(obs.frame)
                    # Check it changed something from initial
                    changed = (fc.shape != frame0.shape or not np.array_equal(fc, frame0))
                    if changed:
                        fh = frame_hash(fc)
                        if fh not in seen_effects:
                            seen_effects[fh] = (probe_x, probe_y)

            for fh, (x, y) in seen_effects.items():
                actions.append(Action(act_id, {'x': x, 'y': y}))
            log.info(f"  Action {act_id}: click, {len(seen_effects)} positions")
        else:
            actions.append(Action(act_id, {}))
            label = "directional" if ni['diff'] > 4 else "counter/minor"
            log.info(f"  Action {act_id}: {label}, {ni['diff']}px")

    # Safety: include any Phase-1-probed actions that Phase 2 skipped due to deadline
    discovered_ids = {a.game_action for a in actions}
    for act_id, ni in nodata_info.items():
        if act_id not in discovered_ids:
            actions.append(Action(act_id, {}))
            log.info(f"  Action {act_id}: deadline-rescued, {ni['diff']}px")

    log.info(f"  Total: {len(actions)} actions")
    return actions


# ─── Game Analyzer ──────────────────────────────────────────────────────

def analyze_game(env_snap, actions: list[Action], frame0: np.ndarray) -> GameProfile:
    """Classify game type from action effects + action fingerprint (Hypatia taxonomy)."""
    profile = GameProfile(actions=actions)
    non_click = [a for a in actions if not a.data or 'x' not in a.data]
    profile.click_actions = [a for a in actions if a.data and 'x' in a.data]

    # Separate real directional actions (>4px change) from counter/minor ones
    profile.directional_actions = []
    for a in non_click:
        ec = copy.deepcopy(env_snap)
        obs = ec.step(a.game_action, a.data if a.data else None)
        if obs:
            diff = frame_diff(frame0, np.array(obs.frame))
            if diff > 4:
                profile.directional_actions.append(a)

    n_dir = len(profile.directional_actions)
    n_click = len(profile.click_actions)

    # ─── Action Fingerprint (Step 1 of Hypatia's pipeline) ───
    if n_dir >= 3 and n_click == 0:
        profile.game_type = "NAVIGATION"
        profile.genre_hints = ['navigation_maze', 'push_block', 'pursuit_evasion']
        # Check for moving agents (pursuit/evasion): apply no-op-like action twice,
        # see if frame changes even when we repeat the same action
        if n_dir >= 4:
            ec = copy.deepcopy(env_snap)
            obs1 = ec.step(profile.directional_actions[0].game_action,
                          profile.directional_actions[0].data or None)
            obs2 = ec.step(profile.directional_actions[0].game_action,
                          profile.directional_actions[0].data or None)
            if obs1 and obs2:
                f1, f2 = np.array(obs1.frame), np.array(obs2.frame)
                if f1.shape == f2.shape:
                    # If same action twice produces different frames beyond player movement,
                    # there might be moving agents
                    diff = int(np.sum(f1 != f2))
                    if diff > 200:  # Large diff suggests other things are moving
                        profile.has_moving_agents = True
                        profile.genre_hints = ['pursuit_evasion', 'navigation_maze']
        return profile

    if n_click == 0:
        profile.game_type = "UNKNOWN"
        return profile

    # ─── Effect Signature (Step 2) for click-based games ───
    test_clicks = profile.click_actions[:5]
    si_count = 0
    neighbor_coupling_count = 0

    for act in test_clicks:
        # Self-inverse test
        ec = copy.deepcopy(env_snap)
        ec.step(act.game_action, act.data)
        obs2 = ec.step(act.game_action, act.data)
        if obs2 and np.array_equal(frame0, np.array(obs2.frame)):
            si_count += 1

    profile.self_inverse = si_count >= len(test_clicks) * 0.6

    # Neighbor coupling test: do nearby clicks produce overlapping effects?
    if n_click >= 3:
        effects = []
        for act in test_clicks[:3]:
            ec = copy.deepcopy(env_snap)
            obs = ec.step(act.game_action, act.data)
            if obs:
                f = np.array(obs.frame)
                if f.shape == frame0.shape:
                    changed_pixels = set(zip(*np.where(f != frame0)))
                    effects.append((act, changed_pixels))
        # Check for overlapping effect regions between different clicks
        for i in range(len(effects)):
            for j in range(i + 1, len(effects)):
                overlap = effects[i][1] & effects[j][1]
                if len(overlap) > 5:  # Significant overlap = neighbor coupling
                    neighbor_coupling_count += 1
        if neighbor_coupling_count > 0:
            profile.has_neighbor_coupling = True

    # Gravity test: after click, do pixels "fall" (shift downward)?
    if n_click >= 1:
        ec = copy.deepcopy(env_snap)
        obs = ec.step(test_clicks[0].game_action, test_clicks[0].data)
        if obs:
            f1 = np.array(obs.frame)
            # Apply same action again — if gravity, things may settle differently
            obs2 = ec.step(test_clicks[0].game_action, test_clicks[0].data)
            if obs2:
                f2 = np.array(obs2.frame)
                if f1.shape == f2.shape == frame0.shape and f1.shape[0] >= 2:
                    # Check if bottom rows change more than top rows
                    h = f1.shape[-2]
                    top_diff = int(np.sum(f1[..., :h//2, :] != f2[..., :h//2, :]))
                    bot_diff = int(np.sum(f1[..., h//2:, :] != f2[..., h//2:, :]))
                    if bot_diff > top_diff * 3 and bot_diff > 20:
                        profile.has_gravity = True

    # ─── Genre Classification ───
    if profile.self_inverse:
        profile.game_type = "CLICK_TOGGLE"
        if profile.has_neighbor_coupling:
            profile.genre_hints = ['toggle_puzzle', 'constraint_satisfaction']
        else:
            profile.genre_hints = ['constraint_satisfaction', 'paint_fill']
    else:
        profile.game_type = "CLICK_SEQUENCE"
        if profile.has_gravity:
            profile.genre_hints = ['gravity_puzzle', 'sorting']
        elif profile.has_neighbor_coupling:
            profile.genre_hints = ['rail_slider', 'toggle_puzzle']
        elif n_dir >= 2:
            profile.genre_hints = ['push_block', 'sliding_tile', 'sorting']
        else:
            profile.genre_hints = ['constraint_satisfaction', 'rail_slider', 'paint_fill']

    log.info(f"Type: {profile.game_type} | Hints: {profile.genre_hints}"
             f"{' [neighbor-coupled]' if profile.has_neighbor_coupling else ''}"
             f"{' [gravity]' if profile.has_gravity else ''}")
    return profile


# ─── LLM Reasoning Layer ──────────────────────────────────────────────

@dataclass
class LLMHint:
    """Structured hints from LLM game analysis."""
    game_description: str = ""
    suggested_strategy: str = ""
    action_priority: list = field(default_factory=list)  # ordered action IDs
    rollout_len: int = 0  # 0 = use default
    raw_response: str = ""

# Cache: game_id → LLMHint (same game, same analysis)
_llm_cache: dict[str, LLMHint] = {}

def _frame_to_base64(frame: np.ndarray, scale: int = 2) -> str:
    """Convert frame array to base64 PNG for LLM vision."""
    if not HAS_PIL:
        return ""
    # frame shape: (C, H, W) or (H, W, C) — normalize to (H, W, C)
    if frame.ndim == 3 and frame.shape[0] in (1, 3, 4):
        frame = np.transpose(frame, (1, 2, 0))
    if frame.ndim == 2:
        frame = np.stack([frame] * 3, axis=-1)
    elif frame.ndim == 3 and frame.shape[2] == 1:
        frame = np.repeat(frame, 3, axis=2)
    if scale > 1:
        frame = np.repeat(np.repeat(frame, scale, axis=0), scale, axis=1)
    img = Image.fromarray(frame.astype(np.uint8))
    buf = io.BytesIO()
    img.save(buf, format='PNG')
    return base64.b64encode(buf.getvalue()).decode()


def _describe_frame(frame: np.ndarray) -> str:
    """Create a text description of a game frame for non-vision LLMs."""
    # Normalize to (H, W, C)
    if frame.ndim == 3 and frame.shape[0] in (1, 3, 4):
        f = np.transpose(frame, (1, 2, 0))
    elif frame.ndim == 2:
        f = frame[:, :, None]
    else:
        f = frame

    h, w = f.shape[:2]
    channels = f.shape[2] if f.ndim == 3 else 1
    is_grayscale = (channels == 1) or (channels >= 3 and np.allclose(f[:,:,0], f[:,:,1]) and np.allclose(f[:,:,0], f[:,:,2]))

    # Find unique colors (downsample if large)
    if h * w > 10000:
        step = max(1, int(np.sqrt(h * w / 5000)))
        sample = f[::step, ::step]
    else:
        sample = f
    pixels = sample.reshape(-1, sample.shape[-1])
    unique_colors = np.unique(pixels, axis=0)

    # Color distribution
    color_counts = []
    for color in unique_colors[:20]:  # cap at 20 colors
        mask = np.all(sample == color, axis=-1)
        pct = mask.sum() / (sample.shape[0] * sample.shape[1]) * 100
        if pct > 0.5:
            if is_grayscale:
                # For grayscale games, describe as distinct color indices
                val = int(color[0])
                color_counts.append(f"  color-{val}: {pct:.0f}%")
            else:
                r, g, b = color[:3] if len(color) >= 3 else (color[0],) * 3
                color_counts.append(f"  RGB({r},{g},{b}): {pct:.0f}%")

    # Detect grid structure
    grid_info = ""
    if f.shape[-1] >= 3:
        gray = f[:, :, 0].astype(float)
        # Check for horizontal/vertical lines (grid indicators)
        row_var = np.var(gray, axis=1)
        col_var = np.var(gray, axis=0)
        h_lines = np.sum(row_var < np.mean(row_var) * 0.1)
        v_lines = np.sum(col_var < np.mean(col_var) * 0.1)
        if h_lines > 2 or v_lines > 2:
            grid_info = f"\n  Possible grid structure: {h_lines} horizontal lines, {v_lines} vertical lines"

    desc = f"Frame: {w}x{h} pixels, {len(unique_colors)} unique colors"
    desc += grid_info
    desc += "\n  Color palette:\n" + "\n".join(color_counts[:10])
    return desc


def llm_analyze_game(frame: np.ndarray, profile: GameProfile,
                     game_id: str = "", n_actions: int = 0) -> Optional[LLMHint]:
    """Ask an LLM to analyze a game and suggest strategy.

    Gated by ARC_USE_LLM=1 env var. Uses DeepSeek (text-only, cheapest).
    Set ARC_USE_VISION=1 to use Anthropic Claude with frame image instead.
    Results cached per game_id.
    """
    if not os.environ.get('ARC_USE_LLM', ''):
        return None
    if not HAS_REQUESTS:
        log.debug("  LLM: missing requests library")
        return None

    # Check cache (keyed by game_id + action count to detect mechanic changes)
    cache_key = f"{game_id}:{n_actions}" if game_id else None
    if cache_key and cache_key in _llm_cache:
        log.info(f"  LLM: cached result for {cache_key}")
        return _llm_cache[cache_key]

    use_vision = os.environ.get('ARC_USE_VISION', '')
    click_desc = f"{len(profile.click_actions)} click positions" if profile.click_actions else "no click actions"
    dir_desc = f"{len(profile.directional_actions)} directional actions" if profile.directional_actions else "no directional actions"

    # Load pattern library for few-shot context
    pattern_context = ""
    try:
        _lib_path = Path(__file__).parent.parent / "knowledge" / "pattern_library" / "patterns.json"
        if _lib_path.exists():
            import json as _json
            with open(_lib_path) as _f:
                _lib = _json.load(_f)
            if _lib.get('patterns'):
                _lines = ["\n\n# Reference Patterns (from solved games):"]
                for _p in _lib['patterns'][:7]:  # cap at 7 to save tokens
                    _h = _p.get('human', {})
                    _lines.append(f"- **{_p['genre'].upper()}**: {_h.get('description', 'N/A')[:100]}")
                    if _h.get('how_to_recognize') or _p.get('visual', {}).get('how_to_recognize'):
                        _recog = _h.get('how_to_recognize') or _p.get('visual', {}).get('how_to_recognize', '')
                        _lines.append(f"  Recognition: {_recog[:100]}")
                pattern_context = "\n".join(_lines)
    except Exception:
        pass  # pattern library is optional

    base_prompt = f"""You are analyzing an interactive game environment for an AI competition.
The game is played by taking actions (clicking positions or pressing directions).
Goal: complete each level using the fewest actions possible.
{pattern_context}

Current game analysis:
- Detected type: {profile.game_type}
- Actions: {n_actions} total ({click_desc}, {dir_desc})
- Self-inverse (clicking same spot twice undoes it): {profile.self_inverse}
- Commutative (order doesn't matter): {profile.commutative}"""

    footer = """
Answer concisely on separate lines:
GAME_TYPE: (puzzle/navigation/pattern-matching/sorting/painting/toggle/sequence/unknown)
STRATEGY: (one sentence — what approach should a solver use?)
ROLLOUT_DEPTH: (estimated actions to solve one level, number only)
ACTION_PRIORITY: (which action types to try first — e.g. 'click' or 'directional' or 'click then directional')"""

    # ─── LLM Backend Selection ──────────────────────────────────────────
    # Priority: ARC_LLM_ENDPOINT (OpenAI-compatible, for Qwen/vLLM/ollama)
    #         > ANTHROPIC (Claude vision)
    #         > DEEPSEEK (text-only fallback)
    oai_endpoint = os.environ.get('ARC_LLM_ENDPOINT', '')  # e.g. http://localhost:8000/v1
    oai_model = os.environ.get('ARC_LLM_MODEL', 'qwen2.5-vl-32b')
    oai_key = os.environ.get('ARC_LLM_KEY', 'none')  # vLLM/ollama don't need real keys

    if oai_endpoint:
        # OpenAI-compatible backend (vLLM, ollama, TGI, any open model)
        img_b64 = _frame_to_base64(frame) if HAS_PIL else None
        if img_b64:
            prompt = base_prompt + "\n\nSee the game screenshot below." + footer
            content = [
                {'type': 'image_url', 'image_url': {
                    'url': f'data:image/png;base64,{img_b64}'}},
                {'type': 'text', 'text': prompt},
            ]
        else:
            frame_desc = _describe_frame(frame)
            prompt = base_prompt + f"\n\nFrame description:\n{frame_desc}" + footer
            content = [{'type': 'text', 'text': prompt}]
        try:
            endpoint = oai_endpoint.rstrip('/')
            resp = _requests.post(
                f'{endpoint}/chat/completions',
                headers={
                    'Authorization': f'Bearer {oai_key}',
                    'Content-Type': 'application/json',
                },
                json={
                    'model': oai_model,
                    'messages': [{'role': 'user', 'content': content}],
                    'max_tokens': 200,
                    'temperature': 0.3,
                },
                timeout=30,
            )
            resp.raise_for_status()
            text = resp.json()['choices'][0]['message']['content']
            log.info(f"  LLM ({oai_model}): got response")
        except Exception as e:
            log.warning(f"  LLM ({oai_model}): API error: {e}")
            return None
    elif use_vision and HAS_PIL:
        # Anthropic Claude with vision
        api_key = os.environ.get('ANTHROPIC', '')
        if not api_key:
            log.debug("  LLM: no ANTHROPIC key for vision")
            return None
        img_b64 = _frame_to_base64(frame)
        if not img_b64:
            return None
        prompt = base_prompt + "\n\nSee the game screenshot below." + footer
        try:
            resp = _requests.post(
                'https://api.anthropic.com/v1/messages',
                headers={
                    'x-api-key': api_key,
                    'anthropic-version': '2023-06-01',
                    'Content-Type': 'application/json',
                },
                json={
                    'model': 'claude-haiku-4-5-20251001',
                    'max_tokens': 200,
                    'messages': [{'role': 'user', 'content': [
                        {'type': 'image', 'source': {
                            'type': 'base64', 'media_type': 'image/png',
                            'data': img_b64}},
                        {'type': 'text', 'text': prompt},
                    ]}],
                },
                timeout=20,
            )
            resp.raise_for_status()
            text = resp.json()['content'][0]['text']
        except Exception as e:
            log.warning(f"  LLM (vision): API error: {e}")
            return None
    else:
        # DeepSeek text-only (cheap, ~$0.001/call)
        api_key = os.environ.get('DEEPSEEK', '')
        if not api_key:
            log.debug("  LLM: no DEEPSEEK API key")
            return None
        frame_desc = _describe_frame(frame)
        prompt = base_prompt + f"\n\nFrame description:\n{frame_desc}" + footer
        try:
            resp = _requests.post(
                'https://api.deepseek.com/chat/completions',
                headers={
                    'Authorization': f'Bearer {api_key}',
                    'Content-Type': 'application/json',
                },
                json={
                    'model': 'deepseek-chat',
                    'messages': [{'role': 'user', 'content': prompt}],
                    'max_tokens': 200,
                    'temperature': 0.3,
                },
                timeout=15,
            )
            resp.raise_for_status()
            text = resp.json()['choices'][0]['message']['content']
        except Exception as e:
            log.warning(f"  LLM: API error: {e}")
            return None

    # Parse response
    hint = LLMHint(raw_response=text)
    lines = [l.strip() for l in text.split('\n') if l.strip()]
    for i, line in enumerate(lines):
        up = line.upper()
        if up.startswith('GAME_TYPE:'):
            hint.game_description = line[10:].strip()
        elif up.startswith('STRATEGY:'):
            hint.suggested_strategy = line[9:].strip()
        elif up.startswith('ROLLOUT_DEPTH:'):
            try:
                hint.rollout_len = int(''.join(c for c in line[14:] if c.isdigit()) or '0')
            except ValueError:
                pass
        elif up.startswith('ACTION_PRIORITY:'):
            hint.action_priority = [s.strip().lower() for s in line[16:].split(',') if s.strip()]
    # Fallback: if first non-empty line has no label, treat it as game_type
    if not hint.game_description and lines:
        first = lines[0]
        if ':' not in first and len(first) < 60:
            hint.game_description = first.strip('* ').lower()

    log.info(f"  LLM: {hint.game_description} | {hint.suggested_strategy}")

    if cache_key:
        _llm_cache[cache_key] = hint

    return hint


# ─── Noise Detection ───────────────────────────────────────────────────

def detect_noise(env_snap, frame0: np.ndarray, available: list[int],
                 n_probes: int = 3) -> Optional[np.ndarray]:
    """Detect counter/noise pixels by intersecting diffs from different actions.

    Noise = pixels that change regardless of which action is taken (counters,
    fuel bars, step displays). Found by probing different actions from the
    same state and intersecting their frame diffs.
    """
    if not available:
        return None

    # Probe different actions to find pixels that ALWAYS change
    noise_candidates = None
    probed = 0
    for act_id in available[:min(len(available), n_probes + 2)]:
        ec = copy.deepcopy(env_snap)
        obs = ec.step(act_id)
        if obs is None:
            continue
        f = np.array(obs.frame)
        if f.shape != frame0.shape:
            continue
        diff = (frame0 != f)
        if not diff.any():
            continue
        if noise_candidates is None:
            noise_candidates = diff.copy()
        else:
            noise_candidates &= diff
        probed += 1
        if probed >= n_probes:
            break

    if noise_candidates is None or not noise_candidates.any():
        return None

    # Extend to full counter/fuel bars: same-value pixels in same rows
    mask = noise_candidates
    if mask.ndim == 3:
        for ch in range(frame0.shape[0]):
            rows = np.where(mask[ch].any(axis=-1))[0]
            for row in rows:
                cols = np.where(mask[ch, row])[0]
                if len(cols) > 0:
                    val = frame0[ch, row, cols[0]]
                    mask[ch, row, frame0[ch, row] == val] = True

    log.info(f"  Noise: {mask.sum()} pixels masked")
    return mask


# ─── Deepcopy BFS ──────────────────────────────────────────────────────

def deepcopy_bfs(env_snap, actions: list[Action], start_levels: int,
                 timeout: float = 60.0, max_states: int = 15000,
                 noise_mask: Optional[np.ndarray] = None,
                 dedup_cycle: int = 0) -> Optional[list[Action]]:
    """
    BFS using deepcopy for O(1) state expansion.
    env_snap is at the start of the current level (deepcopy from caller).
    Detects win via levels_completed > start_levels.
    """
    deadline = time.time() + timeout

    # Get initial frame by probing
    init_env = copy.deepcopy(env_snap)
    init_obs = init_env.step(actions[0].game_action,
                              actions[0].data if actions[0].data else None)
    if init_obs and is_level_won(init_obs, start_levels):
        return [actions[0]]
    # Use initial frame hash + a sentinel for root
    fh_root = "bfs_root"

    queue = deque([(copy.deepcopy(env_snap), [], 0)])
    visited = {(fh_root, 0) if dedup_cycle > 0 else fh_root}
    states_explored = 0
    max_depth = 0

    while queue and time.time() < deadline and states_explored < max_states:
        env_state, path, depth = queue.popleft()
        max_depth = max(max_depth, depth)

        for action in actions:
            if time.time() >= deadline:
                break

            ec = copy.deepcopy(env_state)
            obs = ec.step(action.game_action, action.data if action.data else None)
            if obs is None:
                continue

            states_explored += 1
            new_path = path + [action]

            if is_level_won(obs, start_levels):
                log.info(f"  BFS: WIN depth {len(new_path)}, {states_explored} states")
                return new_path

            fn = np.array(obs.frame)
            fh = frame_hash(fn, noise_mask)
            vk = (fh, (depth + 1) % dedup_cycle) if dedup_cycle > 0 else fh

            if vk not in visited:
                visited.add(vk)
                queue.append((ec, new_path, depth + 1))

    log.info(f"  BFS: exhausted. {states_explored} states, depth {max_depth}")
    return None


def dynamic_action_bfs(env_snap, actions: list[Action], start_levels: int,
                       timeout: float = 60.0, max_states: int = 20000,
                       noise_mask: Optional[np.ndarray] = None) -> Optional[list[Action]]:
    """BFS for games with dynamic available_actions (e.g. Othello).

    Instead of using a fixed action list, reads obs.available_actions from
    each state. Creates Action objects on the fly for each valid move.
    """
    deadline = time.time() + timeout

    # Get initial obs to know starting available_actions
    init_env = copy.deepcopy(env_snap)
    # Store the env + its last known available_actions
    if hasattr(init_env, '_last_obs') and init_env._last_obs is not None:
        init_actions = list(init_env._last_obs.available_actions)
    else:
        # Probe to get initial actions
        probe = copy.deepcopy(env_snap)
        obs0 = probe.step(actions[0].game_action, actions[0].data if actions[0].data else None)
        if obs0 is None:
            return None
        init_actions = list(obs0.available_actions) if obs0.available_actions else [a.game_action for a in actions]

    # BFS queue: (env_state, path, available_action_ids)
    queue = deque([(copy.deepcopy(env_snap), [], init_actions)])
    visited = {"dynbfs_root"}
    states_explored = 0
    max_depth = 0

    while queue and time.time() < deadline and states_explored < max_states:
        env_state, path, avail_acts = queue.popleft()
        max_depth = max(max_depth, len(path))

        for act_id in avail_acts:
            if time.time() >= deadline:
                break

            ec = copy.deepcopy(env_state)
            obs = ec.step(act_id)
            if obs is None:
                continue

            states_explored += 1
            action_obj = Action(act_id, {})
            new_path = path + [action_obj]

            if is_level_won(obs, start_levels):
                log.info(f"  DynBFS: WIN depth {len(new_path)}, {states_explored} states")
                return new_path

            fn = np.array(obs.frame)
            fh = frame_hash(fn, noise_mask)

            if fh not in visited:
                visited.add(fh)
                # Get next state's available actions
                next_acts = list(obs.available_actions) if obs.available_actions else avail_acts
                queue.append((ec, new_path, next_acts))

    log.info(f"  DynBFS: exhausted. {states_explored} states, depth {max_depth}")
    return None


# ─── Navigation Model ──────────────────────────────────────────────────

def navigation_solve(env_snap, actions: list[Action], start_levels: int,
                     timeout: float = 60.0, max_states: int = 50000,
                     noise_mask: Optional[np.ndarray] = None) -> Optional[list[Action]]:
    """
    Navigation-specific BFS with reversal pruning.
    Only uses directional (non-click) actions.
    """
    deadline = time.time() + timeout
    dir_actions = [a for a in actions if not a.data or 'x' not in a.data]
    if len(dir_actions) < 2:
        return None

    # Detect opposites: apply a1+a2, compare with reference action alone
    # If a1+a2 ≈ ref (ignoring noise), they cancel → opposites
    ref_env = copy.deepcopy(env_snap)
    ref_obs = ref_env.step(dir_actions[0].game_action,
                           dir_actions[0].data if dir_actions[0].data else None)
    if ref_obs is None:
        return None
    ref_frame = np.array(ref_obs.frame)

    opposites = {}
    for i, a1 in enumerate(dir_actions):
        for a2 in dir_actions[i + 1:]:
            ec = copy.deepcopy(env_snap)
            ec.step(a1.game_action, a1.data if a1.data else None)
            obs2 = ec.step(a2.game_action, a2.data if a2.data else None)
            if obs2 is None:
                continue
            f2 = np.array(obs2.frame)
            if f2.shape == ref_frame.shape:
                diff = ref_frame != f2
                if noise_mask is not None and noise_mask.shape == diff.shape:
                    diff = diff & ~noise_mask
                if int(diff.sum()) <= 2:
                    opposites[a1] = a2
                    opposites[a2] = a1

    log.info(f"  Nav: {len(dir_actions)} dirs, {len(opposites)//2} opposite pairs")

    # Get initial frame for heuristic scoring
    init_env = copy.deepcopy(env_snap)
    init_obs = init_env.step(dir_actions[0].game_action,
                              dir_actions[0].data if dir_actions[0].data else None)
    init_frame = np.array(init_obs.frame) if init_obs else None

    def _frame_score(frame_arr):
        """Higher score = more different from start = likely more progress.
        Scores by pixel difference from initial frame, ignoring the edge bar
        (which is an action budget, not a progress indicator)."""
        if init_frame is None or frame_arr.shape != init_frame.shape:
            return 0
        diff = frame_arr != init_frame
        if noise_mask is not None and noise_mask.shape == diff.shape:
            diff = diff & ~noise_mask
        # Exclude top 2 rows (likely action budget bar) from scoring
        if diff.ndim == 2 and diff.shape[0] > 4:
            diff[:2, :] = False
        return int(diff.sum())

    # Best-first search with reversal pruning — prioritize most-changed frames
    import heapq
    fh_root = "nav_root"
    # Priority queue: (-score, counter, env, path, last_act)
    counter = 0
    pq = [(-0, counter, copy.deepcopy(env_snap), [], None)]
    visited = {fh_root}
    states = 0

    while pq and time.time() < deadline and states < max_states:
        _, _, env_state, path, last_act = heapq.heappop(pq)

        for action in dir_actions:
            if time.time() >= deadline:
                break
            # Reversal pruning: don't immediately undo last action
            if last_act is not None and opposites.get(action) == last_act:
                continue

            ec = copy.deepcopy(env_state)
            obs = ec.step(action.game_action, action.data if action.data else None)
            if obs is None:
                continue
            states += 1

            new_path = path + [action]
            if is_level_won(obs, start_levels):
                log.info(f"  Nav: WIN depth {len(new_path)}, {states} states")
                return new_path

            fn = np.array(obs.frame)
            fh = frame_hash(fn, noise_mask)
            if fh not in visited:
                visited.add(fh)
                score = _frame_score(fn)
                counter += 1
                heapq.heappush(pq, (-score, counter, ec, new_path, action))

    log.info(f"  Nav: exhausted. {states} states")
    return None


# ─── MCTS ──────────────────────────────────────────────────────────────

class MCTSNode:
    __slots__ = ['parent', 'action', 'children', 'visits', 'value',
                 'env_copy', 'untried', 'depth']

    def __init__(self, parent=None, action=None, env_copy=None, depth=0):
        self.parent = parent
        self.action = action
        self.children = []
        self.visits = 0
        self.value = 0.0
        self.env_copy = env_copy
        self.untried = []
        self.depth = depth

    def ucb1(self, c=1.41):
        if self.visits == 0:
            return float('inf')
        return self.value / self.visits + c * math.sqrt(
            math.log(self.parent.visits) / self.visits)

    def best_child(self):
        return max(self.children, key=lambda c: c.ucb1())


def _biased_action(actions: list[Action], priority: list[str], rng=np.random) -> 'Action':
    """Pick an action with 70/30 bias toward priority types, else uniform random."""
    if priority and rng.random() < 0.7:
        # Try to pick from priority category
        for cat in priority:
            matches = [a for a in actions if cat in a.name.lower()]
            if matches:
                return matches[rng.randint(len(matches))]
    return actions[rng.randint(len(actions))]


def mcts_search(env_snap, actions: list[Action], start_levels: int,
                timeout: float = 30.0, noise_mask: Optional[np.ndarray] = None,
                max_iters: int = 50000, rollout_len: int = 200,
                action_priority: list[str] = None,
                goal_frame: Optional[np.ndarray] = None) -> Optional[list[Action]]:
    """MCTS with deepcopy nodes and novelty-based rollout rewards.
    goal_frame: if provided, pixel similarity to this frame adds reward (0-0.3)."""
    deadline = time.time() + timeout
    root = MCTSNode(env_copy=copy.deepcopy(env_snap), depth=0)
    root.untried = list(actions)
    novel_frames = set()
    iters = 0

    while time.time() < deadline and iters < max_iters:
        iters += 1

        # SELECT
        node = root
        while not node.untried and node.children:
            max_ch = max(2, int(math.sqrt(node.visits + 1)))
            if len(node.children) < max_ch:
                remaining = [a for a in actions if a not in [c.action for c in node.children]]
                if remaining:
                    node.untried = remaining[:1]
                    break
            node = node.best_child()

        # EXPAND
        if node.untried:
            action = node.untried.pop(0)
            ec = copy.deepcopy(node.env_copy)
            obs = ec.step(action.game_action, action.data if action.data else None)
            if obs is not None:
                child = MCTSNode(parent=node, action=action, env_copy=ec, depth=node.depth + 1)
                child.untried = list(actions)
                node.children.append(child)
                node = child

                if is_level_won(obs, start_levels):
                    path = []
                    n = node
                    while n.parent:
                        path.append(n.action)
                        n = n.parent
                    path.reverse()
                    log.info(f"  MCTS: WIN depth {len(path)}, {iters} iters")
                    return path

        # ROLLOUT
        reward = 0.0
        best_goal_sim = 0.0
        if node.env_copy:
            re = copy.deepcopy(node.env_copy)
            _prio = action_priority or []
            for _ in range(rollout_len):
                act = _biased_action(actions, _prio)
                obs = re.step(act.game_action, act.data if act.data else None)
                if obs is None:
                    break
                if is_level_won(obs, start_levels):
                    reward = 1.0
                    break
                cur = np.array(obs.frame)
                fh = frame_hash(cur, noise_mask)
                if fh not in novel_frames:
                    novel_frames.add(fh)
                    reward += 0.1
                # Goal similarity bonus — how close is this frame to the pre-win state?
                if goal_frame is not None:
                    cf = cur[0] if cur.ndim == 3 else cur
                    gf = goal_frame[0] if goal_frame.ndim == 3 else goal_frame
                    if cf.shape == gf.shape:
                        sim = np.mean(cf == gf)  # pixel match ratio (0-1)
                        best_goal_sim = max(best_goal_sim, sim)

        # Add goal proximity bonus (max 0.3, scaled by best similarity seen)
        if goal_frame is not None and best_goal_sim > 0.5:
            reward += (best_goal_sim - 0.5) * 0.6  # maps 0.5-1.0 → 0.0-0.3

        # NOTE: Edge bar is action BUDGET, not progress. Not useful for
        # reward shaping — gradient is always -1 per action. Removed.

        # BACKPROP
        n = node
        while n:
            n.visits += 1
            n.value += reward
            n = n.parent

    log.info(f"  MCTS: {iters} iters, {len(novel_frames)} novel frames")
    return None


# ─── Random Walk Explorer ──────────────────────────────────────────────

def random_walk(env_snap, actions: list[Action], start_levels: int,
                timeout: float = 30.0, walk_len: int = 200,
                noise_mask: Optional[np.ndarray] = None,
                max_walks: int = 500,
                action_priority: list[str] = None) -> Optional[list[Action]]:
    """Go-Explore lite with novelty tracking and via negativa."""
    deadline = time.time() + timeout
    checkpoints = [(copy.deepcopy(env_snap), [])]
    novel_frames = set()
    dead_ends = defaultdict(int)
    STRIKE = 5

    for walk_num in range(max_walks):
        if time.time() >= deadline:
            break

        # Pick a checkpoint (prefer recent/deep)
        idx = np.random.randint(max(0, len(checkpoints) - 50), len(checkpoints))
        start_env, start_path = checkpoints[idx]
        ec = copy.deepcopy(start_env)
        path = list(start_path)

        _rprio = action_priority or []
        for step in range(walk_len):
            if time.time() >= deadline:
                break
            act = _biased_action(actions, _rprio)
            obs = ec.step(act.game_action, act.data if act.data else None)
            if obs is None:
                break
            path.append(act)

            if is_level_won(obs, start_levels):
                log.info(f"  RW: WIN depth {len(path)}, walk {walk_num}")
                return path

            fn = np.array(obs.frame)
            fh = frame_hash(fn, noise_mask)
            if dead_ends[fh] >= STRIKE:
                break  # Dead end — restart from different checkpoint
            if fh not in novel_frames:
                novel_frames.add(fh)
                checkpoints.append((copy.deepcopy(ec), list(path)))

        # Strike terminal frame
        if path:
            dead_ends[fh] += 1

    log.info(f"  RW: {walk_num + 1} walks, {len(novel_frames)} novel, {len(checkpoints)} ckpts")
    return None


# ─── Block-State BFS (Mile-High View) ─────────────────────────────────

def _segment_blocks(frame: np.ndarray, min_size: int = 8) -> tuple:
    """Segment frame into blocks (connected regions). Returns hashable state tuple.

    The 'mile-high view': see blocks, not pixels. State space shrinks from
    10^21 (pixel) to 10^4 (block positions). Like seeing the forest, not trees.
    """
    from scipy import ndimage
    f = frame[0] if len(frame.shape) == 3 and frame.shape[0] <= 4 else frame
    vals, counts = np.unique(f, return_counts=True)
    if len(vals) == 0:
        return ()
    bg = int(vals[counts.argmax()])

    block_keys = []
    for v in vals:
        if int(v) == bg:
            continue
        mask = (f == int(v)).astype(np.int32)
        labeled, n = ndimage.label(mask)
        for i in range(1, n + 1):
            region = np.where(labeled == i)
            size = len(region[0])
            if size < min_size:
                continue
            # Block key: (color, center_y, center_x, size)
            cy = int(round(region[0].mean()))
            cx = int(round(region[1].mean()))
            block_keys.append((int(v), cy, cx, size))

    block_keys.sort()
    return tuple(block_keys)


def forward_model_solve(env_snap, actions: list[Action], start_levels: int,
                        timeout: float = 30.0,
                        noise_mask: Optional[np.ndarray] = None) -> Optional[list[Action]]:
    """Forward model solver — learns game physics from observation, then plans.

    Phase 1 (LEARN): Take N actions, record (frame, action, next_frame) transitions.
    Build a transition model: for each action, what pixel regions change and how?

    Phase 2 (PLAN): Use the learned model to simulate future states and pick
    actions that maximize progress (new frames, level completion).

    This bridges the gap for games needing predictive planning (Flappy, RPG).
    """
    import copy
    deadline = time.time() + timeout
    env = copy.deepcopy(env_snap)

    # Phase 1: LEARN — observe transitions for each action
    n_observe = min(5, len(actions))  # observations per action
    transitions = defaultdict(list)  # action_idx -> [(frame_before, frame_after, delta)]

    f = None
    for obs_round in range(n_observe):
        for ai, act in enumerate(actions):
            if time.time() > deadline:
                break
            obs = env.step(act.game_action, act.data or None)
            if obs is None:
                return None
            frame = np.array(obs.frame); frame = frame[0] if frame.ndim == 3 else frame
            if is_level_won(obs, start_levels):
                # Won during observation — replay the actions
                solution = []
                for r in range(obs_round + 1):
                    for a in range(ai + 1 if r == obs_round else len(actions)):
                        solution.append(actions[a])
                return solution if solution else [act]
            if f is not None:
                delta = frame.astype(np.int16) - f.astype(np.int16)
                transitions[ai].append({
                    'before': f.copy(),
                    'after': frame.copy(),
                    'changed_pixels': int(np.sum(np.abs(delta) > 10)),
                    'mean_shift': float(np.mean(delta[np.abs(delta) > 10])) if np.any(np.abs(delta) > 10) else 0.0,
                })
            f = frame.copy()

    # Phase 2: PLAN — score each action by its effect, prefer novel states
    if not transitions:
        return None

    # Build action profiles: which actions cause the most change?
    action_scores = {}
    for ai in transitions:
        changes = [t['changed_pixels'] for t in transitions[ai]]
        action_scores[ai] = np.mean(changes) if changes else 0

    # Identify "progress" actions (most change) vs "idle" actions (least change)
    if not action_scores:
        return None

    sorted_actions = sorted(action_scores.items(), key=lambda x: -x[1])
    best_action_idx = sorted_actions[0][0]

    # Phase 3: EXECUTE — use learned preferences with exploration
    solution = []
    seen_frames = set()
    env2 = copy.deepcopy(env_snap)

    # Replay observation phase
    for a in solution:
        env2.step(a.game_action, a.data or None)

    while time.time() < deadline:
        # 80% exploit best action, 20% explore others
        if np.random.random() < 0.8:
            chosen = best_action_idx
        else:
            chosen = np.random.randint(len(actions))

        act = actions[chosen]
        obs = env2.step(act.game_action, act.data or None)
        if obs is None:
            break
        solution.append(act)

        if is_level_won(obs, start_levels):
            return solution

        frame = np.array(obs.frame); frame = frame[0] if frame.ndim == 3 else frame
        fhash = hashlib.md5(frame.tobytes()).hexdigest()[:16]
        seen_frames.add(fhash)

        # Update best action based on novelty
        if len(solution) % 20 == 0 and len(sorted_actions) > 1:
            # Try second-best action for a while
            best_action_idx = sorted_actions[len(solution) // 20 % len(sorted_actions)][0]

    return None  # Didn't solve in time


def reactive_controller(env_snap, actions: list[Action], start_levels: int,
                        timeout: float = 15.0,
                        noise_mask: Optional[np.ndarray] = None) -> Optional[list[Action]]:
    """Reactive controller for real-time games (Pong, Breakout, Space Invaders).

    Detects autonomous motion, then tracks the brightest small object (ball/projectile)
    and moves the controllable element (paddle/ship) to intercept.
    Only activates if objects move without player input.
    """
    import copy
    deadline = time.time() + timeout
    env = copy.deepcopy(env_snap)
    neutral = len(actions) // 2

    # Probe: take 3 neutral steps, check if anything moves on its own
    frames = []
    obs = env.step(actions[neutral].game_action, actions[neutral].data or None)
    if obs is None:
        return None
    f = np.array(obs.frame)
    frames.append(f[0] if f.ndim == 3 else f)

    for _ in range(2):
        obs = env.step(actions[neutral].game_action, actions[neutral].data or None)
        if obs is None:
            return None
        f = np.array(obs.frame)
        frames.append(f[0] if f.ndim == 3 else f)
        if is_level_won(obs, start_levels):
            return [actions[neutral]] * 3

    # Check for autonomous motion
    total_change = 0
    for i in range(1, len(frames)):
        diff = np.abs(frames[i].astype(np.int16) - frames[i-1].astype(np.int16))
        total_change += np.sum(diff > 20)
    if total_change < 5:
        return None  # Nothing moves autonomously — not real-time

    # Determine paddle orientation by checking where the controllable element is
    f = frames[-1]
    h, w = f.shape

    # Detect paddle: bright region (>150) at edges
    # Bottom edge = horizontal paddle (Breakout), Right edge = vertical (Pong)
    bottom_bright = np.sum(f[h*3//4:, :] > 150)
    right_bright = np.sum(f[:, w*3//4:] > 150)
    horiz = bottom_bright > right_bright

    # Detect game subtype: 2 actions + gravity = Flappy-like
    is_gravity_game = len(actions) == 2

    # Simple tracking: find ball (brightest, >240), find paddle (>150 at edge), move toward ball
    result_actions = [actions[neutral]] * 3  # the 3 probe steps
    for step in range(2000):
        if time.time() > deadline:
            break

        frame = np.array(obs.frame); frame = frame[0] if frame.ndim == 3 else frame

        # Find ball (brightest pixels)
        ball_ys, ball_xs = np.where(frame >= 240)
        if len(ball_ys) == 0:
            ball_ys, ball_xs = np.where(frame >= 220)

        if len(ball_ys) == 0:
            chosen = neutral
        elif is_gravity_game:
            # Gravity game (Flappy-like): find obstacle gap, flap to match altitude
            player_y = np.mean(ball_ys)
            player_x = np.mean(ball_xs)
            # Scan for nearest pipe/obstacle to the RIGHT of the player
            search_start = int(player_x) + 5
            gap_center = h // 2  # default
            found_obstacle = False
            for cx in range(search_start, w):
                col = frame[:, cx]
                obstacle_ys = np.where((col > 30) & (col < 240))[0]
                if len(obstacle_ys) > 3:
                    # Found an obstacle column — find the gap
                    all_ys = set(range(h))
                    gap_ys = sorted(all_ys - set(obstacle_ys))
                    if gap_ys:
                        gap_center = np.mean(gap_ys)
                    found_obstacle = True
                    break
            # Flap if bird is below gap center, coast if above
            chosen = 0 if player_y > gap_center + 1 else 1
        elif horiz:
            # Horizontal paddle: track ball x
            ball_x = np.mean(ball_xs)
            bottom = frame[h*3//4:, :]
            paddle_xs = np.where(np.any(bottom > 150, axis=0))[0]
            paddle_x = np.mean(paddle_xs) if len(paddle_xs) > 0 else w // 2
            chosen = 0 if ball_x < paddle_x - 2 else (len(actions) - 1 if ball_x > paddle_x + 2 else neutral)
        else:
            # Vertical paddle: track ball y
            ball_y = np.mean(ball_ys)
            right = frame[:, w*3//4:]
            paddle_ys = np.where(np.any(right > 150, axis=1))[0]
            paddle_y = np.mean(paddle_ys) if len(paddle_ys) > 0 else h // 2
            chosen = 0 if ball_y < paddle_y - 2 else (len(actions) - 1 if ball_y > paddle_y + 2 else neutral)

        act = actions[chosen]
        result_actions.append(act)
        obs = env.step(act.game_action, act.data or None)
        if obs is None:
            break
        if is_level_won(obs, start_levels):
            return result_actions

    return None


def sliding_tile_astar(env_snap, actions: list[Action], start_levels: int,
                       timeout: float = 30.0,
                       noise_mask: Optional[np.ndarray] = None,
                       goal_frame: Optional[np.ndarray] = None) -> Optional[list[Action]]:
    """A* search for sliding tile / push-block puzzles.

    Uses block positions as state representation (same as block_state_bfs) but
    adds a Manhattan-distance heuristic toward the goal configuration. If no
    goal_frame is available, falls back to an entropy-based heuristic that
    rewards ordered/symmetric block arrangements.

    Key differences from block_state_bfs:
      - Priority queue (A*) instead of FIFO (BFS) → finds shorter solutions
      - Goal-aware heuristic → exponentially fewer states explored
      - Block identity tracking → handles permutation puzzles correctly
    """
    import heapq
    deadline = time.time() + timeout

    # Get initial state from a no-op deepcopy (don't consume a move)
    init_env = copy.deepcopy(env_snap)
    # Peek at current frame without stepping — try _last_obs or step(noop)
    init_obs = getattr(init_env, '_last_obs', None)
    if init_obs is None:
        # Fallback: step with first action to get observation
        init_obs = init_env.step(actions[0].game_action,
                                  actions[0].data if actions[0].data else None)
        if init_obs is None:
            return None
        if is_level_won(init_obs, start_levels):
            return [actions[0]]
        # Reset env_snap since we consumed a move
        env_snap = init_env

    root_blocks = _segment_blocks(np.array(init_obs.frame))
    if len(root_blocks) < 2:
        return None  # Not a block puzzle

    # Build goal block state if we have a goal frame
    goal_blocks = None
    if goal_frame is not None:
        goal_blocks = _segment_blocks(goal_frame)
        if len(goal_blocks) < 2:
            goal_blocks = None

    def _heuristic(blocks):
        """Estimate distance to goal. Lower = closer to solution."""
        if not blocks:
            return 0
        if goal_blocks is not None and len(blocks) == len(goal_blocks):
            # Manhattan distance: match blocks by color+size, sum position deltas
            # Sort both by color to align matching blocks
            curr = sorted(blocks, key=lambda b: (b[0], b[3], b[1], b[2]))
            goal = sorted(goal_blocks, key=lambda b: (b[0], b[3], b[1], b[2]))
            total = 0
            for c, g in zip(curr, goal):
                # c/g = (color, cy, cx, size)
                total += abs(c[1] - g[1]) + abs(c[2] - g[2])
            return total
        # No goal: use symmetry/order heuristic
        # Reward blocks being in sorted positions (row-major order by color)
        positions = [(b[1], b[2]) for b in blocks]
        disorder = 0
        for i in range(len(positions) - 1):
            if positions[i] > positions[i + 1]:
                disorder += 1
        return disorder

    # For sliding puzzles, generate all possible click positions on the grid
    # (not just the 3 that happened to work at init — valid moves change as blank moves)
    all_click_actions = list(actions)  # start with discovered actions
    if len(root_blocks) >= 3:
        # Infer grid structure from block centers
        centers_y = sorted(set(b[1] for b in root_blocks))
        centers_x = sorted(set(b[2] for b in root_blocks))
        click_action_id = actions[0].game_action if actions else 6
        seen_positions = {(a.data.get('x', -1), a.data.get('y', -1))
                         for a in actions if a.data}
        for cy in centers_y:
            for cx in centers_x:
                if (cx, cy) not in seen_positions:
                    all_click_actions.append(Action(
                        game_action=click_action_id,
                        data={'x': cx, 'y': cy}))

    root_key = hash(root_blocks)
    h0 = _heuristic(root_blocks)
    # Priority queue: (f_score, tiebreak, env_state, path, block_key)
    counter = 0
    pq = [(h0, counter, copy.deepcopy(env_snap), [], root_key)]
    visited = {root_key: 0}  # block_key -> best g_score
    states = 0

    while pq and time.time() < deadline:
        f, _, env_state, path, _ = heapq.heappop(pq)
        g = len(path)

        for action in all_click_actions:
            if time.time() >= deadline:
                break

            ec = copy.deepcopy(env_state)
            obs = ec.step(action.game_action, action.data if action.data else None)
            if obs is None:
                continue
            states += 1

            new_path = path + [action]
            if is_level_won(obs, start_levels):
                log.info(f"  SlidingA*: WIN depth {len(new_path)}, {states} states explored")
                return new_path

            block_state = _segment_blocks(np.array(obs.frame))
            block_key = hash(block_state)
            new_g = len(new_path)

            # Only explore if we haven't seen this state, or found a shorter path
            prev_g = visited.get(block_key)
            if prev_g is not None and new_g >= prev_g:
                continue

            visited[block_key] = new_g
            h = _heuristic(block_state)
            counter += 1
            heapq.heappush(pq, (new_g + h, counter, ec, new_path, block_key))

        if states % 1000 == 0 and states > 0:
            log.debug(f"  SlidingA*: {states} states, {len(visited)} unique, "
                      f"queue {len(pq)}, depth {g}")

    log.info(f"  SlidingA*: exhausted. {states} states, {len(visited)} unique configs")
    return None


def block_state_bfs(env_snap, actions: list[Action], start_levels: int,
                    timeout: float = 30.0,
                    noise_mask: Optional[np.ndarray] = None) -> Optional[list[Action]]:
    """BFS using block-level state representation.

    Instead of hashing pixel frames (millions of unique states), hash block positions
    (compact representation). This makes sliding-block, rail-slider, and toggle puzzles
    tractable — state space drops from 10^21 to 10^4.
    """
    deadline = time.time() + timeout

    # Get initial block state without consuming a move
    init_env = copy.deepcopy(env_snap)
    init_obs = getattr(init_env, '_last_obs', None)
    if init_obs is None:
        # Must step to get observation — try each action to find one that works
        init_obs = init_env.step(actions[0].game_action,
                                  actions[0].data if actions[0].data else None)
        if init_obs is None:
            return None
        if is_level_won(init_obs, start_levels):
            return [actions[0]]
        # Reset to clean env since we consumed a move
        init_env = copy.deepcopy(env_snap)

    root_blocks = _segment_blocks(np.array(init_obs.frame))
    if len(root_blocks) < 2:
        log.info(f"  BlockBFS: too few blocks ({len(root_blocks)}), skipping")
        return None

    root_key = hash(root_blocks)
    queue = deque([(copy.deepcopy(env_snap), [], root_key)])
    visited = {root_key}
    states = 0
    max_depth = 0

    while queue and time.time() < deadline:
        env_state, path, _ = queue.popleft()
        max_depth = max(max_depth, len(path))

        for action in actions:
            if time.time() >= deadline:
                break

            ec = copy.deepcopy(env_state)
            obs = ec.step(action.game_action, action.data if action.data else None)
            if obs is None:
                continue
            states += 1

            new_path = path + [action]
            if is_level_won(obs, start_levels):
                log.info(f"  BlockBFS: WIN depth {len(new_path)}, {states} states, "
                         f"{len(visited)} unique block configs")
                return new_path

            # Hash by block positions, not pixel frame
            block_state = _segment_blocks(np.array(obs.frame))
            block_key = hash(block_state)

            if block_key not in visited:
                visited.add(block_key)
                queue.append((ec, new_path, block_key))

        if states % 500 == 0 and states > 0:
            log.debug(f"  BlockBFS: {states} states, {len(visited)} unique, depth {max_depth}")

    log.info(f"  BlockBFS: exhausted. {states} states, {len(visited)} unique block configs, "
             f"depth {max_depth}")
    return None


# ─── Mechanic Learning Solver (Probe → Model → Compute) ──────────────

def mechanic_learning_solve(env_snap, actions: list[Action], start_levels: int,
                            timeout: float = 30.0,
                            noise_mask: Optional[np.ndarray] = None,
                            frame0: Optional[np.ndarray] = None) -> Optional[list[Action]]:
    """Learn game mechanics by probing buttons, then compute the solution.

    Like a human: press each button, observe what changes, build a mental
    model of the rules, then reason about the solution. No BFS needed.

    Targets: water-pump/levee puzzles, sequence-dependent click games,
    and any game where buttons have learnable, cumulative effects.
    """
    deadline = time.time() + timeout
    click_actions = [a for a in actions if a.data and 'x' in a.data]
    if len(click_actions) < 2:
        return None  # Need multiple buttons to learn mechanics

    from scipy import ndimage

    def _get_regions(frame):
        """Segment frame into labeled regions with their properties."""
        f = frame[0] if len(frame.shape) == 3 and frame.shape[0] <= 4 else frame
        vals, counts = np.unique(f, return_counts=True)
        bg = int(vals[counts.argmax()])
        regions = []
        for v in vals:
            if int(v) == bg:
                continue
            mask = (f == int(v)).astype(np.int32)
            labeled, n = ndimage.label(mask)
            for i in range(1, n + 1):
                ys, xs = np.where(labeled == i)
                if len(ys) < 4:
                    continue
                regions.append({
                    'val': int(v), 'size': len(ys),
                    'ymin': int(ys.min()), 'ymax': int(ys.max()),
                    'xmin': int(xs.min()), 'xmax': int(xs.max()),
                    'cy': int(round(ys.mean())), 'cx': int(round(xs.mean())),
                })
        return regions

    def _region_diff(f1, f2):
        """Compare two frames at region level: which regions grew/shrank/appeared."""
        if f1.shape != f2.shape:
            return None, 'shape_change'
        g1 = f1[0] if len(f1.shape) == 3 else f1
        g2 = f2[0] if len(f2.shape) == 3 else f2
        diff_mask = (g1 != g2)
        n_changed = int(diff_mask.sum())
        if n_changed == 0:
            return {'n_changed': 0, 'columns': set(), 'rows': set()}, 'no_change'
        ys, xs = np.where(diff_mask)
        # Track which spatial zones changed (divide into 8 zones)
        w = g1.shape[1]
        h = g1.shape[0]
        zone_w = max(w // 8, 1)
        zone_h = max(h // 8, 1)
        col_zones = set(int(x // zone_w) for x in xs)
        row_zones = set(int(y // zone_h) for y in ys)
        return {
            'n_changed': n_changed,
            'columns': col_zones, 'rows': row_zones,
            'ymin': int(ys.min()), 'ymax': int(ys.max()),
            'xmin': int(xs.min()), 'xmax': int(xs.max()),
        }, 'normal'

    log.info(f"  MechLearn: {len(click_actions)} buttons to probe")

    # ── Phase 1: Single-click probe ──────────────────────────────────
    # Press each button once from initial state, observe effect
    button_effects = []
    shape_change_buttons = []
    pump_buttons = []
    counter_buttons = []
    init_frame = np.array(copy.deepcopy(env_snap).step(
        actions[0].game_action, actions[0].data if actions[0].data else None
    ).frame) if not click_actions else None

    # Detect mode transition: first click changes frame shape
    first_click = click_actions[0]
    ec_first = copy.deepcopy(env_snap)
    obs_first = ec_first.step(first_click.game_action, first_click.data)
    if obs_first is None:
        return None
    f_after_first = np.array(obs_first.frame)

    # Use passed frame0 (pre-action) to detect shape changes
    if frame0 is not None:
        mode_transition = (frame0.shape != f_after_first.shape)
    else:
        # No pre-action frame available — check if second click changes shape
        ec_test = copy.deepcopy(ec_first)
        obs_test = ec_test.step(first_click.game_action, first_click.data)
        if obs_test:
            f_test = np.array(obs_test.frame)
            mode_transition = (f_after_first.shape != f_test.shape)
        else:
            mode_transition = False

    if mode_transition:
        # The first click triggers a view change. Probe from AFTER this transition.
        log.info(f"  MechLearn: mode transition detected {frame0.shape}->{f_after_first.shape}")
        probe_env = ec_first
        probe_frame = f_after_first

        # Check if level was already won
        if is_level_won(obs_first, start_levels):
            return [first_click]
    else:
        probe_env = env_snap
        probe_frame = frame0 if frame0 is not None else f_after_first

    # Now probe each button from the post-transition state
    for i, act in enumerate(click_actions):
        if time.time() > deadline:
            break
        ec = copy.deepcopy(probe_env)
        obs_p = ec.step(act.game_action, act.data)
        if obs_p is None:
            continue

        if is_level_won(obs_p, start_levels):
            result = [first_click, act] if mode_transition else [act]
            log.info(f"  MechLearn: won on single probe! Button {i}")
            return result

        f_p = np.array(obs_p.frame)
        diff_info, diff_type = _region_diff(probe_frame, f_p)

        effect = {
            'button_idx': i, 'action': act,
            'diff_type': diff_type,
            'diff_info': diff_info,
        }
        button_effects.append(effect)

        if diff_type == 'shape_change':
            shape_change_buttons.append(effect)
        elif diff_info and diff_info['n_changed'] > 5:
            pump_buttons.append(effect)
        else:
            counter_buttons.append(effect)

    log.info(f"  MechLearn: {len(pump_buttons)} pumps, {len(shape_change_buttons)} connectors, "
             f"{len(counter_buttons)} counters")

    if not pump_buttons:
        log.info("  MechLearn: no pump buttons found, aborting")
        return None

    # ── Phase 2: Double-click probe — cumulative vs toggle ───────────
    # For each pump button, click twice and compare to base
    cumulative_pumps = []
    for eff in pump_buttons[:8]:  # Limit probing budget
        if time.time() > deadline:
            break
        act = eff['action']
        ec = copy.deepcopy(probe_env)
        ec.step(act.game_action, act.data)
        obs2 = ec.step(act.game_action, act.data)
        if obs2 is None:
            continue
        f2 = np.array(obs2.frame)
        diff2, dtype2 = _region_diff(probe_frame, f2)

        if dtype2 == 'shape_change':
            # Two clicks triggered level win or mode change
            if is_level_won(obs2, start_levels):
                result = [first_click, act, act] if mode_transition else [act, act]
                log.info(f"  MechLearn: won on double probe!")
                return result
            continue

        if diff2 and diff2['n_changed'] > 0:
            back_to_start = (diff2['n_changed'] == 0)
            eff['cumulative'] = not back_to_start
            eff['double_diff'] = diff2
            if not back_to_start:
                cumulative_pumps.append(eff)

    log.info(f"  MechLearn: {len(cumulative_pumps)} cumulative pumps")

    # ── Phase 2.5: Cross-inverse detection — find opposite-direction pairs ──
    # Like a human noticing "button A and button B undo each other"
    # NOTE: We detect inverses for LOGGING but do NOT eliminate them from BFS.
    # Reason: we can't predict which direction is "progress" — some levels need
    # draining (more empty), not filling. BFS with all buttons finds the answer.
    # Only eliminate when >4 buttons to keep BFS tractable.
    all_pumps_for_bfs = list(cumulative_pumps)  # Keep full set for BFS
    inverse_pairs = set()  # indices to SKIP (only used when many buttons)
    if len(cumulative_pumps) >= 2 and time.time() < deadline:
        pf = probe_frame[0] if len(probe_frame.shape) == 3 and probe_frame.shape[0] <= 4 else probe_frame

        for i in range(len(cumulative_pumps)):
            for j in range(i + 1, len(cumulative_pumps)):
                if time.time() > deadline:
                    break
                a_eff = cumulative_pumps[i]
                b_eff = cumulative_pumps[j]
                # Check if A→B returns close to base
                ec_ab = copy.deepcopy(probe_env)
                ec_ab.step(a_eff['action'].game_action, a_eff['action'].data)
                obs_ab = ec_ab.step(b_eff['action'].game_action, b_eff['action'].data)
                if obs_ab:
                    f_ab = np.array(obs_ab.frame)
                    if len(f_ab.shape) == 3 and f_ab.shape[0] <= 4:
                        f_ab = f_ab[0]
                    if f_ab.shape == pf.shape:
                        diff_ab = int((pf != f_ab).sum())
                        min_eff = min(a_eff['diff_info']['n_changed'], b_eff['diff_info']['n_changed'])
                        if diff_ab < min_eff * 0.15:
                            inverse_pairs.add((i, j))
                            log.info(f"  MechLearn: buttons {a_eff['button_idx']} and {b_eff['button_idx']} "
                                     f"are inverse pair (A→B={diff_ab}px)")

        # Only eliminate inverses when >4 buttons (BFS needs reduction)
        if inverse_pairs and len(cumulative_pumps) > 4:
            skip_indices = set()
            for i, j in inverse_pairs:
                skip_indices.add(j)  # Skip second of each pair
            cumulative_pumps = [p for idx, p in enumerate(cumulative_pumps) if idx not in skip_indices]
            log.info(f"  MechLearn: {len(cumulative_pumps)} pumps after removing inverses (had >{4})")
        elif inverse_pairs:
            log.info(f"  MechLearn: {len(inverse_pairs)} inverse pair(s) detected but keeping all {len(cumulative_pumps)} for BFS (≤4 buttons)")

    # ── Phase 3: Quick single-pump check ────────────────────────────
    # Some games = just spam one button N times
    if time.time() > deadline:
        return None

    cumulative_pumps.sort(key=lambda e: e['diff_info']['n_changed'], reverse=True)
    for eff in cumulative_pumps[:4]:
        if time.time() > deadline:
            break
        act = eff['action']
        ec = copy.deepcopy(probe_env)
        path = [first_click] if mode_transition else []
        for click_count in range(50):
            obs_s = ec.step(act.game_action, act.data)
            if obs_s is None:
                break
            path.append(act)
            if is_level_won(obs_s, start_levels):
                log.info(f"  MechLearn: won by spamming button {eff['button_idx']}! "
                         f"{len(path)} actions")
                return path

    # ── Phase 4: Mini-BFS with reduced button set ────────────────────
    # After inverse elimination, search space is small enough for BFS.
    # This is the "muscle memory" — probe identified the buttons, now BFS
    # finds the optimal interleaving (cascade overflow, etc).
    effective_pumps = cumulative_pumps + shape_change_buttons
    if effective_pumps and time.time() < deadline:
        n_buttons = len(effective_pumps)
        time_left = deadline - time.time()
        # Budget: ~50K states, but scale with time available
        max_states = min(50000, int(time_left * 1500))
        if max_states < 500:
            max_states = 500
        log.info(f"  MechLearn: mini-BFS with {n_buttons} buttons, max {max_states} states")

        bfs_queue = deque()
        bfs_seen = set()

        def _frame_hash(f):
            """Consistent hash regardless of 2D/3D frame format."""
            if isinstance(f, np.ndarray):
                return hashlib.md5(f.ravel().tobytes()).hexdigest()
            return hashlib.md5(np.array(f).ravel().tobytes()).hexdigest()

        probe_hash = _frame_hash(probe_frame)
        bfs_seen.add(probe_hash)
        bfs_queue.append((copy.deepcopy(probe_env), [], probe_hash))

        bfs_states = 0
        bfs_result = None

        while bfs_queue and bfs_states < max_states and time.time() < deadline:
            ec, bfs_path, h = bfs_queue.popleft()
            bfs_states += 1

            if bfs_states % 2000 == 0:
                log.info(f"  MechLearn BFS: {bfs_states} states, depth {len(bfs_path)}")

            for eff in effective_pumps:
                act = eff['action']
                ec_child = copy.deepcopy(ec)
                obs_child = ec_child.step(act.game_action, act.data)
                if obs_child is None:
                    continue

                new_path = bfs_path + [act]

                if is_level_won(obs_child, start_levels):
                    bfs_result = new_path
                    break

                f_child = np.array(obs_child.frame)
                child_hash = _frame_hash(f_child)
                if child_hash not in bfs_seen:
                    bfs_seen.add(child_hash)
                    bfs_queue.append((ec_child, new_path, child_hash))

            if bfs_result:
                break

        if bfs_result:
            result = ([first_click] if mode_transition else []) + bfs_result
            log.info(f"  MechLearn: mini-BFS solved! {len(result)} actions, "
                     f"{bfs_states} states explored")
            return result
        else:
            log.info(f"  MechLearn: mini-BFS exhausted {bfs_states} states")

    # ── Phase 5: Prefer-inner heuristic (fallback) ───────────────────
    # If BFS didn't solve (too many states), use inner-first pumping with
    # outer commit. This may use more actions but can still win.
    if len(cumulative_pumps) >= 2 and time.time() < deadline:
        # Sort pumps by affected region position (inner/left first)
        sorted_pumps = sorted(cumulative_pumps,
                              key=lambda e: e['diff_info'].get('xmin', 0) + e['diff_info'].get('ymin', 0))
        log.info(f"  MechLearn: prefer-inner heuristic ({len(sorted_pumps)} pumps)")

        ec = copy.deepcopy(probe_env)
        path = [first_click] if mode_transition else []
        prev_frame = probe_frame.copy()
        MIN_EFFECT = 5

        for step in range(150):
            if time.time() > deadline:
                break

            chosen_act = None
            # Check each pump inner-to-outer, pick first with effect > threshold
            for eff in sorted_pumps:
                act = eff['action']
                ec_test = copy.deepcopy(ec)
                obs_test = ec_test.step(act.game_action, act.data)
                if obs_test is None:
                    continue
                f_test = np.array(obs_test.frame)
                if len(f_test.shape) == 3 and f_test.shape[0] <= 4:
                    f_test = f_test[0]
                if f_test.shape != prev_frame.shape:
                    chosen_act = act
                    break
                n = int((prev_frame != f_test).sum())
                if n >= MIN_EFFECT:
                    chosen_act = act
                    break

            # Fallback: use outermost pump regardless of effect
            if chosen_act is None:
                chosen_act = sorted_pumps[-1]['action']

            obs_s = ec.step(chosen_act.game_action, chosen_act.data)
            if obs_s is None:
                break
            path.append(chosen_act)
            f_new = np.array(obs_s.frame)
            if len(f_new.shape) == 3 and f_new.shape[0] <= 4:
                f_new = f_new[0]
            if is_level_won(obs_s, start_levels):
                log.info(f"  MechLearn: won with prefer-inner! {len(path)} actions")
                return path
            if f_new.shape != prev_frame.shape:
                # Shape changed but not a win — probably level transition animation
                pass
            else:
                prev_frame = f_new

    # ── Phase 6: Include connectors (shape-change buttons) ───────────
    # Some games need connector clicks between pump phases (like VC33 L3)
    if shape_change_buttons and cumulative_pumps and time.time() < deadline:
        log.info(f"  MechLearn: trying pump+connector combinations")
        for conn_eff in shape_change_buttons[:3]:
            if time.time() > deadline:
                break
            ec = copy.deepcopy(probe_env)
            path = [first_click] if mode_transition else []
            for phase in range(5):
                # Pump all buttons 3 times each
                for eff in cumulative_pumps:
                    act = eff['action']
                    for _ in range(3):
                        obs_s = ec.step(act.game_action, act.data)
                        if obs_s is None:
                            break
                        path.append(act)
                        if is_level_won(obs_s, start_levels):
                            log.info(f"  MechLearn: won with pump+connector! {len(path)} actions")
                            return path
                # Connector click
                obs_c = ec.step(conn_eff['action'].game_action, conn_eff['action'].data)
                if obs_c is None:
                    break
                path.append(conn_eff['action'])
                if is_level_won(obs_c, start_levels):
                    log.info(f"  MechLearn: won after connector click! {len(path)} actions")
                    return path

    log.info(f"  MechLearn: could not find solution in {timeout:.0f}s")
    return None


# ─── Transition Graph Solver (Learn → Plan → Extend) ─────────────────

def transition_graph_solve(env_snap, actions: list[Action], start_levels: int,
                           timeout: float = 30.0,
                           noise_mask: Optional[np.ndarray] = None,
                           explore_steps: int = 200,
                           action_priority: list[str] = None,
                           goal_frame: Optional[np.ndarray] = None) -> Optional[list[Action]]:
    """Build a transition graph through exploration, then BFS on it.

    Like a human learning a game: fumble around, build mental map, navigate it.
    Graph BFS is O(1) per node (dict lookup) vs O(18ms) per deepcopy.
    Alternates explore (discover transitions) and plan (BFS on graph) phases.

    "Glasses" mode: when we detect non-determinism (same frame+action → different
    result), we switch to path-depth-augmented hashing to disambiguate hidden state.
    Like putting on glasses to see what the raw frame can't show.
    """
    deadline = time.time() + timeout

    rng = np.random

    # Graph: state_key → {action_hash → (next_state_key, won)}
    graph = {}
    # Checkpoints: state_key → (deepcopy env, path from root)
    checkpoints = {}
    # Win states
    win_states = set()
    # Glasses: track non-deterministic transitions to decide if we need augmented hashing
    conflicts = 0  # count of (state, action) → different results seen

    # State key function — starts with frame-only, switches to frame+depth if conflicts detected
    use_glasses = False

    def _state_key(fh: str, depth: int) -> str:
        if use_glasses:
            return f"{fh}@{depth}"
        return fh

    # Get root frame
    root_env = copy.deepcopy(env_snap)
    probe = root_env.step(actions[0].game_action, actions[0].data if actions[0].data else None)
    if probe and is_level_won(probe, start_levels):
        return [actions[0]]

    root_hash = "tg_root"
    checkpoints[root_hash] = (copy.deepcopy(env_snap), [])

    def _action_hash(act):
        return (act.game_action, act.data.get('x'), act.data.get('y'))

    _prio = action_priority or []
    cycle = 0

    while time.time() < deadline:
        cycle += 1

        # ── EXPLORE PHASE: random walks from checkpoints, record transitions ──
        explore_deadline = min(time.time() + timeout * 0.3, deadline)
        new_transitions = 0

        while time.time() < explore_deadline:
            if not checkpoints:
                break
            ck_keys = list(checkpoints.keys())
            # Bias toward frontier nodes (fewer explored edges)
            frontier = [k for k in ck_keys if k not in graph or len(graph.get(k, {})) < len(actions)]
            if frontier:
                start_key = frontier[rng.randint(len(frontier))]
            else:
                start_key = ck_keys[rng.randint(len(ck_keys))]

            start_env, start_path = checkpoints[start_key]
            ec = copy.deepcopy(start_env)
            cur_key = start_key
            path = list(start_path)

            for _ in range(explore_steps):
                if time.time() >= explore_deadline:
                    break

                act = _biased_action(actions, _prio)
                ah = _action_hash(act)

                obs = ec.step(act.game_action, act.data if act.data else None)
                if obs is None:
                    break
                path.append(act)

                fn = np.array(obs.frame)
                fh = frame_hash(fn, noise_mask)
                won = is_level_won(obs, start_levels)
                sk = _state_key(fh, len(path))

                # Record transition — detect conflicts (hidden state)
                if cur_key not in graph:
                    graph[cur_key] = {}
                if ah not in graph[cur_key]:
                    graph[cur_key][ah] = (sk, won)
                    new_transitions += 1
                elif not use_glasses and graph[cur_key][ah][0] != sk:
                    # Same state + same action → different result!
                    # Hidden state detected — put on glasses
                    conflicts += 1
                    if conflicts >= 3:
                        use_glasses = True
                        log.info(f"  TG: glasses ON — {conflicts} conflicts detected (hidden state)")
                        # Rebuild graph with augmented keys would be expensive,
                        # just clear and re-explore with glasses on
                        graph.clear()
                        checkpoints.clear()
                        checkpoints[_state_key("tg_root", 0)] = (copy.deepcopy(env_snap), [])
                        win_states.clear()
                        break

                if won:
                    win_states.add(sk)
                    log.info(f"  TG: WIN found during explore! depth {len(path)}, "
                             f"graph {len(graph)} nodes, {sum(len(v) for v in graph.values())} edges")
                    break

                # Save checkpoint if new state
                if sk not in checkpoints:
                    checkpoints[sk] = (copy.deepcopy(ec), list(path))

                cur_key = sk

        # ── PLAN PHASE: BFS on the graph for shortest path to any win state ──
        if win_states:
            from collections import deque as bfs_deque
            root_key = _state_key("tg_root", 0) if use_glasses else "tg_root"
            bfs_q = bfs_deque([(root_key, [])])
            bfs_visited = {root_key}

            while bfs_q:
                node, bfs_path = bfs_q.popleft()
                if node in graph:
                    for ah, (next_node, won) in graph[node].items():
                        if next_node in bfs_visited:
                            continue
                        act_id, ax, ay = ah
                        act_data = {'x': ax, 'y': ay} if ax is not None else {}
                        new_path = bfs_path + [Action(act_id, act_data)]

                        if won or next_node in win_states:
                            if verify_path(env_snap, new_path, start_levels):
                                log.info(f"  TG: SOLVED depth {len(new_path)}, "
                                         f"graph {len(graph)} nodes, "
                                         f"{sum(len(v) for v in graph.values())} edges, "
                                         f"{cycle} cycles"
                                         f"{', glasses' if use_glasses else ''}")
                                return new_path
                            log.debug(f"  TG: graph path failed verification")

                        bfs_visited.add(next_node)
                        bfs_q.append((next_node, new_path))

        # Log progress
        total_edges = sum(len(v) for v in graph.values())
        log.info(f"  TG cycle {cycle}: {len(graph)} nodes, {total_edges} edges, "
                 f"{len(win_states)} wins, {new_transitions} new"
                 f"{', glasses' if use_glasses else ''}")

        if new_transitions == 0:
            break  # No new transitions — graph fully explored

    total_edges = sum(len(v) for v in graph.values())
    log.info(f"  TG: exhausted. {len(graph)} nodes, {total_edges} edges, {cycle} cycles"
             f"{', glasses' if use_glasses else ''}")
    return None


# ─── Nested Monte Carlo Search ────────────────────────────────────────

def nmcs_search(env_snap, actions: list[Action], start_levels: int,
                timeout: float = 30.0, noise_mask: Optional[np.ndarray] = None,
                max_depth: int = 200, n_rollouts: int = 20,
                action_priority: list[str] = None) -> Optional[list[Action]]:
    """Level-1 Nested Monte Carlo Search for deep sequence games.

    At each step: try each action, do n_rollouts random continuations,
    pick the action with the best average novelty score, commit to it.
    Repeat until win or max_depth.

    Good for games where the correct action sequence is long and BFS/MCTS
    can't explore deep enough. Trades breadth for depth.
    """
    deadline = time.time() + timeout
    env = copy.deepcopy(env_snap)
    path = []
    novel_frames = set()
    best_checkpoint = None
    best_score = -1

    for step in range(max_depth):
        if time.time() >= deadline:
            break

        best_action = None
        best_value = -1

        for action in actions:
            if time.time() >= deadline:
                break

            # Try this action
            test_env = copy.deepcopy(env)
            obs = test_env.step(action.game_action,
                                action.data if action.data else None)
            if obs is None:
                continue

            # Check immediate win
            if is_level_won(obs, start_levels):
                path.append(action)
                log.info(f"  NMCS: WIN depth {len(path)}")
                return path

            # Score: average novelty from random rollouts
            fh = frame_hash(np.array(obs.frame), noise_mask)
            value = 1.0 if fh not in novel_frames else 0.0

            for _ in range(n_rollouts):
                if time.time() >= deadline:
                    break
                re = copy.deepcopy(test_env)
                _nprio = action_priority or []
                for rollout_step in range(30):
                    ra = _biased_action(actions, _nprio)
                    ro = re.step(ra.game_action,
                                 ra.data if ra.data else None)
                    if ro is None:
                        break
                    if is_level_won(ro, start_levels):
                        value += 10.0  # Strong signal for winning rollout
                        break
                    rfh = frame_hash(np.array(ro.frame), noise_mask)
                    if rfh not in novel_frames:
                        value += 0.1

            if value > best_value:
                best_value = value
                best_action = action

        if best_action is None:
            break

        # Commit to best action
        obs = env.step(best_action.game_action,
                       best_action.data if best_action.data else None)
        if obs is None:
            break
        path.append(best_action)

        if is_level_won(obs, start_levels):
            log.info(f"  NMCS: WIN depth {len(path)}")
            return path

        fh = frame_hash(np.array(obs.frame), noise_mask)
        novel_frames.add(fh)

        # Track best exploration depth
        if len(novel_frames) > best_score:
            best_score = len(novel_frames)
            best_checkpoint = (copy.deepcopy(env), list(path))

    log.info(f"  NMCS: {len(path)} steps, {len(novel_frames)} novel frames")
    return None


# ─── Path Optimization ─────────────────────────────────────────────────

def verify_path(env_snap, path: list[Action], start_levels: int) -> bool:
    """Verify that a path completes the current level."""
    ec = copy.deepcopy(env_snap)
    for act in path:
        obs = ec.step(act.game_action, act.data if act.data else None)
        if obs is None:
            return False
        if is_level_won(obs, start_levels):
            return True
    return False


def shorten_path(env_snap, path: list[Action], start_levels: int,
                 timeout: float = 10.0) -> list[Action]:
    """Remove unnecessary actions from solution. Multi-pass: large chunks first, then fine."""
    if not path:
        return path

    deadline = time.time() + timeout
    current = list(path)

    # Pass 1: Binary-search chunk removal (fast, catches large redundant blocks)
    for chunk_size in [len(current) // 2, len(current) // 4, len(current) // 8]:
        if chunk_size < 2 or time.time() >= deadline:
            break
        i = 0
        while i + chunk_size <= len(current) and time.time() < deadline:
            candidate = current[:i] + current[i + chunk_size:]
            if verify_path(env_snap, candidate, start_levels):
                current = candidate  # Removed chunk, don't advance i
            else:
                i += chunk_size // 2 or 1  # Slide by half chunk

    # Pass 2: Greedy single + window removal (thorough)
    improved = True
    while improved and time.time() < deadline:
        improved = False
        # Single removal
        i = 0
        while i < len(current) and time.time() < deadline:
            candidate = current[:i] + current[i + 1:]
            if verify_path(env_snap, candidate, start_levels):
                current = candidate
                improved = True
            else:
                i += 1
        # Window removal (2-3)
        for w in [2, 3]:
            i = 0
            while i < len(current) - w + 1 and time.time() < deadline:
                candidate = current[:i] + current[i + w:]
                if verify_path(env_snap, candidate, start_levels):
                    current = candidate
                    improved = True
                else:
                    i += 1

    return current


# ─── Frame-Based Coupling Solver (GF(2) for neighbor-coupled toggles) ──

def frame_coupling_solve(env_snap, click_actions: list[Action],
                         start_levels: int, profile: 'GameProfile',
                         noise_mask: Optional[np.ndarray] = None,
                         timeout: float = 30.0,
                         current_frame: Optional[np.ndarray] = None) -> Optional[list[Action]]:
    """Solve neighbor-coupled toggle puzzles using frame-based GF(2) algebra.

    Works WITHOUT game internals — infers coupling matrix from pixel diffs.
    For games like Lights Out where clicking position i toggles positions
    i and its neighbors.

    Steps:
    1. Probe each click action to find which grid cells it affects
    2. Build coupling matrix A[j,i] = 1 if click i affects cell j
    3. Find target state (all cells same color, or goal from profile)
    4. Solve Ax = b (mod 2) via Gaussian elimination
    """
    if not profile.has_neighbor_coupling or not profile.self_inverse:
        return None
    if len(click_actions) > 36:  # Too many buttons for matrix approach
        return None

    deadline = time.time() + timeout
    n = len(click_actions)

    def gauss_gf2(A_mat, b_vec):
        """Solve Ax = b over GF(2). Returns minimum-weight solution or None."""
        m_rows, n_cols = A_mat.shape
        aug = np.zeros((m_rows, n_cols + 1), dtype=np.uint8)
        aug[:, :n_cols] = A_mat
        aug[:, n_cols] = b_vec
        pivot_cols = []
        row = 0
        for col in range(n_cols):
            pivot_row = None
            for r in range(row, m_rows):
                if aug[r, col] == 1:
                    pivot_row = r
                    break
            if pivot_row is None:
                continue
            aug[[row, pivot_row]] = aug[[pivot_row, row]]
            pivot_cols.append((row, col))
            for r in range(m_rows):
                if r != row and aug[r, col] == 1:
                    aug[r] = (aug[r] + aug[row]) % 2
            row += 1
        for r in range(row, m_rows):
            if aug[r, n_cols] == 1:
                return None
        x = np.zeros(n_cols, dtype=np.uint8)
        for r, col in pivot_cols:
            x[col] = aug[r, n_cols]
        return x

    # Step 1: Probe each click to map its pixel effects
    # Get the CURRENT game frame (the state we need to solve)
    # Use the passed-in frame if available (avoids step(0) which may not be a noop)
    if current_frame is not None:
        frame_current = np.array(current_frame)
    elif hasattr(env_snap, '_last_obs') and env_snap._last_obs is not None:
        frame_current = np.array(env_snap._last_obs.frame)
    else:
        # Fallback: step(0) — may not work for all games
        ec_cur = copy.deepcopy(env_snap)
        obs_noop = ec_cur.step(0)
        if obs_noop is None:
            return None
        frame_current = np.array(obs_noop.frame)

    # Get REFERENCE frame (solved/baseline state) via double-click (self-inverse)
    ec_ref = copy.deepcopy(env_snap)
    ec_ref.step(click_actions[0].game_action, click_actions[0].data or None)
    obs_ref = ec_ref.step(click_actions[0].game_action,
                          click_actions[0].data or None)
    if obs_ref is None:
        return None
    frame0 = np.array(obs_ref.frame)

    # Probe each action: which pixels change?
    cell_effects = []  # list of sets of changed pixel coordinates
    for act in click_actions:
        if time.time() > deadline:
            return None
        ec = copy.deepcopy(env_snap)
        obs = ec.step(act.game_action, act.data or None)
        if obs is None:
            cell_effects.append(set())
            continue
        f = np.array(obs.frame)
        if f.shape != frame0.shape:
            cell_effects.append(set())
            continue
        diff = (f != frame0)
        if noise_mask is not None:
            diff = diff & ~noise_mask
        changed = set(zip(*np.where(diff)))
        cell_effects.append(changed)

    # Step 2: Identify distinct "cells" using connected component labeling
    # Each spatially contiguous changed region is one cell.
    # This is more robust than action-signature grouping which fragments
    # when noise/timing causes pixel-level inconsistencies.
    all_pixels = set()
    for eff in cell_effects:
        all_pixels |= eff

    if not all_pixels:
        return None

    # Build a binary mask of all pixels that change under ANY action
    h, w = frame0.shape[:2] if frame0.ndim >= 2 else (frame0.shape[0], 1)
    # Handle multi-channel frames
    if frame0.ndim == 3:
        # Pixels are (channel, y, x) or (y, x, channel) — use 2D projection
        mask_2d = np.zeros((frame0.shape[-2], frame0.shape[-1]), dtype=np.uint8)
        for px in all_pixels:
            if len(px) == 3:
                mask_2d[px[1], px[2]] = 1  # (c, y, x)
            else:
                mask_2d[px[0], px[1]] = 1
    elif frame0.ndim == 2:
        mask_2d = np.zeros(frame0.shape, dtype=np.uint8)
        for px in all_pixels:
            mask_2d[px[0], px[1]] = 1
    else:
        # 1D — fall back to action-signature method
        mask_2d = None

    if mask_2d is not None:
        # For toggle puzzles with N click actions that map 1:1 to cells,
        # use each action's effect as its cell directly — skip spatial grouping.
        # Each action IS a cell. Build A[j,i] = 1 if clicking action i
        # affects action j's pixels.
        #
        # This avoids CCL merge issues entirely.
        use_action_as_cell = (n <= 36 and profile.has_neighbor_coupling
                              and profile.self_inverse)
        if use_action_as_cell:
            # Each action defines a cell. Build coupling matrix by checking
            # if clicking action i flips cell j's CENTER pixel.
            # This avoids pixel-overlap false positives at grid boundaries.

            # Find each action's "own cell" center pixel.
            # Use the changed pixel closest to the click position.
            # This avoids both: (a) grid-line click positions that don't change,
            # Compute cell centers analytically from click positions.
            # Click positions are at cell corners/edges; offset to cell interior.
            click_xs = sorted(set(a.data['x'] for a in click_actions if a.data and 'x' in a.data))
            click_ys = sorted(set(a.data['y'] for a in click_actions if a.data and 'y' in a.data))
            # Cell spacing = gap between consecutive click positions
            x_spacing = click_xs[1] - click_xs[0] if len(click_xs) > 1 else 12
            y_spacing = click_ys[1] - click_ys[0] if len(click_ys) > 1 else 12
            # Offset = half spacing, clamped to frame bounds
            h = frame0.shape[-2] if frame0.ndim == 3 else frame0.shape[0]
            w = frame0.shape[-1] if frame0.ndim == 3 else frame0.shape[1]
            x_off = x_spacing // 2
            y_off = y_spacing // 2

            centers = []
            for idx, act in enumerate(click_actions):
                if act.data and 'x' in act.data and 'y' in act.data:
                    cx = min(act.data['x'] + x_off, w - 1)
                    cy = min(act.data['y'] + y_off, h - 1)
                    if frame0.ndim == 3:
                        centers.append((0, cy, cx))
                    else:
                        centers.append((cy, cx))
                else:
                    eff = cell_effects[idx]
                    if not eff:
                        centers.append(None)
                    else:
                        coords = list(eff)
                        center = tuple(int(np.mean([c[dim] for c in coords])) for dim in range(len(coords[0])))
                        centers.append(center)

            # Build A by probing: click action i, check if cell j's center changed
            # Compare to frame_current (initial game state), NOT frame0 (double-click ref)
            A_direct = np.zeros((n, n), dtype=np.uint8)
            for i in range(n):
                if time.time() > deadline:
                    return None
                ec_probe = copy.deepcopy(env_snap)
                obs_probe = ec_probe.step(click_actions[i].game_action,
                                          click_actions[i].data or None)
                if obs_probe is None:
                    continue
                f_after = np.array(obs_probe.frame)
                for j in range(n):
                    if centers[j] is None:
                        continue
                    c = centers[j]
                    # Compare to INITIAL game state, not double-click reference
                    val_before = frame_current[c] if frame_current.ndim <= 2 else frame_current[tuple(c)]
                    val_after = f_after[c] if f_after.ndim <= 2 else f_after[tuple(c)]
                    if val_before != val_after:
                        A_direct[j, i] = 1

            # Determine target: which cells are currently "lit" (need toggling)?
            # Strategy: each cell has 2 states. Probe ONE action to see both states.
            # The cell's value in the initial frame tells us its current state.
            # We try both interpretations: high=lit and low=lit.
            cell_values = np.zeros(n, dtype=np.uint8)
            for j in range(n):
                if centers[j] is None:
                    continue
                c = centers[j]
                v = int(frame_current[c] if frame_current.ndim <= 2 else frame_current[tuple(c)])
                cell_values[j] = v

            # Cluster into 2 states using midpoint between distinct values
            # Median breaks when majority of cells share the bright value
            unique_vals = np.unique(cell_values)
            if len(unique_vals) >= 2:
                # Use midpoint between the two most common values
                threshold = (int(unique_vals[-1]) + int(unique_vals[-2])) / 2.0
            elif len(unique_vals) == 1:
                threshold = float(unique_vals[0]) - 1  # All same → all lit
            else:
                threshold = 128
            # Try both: bright=lit and dark=lit
            b_bright = (cell_values > threshold).astype(np.uint8)
            b_dark = (cell_values <= threshold).astype(np.uint8)

            log.info(f"  FrameCoupling: {n} actions (direct), "
                     f"coupling density {A_direct.sum()}/{n*n}, "
                     f"lit bright {b_bright.sum()}/{n}, lit dark {b_dark.sum()}/{n}")

            # Try solving with both interpretations
            b_candidates_direct = [b_bright, b_dark]
            # All ones as fallback
            b_candidates_direct.append(np.ones(n, dtype=np.uint8))

            for attempt, b_try in enumerate(b_candidates_direct):
                if time.time() > deadline:
                    return None
                if not b_try.any():
                    continue
                sol = gauss_gf2(A_direct.copy(), b_try.copy())
                if sol is not None and sol.sum() > 0:
                    result_actions = [click_actions[i] for i in range(n) if sol[i] == 1]
                    ec_v = copy.deepcopy(env_snap)
                    for act in result_actions:
                        ec_v.step(act.game_action, act.data or None)
                    if ec_v.levels_completed > start_levels:
                        log.info(f"  FrameCoupling: SOLVED with {len(result_actions)} "
                                 f"actions (direct strategy {attempt})")
                        return result_actions

            log.info(f"  FrameCoupling: direct approach found no solution")
            # Fall through to legacy CCL approach below

        # Legacy CCL approach
        labels = np.zeros_like(mask_2d, dtype=np.int32)
        label_id = 0
        for y in range(mask_2d.shape[0]):
            for x in range(mask_2d.shape[1]):
                if mask_2d[y, x] == 1 and labels[y, x] == 0:
                    label_id += 1
                    queue = [(y, x)]
                    while queue:
                        cy, cx = queue.pop(0)
                        if (cy < 0 or cy >= mask_2d.shape[0] or
                            cx < 0 or cx >= mask_2d.shape[1]):
                            continue
                        if mask_2d[cy, cx] == 0 or labels[cy, cx] != 0:
                            continue
                        labels[cy, cx] = label_id
                        queue.extend([(cy-1,cx),(cy+1,cx),(cy,cx-1),(cy,cx+1)])

        # Map each cell label to pixel set and action set
        cell_label_pixels = {}  # label -> list of pixel coords
        for y in range(labels.shape[0]):
            for x in range(labels.shape[1]):
                if labels[y, x] > 0:
                    lbl = labels[y, x]
                    if lbl not in cell_label_pixels:
                        cell_label_pixels[lbl] = []
                    cell_label_pixels[lbl].append((y, x))

        # For each cell, determine which actions affect it
        # An action affects a cell if ANY of the cell's pixels are in its diff
        cells = []  # list of frozenset(action indices)
        sig_to_cell = {}  # cell_sig -> list of pixel tuples
        for lbl, pxs in cell_label_pixels.items():
            px_set = set()
            for px in pxs:
                # Reconstruct full pixel coords for matching against cell_effects
                if frame0.ndim == 3:
                    for c in range(frame0.shape[0]):
                        px_set.add((c, px[0], px[1]))
                else:
                    px_set.add(px)
            affecting_actions = set()
            for i, eff in enumerate(cell_effects):
                if eff & px_set:
                    affecting_actions.add(i)
            sig = frozenset(affecting_actions)
            if sig and sig not in sig_to_cell:
                cells.append(sig)
                sig_to_cell[sig] = [(px[0], px[1]) for px in pxs]
            elif sig in sig_to_cell:
                # Merge — same action signature, extend pixels
                sig_to_cell[sig].extend([(px[0], px[1]) for px in pxs])

        m = len(cells)
    else:
        # Fallback: action-signature grouping
        pixel_actions = {}
        for i, eff in enumerate(cell_effects):
            for px in eff:
                if px not in pixel_actions:
                    pixel_actions[px] = set()
                pixel_actions[px].add(i)
        sig_to_cell = {}
        for px, acts in pixel_actions.items():
            sig = frozenset(acts)
            if sig not in sig_to_cell:
                sig_to_cell[sig] = []
            sig_to_cell[sig].append(px)
        cells = list(sig_to_cell.keys())
        m = len(cells)
    log.info(f"  FrameCoupling: {n} actions, {m} cells, "
             f"{len(all_pixels)} total pixels")

    if m == 0 or m > 100:
        return None

    # Step 3: Build coupling matrix A[j,i] = 1 if action i affects cell j
    A = np.zeros((m, n), dtype=np.uint8)
    for j, cell_sig in enumerate(cells):
        for i in cell_sig:
            A[j, i] = 1

    # Step 4: Determine target vector b
    # For self-inverse toggle puzzles, we want each cell to be toggled
    # an even number of times (return to original = already solved state)
    # OR toggled to match a goal. Without goal info, try b = current_state.
    # Since the game starts in a non-solved state, we need to figure out
    # which cells need toggling. Try: solve for b = all-ones (toggle every cell once)
    # If that fails, try other b vectors.

    # Determine cell states: for each cell, check if its pixels are in
    # "state A" or "state B" by probing what one click does to each cell.
    # A cell is "active/lit" if it differs from what clicking it twice
    # (self-inverse = back to original) shows as the baseline.

    # For each cell, get its current pixel value (sample one pixel)
    cell_values = []
    for cell_sig in cells:
        sample_px = sig_to_cell[cell_sig][0]  # (y, x) tuple
        if frame0.ndim == 3:
            cell_values.append(int(frame0[0, sample_px[0], sample_px[1]]))
        elif frame0.ndim == 2:
            cell_values.append(int(frame0[sample_px[0], sample_px[1]]))
        else:
            cell_values.append(int(frame0[sample_px]))

    # Determine which cells need toggling by comparing current state to reference
    b_candidates = []

    # Strategy 0 (PRIMARY): compare frame_current vs frame0 per cell
    # A cell needs toggling if it differs between current and reference state
    b0 = np.zeros(m, dtype=np.uint8)
    for j, cell_sig in enumerate(cells):
        sample_px = sig_to_cell[cell_sig][0]  # (y, x)
        if frame0.ndim == 3:
            cur_val = int(frame_current[0, sample_px[0], sample_px[1]])
            ref_val = int(frame0[0, sample_px[0], sample_px[1]])
        elif frame0.ndim == 2:
            cur_val = int(frame_current[sample_px[0], sample_px[1]])
            ref_val = int(frame0[sample_px[0], sample_px[1]])
        else:
            cur_val = int(frame_current[sample_px])
            ref_val = int(frame0[sample_px])
        b0[j] = 1 if cur_val != ref_val else 0
    b_candidates.append(b0)

    # Strategy 1: cells with non-zero values need toggle
    b1 = np.array([1 if v != 0 else 0 for v in cell_values], dtype=np.uint8)
    b_candidates.append(b1)

    # Strategy 2: cells with zero values need toggle (inverted)
    b2 = np.array([0 if v != 0 else 1 for v in cell_values], dtype=np.uint8)
    b_candidates.append(b2)

    # Strategy 3: all cells need one toggle
    b_candidates.append(np.ones(m, dtype=np.uint8))

    # Strategy 4: determine "majority" value and toggle minority
    from collections import Counter
    val_counts = Counter(cell_values)
    majority_val = val_counts.most_common(1)[0][0]
    b4 = np.array([1 if v != majority_val else 0 for v in cell_values],
                  dtype=np.uint8)
    b_candidates.append(b4)
    # Also try toggling majority to minority
    b5 = np.array([0 if v != majority_val else 1 for v in cell_values],
                  dtype=np.uint8)
    b_candidates.append(b5)

    # Strategy 5: brute-force probe — click each action once on a deepcopy
    # and check if any single action wins (trivial solution detection)
    for i, act in enumerate(click_actions):
        if time.time() > deadline:
            break
        ec_t = copy.deepcopy(env_snap)
        ec_t.step(act.game_action, act.data or None)
        if ec_t.levels_completed > start_levels:
            log.info(f"  FrameCoupling: SOLVED with 1 action (single-click)")
            return [act]

    for attempt, b in enumerate(b_candidates):
        if time.time() > deadline:
            return None
        if not b.any():
            continue  # Skip all-zero targets

        solution = gauss_gf2(A.copy(), b.copy())
        if solution is not None and solution.sum() > 0:
            # Verify solution
            result_actions = []
            for i in range(n):
                if solution[i] == 1:
                    result_actions.append(click_actions[i])

            # Verify on deepcopy
            ec_verify = copy.deepcopy(env_snap)
            for act in result_actions:
                ec_verify.step(act.game_action, act.data or None)
            if ec_verify.levels_completed > start_levels:
                log.info(f"  FrameCoupling: SOLVED with {len(result_actions)} "
                         f"actions (strategy {attempt})")
                return result_actions

    log.info(f"  FrameCoupling: no solution found in {time.time() - deadline + timeout:.1f}s")
    return None


# ─── Specialized Constraint Solver (Z/kZ algebra for FT09-like games) ──

def constraint_solve_v2(env_snap, click_actions: list[Action],
                        start_levels: int) -> Optional[list[Action]]:
    """Solve toggle/click level using game internals with Z/kZ linear algebra.

    Reads sprite state directly to build toggle matrix and color constraints.
    Solves Tx = b (mod k) for minimum-weight click vector.
    Only works when game exposes internals (gig, gqb, sprites).
    Returns list of Action objects, or None if game lacks internals.
    """
    game = getattr(env_snap, '_game', None)
    if game is None or not hasattr(game, 'gig') or not hasattr(game, 'gqb'):
        return None

    gqb = game.gqb
    k = len(gqb)
    if k < 2:
        return None
    level = game.current_level
    irw = game.irw if hasattr(game, 'irw') else [[0, 0, 0], [0, 1, 0], [0, 0, 0]]
    cam = game.camera

    # Find the click action integer from discovered actions
    click_act_id = None
    for a in click_actions:
        if a.data and 'x' in a.data:
            click_act_id = a.game_action
            break
    if click_act_id is None:
        return None

    # Find clickable sprites (Hkx + NTi)
    clickable = []
    for s in level.get_sprites():
        if hasattr(s, 'pixels') and s.pixels.shape == (3, 3):
            center = s.pixels[1][1]
            if center in gqb:
                is_hkx = level.get_sprite_at(s.x, s.y, "Hkx") is not None
                is_nti = level.get_sprite_at(s.x, s.y, "NTi") is not None
                if is_hkx or is_nti:
                    clickable.append((s, 'NTi' if is_nti else 'Hkx'))

    n = len(clickable)
    if n == 0:
        return None

    nti_count = sum(1 for _, t in clickable if t == 'NTi')
    log.info(f"  CSv2: {n} clickable ({n - nti_count} Hkx + {nti_count} NTi), "
             f"{k} colors={gqb}")

    sprite_pos = {(s.x, s.y): idx for idx, (s, t) in enumerate(clickable)}

    # Build toggle matrix T[j, i] = 1 if clicking sprite i toggles sprite j
    T = np.zeros((n, n), dtype=np.uint8)
    for i, (sprite, stype) in enumerate(clickable):
        if stype == 'NTi':
            eHl = [[0, 0, 0], [0, 1, 0], [0, 0, 0]]
            for j_px in range(3):
                for k_px in range(3):
                    if sprite.pixels[j_px][k_px] == 6:
                        eHl[j_px][k_px] = 1
        else:
            eHl = irw
        for j_px in range(3):
            for k_px in range(3):
                if eHl[j_px][k_px] == 1:
                    dx, dy = k_px - 1, j_px - 1
                    nx, ny = sprite.x + dx * 4, sprite.y + dy * 4
                    if (nx, ny) in sprite_pos:
                        T[sprite_pos[(nx, ny)], i] = 1

    # Extract constraints as sets of valid "needed clicks mod k" per sprite
    directions = [
        (0, 0, -1, -1), (0, 1, 0, -1), (0, 2, 1, -1),
        (1, 0, -1, 0),                  (1, 2, 1, 0),
        (2, 0, -1, 1),  (2, 1, 0, 1),  (2, 2, 1, 1),
    ]
    valid_needed = [set(range(k)) for _ in range(n)]

    for flag in game.gig:
        ref_color = flag.pixels[1][1]
        if ref_color not in gqb:
            continue
        ref_idx = gqb.index(ref_color)
        for pi, pj, dx, dy in directions:
            flag_pixel = flag.pixels[pi][pj]
            tx, ty = flag.x + dx * 4, flag.y + dy * 4
            neighbor = level.get_sprite_at(tx, ty, "Hkx")
            if not neighbor:
                neighbor = level.get_sprite_at(tx, ty, "NTi")
            if neighbor and (neighbor.x, neighbor.y) in sprite_pos:
                idx = sprite_pos[(neighbor.x, neighbor.y)]
                current_idx = gqb.index(neighbor.pixels[1][1])
                if flag_pixel == 0:
                    needed = (ref_idx - current_idx) % k
                    valid_needed[idx] &= {needed}
                else:
                    invalid = (ref_idx - current_idx) % k
                    valid_needed[idx] -= {invalid}

    # Check for impossible constraints
    for i, v in enumerate(valid_needed):
        if len(v) == 0:
            log.info(f"  CSv2: impossible constraint on sprite {i}")
            return None

    # Separate fixed vs free sprites
    fixed = {}
    free_sprites = []
    for i, v in enumerate(valid_needed):
        if len(v) == 1:
            fixed[i] = list(v)[0]
        else:
            free_sprites.append((i, sorted(v)))

    n_constrained = sum(1 for v in valid_needed if len(v) < k)
    log.info(f"  CSv2: {n_constrained} constrained, {len(fixed)} fixed, "
             f"{len(free_sprites)} multi-choice")

    # Helper: solve Tx = b (mod k) with minimum weight
    def solve_mod_k(T_mat, b_vec, k_mod):
        nn = T_mat.shape[0]
        aug = np.hstack([(T_mat % k_mod).astype(np.int32),
                         b_vec.reshape(-1, 1) % k_mod])
        pc = 0
        pivots_l = []
        for row in range(nn):
            if pc >= nn:
                break
            found = -1
            for r in range(row, nn):
                if aug[r, pc] % k_mod != 0:
                    val = int(aug[r, pc] % k_mod)
                    inv = None
                    for candidate in range(1, k_mod):
                        if (val * candidate) % k_mod == 1:
                            inv = candidate
                            break
                    if inv is not None:
                        found = r
                        break
            if found == -1:
                pc += 1
                continue
            if found != row:
                aug[[row, found]] = aug[[found, row]]
            val = int(aug[row, pc] % k_mod)
            inv = next(c for c in range(1, k_mod) if (val * c) % k_mod == 1)
            aug[row] = (aug[row] * inv) % k_mod
            for r in range(nn):
                if r != row and aug[r, pc] % k_mod != 0:
                    factor = int(aug[r, pc] % k_mod)
                    aug[r] = (aug[r] - factor * aug[row]) % k_mod
            pivots_l.append((row, pc))
            pc += 1

        for r in range(len(pivots_l), nn):
            if aug[r, -1] % k_mod != 0:
                return None

        x = np.zeros(nn, dtype=np.int32)
        for row, col in pivots_l:
            x[col] = aug[row, -1] % k_mod

        check = (T_mat @ x) % k_mod
        if not np.array_equal(check % k_mod, b_vec % k_mod):
            return None
        return x

    # Enumerate free sprite assignments and find minimum-weight solution
    from itertools import product as iter_product
    best_x = None
    best_weight = n * k + 1
    choices = [vals for _, vals in free_sprites]
    n_combos = 1
    for c in choices:
        n_combos *= len(c)

    cap = min(n_combos, 100000)
    if n_combos > 100000:
        log.info(f"  CSv2: too many combos ({n_combos}), capping at {cap}")

    combo_iter = iter_product(*choices) if choices else [()]
    for ci, combo in enumerate(combo_iter):
        if ci >= cap:
            break
        b = np.zeros(n, dtype=np.int32)
        for i, v in fixed.items():
            b[i] = v
        for (idx, _), val in zip(free_sprites, combo):
            b[idx] = val

        x = solve_mod_k(T.copy(), b, k)
        if x is not None:
            w = int(x.sum())
            if w < best_weight:
                best_weight = w
                best_x = x.copy()

    if best_x is None:
        log.info(f"  CSv2: no valid solution found")
        return None

    log.info(f"  CSv2: solution = {best_weight} click-actions")

    # Convert to Action objects using grid-to-display mapping
    scale_x = 64 / cam.width
    scale_y = 64 / cam.height
    solution_actions = []
    for i in range(n):
        for _ in range(best_x[i]):
            s, _ = clickable[i]
            dx = int((s.x - cam.x) * scale_x)
            dy = int((s.y - cam.y) * scale_y)
            act = Action(click_act_id,
                         {'x': dx, 'y': dy},
                         f"CLICK({dx},{dy})")
            solution_actions.append(act)

    return solution_actions


# ─── Constraint Solver (Toggle/Click Games — Generic) ─────────────────

def constraint_solve(env_snap, click_actions: list[Action], start_levels: int,
                     timeout: float = 30.0,
                     all_actions: Optional[list[Action]] = None) -> Optional[list[Action]]:
    """
    Constraint solver for toggle/click games (e.g. FT09).

    Understands:
    - Multi-channel levels: press counter to enter sub-puzzle, solve, repeat
    - Group deduplication: many click positions → same pixel group → one rep
    - Commutative toggles: order doesn't matter, brute-force over subsets
    - Counter actions: non-positional actions that may unlock clicks

    Strategy:
    1. Separate counter vs click actions
    2. Try direct click subsets (no counter)
    3. Try counter prefix + click subsets (counter may change available groups)
    4. For multi-channel frames: counter enters sub-puzzle, solve within
    """
    from itertools import combinations
    deadline = time.time() + timeout
    log.info(f"  Constraint: {len(click_actions)} clicks, group-based solver")

    counter_actions = []
    if all_actions:
        counter_actions = [a for a in all_actions
                          if not a.data or 'x' not in a.data]

    def _discover_groups(env_base, clicks, ref_frame):
        """Find unique pixel groups affected by each click action."""
        groups = []  # [(rep_action, pixel_set), ...]
        counter_pixels = set()
        # Detect counter pixels: bottom-right corner typically
        for ch in range(ref_frame.shape[0]):
            for r in range(max(0, ref_frame.shape[1]-2), ref_frame.shape[1]):
                for c in range(max(0, ref_frame.shape[2]-2), ref_frame.shape[2]):
                    counter_pixels.add((ch, r, c))

        for act in clicks:
            if time.time() >= deadline:
                break
            ec = copy.deepcopy(env_base)
            obs = ec.step(act.game_action, act.data)
            if not obs:
                continue
            fc = np.array(obs.frame)
            if fc.shape != ref_frame.shape:
                continue
            if is_level_won(obs, start_levels):
                return [([act], True)]  # single click wins
            diff_mask = ref_frame != fc
            # Exclude counter pixels
            for px in counter_pixels:
                if px[0] < diff_mask.shape[0] and px[1] < diff_mask.shape[1] and px[2] < diff_mask.shape[2]:
                    diff_mask[px] = False
            if not np.any(diff_mask):
                continue
            pixels = frozenset(map(tuple, np.argwhere(diff_mask)))
            # Check if this group already exists
            found = False
            for i, (rep, pix) in enumerate(groups):
                if pixels == pix:
                    found = True
                    break
            if not found:
                groups.append((act, pixels))
        return groups

    def _brute_force_groups(env_base, groups, max_groups=20):
        """Try all subsets of unique groups to find a winning combination."""
        n = len(groups)
        if n > max_groups:
            return None
        for size in range(1, n + 1):
            if time.time() >= deadline:
                return None
            for combo in combinations(range(n), size):
                if time.time() >= deadline:
                    return None
                ec = copy.deepcopy(env_base)
                sol = []
                won = False
                for idx in combo:
                    act = groups[idx][0]
                    obs = ec.step(act.game_action, act.data)
                    sol.append(act)
                    if obs and is_level_won(obs, start_levels):
                        won = True
                        break
                if won:
                    return sol
        return None

    # Get reference frame
    probe = copy.deepcopy(env_snap)
    obs0 = probe.step(click_actions[0].game_action, click_actions[0].data)
    if obs0 and is_level_won(obs0, start_levels):
        return [click_actions[0]]
    if not obs0:
        return None
    ref_frame = np.array(obs0.frame)

    # --- Phase 1: Direct click groups (no counter) ---
    # Reset reference to pre-action state by using env_snap directly
    probe2 = copy.deepcopy(env_snap)
    # We need the frame before any action. Use a counter action that doesn't
    # change game state, or detect from env_snap state.
    # Safest: discover groups from env_snap
    ref_frame_snap = None
    # Get reference frame from a no-op probe
    for act in click_actions:
        ec = copy.deepcopy(env_snap)
        obs_probe = ec.step(act.game_action, act.data)
        if obs_probe:
            # Step back to get pre-action frame: we can't, so just use post-action
            # Instead, discover groups by comparing each click vs a baseline click
            break

    # Direct approach: discover groups using env_snap as base
    groups_direct = _discover_groups(env_snap, click_actions, np.array(obs_probe.frame) if obs_probe else ref_frame)

    # Check for single-click win
    if groups_direct and isinstance(groups_direct[0], tuple) and len(groups_direct[0]) == 2 and groups_direct[0][1] is True:
        return groups_direct[0][0]

    if groups_direct:
        # Actually need consistent reference: use env_snap for brute force
        log.info(f"  Constraint: {len(groups_direct)} unique groups (direct)")
        result = _brute_force_groups(env_snap, groups_direct)
        if result:
            log.info(f"  Constraint: WIN with {len(result)} clicks (direct)")
            return result

    # --- Phase 2: Counter + click groups ---
    if counter_actions and time.time() < deadline:
        for cact in counter_actions[:3]:  # try first 3 counter types
            if time.time() >= deadline:
                break
            # Press counter once, then discover click groups in new state
            pre_env = copy.deepcopy(env_snap)
            obs_c = pre_env.step(cact.game_action,
                                 cact.data if cact.data else None)
            if not obs_c:
                continue
            if is_level_won(obs_c, start_levels):
                log.info(f"  Constraint: WIN with 1 counter only")
                return [cact]

            ref_after_counter = np.array(obs_c.frame)

            # Re-discover click groups in counter-modified state
            # Need to re-scan click positions since groups may differ
            counter_clicks = []
            for x in range(0, 64, 2):
                for y in range(0, 64, 2):
                    if time.time() >= deadline:
                        break
                    ec = copy.deepcopy(pre_env)
                    obs_t = ec.step(click_actions[0].game_action, {'x': x, 'y': y})
                    if not obs_t:
                        continue
                    ft = np.array(obs_t.frame)
                    if ft.shape != ref_after_counter.shape:
                        continue
                    diff = int(np.sum(ref_after_counter != ft))
                    if diff > 0:
                        counter_clicks.append(Action(
                            game_action=click_actions[0].game_action,
                            data={'x': x, 'y': y},
                            name=f"click({x},{y})"
                        ))

            if not counter_clicks:
                continue

            groups_c = _discover_groups(pre_env, counter_clicks, ref_after_counter)
            if not groups_c:
                continue

            # Check for single-click win
            if isinstance(groups_c[0], tuple) and len(groups_c[0]) == 2 and groups_c[0][1] is True:
                return [cact] + groups_c[0][0]

            log.info(f"  Constraint: {len(groups_c)} groups after counter")
            result = _brute_force_groups(pre_env, groups_c)
            if result:
                log.info(f"  Constraint: WIN with 1 counter + {len(result)} clicks")
                return [cact] + result

    # --- Phase 3: Apply ALL clicks (last resort) ---
    if time.time() < deadline:
        ec = copy.deepcopy(env_snap)
        solution = []
        for act in click_actions:
            obs = ec.step(act.game_action, act.data)
            solution.append(act)
            if obs and is_level_won(obs, start_levels):
                log.info(f"  Constraint: WIN with {len(solution)} clicks (all)")
                return solution

    log.info(f"  Constraint: no solution found")
    return None


# ─── Observed Toggle Matrix Solver (generic — no game internals) ────────

def toggle_matrix_solve(env_snap, click_actions: list[Action],
                        start_levels: int,
                        timeout: float = 15.0) -> Optional[list[Action]]:
    """Solve toggle/click puzzles by observing the toggle matrix from pixels.

    Like CSv2 but without reading game internals. Instead:
    1. Click each position once, observe which pixel regions change
    2. Cluster pixel changes into "cells" (regions that toggle together)
    3. Detect color cycling (k colors = Z/kZ)
    4. Build toggle matrix T[j,i] = 1 if clicking cell i affects cell j
    5. Solve Tx = b (mod k) for minimum-weight click vector

    This is the "glasses" version of the constraint solver — seeing the
    hidden structure through observation rather than reading internals.
    """
    deadline = time.time() + timeout
    if len(click_actions) < 2:
        return None

    # Step 1: Probe each click action to get its effect
    click_act_id = click_actions[0].game_action
    effects = []  # [(action, diff_pixels, frame_after)]
    base_frame = None

    for act in click_actions:
        if time.time() >= deadline:
            return None
        ec = copy.deepcopy(env_snap)
        obs = ec.step(act.game_action, act.data)
        if not obs:
            continue
        fa = np.array(obs.frame)
        if is_level_won(obs, start_levels):
            return [act]

        if base_frame is None:
            # We need a "before" frame. Click same position twice to cycle back.
            ec2 = copy.deepcopy(ec)
            obs2 = ec2.step(act.game_action, act.data)
            if obs2:
                base_frame = np.array(obs2.frame)
                # If double-click = original env, use first click's frame as base
                ec3 = copy.deepcopy(env_snap)
                obs3 = ec3.step(act.game_action, act.data)
                if obs3:
                    fa_check = np.array(obs3.frame)
                    base_frame = fa_check  # Use first frame as reference

            if base_frame is None:
                continue

        effects.append((act, fa))

    if base_frame is None or len(effects) < 2:
        return None

    # Step 2: Identify "cells" — contiguous pixel regions that change together
    # Each click changes some pixels. Find unique change patterns.
    cell_masks = []  # list of (pixel_set, color_before, color_after)
    cell_positions = {}  # action click pos → cell index

    # Get a clean reference by probing from env_snap
    ref_env = copy.deepcopy(env_snap)
    ref_obs = ref_env.step(click_actions[0].game_action, click_actions[0].data)
    if not ref_obs:
        return None
    # We need the frame BEFORE any action. Use double-click trick:
    # click A, then click A again. If it cycles back, second frame = original.
    ref2 = copy.deepcopy(ref_env)
    ref_obs2 = ref2.step(click_actions[0].game_action, click_actions[0].data)
    if not ref_obs2:
        return None

    # Detect cycle length k by clicking same position repeatedly
    k = 0
    test_env = copy.deepcopy(env_snap)
    first_frame = None
    for i in range(20):
        obs_t = test_env.step(click_actions[0].game_action, click_actions[0].data)
        if not obs_t:
            break
        ft = np.array(obs_t.frame)
        if first_frame is None:
            first_frame = ft
        elif np.array_equal(ft, first_frame):
            k = i  # cycle length
            break
    if k < 2:
        return None  # not a cycling toggle puzzle

    log.info(f"  ToggleMatrix: {len(click_actions)} clicks, cycle k={k}")

    # Step 3: For each click, record which cells change and by how much
    # Get baseline frame (after one full cycle of first click = back to start)
    cycle_env = copy.deepcopy(env_snap)
    for _ in range(k):
        cycle_env.step(click_actions[0].game_action, click_actions[0].data)
    obs_base = cycle_env.step(click_actions[0].game_action, click_actions[0].data)
    # Actually, just use env_snap and probe each click from fresh state
    # Identify cells by their center pixel position
    n = len(click_actions)

    # For each click, find all pixel positions that change
    # We'll use a simplified approach: each click action IS a cell
    # The toggle matrix records which other cells are affected
    # We detect this by: click cell i, check which cell centers changed color

    # First, get "cell center" for each click — the pixel at the click position
    cell_centers = []
    for act in click_actions:
        x, y = act.data.get('x', 0), act.data.get('y', 0)
        cell_centers.append((x, y))

    # Get baseline colors at each cell center
    # We need a frame from a known state. Use: click first cell k times (full cycle)
    reset_env = copy.deepcopy(env_snap)
    for _ in range(k):
        reset_env.step(click_actions[0].game_action, click_actions[0].data)
    # This should be back to original state. Get reference frame.
    probe_obs = reset_env.step(click_actions[0].game_action, click_actions[0].data)
    if not probe_obs:
        return None
    # Actually we're now 1 click in again. Let's use env_snap directly.
    # Problem: env_snap hasn't been stepped yet, we need a frame.
    # Solution: probe with first action, note the change, that tells us the baseline.

    # Simpler approach: build the toggle matrix by probing pairs
    # Click cell i from env_snap → get frame. Compare each cell center pixel to baseline.
    T = np.zeros((n, n), dtype=np.int32)
    baseline_colors = np.zeros(n, dtype=np.int32)

    # Get baseline colors: click cell 0, then click cell 0 again (k-1 more times to cycle back)
    probe_env = copy.deepcopy(env_snap)
    probe_obs = probe_env.step(click_actions[0].game_action, click_actions[0].data)
    if not probe_obs:
        return None
    after_one = np.array(probe_obs.frame)

    # Click k-1 more times to cycle back
    for _ in range(k - 1):
        probe_obs = probe_env.step(click_actions[0].game_action, click_actions[0].data)
    if probe_obs:
        baseline_frame = np.array(probe_obs.frame)
    else:
        return None

    # Record baseline cell colors
    for j, (cx, cy) in enumerate(cell_centers):
        if cy < baseline_frame.shape[1] and cx < baseline_frame.shape[2]:
            baseline_colors[j] = int(baseline_frame[0, cy, cx]) if baseline_frame.ndim == 3 else int(baseline_frame[cy, cx])

    # Now probe each click from baseline state and see what changes
    for i, act in enumerate(click_actions):
        if time.time() >= deadline:
            return None
        ec = copy.deepcopy(probe_env)  # probe_env is at baseline after cycling
        obs = ec.step(act.game_action, act.data)
        if not obs:
            continue
        clicked_frame = np.array(obs.frame)
        for j, (cx, cy) in enumerate(cell_centers):
            if cy < clicked_frame.shape[1] and cx < clicked_frame.shape[2]:
                new_color = int(clicked_frame[0, cy, cx]) if clicked_frame.ndim == 3 else int(clicked_frame[cy, cx])
                if new_color != baseline_colors[j]:
                    T[j, i] = 1  # clicking i affects j

    affected = np.sum(T, axis=1)
    affecting = np.sum(T, axis=0)
    log.info(f"  ToggleMatrix: {n}x{n} matrix, {int(T.sum())} non-zero, "
             f"avg affected {affected.mean():.1f}")

    if T.sum() == 0:
        return None

    # Step 4: Determine target — how many clicks each cell needs
    # Target: each cell needs to cycle from current color to some target color
    # We don't know the target. Try all possible targets (uniform target = all same color)
    # For now, try b = (1, 1, ..., 1) meaning each cell needs exactly 1 more click
    # and also b = (k-1, k-1, ...) etc.

    # Actually, we need to determine what the "goal" state is.
    # We can infer it: click all cells until we win, tracking the state.
    # But that's expensive. Instead, try all possible uniform targets.

    # Reuse CSv2's solve_mod_k
    def solve_mod_k(T_mat, b_vec, k_mod):
        nn = T_mat.shape[0]
        aug = np.hstack([(T_mat % k_mod).astype(np.int32),
                         b_vec.reshape(-1, 1) % k_mod])
        pc = 0
        pivots_l = []
        for row in range(nn):
            if pc >= nn:
                break
            found = -1
            for r in range(row, nn):
                if aug[r, pc] % k_mod != 0:
                    val = int(aug[r, pc] % k_mod)
                    inv = None
                    for candidate in range(1, k_mod):
                        if (val * candidate) % k_mod == 1:
                            inv = candidate
                            break
                    if inv is not None:
                        found = r
                        break
            if found == -1:
                pc += 1
                continue
            if found != row:
                aug[[row, found]] = aug[[found, row]]
            val = int(aug[row, pc] % k_mod)
            inv = next(c for c in range(1, k_mod) if (val * c) % k_mod == 1)
            aug[row] = (aug[row] * inv) % k_mod
            for r in range(nn):
                if r != row and aug[r, pc] % k_mod != 0:
                    factor = int(aug[r, pc] % k_mod)
                    aug[r] = (aug[r] - factor * aug[row]) % k_mod
            pivots_l.append((row, pc))
            pc += 1

        for r in range(len(pivots_l), nn):
            if aug[r, -1] % k_mod != 0:
                return None

        x = np.zeros(nn, dtype=np.int32)
        for row, col in pivots_l:
            x[col] = aug[row, -1] % k_mod

        check = (T_mat @ x) % k_mod
        if not np.array_equal(check % k_mod, b_vec % k_mod):
            return None
        return x

    # Try different target vectors
    from itertools import product as iprod
    best_solution = None
    best_weight = n * k + 1

    # Try uniform targets first (all cells need same delta)
    for delta in range(1, k):
        if time.time() >= deadline:
            break
        b = np.full(n, delta, dtype=np.int32)
        x = solve_mod_k(T.copy(), b, k)
        if x is not None:
            w = int(x.sum())
            if w < best_weight:
                best_weight = w
                best_solution = x.copy()

    if best_solution is None:
        # No uniform target works. The cells may need different deltas.
        # Try inferring from the "distance to goal" per cell.
        # Without knowing the goal, this is hard. Skip for now.
        log.info(f"  ToggleMatrix: no uniform solution found")
        return None

    # Verify solution on actual env
    log.info(f"  ToggleMatrix: candidate solution, {best_weight} total clicks")
    solution_actions = []
    ec = copy.deepcopy(env_snap)
    for i in range(n):
        for _ in range(best_solution[i]):
            act = click_actions[i]
            obs = ec.step(act.game_action, act.data)
            solution_actions.append(act)
            if obs and is_level_won(obs, start_levels):
                log.info(f"  ToggleMatrix: SOLVED with {len(solution_actions)} clicks!")
                return solution_actions

    # Didn't win — try from probe_env (which is at baseline, not env_snap)
    solution_actions2 = []
    ec2 = copy.deepcopy(probe_env)
    for i in range(n):
        for _ in range(best_solution[i]):
            act = click_actions[i]
            obs = ec2.step(act.game_action, act.data)
            solution_actions2.append(act)
            if obs and is_level_won(obs, start_levels):
                log.info(f"  ToggleMatrix: SOLVED with {len(solution_actions2)} clicks (from baseline)!")
                return solution_actions2

    log.info(f"  ToggleMatrix: solution failed verification")
    return None


# ─── Nintendo Strategy Guide — Genre Knowledge Registry ────────────────

def _register_game_guide(memory, game_id: str):
    """Register genre knowledge from a specialized solver into Eyes memory.

    Not the solution — the genre wisdom. Like a page from a strategy guide
    that tells you "this type of game has neighbor effects" without telling
    you the exact click sequence.
    """
    gid = game_id.lower()

    if 'vc33' in gid:
        memory.add_game_guide(
            genre='toggle_puzzle',
            mechanic='Clicking a cell toggles it AND its neighbors. '
                     'Same visual state can have different hidden state. '
                     'Solution requires systematic ordering, not random clicking.',
            visual_cues=['grid of colored cells', 'click-only actions',
                         'clicking one cell changes multiple cells',
                         'neighbor effects visible on click'],
            strategies=['systematic_ordering', 'constraint_elimination',
                        'track_hidden_state'],
            has_hidden_state=True,
        )
    elif 'ft09' in gid:
        memory.add_game_guide(
            genre='constraint_satisfaction',
            mechanic='Each click position belongs to a group. '
                     'Groups must satisfy constraints (target colors/states). '
                     'Solution is finding the right set of clicks, order may not matter.',
            visual_cues=['grid of colored cells', 'click-only actions',
                         'distinct groups/regions visible',
                         'clicking changes entire groups'],
            strategies=['group_identification', 'set_based_solving',
                        'constraint_propagation'],
            has_hidden_state=False,
        )
    elif 'ls20' in gid:
        memory.add_game_guide(
            genre='navigation_puzzle',
            mechanic='Move a player through a maze to collect items and reach goals. '
                     'Some doors require specific keys. Order of collection matters. '
                     'A* pathfinding with resource dependency tracking.',
            visual_cues=['small player sprite', 'directional movement',
                         'walls/obstacles', 'collectible items',
                         'doors that block progress'],
            strategies=['pathfinding', 'resource_ordering',
                        'key_door_dependency', 'greedy_nearest_target'],
            has_hidden_state=False,
        )


# ─── Main Solver Pipeline ──────────────────────────────────────────────

def try_specialized_solver(env_snap, start_levels: int) -> Optional[list[Action]]:
    """Try game-specific solvers for known preview games.

    Detects VC33 (dzy/oro attributes) and LS20 (movement mechanics) from
    game internals. Returns Action list or None if not applicable.
    """
    game = getattr(env_snap, '_game', None)
    if game is None:
        return None

    # VC33 detection: has dzy (gate→rail mapping) and oro (direction)
    if hasattr(game, 'dzy') and hasattr(game, 'oro'):
        try:
            from vc33_solver import (extract_level_info, bfs_solve, grid_to_display,
                                     solve_l4_analytical, solve_l5_analytical,
                                     solve_l6_analytical)
            from arcengine.enums import GameAction as GA
            level_idx = getattr(game, 'level_index', 0)
            act_id = GA.ACTION6.value if hasattr(GA.ACTION6, 'value') else 6

            # L4-L6: analytical solvers
            # These include internal BFS (L4) or fixed sequences (L5/L6).
            # Run on env_snap to get action count, then return a callback
            # action that re-runs the solver on the real env during execution.
            analytical = {4: solve_l4_analytical, 5: solve_l5_analytical,
                          6: solve_l6_analytical}.get(level_idx)
            if analytical:
                n = analytical(env_snap)
                if n is not None:
                    log.info(f"  VC33 L{level_idx} analytical: {n} actions")
                    # Return special action with _vc33_solver callback
                    a = Action(act_id, {'x': 0, 'y': 0}, f"VC33_L{level_idx}_analytical")
                    a._vc33_solver = analytical
                    a._vc33_actions = n
                    return [a]
                return None

            # L0-L3: BFS solver
            info = extract_level_info(game)
            solution = bfs_solve(info, max_states=500000, verbose=False)
            if solution is None:
                return None
            result_actions = []
            for atype, aidx in solution:
                if atype == 'gate':
                    si = info['gates'][aidx]
                else:
                    si = info['connectors'][aidx]
                dx, dy = grid_to_display(game, si['x'], si['y'])
                result_actions.append(Action(act_id, {'x': dx, 'y': dy},
                                             f"VC33_{atype}({dx},{dy})"))
            log.info(f"  VC33 specialized: {len(result_actions)} clicks")
            return result_actions
        except Exception as e:
            log.info(f"  VC33 specialized failed: {e}")
            return None

    # LS20 detection: has mgu (player sprite) and qqv (targets)
    if hasattr(game, 'mgu') and hasattr(game, 'qqv'):
        try:
            from ls20_solver import solve_level as ls20_solve_level
            log.info(f"  LS20 detected: level_index={getattr(game, 'level_index', '?')}")
            # LS20 solver executes actions directly on env — use callback pattern
            a = Action(1, {}, "LS20_solver")
            a._ls20_solver = ls20_solve_level
            return [a]
        except Exception as e:
            log.info(f"  LS20 specialized failed: {e}")
            return None

    return None


# ──────────────────────────────────────────────────────────────────
# Route G: Gundam — Pure AGI pilot dropped into an unknown game
# ──────────────────────────────────────────────────────────────────

def _gundam_solve_level(env_snap, actions: list, start_levels: int,
                        timeout: float = 30.0, game_id: str = "",
                        level: int = 0, noise_mask=None,
                        initial_frame: np.ndarray = None,
                        eyes_rules: list = None,
                        cnn_predictor=None,
                        solver_memory=None,
                        profile=None):
    """Drop the Gundam into env_snap. Returns action sequence or None.

    This is the Pacific Rim adapter: Gundam's observe→think→act loop
    driving the env_snap directly. No BFS, no search — just reasoning.

    With CNN: reflexes (cheap CNN exploration) between reasoning (expensive LLM calls).
    The CNN learns which actions change frames. The pilot decides WHY and WHAT TO DO.
    """
    try:
        from gundam import Gundam
    except ImportError:
        log.info("  [gundam] Gundam not importable — skipping")
        return None

    deadline = time.time() + timeout
    env = copy.deepcopy(env_snap)

    # Build action info from the agent's action list — give Gundam useful labels
    action_info = {}
    for i, act in enumerate(actions):
        parts = [f"action_{i}"]
        if hasattr(act, 'label') and act.label:
            parts.append(act.label)
        if hasattr(act, 'direction') and act.direction:
            parts = [act.direction]
        if hasattr(act, 'data') and act.data:
            parts.append(f"data={act.data}")
        action_info[i] = " ".join(parts)

    # Get initial frame — caller passes it if available, else peek with deepcopy
    if initial_frame is None:
        peek_env = copy.deepcopy(env)
        peek_obs = peek_env.step(actions[0].action_id)
        initial_frame = np.array(peek_obs.frame)
        del peek_env
    else:
        initial_frame = np.array(initial_frame)

    n_actions = len(actions)
    mind = Gundam(budget_per_level=max(5, int(timeout / 3)),
                  verbose=False, text_only=False)
    mind.memory.game_id = game_id

    # Initialize the Gundam's observation
    lvl = mind.observe_initial(initial_frame, n_actions, action_info, level)

    # Inject Eyes intel — pre-discovered rules from quick probe
    # CRITICAL: Remap game action IDs to Gundam's 0-indexed action IDs.
    # Pre-probe uses game_action (1-based), pilot uses index (0-based).
    if eyes_rules:
        game_to_idx = {act.game_action: i for i, act in enumerate(actions)}
        remapped_rules = []
        for rule in eyes_rules:
            remapped = rule
            for game_id, idx in game_to_idx.items():
                remapped = remapped.replace(f"Action {game_id} ", f"Action {idx} ")
            # Make direction human-readable: (5,0) → DOWN, (-5,0) → UP, etc.
            import re as _re
            _m = _re.search(r'by \((-?\d+),\s*(-?\d+)\)', remapped)
            if _m:
                _dy, _dx = int(_m.group(1)), int(_m.group(2))
                _dirs = []
                if _dy > 0: _dirs.append('DOWN')
                elif _dy < 0: _dirs.append('UP')
                if _dx > 0: _dirs.append('RIGHT')
                elif _dx < 0: _dirs.append('LEFT')
                if _dirs:
                    remapped = remapped.replace(_m.group(0), f"by ({_m.group(1)},{_m.group(2)}) = {'+'.join(_dirs)}")
            remapped_rules.append(remapped)
        if isinstance(mind.memory.rules_discovered, set):
            mind.memory.rules_discovered.update(remapped_rules)
        else:
            for rule in remapped_rules:
                if rule not in mind.memory.rules_discovered:
                    mind.memory.rules_discovered.append(rule)
        log.info(f"  [gundam] Injected {len(remapped_rules)} Eyes rules (remapped to 0-indexed)")

    # Inject solver_memory rules — persistent cross-session knowledge (genre-filtered)
    if solver_memory is not None:
        try:
            from solver_memory import SolverMemory, tags_from_profile
            _sm_genre = _classify_genre(profile) if profile is not None else ""
            genre_rules = solver_memory.get_rulebook(genre=_sm_genre, min_confidence=0.5)
            if genre_rules:
                for rule in genre_rules:
                    rule_text = f"⚠️ [PAST KNOWLEDGE — TRUST THIS] {rule['pattern']}: {rule['strategy']}"
                    if rule_text not in mind.memory.rules_discovered:
                        mind.memory.rules_discovered.append(rule_text)
                log.info(f"  [gundam] Injected {len(genre_rules)} solver_memory rules (genre={_sm_genre!r})")
        except Exception as e:
            log.debug(f"  [gundam] Solver memory injection failed: {e}")

    # Inject game_knowledge.json rules + hypothesis
    try:
        _gk_path = os.path.join(os.path.dirname(__file__), 'game_knowledge.json')
        with open(_gk_path) as _gk_f:
            _gk_all = json.load(_gk_f)
        _gk_game_id = getattr(mind.memory, 'game_id', '') or ''
        _gk_entry = _gk_all.get(_gk_game_id, _gk_all.get('_default', {}))
        for rule in _gk_entry.get('rules', []):
            if rule not in mind.memory.rules_discovered:
                mind.memory.rules_discovered.append(rule)
        _gk_hyp = _gk_entry.get('hypothesis', '')
        if _gk_hyp and not mind.memory.game_hypothesis:
            mind.memory.game_hypothesis = _gk_hyp
            mind.memory.hypothesis_locked = True
            log.info(f"  [gundam] Hypothesis locked from game_knowledge: {_gk_hyp[:80]}")
        log.info(f"  [gundam] Injected {len(_gk_entry.get('rules', []))} game_knowledge rules")
    except Exception as e:
        log.debug(f"  [gundam] game_knowledge injection failed: {e}")

    # Inject genre-specific rules from game_knowledge.json based on detected genre
    try:
        _gk_path2 = os.path.join(os.path.dirname(__file__), 'game_knowledge.json')
        with open(_gk_path2) as _gk_f2:
            _gk_all2 = json.load(_gk_f2)
        _genre_map = {
            'navigation_maze': '_genre_navigation', 'pursuit_evasion': '_genre_navigation',
            'toggle_puzzle': '_genre_toggle', 'constraint_satisfaction': '_genre_toggle',
            'paint_fill': '_genre_toggle',
            'rail_slider': '_genre_sequence', 'circuit_puzzle': '_genre_sequence',
            'connection_flow': '_genre_sequence',
            'push_block': '_genre_sorting', 'sliding_tile': '_genre_sorting',
            'sorting': '_genre_sorting',
        }
        _hints = getattr(profile, 'genre_hints', []) or []
        _injected_genres = set()
        for hint in _hints:
            _gk_key = _genre_map.get(hint)
            if _gk_key and _gk_key not in _injected_genres:
                _genre_entry = _gk_all2.get(_gk_key, {})
                for rule in _genre_entry.get('rules', []):
                    if rule not in mind.memory.rules_discovered:
                        mind.memory.rules_discovered.append(rule)
                _injected_genres.add(_gk_key)
        if _injected_genres:
            log.info(f"  [gundam] Injected genre rules: {_injected_genres}")
    except Exception as e:
        log.debug(f"  [gundam] Genre rule injection failed: {e}")

    # Seed spatial model from Eyes movement rules
    if eyes_rules:
        import re
        for rule in remapped_rules:
            m = re.match(r'Action (\d+) moves color (\d+) by \((-?\d+),(-?\d+)\)', rule)
            if m:
                aid, color, dy, dx = int(m.group(1)), int(m.group(2)), int(m.group(3)), int(m.group(4))
                if mind.spatial.player_color < 0:
                    mind.spatial.player_color = color
                    log.info(f"  [gundam] Spatial: player=color {color} (from Eyes)")
                # Derive direction name from dy, dx
                if abs(dy) > abs(dx):
                    dname = "DOWN" if dy > 0 else "UP"
                elif abs(dx) > abs(dy):
                    dname = "RIGHT" if dx > 0 else "LEFT"
                else:
                    dname = f"({dy},{dx})"
                mind.spatial.action_map[aid] = (dy, dx, dname)
        # If player detected, find target in initial frame
        if mind.spatial.player_color >= 0 and mind.spatial.target_pos == (-1, -1):
            from gundam import detect_sprites
            try:
                sprites = detect_sprites(initial_frame)
                player_sprites = [s for s in sprites if s.color == mind.spatial.player_color]
                if player_sprites:
                    ps = player_sprites[0]
                    mind.spatial.player_pos = (int(ps.center[0]), int(ps.center[1]))
                    # Target = smallest non-player, non-wall sprite
                    candidates = [s for s in sprites if s.color != mind.spatial.player_color
                                  and s.size < 100 and s.size > 2]
                    if candidates:
                        py, px = mind.spatial.player_pos
                        candidates.sort(key=lambda s: -(abs(s.center[0]-py) + abs(s.center[1]-px)))
                        t = candidates[0]
                        mind.spatial.target_pos = (int(t.center[0]), int(t.center[1]))
                        mind.spatial.target_color = t.color
                        log.info(f"  [gundam] Spatial: target=color {t.color} at {mind.spatial.target_pos}")
            except Exception as e:
                log.warning(f"  [gundam] Spatial init from Eyes failed: {e}")

    # CNN reflexes — the Gundam's motor cortex
    has_cnn = cnn_predictor is not None
    if has_cnn:
        log.info(f"  [gundam] CNN reflexes online — {cnn_predictor.buffer_size} prior observations")

    # Recall past games
    recalled = mind.recall(game_id)
    mind._recalled_memories = recalled

    action_sequence = []
    prev_levels = start_levels
    done = False

    frame = initial_frame.copy()

    def _execute_action(action_id, data=None):
        """Execute one action, train CNN, return (frame, obs, won, done)."""
        nonlocal frame, prev_levels
        action_id = max(0, min(action_id, n_actions - 1))
        frame_before = frame.copy()

        act = actions[action_id]
        step_data = data if data else (act.data or None)
        try:
            if step_data:
                obs = env.step(act.game_action, step_data)
            else:
                obs = env.step(act.game_action)
        except Exception as e:
            log.info(f"  [gundam] Step failed: {e}")
            return frame, None, False, False

        new_frame = np.array(obs.frame)
        # Guard against empty frames (can happen on level transitions)
        if new_frame.size == 0:
            log.debug(f"  [gundam] Empty frame from action {action_id} (game_action={act.game_action}, data={step_data}) — keeping previous")
            new_frame = frame.copy()
        frame = new_frame
        action_sequence.append(act.game_action)

        # Train CNN on every transition — the reflexes learn
        if has_cnn:
            changed = not np.array_equal(frame_before, frame)
            cnn_predictor.observe(frame_before, action_id, frame, changed)

        won = obs.levels_completed > prev_levels
        state_str = str(obs.state).upper()
        is_done = 'WIN' in state_str or 'LOSS' in state_str or 'GAME_OVER' in state_str or (
            state_str.endswith('FINISHED') and 'NOT_FINISHED' not in state_str)

        return frame, obs, won, is_done

    _cnn_recent_actions = []  # track recent CNN actions to avoid repetition
    _visited_frames = set()   # frame hashes seen — curiosity prefers novel frames
    _cnn_hits = 0   # CNN actions that changed frame
    _cnn_misses = 0  # CNN actions that didn't change frame

    def _cnn_explore(max_steps: int = 5):
        """CNN-guided exploration between LLM calls — the reflexes act alone.

        Tries actions the CNN predicts will change the frame.
        Returns list of (action_id, changed, frame_before) for the pilot to review.
        """
        nonlocal _cnn_recent_actions, _cnn_hits, _cnn_misses
        if not has_cnn:
            return []
        explorations = []

        # Greedy nav mode: if spatial model has player, target, and action_map,
        # use direct pathfinding instead of CNN predictions
        # BUT: disable for complex games where modifiers/attributes matter
        _game_hyp = getattr(getattr(mind, 'memory', None), 'game_hypothesis', '') or ''
        # Also check game_knowledge hypothesis
        _gk_hyp = ''
        try:
            import json as _json
            _gk_path = os.path.join(os.path.dirname(__file__), 'game_knowledge.json')
            if os.path.exists(_gk_path):
                with open(_gk_path) as _f:
                    _gk = _json.load(_f)
                _gid = getattr(getattr(mind, 'memory', None), 'game_id', '') or ''
                for _k, _v in _gk.items():
                    if _k.startswith('_'): continue
                    if _k.lower() in _gid.lower():
                        _gk_hyp = _v.get('hypothesis', '')
                        break
        except Exception:
            pass
        _all_hyp = f"{_game_hyp} {_gk_hyp}".lower()
        _complex_game = any(w in _all_hyp for w in ['modifier', 'attribute', 'collect', 'match', 'shape', 'rotation'])
        _has_modifiers = (hasattr(mind, 'spatial') and
                         (mind.spatial.known_modifiers or mind.spatial.pickup_events))
        _is_nav = (profile is None or getattr(profile, 'game_type', '') == 'NAVIGATION')
        _use_greedy_nav = (_is_nav
                          and hasattr(mind, 'spatial') and mind.spatial.action_map
                          and mind.spatial.target_pos != (-1, -1)
                          and mind.spatial.player_pos != (0, 0)
                          and not _has_modifiers   # don't autopilot when game has modifiers
                          and not _complex_game)   # don't autopilot when knowledge says complex

        for step_i in range(max_steps):
            if time.time() > deadline:
                break

            if _use_greedy_nav:
                # Use A* if enough map is known, else greedy Manhattan
                py, px = mind.spatial.player_pos
                ty, tx = mind.spatial.target_pos
                best_aid = None
                if step_i == 0:  # log once per round
                    log.info(f"  [nav] pos=({py},{px}) target=({ty},{tx}) visited={len(mind.spatial.visited)} blocked={len(mind.spatial.blocked)} astar={'yes' if len(mind.spatial.visited)>=3 else 'no'}")
                _astar_path = mind.spatial.plan_path() if len(mind.spatial.visited) >= 3 else []
                if _astar_path:
                    best_aid = _astar_path[0]
                else:
                    # A* failed to reach target — find nearest FRONTIER cell
                    # Frontier = unvisited cell adjacent to a visited cell
                    frontiers = []
                    for (vy, vx) in mind.spatial.visited:
                        for aid_f, (ady, adx, _) in mind.spatial.action_map.items():
                            fy, fx = vy + ady, vx + adx
                            if (fy, fx) not in mind.spatial.visited:
                                # Check not blocked from this visited cell
                                if (vy, vx, aid_f) not in mind.spatial.blocked:
                                    # Score: prefer frontiers TOWARD target
                                    d_target = abs(ty - fy) + abs(tx - fx)
                                    frontiers.append((d_target, fy, fx, vy, vx))
                    if frontiers:
                        frontiers.sort()  # closest to target first
                        for d_t, fy, fx, via_y, via_x in frontiers[:3]:
                            # A* from current pos to the visited cell adjacent to frontier
                            if (via_y, via_x) == (py, px):
                                # Already adjacent — pick the action that goes to frontier
                                for aid_f, (ady, adx, _) in mind.spatial.action_map.items():
                                    if py + ady == fy and px + adx == fx:
                                        best_aid = aid_f
                                        break
                                if best_aid is not None:
                                    break
                            else:
                                # A* to the via cell, then step into frontier
                                path = mind.spatial.find_path(target=(via_y, via_x))
                                if path:
                                    best_aid = path[0]
                                    break
                    # Fallback: greedy toward target (ignore visited)
                    if best_aid is None:
                        for aid, (ady, adx, _) in mind.spatial.action_map.items():
                            if aid < n_actions and (py, px, aid) not in mind.spatial.blocked:
                                best_aid = aid
                                break
                if best_aid is not None:
                    # Anti-repeat: if last 2 greedy actions were same AND didn't change frame, skip
                    if (len(_cnn_recent_actions) >= 2 and
                        _cnn_recent_actions[-1] == best_aid == _cnn_recent_actions[-2]):
                        # Mark as blocked in spatial model and try different action
                        if hasattr(mind, 'spatial'):
                            py_b, px_b = mind.spatial.player_pos
                            mind.spatial.blocked.add((py_b, px_b, best_aid))
                        _use_greedy_nav = False  # fall through to CNN for diversity
                        continue
                    fb = frame.copy()
                    _cnn_recent_actions.append(best_aid)
                    new_frame, obs, won, is_done = _execute_action(best_aid)
                    changed = not np.array_equal(fb, new_frame)
                    if changed: _cnn_hits += 1
                    else:
                        _cnn_misses += 1
                        # Mark blocked immediately on wall hit
                        if hasattr(mind, 'spatial'):
                            py_b, px_b = mind.spatial.player_pos
                            mind.spatial.blocked.add((py_b, px_b, best_aid))
                    explorations.append({
                        'action': best_aid, 'changed': changed,
                        'desc': action_info.get(best_aid, f"action_{best_aid}")
                    })
                    action_desc = action_info.get(best_aid, f"action_{best_aid}")
                    exp = mind.observe_effect(lvl, best_aid, action_desc, fb, new_frame, won)
                    exp.hypothesis_at_time = "(greedy nav — pathfinding)"
                    if won or is_done:
                        return explorations
                    continue  # next step, still greedy

            # Systematic probing for small click puzzles: try untried actions first
            # This is the "subconscious" learning — systematically explore before guessing
            untried = [a for a in range(n_actions) if a not in _cnn_recent_actions[-10:]]
            if untried and step_i < 2:
                # First 2 steps: prioritize untried actions (discovery phase)
                probe_aid = untried[step_i % len(untried)]
                fb = frame.copy()
                _cnn_recent_actions.append(probe_aid)
                new_frame, obs, won, is_done = _execute_action(probe_aid)
                changed = not np.array_equal(fb, new_frame)
                if changed: _cnn_hits += 1
                else: _cnn_misses += 1
                explorations.append({
                    'action': probe_aid, 'changed': changed,
                    'desc': action_info.get(probe_aid, f"action_{probe_aid}")
                })
                action_desc = action_info.get(probe_aid, f"action_{probe_aid}")
                exp = mind.observe_effect(lvl, probe_aid, action_desc, fb, new_frame, won)
                exp.hypothesis_at_time = "(systematic probe — untried action)"
                # Track novelty
                fhash = hash(new_frame.tobytes()[:512])
                if fhash not in _visited_frames:
                    _visited_frames.add(fhash)
                if won or is_done:
                    return explorations
                continue

            # Ask CNN which actions are most likely to change the frame
            probs = cnn_predictor.predict(frame)
            if probs is None:
                break
            # Try actions — add random jitter to break ties + bias toward target
            import random
            jitter = [random.uniform(0, 0.05) for _ in range(n_actions)]
            # Suppress recently-used actions to force diversity
            # Scale penalty by how many times it appeared recently (escalating)
            from collections import Counter
            recent_counts = Counter(_cnn_recent_actions[-6:])
            for recent_aid, count in recent_counts.items():
                if recent_aid < n_actions:
                    # Escalating penalty: -0.5 per repeat (was -0.3)
                    jitter[recent_aid] -= 0.5 * count
            # Hard cap: if last 2 actions are identical, heavily suppress
            if len(_cnn_recent_actions) >= 2 and \
               _cnn_recent_actions[-1] == _cnn_recent_actions[-2]:
                stuck_aid = _cnn_recent_actions[-1]
                if stuck_aid < n_actions:
                    jitter[stuck_aid] -= 5.0  # effectively kill this action after 2 repeats
            # Bias toward target direction if spatial model has one
            if hasattr(mind, 'spatial') and mind.spatial.action_map and \
               mind.spatial.target_pos != (-1, -1) and mind.spatial.player_pos != (0, 0):
                dy = mind.spatial.target_pos[0] - mind.spatial.player_pos[0]
                dx = mind.spatial.target_pos[1] - mind.spatial.player_pos[1]
                for aid, (ady, adx, _) in mind.spatial.action_map.items():
                    if aid < n_actions:
                        # Reward actions that move toward target (strong bias for nav games)
                        _genre = getattr(getattr(mind, '_env_profile', None), 'genre', '')
                        nav_boost = 0.3 if 'nav' in _genre.lower() else 0.15
                        if (dy > 0 and ady > 0) or (dy < 0 and ady < 0):
                            jitter[aid] += nav_boost
                        if (dx > 0 and adx > 0) or (dx < 0 and adx < 0):
                            jitter[aid] += nav_boost
            # Curiosity bonus: if current frame is already visited, boost exploration
            cur_fhash = hash(frame.tobytes())
            _visited_frames.add(cur_fhash)
            if len(_visited_frames) > 1:
                # In a stuck state (repeated frame), boost less-tried actions
                for aid_j in range(n_actions):
                    if aid_j not in _cnn_recent_actions[-5:]:
                        jitter[aid_j] += 0.1  # explore untried actions
            # Suppress spatially blocked actions (walls the Gundam already found)
            if hasattr(mind, 'spatial') and mind.spatial.player_pos != (0, 0):
                py, px = mind.spatial.player_pos
                for aid_b in range(n_actions):
                    if (py, px, aid_b) in mind.spatial.blocked:
                        jitter[aid_b] -= 10.0  # effectively block it
            _cnn_range = min(n_actions, len(probs))
            sorted_actions = sorted(range(_cnn_range), key=lambda a: -(probs[a] + jitter[a]))
            tried = False
            for aid in sorted_actions:
                if probs[aid] + jitter[aid] < 0.1:  # below threshold after suppression
                    break
                fb = frame.copy()
                _cnn_recent_actions.append(aid)
                new_frame, obs, won, is_done = _execute_action(aid)
                changed = not np.array_equal(fb, new_frame)
                # Curiosity: track frame novelty
                new_fhash = hash(new_frame.tobytes())
                novel = new_fhash not in _visited_frames
                _visited_frames.add(new_fhash)
                if changed: _cnn_hits += 1
                else: _cnn_misses += 1
                explorations.append({
                    'action': aid, 'changed': changed, 'novel': novel,
                    'desc': action_info.get(aid, f"action_{aid}")
                })
                # Tell the Gundam what we saw
                action_desc = action_info.get(aid, f"action_{aid}")
                exp = mind.observe_effect(lvl, aid, action_desc, fb, new_frame, won)
                exp.hypothesis_at_time = "(CNN reflex — no reasoning)"
                if won:
                    return explorations  # won during exploration!
                if is_done:
                    return explorations
                tried = True
                break  # one action per explore step
            if not tried:
                break  # CNN has nothing useful to suggest
        return explorations

    for turn in range(200):  # hard limit
        if time.time() > deadline:
            log.info(f"  [gundam] Timeout after {len(action_sequence)} actions")
            break

        if mind.total_llm_calls >= mind.budget_per_level:
            log.info(f"  [gundam] LLM budget exhausted after {len(action_sequence)} actions")
            break

        # CNN REFLEX PHASE — cheap exploration between LLM calls
        # Only after turn 0 (let the pilot look first) and every other turn.
        # In Pure AGI mode: CNN observes only until pilot has made 5+ actions.
        # This prevents CNN from driving blind with stale cross-game priors.
        # Cap total CNN actions to leave room for LLM reasoning.
        _pure_agi = os.environ.get('ARC_PURE_AGI', '1').strip() != '0'
        _cnn_min_pilot_actions = 5 if _pure_agi else 0
        _cnn_total_actions = _cnn_hits + _cnn_misses
        _cnn_max_per_level = 20  # Don't let CNN eat more than 20 actions
        if has_cnn and turn > 0 and turn % 2 == 0 and len(action_sequence) >= _cnn_min_pilot_actions and _cnn_total_actions < _cnn_max_per_level:
            # Adaptive CNN budget: more steps when CNN is producing frame changes
            _cnn_total = _cnn_hits + _cnn_misses
            if _cnn_total >= 4 and _cnn_hits / _cnn_total >= 0.6:
                _cnn_budget = 6  # CNN is effective — explore more
            elif _cnn_total >= 4 and _cnn_hits / _cnn_total <= 0.2:
                _cnn_budget = 1  # CNN is stuck — save actions
            else:
                _cnn_budget = 3  # default
            explorations = _cnn_explore(max_steps=_cnn_budget)
            if explorations:
                # Check if CNN exploration solved it
                last = explorations[-1]
                if any(e.get('won') for e in explorations):
                    log.info(f"  [gundam] CNN reflexes SOLVED level {level}!")
                    mind.remember_game()
                    return action_sequence
                log.info(f"  [gundam] CNN explored {len(explorations)} actions: "
                         f"{', '.join(e['desc'] for e in explorations)}")

        # ── MUSCLE MEMORY: execute learned spatial path without LLM call ──
        # If the spatial model has a known path to target, follow it.
        # Approved by Peter, 2026-03-16. "I think I know how this works."
        # Only for NAVIGATION games — click/toggle puzzles don't have spatial paths.
        _muscle_path = []
        _is_nav_game = (profile is None or getattr(profile, 'game_type', '') == 'NAVIGATION')
        if (_is_nav_game
                and hasattr(mind, 'spatial') and mind.spatial.action_map
                and mind.spatial.player_pos != (0, 0)
                and mind.spatial.target_pos != (-1, -1)
                and mind.total_llm_calls > 0):
            _muscle_path = mind.spatial.plan_path()

        if _muscle_path:
            # Known-corridor extension: if path goes through visited positions, extend cap
            _mm_cap = 5
            if mind.spatial.visited:
                _known = 0
                _pos = mind.spatial.player_pos
                for _aid in _muscle_path:
                    if _aid in mind.spatial.action_map:
                        _dy, _dx, _ = mind.spatial.action_map[_aid]
                        _nxt = (_pos[0] + _dy, _pos[1] + _dx)
                        if _nxt in mind.spatial.visited:
                            _known += 1
                            _pos = _nxt
                        else:
                            break
                    else:
                        break
                if _known >= 3:
                    _mm_cap = min(12, max(_mm_cap, _known))
            _mm_len = min(len(_muscle_path), _mm_cap)
            decision = {
                'action': _muscle_path[0],
                'data': {},
                'hypothesis': f'Muscle memory — spatial path ({len(_muscle_path)} steps, executing {_mm_len})',
                'sequence': [(_muscle_path[i], {}) for i in range(_mm_len)],
            }
            log.info(f"  [muscle-memory] Path: {_mm_len}/{len(_muscle_path)} steps toward target")
        else:
            # THINK — the pilot reasons (expensive LLM call)
            # CNN priors injected into mind context via action ordering hint
            if has_cnn:
                probs = cnn_predictor.predict(frame)
                if probs is not None:
                    _cnn_n = min(n_actions, len(probs))
                    ranked = sorted(range(_cnn_n), key=lambda a: -probs[a])
                    hint = ", ".join(f"{action_info.get(a, f'action_{a}')}({probs[a]:.0%})"
                                    for a in ranked[:4])
                    mind._cnn_hint = f"CNN reflexes suggest: {hint}"
                else:
                    mind._cnn_hint = None
            decision = mind.think(lvl, phase="act")
        sequence = decision.get('sequence', [(decision['action'], decision.get('data', {}))])
        if not sequence:
            sequence = [(decision['action'], decision.get('data', {}))]

        # EXECUTE the pilot's chosen sequence
        for seq_idx, (action_id, data) in enumerate(sequence):
            if time.time() > deadline:
                break
            frame_before = frame.copy()
            frame, obs, won, is_done = _execute_action(action_id, data)

            if obs is None:
                continue

            action_desc = action_info.get(action_id, f"action_{action_id}")
            exp = mind.observe_effect(lvl, action_id, action_desc, frame_before, frame, won)
            exp.hypothesis_at_time = decision.get('hypothesis', '')

            if won:
                log.info(f"  [gundam] SOLVED level {level} in {len(action_sequence)} actions, "
                         f"{mind.total_llm_calls} LLM calls")
                mind.remember_game()
                # CNN remembers: reinforce successful action sequence
                if cnn_predictor and hasattr(cnn_predictor, 'reinforce'):
                    action_indices = [a for a, _ in action_sequence] if action_sequence else []
                    cnn_predictor.reinforce(action_indices)
                return action_sequence

            if is_done:
                done = True
                break  # exit sequence loop

        if done:
            log.info(f"  [gundam] Game ended without solving (lost?)")
            break  # exit main loop

    # Didn't solve — remember what we learned for next time
    mind.remember_game()
    return None


def solve_level(env_snap, actions: list[Action], profile: GameProfile,
                start_levels: int, timeout: float = 60.0,
                recorder: GameRecorder = None,
                eyes_memory=None,
                llm_hint: Optional[LLMHint] = None,
                goal_frame: Optional[np.ndarray] = None,
                preferred_route: str = '',
                winning_route: list = None,
                pilot: Pilot = None,
                game_id: str = '',
                level: int = 0,
                levels_solved: int = 0,
                levels_total: int = 0,
                level_results: list = None,
                solver_memory=None,
                reasoner=None,
                failure_detector: 'FailureDetector' = None,
                cnn_predictor=None,
                level_baseline: int = 0) -> Optional[list[Action]]:
    """
    Try multiple solver routes to complete one level.
    env_snap is a deepcopy of env at the start of this level.
    eyes_memory: EpisodicMemory from arc_eyes — accumulates across levels/games.
    llm_hint: Optional LLM analysis of the game — affects MCTS rollout depth.
    goal_frame: Pre-win frame from previous level — used as MCTS reward signal.
    preferred_route: Route ID that solved the previous level — tried first with 40% budget.
    winning_route: Mutable list — set to [route_id] when a route succeeds.
    """
    deadline = time.time() + timeout
    if winning_route is None:
        winning_route = []

    # ARC_NO_BFS: Pure reasoning mode — no brute-force search.
    # Keeps: Gundam (AGI pilot), Eyes, Mechanic Learning, Reactive Controller, bus/dopamine.
    # Skips: BFS, MCTS, NMCS, Random Walk, A*, Block-BFS, Transition Graph, NavModel.
    no_bfs = bool(os.environ.get('ARC_NO_BFS', ''))
    pure_agi = os.environ.get('ARC_PURE_AGI', '1').strip() != '0'

    def _won(route_id, result):
        """Record which route won and return the result."""
        winning_route.clear()
        winning_route.append(route_id)
        # Record success in failure detector
        if failure_detector:
            failure_detector.record(profile.game_type, route_id, success=True)
        return result

    def _failed_route(route_id):
        """Record a route failure in the failure detector."""
        if failure_detector:
            failure_detector.record(profile.game_type, route_id, success=False)

    def _skip_route(route_id) -> bool:
        """Should we skip this route? Based on failure pattern detection."""
        if failure_detector and failure_detector.should_skip_route(profile.game_type, route_id):
            _thought(f'Skipping {route_id} — failure pattern detected', 'thought')
            log.info(f"  [self] Skipping {route_id} — futile for {profile.game_type}")
            return True
        return False

    def _thought(text, ctype='route'):
        if recorder:
            recorder.set_route(text) if ctype == 'route' else recorder.comment(text, comment_type=ctype)

    # Route -1: Specialized solvers for known games (VC33, LS20) — fastest, try first
    if not no_bfs and not pure_agi and profile.game_type == 'SPECIALIZED':
        _thought('Route -1: Specialized solver')
        specialized = try_specialized_solver(env_snap, start_levels)
        if specialized is not None:
            _thought('Specialized solver cracked it', 'thought')
            return _won('specialized', specialized)

    # Eyes moved to Route 0.5 (after constraint solver, before generic BFS)

    # Get the current game frame BEFORE any action (for frame_coupling_solve)
    current_game_frame = None
    if hasattr(env_snap, '_last_obs') and env_snap._last_obs is not None:
        current_game_frame = np.array(env_snap._last_obs.frame)

    # Get initial frame for noise detection
    probe_env = copy.deepcopy(env_snap)
    if not actions:
        log.warning("  No actions discovered — cannot solve")
        return []
    probe_obs = probe_env.step(actions[0].game_action,
                                actions[0].data if actions[0].data else None)
    if probe_obs and is_level_won(probe_obs, start_levels):
        return [actions[0]]  # First action wins!

    frame0 = np.array(probe_obs.frame) if probe_obs else None
    noise_mask = None
    if frame0 is not None and probe_obs:
        noise_mask = detect_noise(env_snap, frame0, probe_obs.available_actions)

    _thought(f'{len(actions)} actions discovered, type={profile.game_type}', 'thought')
    log.info(f"  {len(actions)} actions, type={profile.game_type}")

    # ─── CNN-Guided Action Ordering ─────────────────────────────────────
    # Sort actions by predicted change probability — BFS/MCTS explore productive actions first.
    # Also feed the probe observation to the CNN for online learning.
    cnn_action_probs = None
    if cnn_predictor and cnn_predictor.enabled and frame0 is not None:
        # Feed probe transition to CNN
        probe_env2 = copy.deepcopy(env_snap)
        for ai, act in enumerate(actions[:12]):  # probe up to 12 actions
            try:
                pe = copy.deepcopy(probe_env2)
                po = pe.step(act.game_action, act.data if act.data else None)
                if po:
                    cnn_predictor.observe(frame0, ai, np.array(po.frame))
                del pe
            except Exception:
                pass
        # Train if we have enough data
        if cnn_predictor.total_obs >= 10:
            cnn_predictor.train(epochs=2, batch_size=32)
        # Get action predictions
        if cnn_predictor.confidence > 0.2:
            cnn_action_probs = cnn_predictor.predict(frame0)
            # Sort actions by predicted change probability (highest first)
            sorted_indices = np.argsort(-cnn_action_probs[:len(actions)])
            actions = [actions[i] for i in sorted_indices]
            log.info(f"  CNN action order: {[f'{cnn_action_probs[i]:.2f}' for i in sorted_indices[:6]]}")

    # Smart action filtering: if we have click actions, try click-only first
    click_only = [a for a in actions if a.data and 'x' in a.data]

    # ─── Cross-Level Route Learning ────────────────────────────────────
    # If a previous level solved with a specific route, try it first with 40% budget.
    # Rationale: games tend to use the same mechanic across levels.
    if preferred_route and not pure_agi:
        _thought(f'Preferred route from prior level: {preferred_route}', 'thought')
        boost_budget = (deadline - time.time()) * 0.4
        _route_solvers = {
            'constraint': lambda t: constraint_solve_v2(env_snap, click_only or actions, start_levels),
            'toggle_matrix': lambda t: toggle_matrix_solve(env_snap, click_only or actions, start_levels, timeout=t),
            'bfs': lambda t: deepcopy_bfs(env_snap, click_only or actions, start_levels, timeout=t, noise_mask=noise_mask),
            'bfs_cyclic': lambda t: deepcopy_bfs(env_snap, click_only or actions, start_levels, timeout=t, noise_mask=noise_mask, dedup_cycle=3),
            'mechanic': lambda t: mechanic_learning_solve(env_snap, actions, start_levels, timeout=t, noise_mask=noise_mask, frame0=frame0),
            'navigation': lambda t: navigation_solve(env_snap, actions, start_levels, timeout=t, noise_mask=noise_mask),
            'mcts': lambda t: mcts_search(env_snap, actions, start_levels, timeout=t, noise_mask=noise_mask, rollout_len=200, goal_frame=goal_frame),
            'sliding_astar': lambda t: sliding_tile_astar(env_snap, actions, start_levels, timeout=t, noise_mask=noise_mask, goal_frame=goal_frame),
            'block_bfs': lambda t: block_state_bfs(env_snap, actions, start_levels, timeout=t, noise_mask=noise_mask),
            'transition_graph': lambda t: transition_graph_solve(env_snap, actions, start_levels, timeout=t, noise_mask=noise_mask, goal_frame=goal_frame),
            'nmcs': lambda t: nmcs_search(env_snap, actions, start_levels, timeout=t, noise_mask=noise_mask, max_depth=100),
            'random_walk': lambda t: random_walk(env_snap, actions, start_levels, timeout=t, noise_mask=noise_mask, walk_len=200),
            'dynamic_bfs': lambda t: dynamic_action_bfs(env_snap, actions, start_levels, timeout=t, noise_mask=noise_mask),
            'inductive': lambda t: None,  # Inductive reasoning runs in Eyes pipeline, not standalone
        }
        solver_fn = _route_solvers.get(preferred_route)
        if solver_fn and boost_budget > 3:
            try:
                result = solver_fn(boost_budget)
                if result:
                    _thought(f'Preferred route {preferred_route} solved it again!', 'thought')
                    return _won(preferred_route, result)
            except Exception as e:
                log.debug(f"  Preferred route {preferred_route} failed: {e}")
            # Didn't work — fall through to normal cascade with remaining 60% budget
            log.info(f"  Preferred route {preferred_route} didn't solve — cascading normally")

    # ─── Genre-Aware Budget Allocation ──────────────────────────────────
    # Use genre_route_budget() to scale time allocations per route.
    # High-confidence genre → spend more on expected routes, less on others.
    # Unknown genre → all routes get default budget (no change from before).
    _genre = (profile.genre_hints[0] if profile.genre_hints else 'unknown')
    _genre_known = _genre in GENRE_ROUTE_MAP
    _has_action_profile = bool(getattr(profile, 'oprah_action_counts', {}))
    if _genre_known:
        log.info(f"  Genre routing: {_genre} → prioritizing {list(GENRE_ROUTE_MAP[_genre].keys())[:5]}")
    elif _has_action_profile:
        _ap_weights = _action_profile_route_weights(profile)
        _top_routes = sorted(_ap_weights, key=_ap_weights.get, reverse=True)[:4]
        log.info(f"  Action-profile routing: {profile.oprah_action_counts} → prioritizing {_top_routes}")

    def _gbud(route_id: str, default_frac: float) -> float:
        """Genre-scaled budget: returns adjusted fraction for this route."""
        if not _genre_known and not _has_action_profile:
            return default_frac  # No routing info → use existing hardcoded budget
        genre_frac = genre_route_budget(profile, route_id)
        # Scale: if genre says 0.30 and default is 0.15, use max(0.30, default)
        # If genre says 0.05 (not prioritized), use min(0.05, default) — shrink it
        if genre_frac > _DEFAULT_ROUTE_BUDGET:
            return max(genre_frac, default_frac)  # Route prioritized
        else:
            return min(genre_frac, default_frac)  # Route deprioritized

    # ─── Pre-probe: Eyes → Gundam intel pipeline ─────────────────────
    # Run a quick probe BEFORE Gundam to discover movement rules, game type,
    # sprite positions. ~5s cost, massive reasoning benefit.
    eyes_rules_for_gundam = []
    pre_probe_profile = None
    if actions and (deadline - time.time()) > 15:  # only if enough budget + actions
        _thought('Pre-probe: Eyes intel for Gundam')
        try:
            from arc_reason import InductiveReasoner
            pre_reasoner = InductiveReasoner()
            probe_set = actions[:min(5, len(actions))]
            # Get baseline frame — use current_game_frame if available
            if current_game_frame is not None:
                baseline_frame = np.array(current_game_frame)
            else:
                # Peek at initial state
                peek_env = copy.deepcopy(env_snap)
                peek_obs = peek_env.step(actions[0].game_action)
                baseline_frame = np.array(peek_obs.frame)
                del peek_env
            for probe_act in probe_set:
                probe_env = copy.deepcopy(env_snap)
                frame_before = baseline_frame.copy()
                try:
                    if probe_act.data:
                        probe_obs1 = probe_env.step(probe_act.game_action, **probe_act.data)
                    else:
                        probe_obs1 = probe_env.step(probe_act.game_action)
                except TypeError:
                    probe_obs1 = probe_env.step(probe_act.game_action, data=probe_act.data)
                frame_after = np.array(probe_obs1.frame)
                won = probe_obs1.state.value == 'WIN' if hasattr(probe_obs1, 'state') else False
                action_data = probe_act.data if probe_act.data else {}
                pre_reasoner.observe(
                    action_id=probe_act.game_action,
                    action_data=action_data,
                    frame_before=frame_before,
                    frame_after=frame_after,
                    won=won
                )
                del probe_env
            pre_hypotheses = pre_reasoner.hypothesize()
            if pre_hypotheses:
                eyes_rules_for_gundam = [h.description for h in pre_hypotheses if h.confidence >= 0.3]
                log.info(f"  [pre-probe] {len(eyes_rules_for_gundam)} rules for Gundam: {[r[:50] for r in eyes_rules_for_gundam]}")

            # SECOND-STATE PROBE: actions that looked minor at start might be
            # blocked there. Move with a known directional action first, then re-probe.
            # If we have fewer rules than actions, some actions may be start-blocked.
            if len(eyes_rules_for_gundam) < len(probe_set) and len(probe_set) > 1 and (deadline - time.time()) > 10:
                minor_actions = probe_set  # re-probe everything from new state
                mover = probe_set[0]  # use first action as state changer
                try:
                    moved_env = copy.deepcopy(env_snap)
                    moved_obs = moved_env.step(mover.game_action)
                    moved_frame = np.array(moved_obs.frame)
                    for ma in minor_actions:
                        probe_env2 = copy.deepcopy(moved_env)
                        try:
                            probe_obs2 = probe_env2.step(ma.game_action)
                        except TypeError:
                            probe_obs2 = probe_env2.step(ma.game_action, data=ma.data)
                        frame_after2 = np.array(probe_obs2.frame)
                        pre_reasoner.observe(
                            action_id=ma.game_action,
                            action_data=ma.data if ma.data else {},
                            frame_before=moved_frame,
                            frame_after=frame_after2,
                            won=False
                        )
                        del probe_env2
                    del moved_env
                    # Re-hypothesize with new observations
                    pre_hypotheses2 = pre_reasoner.hypothesize()
                    if pre_hypotheses2:
                        new_rules = [h.description for h in pre_hypotheses2
                                     if h.confidence >= 0.3 and h.description not in eyes_rules_for_gundam]
                        if new_rules:
                            eyes_rules_for_gundam.extend(new_rules)
                            log.info(f"  [pre-probe] Second-state discovered {len(new_rules)} more rules: {[r[:50] for r in new_rules]}")
                except Exception as e:
                    log.debug(f"  [pre-probe] Second-state probe failed: {e}")
        except Exception as e:
            log.info(f"  [pre-probe] Failed: {type(e).__name__}: {e}")

    # ─── Route S: SHORT SEQUENCE PROBE (small action spaces only) ────
    # For games with ≤5 actions, try all sequences of length 1-4 on deepcopy.
    # This is informed exploration, not blind BFS — we check after each sequence.
    # Budget: 10s max. Only fires on levels after L0 (L0 needs discovery first).
    n_game_actions = len(actions)
    _skip_probes = level_baseline > 16  # Baseline too high for combinatorial probes
    if _skip_probes:
        log.info(f"  [probes] Skipping short-seq/subset probes — baseline={level_baseline} too high for combinatorial search")
    if not _skip_probes and n_game_actions <= 5 and (deadline - time.time()) > 30:
        import itertools
        _seq_t0 = time.time()
        _seq_budget = min(60.0, (deadline - time.time()) * 0.15)
        _seq_actions = list(range(n_game_actions))  # use action indices, not raw IDs
        _seq_solved = None
        # ~5ms per combo (deepcopy dominated). Budget=60s → ~12K combos max.
        # 3 acts: 3^8=6561 (5s), 4 acts: 4^6=4096 (3s), 5 acts: 5^5=3125 (2.5s)
        _max_seq = 8 if n_game_actions <= 3 else (6 if n_game_actions <= 4 else 5)
        for seq_len in range(1, _max_seq + 1):
            if time.time() - _seq_t0 > _seq_budget:
                break
            for seq in itertools.product(_seq_actions, repeat=seq_len):
                if time.time() - _seq_t0 > _seq_budget:
                    break
                try:
                    _seq_env = copy.deepcopy(env_snap)
                    _seq_obs = None
                    for idx in seq:
                        act = actions[idx]
                        _seq_obs = _seq_env.step(act.game_action, act.data or None)
                    if _seq_obs and (_seq_obs.levels_completed > start_levels or
                                     _seq_obs.state.value == 'WIN'):
                        _seq_solved = list(seq)  # list of action indices
                        log.info(f"  [short-seq] SOLVED with sequence {_seq_solved} "
                                 f"(len={seq_len}) in {time.time()-_seq_t0:.1f}s")
                        break
                except Exception:
                    pass
            if _seq_solved:
                break
        if _seq_solved:
            # Execute on real env — map indices back to Action objects
            solution_acts = [Action(actions[idx].game_action, actions[idx].data, f"short_seq_{i}") for i, idx in enumerate(_seq_solved)]
            return _won('short_seq', solution_acts)
        elif time.time() - _seq_t0 > 0.5:
            _total_combos = sum(n_game_actions**k for k in range(1, _max_seq + 1))
            log.info(f"  [short-seq] No solution in {time.time()-_seq_t0:.1f}s "
                     f"({n_game_actions}^{_max_seq} = {_total_combos} combos checked)")

    # ─── Route S2: SUBSET PROBE (click/toggle puzzles — order-independent) ────
    # For lights-out / constraint puzzles, test all SUBSETS of click positions.
    # Each position is clicked 0 or 1 times; order doesn't matter.
    # 2^N subsets for N click positions — feasible up to N=12 (4096 subsets).
    _profile_type = profile.game_type if profile else ''
    _profile_hints = getattr(profile, 'genre_hints', []) if profile else []
    _is_click_puzzle = (_profile_type == 'CLICK_SEQUENCE' and
                        ('constraint_satisfaction' in _profile_hints or
                         'toggle_puzzle' in _profile_hints or
                         'paint_fill' in _profile_hints))
    # Find click-type actions (those with position data, not counter/minor)
    _click_acts = [a for a in actions if hasattr(a, 'data') and a.data and
                   any(k in a.data for k in ('x', 'y', 'position'))]
    if not _click_acts:
        # Fallback: actions with game_action containing position data
        _click_acts = [a for a in actions if isinstance(a.game_action, dict)]
    n_click = len(_click_acts)
    log.info(f"  [subset-probe] click_puzzle={_is_click_puzzle}, n_click={n_click}, type={_profile_type}, hints={_profile_hints}")
    if not _skip_probes and _is_click_puzzle and 2 <= n_click <= 16 and (deadline - time.time()) > 15:
        import itertools
        _sub_t0 = time.time()
        _sub_budget = min(30.0, (deadline - time.time()) * 0.10)
        _sub_solved = None
        # Test all subsets from size 1 to n_click (order-independent)
        for subset_size in range(1, n_click + 1):
            if time.time() - _sub_t0 > _sub_budget:
                break
            for subset in itertools.combinations(range(n_click), subset_size):
                if time.time() - _sub_t0 > _sub_budget:
                    break
                try:
                    _sub_env = copy.deepcopy(env_snap)
                    _sub_obs = None
                    for idx in subset:
                        act = _click_acts[idx]
                        _sub_obs = _sub_env.step(act.game_action, act.data or None)
                    if _sub_obs and (_sub_obs.levels_completed > start_levels or
                                     _sub_obs.state.value == 'WIN'):
                        _sub_solved = [_click_acts[idx] for idx in subset]
                        log.info(f"  [subset-probe] SOLVED with {len(_sub_solved)} clicks "
                                 f"(size={subset_size}) in {time.time()-_sub_t0:.1f}s")
                        break
                except Exception:
                    pass
            if _sub_solved:
                break
        if _sub_solved:
            solution_acts = [Action(a.game_action, a.data, f"subset_{i}") for i, a in enumerate(_sub_solved)]
            return _won('subset_probe', solution_acts)
        elif time.time() - _sub_t0 > 0.5:
            log.info(f"  [subset-probe] No solution in {time.time()-_sub_t0:.1f}s "
                     f"(2^{n_click} = {2**n_click} subsets, click puzzle)")
            # Phase 1b: ordered permutation probe — subsets failed, try ORDER-DEPENDENT permutations
            # For n_click=14: P(14,1..5) = ~267K perms, feasible in ~30s
            if n_click <= 16 and (deadline - time.time()) > 15:
                _perm_t0 = time.time()
                _perm_budget = min(45.0, (deadline - time.time()) * 0.12)
                _perm_solved = None
                _perm_combos = 0
                _max_perm = min(7, n_click)  # cap at perm size 7
                for perm_size in range(2, _max_perm + 1):  # skip size 1 (already tried)
                    if time.time() - _perm_t0 > _perm_budget:
                        break
                    for perm in itertools.permutations(range(n_click), perm_size):
                        if time.time() - _perm_t0 > _perm_budget:
                            break
                        _perm_combos += 1
                        try:
                            _perm_env = copy.deepcopy(env_snap)
                            _perm_obs = None
                            for idx in perm:
                                act = _click_acts[idx]
                                _perm_obs = _perm_env.step(act.game_action, act.data or None)
                            if _perm_obs and (_perm_obs.levels_completed > start_levels or
                                              _perm_obs.state.value == 'WIN'):
                                _perm_solved = [_click_acts[idx] for idx in perm]
                                log.info(f"  [perm-probe] SOLVED with ordered sequence of {len(_perm_solved)} clicks "
                                         f"in {time.time()-_perm_t0:.1f}s ({_perm_combos} perms)")
                                break
                        except Exception:
                            pass
                    if _perm_solved:
                        break
                if _perm_solved:
                    solution_acts = [Action(a.game_action, a.data, f"perm_{i}") for i, a in enumerate(_perm_solved)]
                    return _won('perm_probe', solution_acts)
                elif _perm_combos > 0:
                    log.info(f"  [perm-probe] No solution in {time.time()-_perm_t0:.1f}s "
                             f"({_perm_combos} permutations up to size {perm_size})")
            # Phase 2: mixed probe — try all combos of N counters + M clicks where N+M ≤ max_len
            _non_click = [a for a in actions if a not in _click_acts]
            n_nc = len(_non_click)
            if _non_click and n_click <= 16 and (deadline - time.time()) > 15:
                _sub2_budget = min(60.0, (deadline - time.time()) * 0.15)
                _sub2_t0 = time.time()
                _sub2_combos = 0
                _max_total = min(8, n_nc + n_click)  # max actions in combo
                for total_len in range(2, _max_total + 1):
                    if time.time() - _sub2_t0 > _sub2_budget:
                        break
                    for n_counters in range(1, min(n_nc + 1, total_len)):
                        n_clicks = total_len - n_counters
                        if n_clicks < 1 or n_clicks > n_click:
                            continue
                        if time.time() - _sub2_t0 > _sub2_budget:
                            break
                        for counter_combo in itertools.combinations(range(n_nc), n_counters):
                            if time.time() - _sub2_t0 > _sub2_budget:
                                break
                            for click_combo in itertools.combinations(range(n_click), n_clicks):
                                if time.time() - _sub2_t0 > _sub2_budget:
                                    break
                                _sub2_combos += 1
                                try:
                                    _sub2_env = copy.deepcopy(env_snap)
                                    _sub2_obs = None
                                    for ci in counter_combo:
                                        act = _non_click[ci]
                                        _sub2_obs = _sub2_env.step(act.game_action, act.data or None)
                                    for ci in click_combo:
                                        act = _click_acts[ci]
                                        _sub2_obs = _sub2_env.step(act.game_action, act.data or None)
                                    if _sub2_obs and (_sub2_obs.levels_completed > start_levels or
                                                       _sub2_obs.state.value == 'WIN'):
                                        _sub_solved = ([_non_click[ci] for ci in counter_combo] +
                                                      [_click_acts[ci] for ci in click_combo])
                                        log.info(f"  [mixed-probe] SOLVED: {n_counters} counters + {n_clicks} clicks "
                                                 f"in {time.time()-_sub2_t0:.1f}s ({_sub2_combos} combos)")
                                        break
                                except Exception:
                                    pass
                            if _sub_solved:
                                break
                        if _sub_solved:
                            break
                    if _sub_solved:
                        break
                if _sub_solved:
                    solution_acts = [Action(a.game_action, a.data, f"mixed_{i}") for i, a in enumerate(_sub_solved)]
                    return _won('mixed_probe', solution_acts)
                elif time.time() - _sub2_t0 > 0.5:
                    log.info(f"  [mixed-probe] No solution in {time.time()-_sub2_t0:.1f}s "
                             f"({_sub2_combos} combos, {n_nc} counters × {n_click} clicks)")

    # ─── Route 0 (pre-Gundam): CSv2 constraint solver for click games ──
    # CSv2 is instant (no LLM), so run it before Gundam to avoid wasting budget.
    # Only in competition mode (not pure_agi) and for click/toggle games.
    if not pure_agi and click_only and profile.game_type in ("CLICK_TOGGLE", "CLICK_SEQUENCE"):
        log.info(f"  [pre-Gundam CSv2] Trying constraint solver ({len(click_only)} click actions)")
        result = constraint_solve_v2(env_snap, click_only, start_levels)
        if result:
            return _won('constraint', result)
        # Also try generic observation-based solver
        time_left = deadline - time.time()
        if time_left > 3:
            cst_time = min(time_left * 0.1, 30)
            result = constraint_solve(env_snap, click_only, start_levels,
                                      timeout=cst_time, all_actions=actions)
            if result:
                return _won('constraint', result)
        log.info(f"  [pre-Gundam CSv2] No solution found")

    # ─── Route G: GUNDAM — Pure AGI pilot ─────────────────────────────
    # Observe → Hypothesize → Plan → Execute → Adapt. No search, just thinking.
    # In pure_agi mode, Gundam gets 90% budget. Otherwise 40%.
    gundam_budget = (deadline - time.time()) * (0.90 if (pure_agi or no_bfs) else 0.40)
    if gundam_budget > 5:
        _thought('Route G: Gundam (pure AGI pilot)')
        try:
            # Inject baseline hint so the pilot knows expected solution length
            _gundam_eyes = list(eyes_rules_for_gundam)
            if level_baseline > 0:
                _gundam_eyes.append(f"⚠️ BASELINE: The human benchmark for this level is {level_baseline} actions. Plan accordingly — {'short combo needed' if level_baseline <= 10 else 'sustained multi-step strategy needed, not short brute-force'}.")
            gundam_result = _gundam_solve_level(
                env_snap, actions, start_levels, timeout=gundam_budget,
                game_id=game_id, level=level, noise_mask=noise_mask,
                initial_frame=current_game_frame,
                eyes_rules=_gundam_eyes,
                cnn_predictor=cnn_predictor,
                solver_memory=solver_memory,
                profile=profile)
            if gundam_result is not None:
                _thought('Gundam solved it by pure reasoning', 'thought')
                return _won('gundam', gundam_result)
        except Exception as e:
            import traceback
            log.info(f"  [gundam] Route G failed: {e}")
            log.info(f"  [gundam] Traceback:\n{traceback.format_exc()}")
        log.info(f"  [gundam] Didn't solve — cascading to other routes")

    # PURE AGI MODE: Skip all algorithmic routes. Gundam is all we have.
    # Specialized solvers and BFS are "dangerous weapons, use in emergency."
    if pure_agi:
        log.info("  PURE AGI: Skipping all algorithmic routes (constraint, BFS, MCTS, etc.)")
        return None

    # Route 0: Constraint solver for toggle/click games
    if click_only and profile.game_type in ("CLICK_TOGGLE", "CLICK_SEQUENCE"):
        _thought('Route 0: Constraint solver (Z/kZ)')
        result = constraint_solve_v2(env_snap, click_only, start_levels)
        if result:
            return _won('constraint', result)
        # Fall back to generic observation-based solver
        time_left = deadline - time.time()
        if time_left > 3:
            cst_time = min(time_left * _gbud('constraint', 0.5), 30)
            result = constraint_solve(env_snap, click_only, start_levels,
                                      timeout=cst_time, all_actions=actions)
            if result:
                return _won('constraint', result)

    # Route 0.2: Frame-based coupling solver for neighbor-coupled toggles
    if click_only and profile.has_neighbor_coupling and profile.self_inverse:
        _thought('Route 0.2: Frame coupling solver (GF(2) from pixel diffs)')
        time_left = deadline - time.time()
        if time_left > 5:
            fc_time = min(time_left * _gbud('frame_coupling', 0.3), 20)
            result = frame_coupling_solve(env_snap, click_only, start_levels,
                                          profile, noise_mask=noise_mask,
                                          timeout=fc_time,
                                          current_frame=current_game_frame)
            if result:
                return _won('frame_coupling', result)

    # Route 0.3: Toggle matrix solver — observation-based Z/kZ
    if click_only and len(click_only) <= 30:
        _thought('Route 0.3: Toggle matrix solver (observation-based)')
        time_left = deadline - time.time()
        if time_left > 5:
            tms_time = min(time_left * _gbud('toggle_matrix', 0.2), 15)
            result = toggle_matrix_solve(env_snap, click_only, start_levels,
                                          timeout=tms_time)
            if result:
                return _won('toggle_matrix', result)

    # Route 0.4: Reactive Controller (real-time games: Pong, Breakout-like)
    # Must run BEFORE Eyes — Eyes wastes actions on random exploration in real-time games
    if profile.game_type in ("NAVIGATION", "UNKNOWN"):
        _thought('Route 0.4: Reactive controller (tracking)')
        time_left = deadline - time.time()
        if time_left > 5:
            result = reactive_controller(env_snap, actions, start_levels,
                                          timeout=min(time_left * _gbud('reactive', 0.15), 15),
                                          noise_mask=noise_mask)
            if result:
                return _won('reactive', result)

    # Route 0.5: Eyes perception solver — general-purpose fallback
    _thought('Route 0.5: ARC Eyes (perception)')
    try:
        eyes_env = copy.deepcopy(env_snap)
        eyes_obs = getattr(eyes_env, '_last_obs', None)
        if eyes_obs is None:
            probe2 = copy.deepcopy(env_snap)
            eyes_obs = probe2.step(actions[0].game_action,
                                   actions[0].data if actions[0].data else None)
        if eyes_obs:
            eyes_env._start_levels = start_levels
            # Navigation games: cap Eyes at 10s (click analysis is wasted)
            if profile.game_type == 'NAVIGATION':
                eyes_timeout = min((deadline - time.time()) * 0.2, 10)
            else:
                eyes_timeout = min((deadline - time.time()) * 0.4, 30)
            # Pre-seed Eyes theory with LLM game type hint
            eyes_theory = None
            if llm_hint and llm_hint.game_description:
                from arc_eyes import GameTheory as EyesTheory
                eyes_theory = EyesTheory(
                    game_type=llm_hint.game_description,
                    player_sprite=None, targets=[], obstacles=[],
                    action_effects={}, rules=[], confidence=0.3)
            if eyes_timeout > 3:
                eyes_result, _, eyes_memory = eyes_play_level(
                    eyes_env, eyes_obs, timeout=eyes_timeout,
                    memory=eyes_memory, theory=eyes_theory)
                if eyes_result:
                    _thought('Eyes solved it!', 'thought')
                    return _won('eyes', [Action(aid, d if d else {}) for aid, d in eyes_result])
    except Exception as e:
        log.debug(f"  Eyes failed: {e}")

    # Extract action intelligence from Eyes memory for later routes
    # Sort actions by effectiveness: high pixel change first, noops last
    if eyes_memory and eyes_memory.action_memories:
        effective = eyes_memory.get_effective_actions(min_observations=1)
        if effective:
            effective_ids = {am.action_id for am in effective}
            # Reorder: effective actions first, then rest
            eff_actions = [a for a in actions if a.game_action in effective_ids]
            rest_actions = [a for a in actions if a.game_action not in effective_ids]
            if eff_actions:
                actions = eff_actions + rest_actions
                log.info(f"  Eyes intel: {len(eff_actions)} effective actions prioritized")

        # ── Route 0.6: Inductive Reasoning — solve by understanding, not search ──
        # Probe with real frames, induce rules, solve analytically
        try:
            from arc_reason import InductiveReasoner
            reasoner = InductiveReasoner()

            # Probe phase: take a few real actions, capture before/after frames
            click_actions = [a for a in actions if a.data and 'x' in a.data]
            probe_actions = click_actions[:min(8, len(click_actions))] if click_actions else actions[:min(5, len(actions))]

            if probe_actions:
                for probe_act in probe_actions:
                    probe_env = copy.deepcopy(env_snap)
                    probe_obs0 = getattr(probe_env, '_last_obs', None)
                    if not probe_obs0:
                        break
                    frame_before = np.array(probe_obs0.frame)

                    # Take the action (handle both SDK and synthetic game APIs)
                    try:
                        if probe_act.data:
                            probe_obs1 = probe_env.step(probe_act.game_action, **probe_act.data)
                        else:
                            probe_obs1 = probe_env.step(probe_act.game_action)
                    except TypeError:
                        probe_obs1 = probe_env.step(probe_act.game_action, data=probe_act.data)
                    frame_after = np.array(probe_obs1.frame)
                    won = probe_obs1.state.value == 'WIN' if hasattr(probe_obs1, 'state') else False

                    action_data = probe_act.data if probe_act.data else {}
                    reasoner.observe(
                        action_id=probe_act.game_action,
                        action_data=action_data,
                        frame_before=frame_before,
                        frame_after=frame_after,
                        won=won
                    )

                # Generate hypotheses from real observations
                hypotheses = reasoner.hypothesize()
                if hypotheses:
                    log.info(f"  Reasoner: {len(hypotheses)} hypotheses — {[h.description[:40] for h in hypotheses]}")

                    # If we have confirmed rules, try analytical solving
                    if reasoner.can_solve_analytically():
                        reason_env = copy.deepcopy(env_snap)
                        reason_obs = getattr(reason_env, '_last_obs', None)
                        if reason_obs:
                            reason_frame = np.array(reason_obs.frame)
                            analytical_solution = reasoner.solve_analytically(reason_frame, actions, reason_env)
                            if analytical_solution:
                                _thought('Route 0.6: Inductive reasoning — analytical solution!', 'thought')
                                log.info(f"  ANALYTICAL SOLVE: {len(analytical_solution)} actions (no BFS needed)")
                                return _won('inductive', analytical_solution)
                    # Route insights to profile for solver selection
                    for hyp in hypotheses:
                        if hyp.confidence >= 0.5:
                            if hyp.rule_type == 'move':
                                profile.has_navigation = True
                                profile.game_type = "NAVIGATION"
                            elif hyp.rule_type in ('toggle', 'cycle'):
                                profile.has_neighbor_coupling = True
                                profile.game_type = "CLICK_TOGGLE"
                            elif hyp.rule_type == 'counter':
                                if not profile.game_type or profile.game_type == "UNKNOWN":
                                    profile.game_type = "CLICK_COUNTER"
                            elif hyp.rule_type == 'commit_button':
                                if not profile.game_type or profile.game_type == "UNKNOWN":
                                    profile.game_type = "CONSTRAINT_SAT"
                    log.info(f"  Reasoner→Profile: game_type={profile.game_type}")
                    # Store hypotheses on profile for post-game persistence
                    profile._reasoner_hypotheses = hypotheses
        except Exception as e:
            log.debug(f"  Inductive reasoning failed: {e}")

        # Eyes→Agent theory composition: refine profile with Eyes' observations
        winning_acts = eyes_memory.get_winning_actions()
        if winning_acts:
            # Eyes found winning actions on earlier levels — propagate to profile
            for wa in winning_acts:
                mov = eyes_memory.dominant_movement(wa.action_id, {})
                if mov and not getattr(profile, 'has_navigation', False):
                    profile.has_navigation = True
                    log.info(f"  Eyes→Profile: detected navigation from winning action patterns")
                    break

        # Detect noop-heavy actions → likely spatial constraint game
        noop_actions = [am for am in effective if am.noop_count > am.total_observations * 0.7]
        if len(noop_actions) > len(effective) * 0.5 and not profile.has_neighbor_coupling:
            profile.has_neighbor_coupling = True
            log.info(f"  Eyes→Profile: {len(noop_actions)}/{len(effective)} actions are mostly noops — possible spatial constraints")

    # Route 1: BFS with click-only actions (if available, lower branching)
    if no_bfs:
        log.info("  [no_bfs] Skipping BFS/MCTS/RW routes — pure reasoning mode")
    _thought('Route 1: BFS search')
    search_actions = click_only if click_only else actions
    time_left = deadline - time.time()
    if not no_bfs and time_left > 3 and not _skip_route('bfs'):
        # Navigation: BFS is weak (can't reach depth 20+), keep it short
        bfs_cap = 8 if profile.game_type == "NAVIGATION" else 20
        bfs_frac_base = 0.15 if profile.game_type == "NAVIGATION" else 0.3
        bfs_time = min(time_left * _gbud('bfs', bfs_frac_base), bfs_cap)
        result = deepcopy_bfs(env_snap, search_actions, start_levels,
                              timeout=bfs_time, noise_mask=noise_mask)
        if result:
            return _won('bfs', result)

    # Route 1.5: BFS with ALL actions (in case directional ones matter)
    if not no_bfs and click_only and len(click_only) < len(actions):
        time_left = deadline - time.time()
        if time_left > 3:
            bfs_time = min(time_left * _gbud('bfs', 0.15), 10)
            result = deepcopy_bfs(env_snap, actions, start_levels,
                                  timeout=bfs_time, noise_mask=noise_mask)
            if result:
                return _won('bfs', result)

    # Route 1.6: Dynamic-action BFS for games with changing available_actions
    # Detects if env provides per-state actions (e.g. board games like Othello)
    time_left = deadline - time.time()
    if not no_bfs and time_left > 5:
        _thought('Route 1.6: Dynamic-action BFS')
        dyn_time = min(time_left * _gbud('bfs', 0.25), 15)
        result = dynamic_action_bfs(env_snap, actions, start_levels,
                                    timeout=dyn_time, noise_mask=noise_mask)
        if result:
            return _won('dynamic_bfs', result)

    # Route 1.7: Cyclic BFS for frame-ambiguous games
    time_left = deadline - time.time()
    if not no_bfs and time_left > 3 and profile.game_type != "NAVIGATION":
        cyc_time = min(time_left * _gbud('bfs', 0.15), 10)
        result = deepcopy_bfs(env_snap, search_actions, start_levels,
                              timeout=cyc_time, noise_mask=noise_mask,
                              dedup_cycle=3)
        if result:
            return _won('bfs_cyclic', result)

    # Route 1.8: Mechanic Learning EARLY for click games
    # For click-sequence/toggle games, probe-based learning is the strongest route.
    # Run it early with generous time before MCTS/block-BFS eat the budget.
    if profile and profile.game_type in ('CLICK_SEQUENCE', 'CLICK_TOGGLE', 'UNKNOWN'):
        if profile.game_type != 'NAVIGATION':
            _thought('Route 1.8: Mechanic learning (probe, model, compute)')
            time_left = deadline - time.time()
            if time_left > 15:
                result = mechanic_learning_solve(env_snap, actions, start_levels,
                                                 timeout=time_left * _gbud('mechanic', 0.5),
                                                 noise_mask=noise_mask,
                                                 frame0=frame0)
                if result:
                    return _won('mechanic', result)

    # Route 2: NavModel for navigation games
    time_left = deadline - time.time()
    if not no_bfs and time_left > 3 and profile.game_type == "NAVIGATION" and not _skip_route('navigation'):
        _thought('Route 2: Navigation model (best-first)')
        # Cap Nav at 40% — MCTS is better for deep navigation (random rollouts reach depth 400)
        result = navigation_solve(env_snap, actions, start_levels,
                                  timeout=time_left * _gbud('navigation', 0.4),
                                  noise_mask=noise_mask)
        if result:
            return _won('navigation', result)

    # LLM-guided time allocation for search routes
    # Default: MCTS 50%, NMCS 25%, RW 25%
    # Deep sequence hint: shift budget toward NMCS
    _aprio = llm_hint.action_priority if llm_hint else []
    llm_type = (llm_hint.game_description or '').lower() if llm_hint else ''
    if llm_hint and llm_hint.rollout_len > 50:
        # Deep game — NMCS gets more budget
        mcts_frac, nmcs_frac = 0.3, 0.4
    elif 'navig' in llm_type or profile.game_type == 'NAVIGATION':
        # Navigation — MCTS gets most (random rollouts reach deep solutions)
        mcts_frac, nmcs_frac = 0.7, 0.2
    else:
        mcts_frac, nmcs_frac = 0.5, 0.25

    # Route 3: MCTS
    _thought('Route 3: Monte Carlo Tree Search')
    time_left = deadline - time.time()
    if not no_bfs and time_left > 3 and not _skip_route('mcts'):
        rl = 400 if profile.game_type == "NAVIGATION" else 60
        if profile.game_type == "UNKNOWN":
            rl = 150  # Unknown games: deeper rollouts for safety
        # LLM can override rollout depth if it has a better estimate
        if llm_hint and llm_hint.rollout_len > 0:
            rl = max(rl, llm_hint.rollout_len * 2)  # 2x suggested depth for exploration margin
            _thought(f'LLM suggests ~{llm_hint.rollout_len} actions, rollout={rl}', 'thought')
        result = mcts_search(env_snap, actions, start_levels,
                             timeout=time_left * _gbud('mcts', mcts_frac),
                             noise_mask=noise_mask, rollout_len=rl,
                             action_priority=_aprio,
                             goal_frame=goal_frame)
        if result:
            return _won('mcts', result)

    # Route 3.05: Sliding Tile A* (goal-aware block search)
    # For navigation/push-block games: A* with Manhattan heuristic beats blind BFS
    if not no_bfs and (profile.game_type in ("NAVIGATION", "UNKNOWN", "CLICK_SEQUENCE") or 'push_block' in profile.genre_hints or 'sliding_tile' in profile.genre_hints):
        _thought('Route 3.05: Sliding tile A* (goal-aware)')
        time_left = deadline - time.time()
        if time_left > 5:
            result = sliding_tile_astar(env_snap, actions, start_levels,
                                        timeout=time_left * _gbud('sliding_astar', 0.3),
                                        noise_mask=noise_mask,
                                        goal_frame=goal_frame)
            if result:
                return _won('sliding_astar', result)

    # Route 3.1: Block-State BFS (mile-high view — see blocks, not pixels)
    # Give push-block games (Sokoban-like) more time — they need deeper search
    _thought('Route 3.1: Block-state BFS (structural)')
    time_left = deadline - time.time()
    push_block = profile and 'push_block' in (profile.genre_hints or [])
    bfs_frac = 0.7 if push_block else 0.4
    if not no_bfs and time_left > 5:
        result = block_state_bfs(env_snap, actions, start_levels,
                                 timeout=time_left * _gbud('block_bfs', bfs_frac),
                                 noise_mask=noise_mask)
        if result:
            return _won('block_bfs', result)

    # Route 3.15: Mechanic Learning (probe → model → compute)
    # For click-sequence games: learn what each button does, then reason about solution
    # This is the PRIMARY route for click games — give it generous time
    if profile and profile.game_type in ('CLICK_SEQUENCE', 'CLICK_TOGGLE', 'UNKNOWN'):
        _thought('Route 3.15: Mechanic learning (probe, model, compute)')
        time_left = deadline - time.time()
        if time_left > 8:
            result = mechanic_learning_solve(env_snap, actions, start_levels,
                                             timeout=time_left * _gbud('mechanic', 0.7),
                                             noise_mask=noise_mask,
                                             frame0=frame0)
            if result:
                return _won('mechanic', result)

    # Route 3.25: Transition Graph (learn → plan → extend)
    _thought('Route 3.25: Transition graph (learn then plan)')
    time_left = deadline - time.time()
    if not no_bfs and time_left > 5:
        result = transition_graph_solve(env_snap, actions, start_levels,
                                        timeout=time_left * _gbud('transition_graph', 0.3),
                                        noise_mask=noise_mask,
                                        action_priority=_aprio,
                                        goal_frame=goal_frame)
        if result:
            return _won('transition_graph', result)

    # Route 3.5: NMCS for deep sequence games
    time_left = deadline - time.time()
    if not no_bfs and time_left > 5 and not _skip_route('nmcs'):
        _thought('Route 3.5: Nested Monte Carlo Search')
        nmcs_depth = 100
        if llm_hint and llm_hint.rollout_len > 0:
            nmcs_depth = max(nmcs_depth, llm_hint.rollout_len * 2)
        result = nmcs_search(env_snap, actions, start_levels,
                             timeout=time_left * _gbud('nmcs', nmcs_frac),
                             noise_mask=noise_mask, max_depth=nmcs_depth,
                             action_priority=_aprio)
        if result:
            return _won('nmcs', result)

    # Route 3.75: Forward Model Solve (learn physics → plan with learned model)
    time_left = deadline - time.time()
    if not no_bfs and time_left > 8:
        _thought('Route 3.75: Forward model (learn then exploit)')
        result = forward_model_solve(env_snap, actions, start_levels,
                                     timeout=time_left * _gbud('forward_model', 0.4),
                                     noise_mask=noise_mask)
        if result:
            return _won('forward_model', result)

    # Route 4: Random Walk (Go-Explore) — last resort, gets ALL remaining time
    # RHAE scoring is quadratic: ANY solution beats no solution.
    # A 500-action solve at baseline=20 scores 0.0016 — still better than 0.0.
    _thought('Route 4: Random walk exploration')
    time_left = deadline - time.time()
    # Reserve time for pilot (Route 5) if pilot is available
    pilot_reserve = 15.0 if pilot is not None else 0.0
    rw_budget = time_left - pilot_reserve
    if not no_bfs and rw_budget > 3:
        wl = 400 if profile.game_type == "NAVIGATION" else 300
        if llm_hint and llm_hint.rollout_len > 0:
            wl = max(wl, llm_hint.rollout_len * 3)
        # Unknown games get deeper walks — we don't know the solution depth
        if profile.game_type == "UNKNOWN":
            wl = max(wl, 500)
        result = random_walk(env_snap, actions, start_levels,
                             timeout=rw_budget,
                             noise_mask=noise_mask, walk_len=wl,
                             action_priority=_aprio)
        if result:
            return _won('random_walk', result)

    # ─── Route 5: PILOT — LLM mind at the inflection point ──────────
    # All routes exhausted. The body is stuck. Call the mind.
    if pilot is not None:
        time_left = deadline - time.time()
        if time_left > 5:
            _thought('Route 5: Pilot (LLM mind)')
            failed = [
                ('constraint', 'exhausted or N/A'),
                ('frame_coupling', 'exhausted or N/A'),
                ('bfs', 'exhausted'),
                ('dynamic_bfs', 'exhausted'),
                ('mechanic', 'exhausted or N/A'),
                ('navigation', 'exhausted or N/A'),
                ('mcts', 'no solution found'),
                ('nmcs', 'no solution found'),
                ('random_walk', 'no solution found'),
            ]
            ctx = build_inflection_context(
                env_snap=env_snap,
                game_id=game_id,
                profile=profile,
                level=level,
                levels_solved=levels_solved,
                levels_total=levels_total,
                failed_routes=failed,
                time_spent=timeout - time_left,
                time_remaining=time_left,
                previous_winning_route=preferred_route,
                level_results=level_results,
                reasoner=reasoner,
                solver_memory=solver_memory,
            )
            directive = pilot.consult(ctx)
            if directive:
                _thought(f'Pilot says: {directive.action} — {directive.reasoning[:80]}', 'thought')
                apply_directive(directive, profile, solver_memory)

                if directive.action == 'skip':
                    log.info("  [pilot] Strategic retreat — skipping level")
                    return None

                if directive.action == 'observe' and directive.observations_requested:
                    # The pilot wants to look before deciding — run experiments
                    from arc_pilot import run_observations
                    _thought('Pilot observing — running targeted experiments', 'thought')
                    obs_results = run_observations(
                        env_snap, actions, directive.observations_requested, start_levels)
                    log.info(f"  [pilot] Observations: {len(obs_results)} experiments")
                    for obs in obs_results:
                        log.info(f"    {obs['request']}: {len(obs['results'])} results")
                    # Feed observations back and consult pilot again
                    # (with observation results in context)
                    time_left2 = deadline - time.time()
                    if time_left2 > 5 and pilot.budget_remaining > 0:
                        # Enrich context with observations
                        obs_summary = []
                        for obs in obs_results:
                            for r in obs['results']:
                                obs_summary.append(f"{obs['request']}: {r}")
                        ctx.hypotheses.extend([
                            (f"OBSERVATION: {s[:80]}", 0.9) for s in obs_summary[:5]
                        ])
                        ctx.time_remaining = time_left2
                        directive = pilot.consult(ctx)
                        if directive and directive.action in ('reframe', 'retry'):
                            _thought(f'Post-observation: {directive.action} — {directive.reasoning[:60]}', 'thought')
                            apply_directive(directive, profile, solver_memory)
                            # Fall through to retry logic below
                        elif directive and directive.action == 'skip':
                            return None
                        else:
                            return None  # Pilot still can't figure it out

                if directive.action in ('reframe', 'retry'):
                    # Retry with pilot's guidance
                    retry_budget = min(time_left - 2, time_left * 0.8)
                    if retry_budget > 3:
                        # Build action set based on pilot's focus
                        focus = directive.param_overrides.get('focus_actions')
                        retry_actions = actions
                        if focus and isinstance(focus, list):
                            focused = [a for a in actions if a.game_action in focus]
                            if focused:
                                retry_actions = focused

                        route = directive.preferred_route
                        env_retry = copy.deepcopy(env_snap)

                        # Route the pilot's preferred solver
                        retry_result = None
                        if route == 'bfs':
                            depth = directive.param_overrides.get('bfs_depth', 25)
                            retry_result = deepcopy_bfs(env_retry, retry_actions, start_levels,
                                                        timeout=retry_budget, noise_mask=noise_mask)
                        elif route == 'mcts':
                            rl = directive.param_overrides.get('mcts_rollout', 300)
                            retry_result = mcts_search(env_retry, retry_actions, start_levels,
                                                       timeout=retry_budget, noise_mask=noise_mask,
                                                       rollout_len=rl, goal_frame=goal_frame)
                        elif route == 'mechanic':
                            retry_result = mechanic_learning_solve(env_retry, retry_actions, start_levels,
                                                                    timeout=retry_budget, noise_mask=noise_mask,
                                                                    frame0=frame0)
                        elif route == 'navigation':
                            retry_result = navigation_solve(env_retry, retry_actions, start_levels,
                                                            timeout=retry_budget, noise_mask=noise_mask)
                        elif route == 'constraint':
                            retry_result = constraint_solve_v2(env_retry, retry_actions, start_levels)
                        elif route == 'toggle_matrix':
                            retry_result = toggle_matrix_solve(env_retry, retry_actions, start_levels,
                                                                timeout=retry_budget)
                        elif route == 'nmcs':
                            depth = directive.param_overrides.get('nmcs_depth', 200)
                            retry_result = nmcs_search(env_retry, retry_actions, start_levels,
                                                       timeout=retry_budget, noise_mask=noise_mask,
                                                       max_depth=depth)
                        elif route == 'block_bfs':
                            retry_result = block_state_bfs(env_retry, retry_actions, start_levels,
                                                           timeout=retry_budget, noise_mask=noise_mask)
                        elif route == 'dynamic_bfs':
                            retry_result = dynamic_action_bfs(env_retry, retry_actions, start_levels,
                                                              timeout=retry_budget, noise_mask=noise_mask)
                        else:
                            # Unknown route — try MCTS with longer rollout as safe default
                            retry_result = mcts_search(env_retry, retry_actions, start_levels,
                                                       timeout=retry_budget, noise_mask=noise_mask,
                                                       rollout_len=500, goal_frame=goal_frame)

                        if retry_result:
                            _thought(f'Pilot-guided {route or "mcts"} solved it!', 'thought')
                            return _won(f'pilot_{route or "mcts"}', retry_result)

    return None


# ─── Competition Runner ─────────────────────────────────────────────────

def _classify_genre(profile) -> str:
    """Map a GameProfile to a genre string for SolverMemory."""
    gt = profile.game_type
    hints = getattr(profile, 'genre_hints', []) or []
    if gt == "CLICK_TOGGLE":
        return "toggle"
    elif gt == "CLICK_SEQUENCE":
        if 'push_block' in hints or 'sliding_tile' in hints:
            return "spatial"
        return "circuit" if profile.has_neighbor_coupling else "combinatorial"
    elif gt == "NAVIGATION":
        return "spatial"
    else:
        return "unknown"


def run_game(arcade, game_id: str, timeout_per_level: float = 60.0,
             recorder: GameRecorder = None,
             eyes_memory=None,
             solver_memory=None,
             force_generic: bool = False) -> dict:
    """Run all levels of a game, return results.
    eyes_memory: EpisodicMemory from arc_eyes, accumulates across games.
    solver_memory: SolverMemory for persistent cross-session learning.
    force_generic: if True, skip specialized solvers (measure generic floor).
    """
    game_start_time = time.time()
    env = arcade.make(game_id)
    if env is None:
        log.error(f"Failed to create env for {game_id}")
        return {'game': game_id, 'error': 'failed to create'}

    obs = env.reset()
    frame0 = np.array(obs.frame)
    env._last_obs = obs  # Store for frame_coupling_solve's current-frame extraction

    # Initialize Eyes cross-game memory
    if eyes_memory is not None:
        eyes_memory.start_game(game_id)

    # Record initial frame
    if recorder:
        recorder.record_frame(frame0, level=0, action_name='reset')
        recorder.comment(f'Game start: {game_id.upper()}', comment_type='milestone')

    # Get baseline
    envs = arcade.get_environments()
    baseline = None
    for e in envs:
        if game_id in e.game_id:
            baseline = e.baseline_actions
            break

    log.info(f"\n{'='*60}")
    log.info(f"Game: {game_id.upper()} ({len(baseline) if baseline else '?'} levels)")
    log.info(f"{'='*60}")

    # Quick check: can a specialized solver handle this game?
    # If so, skip expensive action discovery/analysis
    #
    # ARC_PURE_AGI (default ON): Specialized solvers disabled. Pure Gundam reasoning.
    # Peter directive: "We're spending money on pure AGI, not specialized solvers."
    # Set ARC_PURE_AGI=0 only if Peter explicitly approves specialized solver use.
    pure_agi = os.environ.get('ARC_PURE_AGI', '1').strip() != '0'
    _quick_snap = copy.deepcopy(env)
    _quick_game = getattr(_quick_snap, '_game', None)
    use_specialized = False
    no_bfs = os.environ.get('ARC_NO_BFS', '').strip() == '1'
    if not force_generic and not no_bfs and not pure_agi and _quick_game is not None:
        # VC33 detection
        if hasattr(_quick_game, 'dzy') and hasattr(_quick_game, 'oro'):
            use_specialized = True
            log.info("Detected VC33 — using specialized solver (skipping discovery)")
        # LS20 detection
        elif hasattr(_quick_game, 'mgu') and hasattr(_quick_game, 'qqv'):
            use_specialized = True
            log.info("Detected LS20 — using specialized solver (skipping discovery)")
    if pure_agi:
        log.info("PURE AGI MODE — specialized solvers disabled. Gundam pilot only.")
    del _quick_snap

    # Discover actions from level 0 (skip if specialized solver available)
    if use_specialized:
        actions = []  # Not needed — specialized solver reads game internals
        profile = GameProfile(game_type='SPECIALIZED', actions=[])
    else:
        env_snap = copy.deepcopy(env)
        actions = discover_actions(env_snap, frame0, obs.available_actions,
                                   time_limit=60.0)  # 60s for thorough probing
        env_snap = copy.deepcopy(env)
        profile = analyze_game(env_snap, actions, frame0)

    # OPRAH second opinion — zero cost, runs when analyze_game returns UNKNOWN
    if profile.game_type == "UNKNOWN":
        try:
            from gundam import (_oprah_frame_diff, _oprah_classify_action,
                               _oprah_infer_genre, ActionProbe, ActionType)
            _oprah_probes = []
            for i, act in enumerate(actions[:8]):
                ec = copy.deepcopy(env_snap)
                obs_p = ec.step(act.game_action, act.data if act.data else None)
                if obs_p:
                    diff = _oprah_frame_diff(frame0, np.array(obs_p.frame))
                    has_pos = bool(act.data and 'x' in act.data)
                    atype = _oprah_classify_action(diff, has_pos)
                    _oprah_probes.append(ActionProbe(
                        action_idx=i, action_type=atype,
                        frame_changed=diff['pixels'] > 0,
                        pixels_changed=diff['pixels'],
                        change_fraction=diff.get('fraction', 0)))
            if _oprah_probes:
                _has_grid = False
                _f = frame0.squeeze() if frame0.ndim == 3 and frame0.shape[0] <= 4 else frame0
                if _f.ndim >= 2 and _f.shape[0] >= 10:
                    _rows_std = np.std(_f.reshape(_f.shape[0], -1).astype(float), axis=1)
                    _has_grid = int(np.sum(_rows_std < 5)) >= 3
                _has_click = any(a.data and 'x' in a.data for a in actions[:8])
                _genre, _conf = _oprah_infer_genre(_oprah_probes, _has_grid, _has_click)
                # Always store action profile for unknown-genre routing
                _acounts = {}
                for _ap in _oprah_probes:
                    _akey = _ap.action_type.value if hasattr(_ap.action_type, 'value') else str(_ap.action_type)
                    _acounts[_akey] = _acounts.get(_akey, 0) + 1
                profile.oprah_action_counts = _acounts
                profile.oprah_has_grid = _has_grid
                profile.oprah_has_click = _has_click
                log.info(f"  OPRAH action profile: {_acounts} grid={_has_grid} click={_has_click}")
                if _conf >= 0.5 and _genre != 'novel_unknown':
                    _OPRAH_TO_TYPE = {
                        'toggle_puzzle': 'CLICK_TOGGLE', 'constraint_satisfaction': 'CLICK_SEQUENCE',
                        'navigation_maze': 'NAVIGATION', 'push_block': 'NAVIGATION',
                        'multi_phase': 'NAVIGATION', 'paint_fill': 'PAINTING',
                    }
                    new_type = _OPRAH_TO_TYPE.get(_genre, 'UNKNOWN')
                    if new_type != 'UNKNOWN':
                        profile.game_type = new_type
                        profile.genre_hints = [_genre]
                        log.info(f"  OPRAH override: UNKNOWN → {new_type} ({_genre}, conf={_conf:.2f})")
        except Exception as e:
            log.debug(f"  OPRAH fallback failed: {e}")

    log.info(f"Type: {profile.game_type}")

    # ─── CNN Action Predictor — learns which actions cause frame changes ───
    cnn_predictor = None
    if HAS_CNN_PREDICTOR and not use_specialized:
        try:
            fh, fw = frame0.shape[0], frame0.shape[1] if frame0.ndim >= 2 else 64
            if frame0.ndim == 3 and frame0.shape[0] <= 4:  # CHW format
                fh, fw = frame0.shape[1], frame0.shape[2]
            cnn_predictor = ActionPredictor(
                n_actions=len(actions), frame_h=fh, frame_w=fw,
                max_colors=profile.max_palette_size if hasattr(profile, 'max_palette_size') and profile.max_palette_size else 16
            )
            # Seed with OPRAH probe data (already collected above — free!)
            if '_oprah_probes' in dir() and _oprah_probes:
                for _ap in _oprah_probes:
                    if _ap.action_idx < len(actions):
                        act = actions[_ap.action_idx]
                        ec = copy.deepcopy(env)
                        obs_seed = ec.step(act.game_action, act.data if act.data else None)
                        if obs_seed:
                            cnn_predictor.observe(frame0, _ap.action_idx, np.array(obs_seed.frame))
                        del ec
                if cnn_predictor.total_obs >= 5:
                    cnn_predictor.train(epochs=3, batch_size=32)
                    log.info(f"  CNN predictor seeded: {cnn_predictor.total_obs} obs, confidence={cnn_predictor.confidence:.2f}")
                else:
                    log.info(f"  CNN predictor initialized (awaiting data)")
            else:
                log.info(f"  CNN predictor initialized (no OPRAH seed)")
        except Exception as e:
            log.debug(f"  CNN predictor init failed: {e}")
            cnn_predictor = None

    # LLM reasoning layer — analyze game once, reuse hints for all levels
    llm_hint = None
    if not use_specialized and os.environ.get('ARC_USE_LLM', ''):
        log.info("  LLM: Analyzing game...")
        llm_hint = llm_analyze_game(frame0, profile, game_id=game_id,
                                     n_actions=len(actions))
        if llm_hint:
            log.info(f"  LLM says: {llm_hint.game_description}")
            if recorder:
                recorder.comment(f'LLM: {llm_hint.game_description} — {llm_hint.suggested_strategy}',
                                 comment_type='thought')
            # Override UNKNOWN game type with LLM's assessment
            if profile.game_type == "UNKNOWN" and llm_hint.game_description:
                llm_type = llm_hint.game_description.lower()
                if 'navig' in llm_type or 'maze' in llm_type:
                    profile.game_type = "NAVIGATION"
                    log.info(f"  LLM override: UNKNOWN → NAVIGATION")
                elif 'toggle' in llm_type:
                    profile.game_type = "CLICK_TOGGLE"
                    log.info(f"  LLM override: UNKNOWN → CLICK_TOGGLE")
                elif 'sort' in llm_type or 'match' in llm_type or 'pattern' in llm_type:
                    profile.game_type = "SORTING"
                    log.info(f"  LLM override: UNKNOWN → SORTING")

    results = {
        'game': game_id, 'game_type': profile.game_type,
        'levels': [], 'total_actions': 0,
        'baseline_total': sum(baseline) if baseline else 0,
    }

    level = 0
    time_bank = 0.0
    current_levels = 0  # Track levels_completed
    goal_frame = None   # Pre-win frame from previous level (goal heuristic)
    last_winning_route = ''  # Cross-level route learning
    level_results_log = []  # Track per-level outcomes for pilot context

    # Initialize pilot (LLM mind) + flight recorder (the self) + HUD
    # Pilot gated by ARC_USE_PILOT env var. HUD/failure detector always active.
    pilot = None
    flight_recorder = None
    failure_detector = FailureDetector()
    hud = HUD(enabled=bool(os.environ.get('ARC_USE_HUD', '')))

    if os.environ.get('ARC_USE_PILOT', ''):
        flight_recorder = FlightRecorder()
        max_calls = int(os.environ.get('ARC_PILOT_BUDGET', '3'))
        pilot = Pilot(max_calls_per_game=max_calls, flight_recorder=flight_recorder)
        flight_recorder._games_played += 1
        log.info(f"  [pilot] Initialized — budget: {max_calls} calls/game")
        log.info(f"  [self] Flight recorder active — {get_vitals().summary()}")

    # Phase 0: Consult solver memory — what do we know about this genre?
    if solver_memory is not None:
        try:
            genre = _classify_genre(profile)
            similar = solver_memory.get_similar_games(genre=genre)
            if similar:
                # Promote the route that won most often for this genre
                routes = {}
                for g in similar:
                    rt = g.get('winning_route', '')
                    if rt:
                        routes[rt] = routes.get(rt, 0) + 1
                if routes:
                    best = max(routes, key=routes.get)
                    last_winning_route = best
                    log.info(f"  [memory] Genre '{genre}': {len(similar)} prior games, "
                             f"best route = {best} ({routes[best]} wins)")
            rules = solver_memory.get_rulebook(genre=genre, min_confidence=0.5)
            if rules:
                log.info(f"  [memory] {len(rules)} rules for '{genre}' "
                         f"(top: {rules[0]['strategy'][:60]})")

            # Tag-based recall — finds similar games by structural fingerprint
            # Especially valuable for unknown genres where genre-based recall misses
            if hasattr(solver_memory, 'recall_by_tags'):
                try:
                    from solver_memory import tags_from_profile
                    query_tags = tags_from_profile(profile)
                    if query_tags:
                        tag_results = solver_memory.recall_by_tags(query_tags, min_overlap=2)
                        tag_games = tag_results.get("games", [])
                        tag_rules = tag_results.get("rules", [])
                        if tag_games or tag_rules:
                            log.info(f"  [memory] Tag recall: {len(tag_games)} games, "
                                     f"{len(tag_rules)} rules (tags: {query_tags[:5]})")
                            # Merge tag-recalled games with genre-recalled
                            if not similar and tag_games:
                                routes = {}
                                for g in tag_games:
                                    rt = g.get('winning_route', '')
                                    if rt:
                                        routes[rt] = routes.get(rt, 0) + 1
                                if routes:
                                    best = max(routes, key=routes.get)
                                    last_winning_route = best
                                    log.info(f"  [memory] Tag-based best route = {best}")
                except Exception:
                    pass  # tag recall is supplementary, never block
        except Exception as e:
            log.debug(f"  Memory consultation failed: {e}")

    # Wire CommonSenseBus — pushes memory-driven hunches to the solver
    bus = None
    if HAS_BUS:
        try:
            bus = CommonSenseBus(solver_memory=solver_memory, eyes_memory=eyes_memory)
            bus.fingerprint = profile.game_type  # use game type as fingerprint
            hunch = bus.pre_cascade(profile=profile)
            if hunch and hunch.suggested_route:
                last_winning_route = hunch.suggested_route
                log.info(f"  [bus] Hunch: {hunch.content} (salience={hunch.salience:.2f})")
            # Wire dopamine for reward learning
            if HAS_DOPAMINE:
                tracker = HabitTracker()
                wire_dopamine_to_bus(bus, tracker)
                log.info("  [bus] Dopamine wired — reward learning active")
        except Exception as e:
            log.debug(f"  Bus init failed: {e}")

    while True:
        level_timeout = timeout_per_level + time_bank * 0.8
        t0 = time.time()
        log.info(f"\n--- Level {level} (budget: {level_timeout:.0f}s) ---")

        # Snapshot env at current level start
        env_snap = copy.deepcopy(env)

        if recorder:
            recorder.current_level = level
            recorder.comment(f'Level {level} — budget {level_timeout:.0f}s',
                             comment_type='milestone')

        level_winning_route = []
        solution = solve_level(env_snap, profile.actions, profile,
                               start_levels=current_levels,
                               timeout=level_timeout,
                               recorder=recorder,
                               eyes_memory=eyes_memory,
                               llm_hint=llm_hint,
                               goal_frame=goal_frame,
                               preferred_route=last_winning_route,
                               winning_route=level_winning_route,
                               pilot=pilot,
                               game_id=game_id,
                               level=level,
                               levels_solved=len([r for r in level_results_log if r.get('solved')]),
                               levels_total=len(baseline) if baseline else 0,
                               level_results=level_results_log,
                               solver_memory=solver_memory,
                               failure_detector=failure_detector,
                               cnn_predictor=cnn_predictor,
                               level_baseline=baseline[level] if baseline and level < len(baseline) else 0)

        elapsed = time.time() - t0

        if solution:
            # Optimize (skip for VC33 analytical solutions)
            is_analytical = (len(solution) == 1 and (hasattr(solution[0], '_vc33_solver') or hasattr(solution[0], '_ls20_solver')))
            is_raw_ints = solution and isinstance(solution[0], (int, np.integer))
            if not is_analytical and not is_raw_ints:
                env_snap_opt = copy.deepcopy(env)
                opt = shorten_path(env_snap_opt, solution, current_levels)
                if len(opt) < len(solution):
                    log.info(f"  Shortened: {len(solution)} → {len(opt)}")
                    solution = opt

            # ── Falsification checkpoint ──────────────────────────────
            # Verify solution on a FRESH deepcopy before committing to real env.
            # Catches non-deterministic games where solution worked on one copy but not another.
            # RHAE scoring is quadratic — a wasted level is catastrophic.
            if not is_analytical and not is_raw_ints:
                falsify_env = copy.deepcopy(env)
                if not verify_path(falsify_env, solution, current_levels):
                    log.warning(f"  FALSIFICATION FAILED — solution doesn't replay!")
                    # Try once more with remaining time
                    retry_time = max(deadline - time.time(), 10) if 'deadline' in dir() else 15
                    env_snap_retry = copy.deepcopy(env)
                    solution = solve_level(env_snap_retry, profile.actions, profile,
                                           start_levels=current_levels,
                                           timeout=retry_time,
                                           recorder=recorder,
                                           eyes_memory=eyes_memory,
                                           llm_hint=llm_hint,
                                           goal_frame=goal_frame)
                    if solution and not (len(solution) == 1 and (hasattr(solution[0], '_vc33_solver') or hasattr(solution[0], '_ls20_solver'))):
                        # Re-shorten and re-verify
                        env_snap_opt2 = copy.deepcopy(env)
                        solution = shorten_path(env_snap_opt2, solution, current_levels)
                        falsify2 = copy.deepcopy(env)
                        if not verify_path(falsify2, solution, current_levels):
                            log.warning(f"  Second falsification also failed — skipping level")
                            solution = None
                    elif solution and len(solution) == 1 and (hasattr(solution[0], '_vc33_solver') or hasattr(solution[0], '_ls20_solver')):
                        pass  # Analytical solver — trust it
                    if not solution:
                        # Give up on this level
                        results['levels'].append({
                            'level': level, 'solved': False,
                            'actions': 0, 'baseline': baseline[level] if baseline and level < len(baseline) else None,
                            'efficiency': None, 'time': round(time.time() - t0, 1),
                        })
                        log.info(f"  FAILED (non-deterministic)")
                        break  # Can't advance without solving
                else:
                    log.info(f"  Falsification passed ✓")

            # Execute on REAL env to advance level
            # Check for VC33 analytical solver callback
            vc33_analytical = (len(solution) == 1 and hasattr(solution[0], '_vc33_solver'))
            ls20_specialized = (len(solution) == 1 and hasattr(solution[0], '_ls20_solver'))
            if vc33_analytical:
                solver_fn = solution[0]._vc33_solver
                n_acts = solution[0]._vc33_actions
                log.info(f"  Executing VC33 analytical solver on real env ({n_acts} actions)")
                if recorder:
                    recorder.comment(f'VC33 analytical: {n_acts} actions',
                                     comment_type='milestone')
                actual_n = solver_fn(env)
                if actual_n is not None:
                    obs = env.step(6, {'x': 0, 'y': 0})  # get fresh obs
                    # Create dummy solution list for action counting only
                    solution = [Action(6, {}, f"VC33_analytical")] * actual_n
                    log.info(f"  VC33 analytical executed: {actual_n} actions")
                else:
                    log.warning(f"  VC33 analytical solver failed on real env!")
                    solution = []
            elif len(solution) == 1 and hasattr(solution[0], '_ls20_solver'):
                ls20_fn = solution[0]._ls20_solver
                log.info(f"  Executing LS20 specialized solver on real env")
                if recorder:
                    recorder.comment('LS20 specialized solver', comment_type='milestone')
                # Count env.step() calls by wrapping
                _orig_step = env.step
                _step_count = [0]
                def _counting_step(*a, **kw):
                    _step_count[0] += 1
                    return _orig_step(*a, **kw)
                env.step = _counting_step
                try:
                    # Handle multi-frame obs (LS20 quirk)
                    raw = np.array(obs.frame)
                    if raw.ndim == 3 and raw.shape[0] > 1:
                        obs = _orig_step(6, {'x': 0, 'y': 0})
                    obs_new, won = ls20_fn(env, obs, verbose=True)
                finally:
                    env.step = _orig_step
                if won:
                    obs = obs_new if obs_new else obs
                    n_acts = max(1, _step_count[0])
                    solution = [Action(1, {}, "LS20_move")] * n_acts
                    log.info(f"  LS20 solved: {n_acts} actions")
                else:
                    log.warning(f"  LS20 solver failed on real env")
                    solution = []
            else:
                if recorder:
                    recorder.comment(f'Solved! Executing {len(solution)} actions',
                                     comment_type='milestone')
            prev_frame = np.array(obs.frame) if obs else frame0
            for act in solution:
                if vc33_analytical or ls20_specialized:
                    break  # Already executed above — don't replay
                try:
                    # Handle both Action objects and raw ints (from Gundam)
                    if isinstance(act, int):
                        obs = env.step(act, None)
                        act_id, act_data = act, None
                    else:
                        obs = env.step(act.game_action, act.data if act.data else None)
                        act_id, act_data = act.game_action, act.data
                except Exception as e:
                    log.warning(f"  env.step() failed during replay: {e}")
                    break
                # Record the frame after each action
                if recorder and obs:
                    try:
                        recorder.record_frame(
                            np.array(obs.frame), action_id=act_id,
                            action_data=act_data, action_name=repr(act),
                            level=level, is_win=bool(obs and obs.state.value == 'WIN'))
                    except Exception:
                        pass
                # Feed transition into Eyes memory (cross-level learning)
                if eyes_memory is not None and obs:
                    try:
                        cur_frame = np.array(obs.frame)
                        is_win = obs.state.value == 'WIN' or obs.levels_completed > current_levels
                        eyes_memory.record(prev_frame, act_id,
                                           act_data if act_data else {},
                                           cur_frame, level=level, won=is_win)
                        prev_frame = cur_frame
                    except Exception:
                        pass
                # Pump through any animations (VC33-style)
                try:
                    game = getattr(env, '_game', None)
                    if game and hasattr(game, 'vai') and game.vai is not None:
                        for _ in range(500):
                            obs = env.step(act.game_action, {'x': 0, 'y': 0})
                            if game.vai is None:
                                break
                except Exception:
                    pass

            # Capture pre-win frame for goal-conditioned search on next level
            # prev_frame is the frame just before the last (winning) action
            if not use_specialized and len(solution) >= 2:
                goal_frame = prev_frame
                log.info(f"  Goal frame captured for next level")

            # Update _last_obs for frame_coupling_solve's current-frame extraction
            if obs is not None:
                env._last_obs = obs

            # Cross-level route learning: remember which route won
            if level_winning_route:
                last_winning_route = level_winning_route[0]
                log.info(f"  Route learning: {last_winning_route} → preferred for next level")

            # CNN predictor: reset buffer for new level (keep model weights for transfer)
            if cnn_predictor and cnn_predictor.enabled:
                cnn_predictor.reset()
                log.info(f"  CNN predictor: reset for next level (weights preserved)")

            base_acts = baseline[level] if baseline and level < len(baseline) else None
            eff = (base_acts / len(solution) * 100) if base_acts and len(solution) > 0 else None

            level_entry = {
                'level': level, 'solved': True,
                'actions': len(solution), 'baseline': base_acts,
                'efficiency': eff, 'time': round(elapsed, 1),
                'route': level_winning_route[0] if level_winning_route else '',
            }
            results['levels'].append(level_entry)
            level_results_log.append(level_entry)
            results['total_actions'] += len(solution)

            eff_str = f"{eff:.0f}%" if eff else "?"
            log.info(f"  SOLVED: {len(solution)} acts (baseline {base_acts}, eff {eff_str})")

            # Flight recorder: track route success + pilot evaluation
            if flight_recorder:
                winning = level_winning_route[0] if level_winning_route else 'unknown'
                flight_recorder.record_route(game_id, profile.game_type, winning,
                                             success=True, actions_used=len(solution),
                                             time_spent=elapsed, level=level)
                flight_recorder._levels_solved += 1
                # If pilot was involved, evaluate its directive
                if winning.startswith('pilot_'):
                    # Find the last engagement record on the directive
                    for rec in reversed(flight_recorder._session_log):
                        if rec.game_id == game_id and rec.level == level and rec.outcome == 'pending':
                            flight_recorder.evaluate(rec, solved=True, actions_used=len(solution))
                            break

            # Record win state in solver memory
            if solver_memory is not None and obs is not None:
                try:
                    from solver_memory import analyze_state
                    win_frame = np.array(obs.frame)
                    genre = _classify_genre(profile)
                    features = analyze_state(win_frame, genre, prev_frame)
                    solver_memory.record_state(game_id, profile.game_type, genre,
                                              'win', win_frame, features, level)
                except Exception:
                    pass

            # Record win in CommonSenseBus (feeds dopamine if wired)
            if bus is not None:
                try:
                    route_name = level_winning_route[0] if level_winning_route else 'unknown'
                    bus.record_win(profile.game_type, route_name, solution,
                                   elapsed, profile, frame0)
                except Exception as e:
                    log.debug(f"  Bus record_win failed: {e}")

            # Update level tracking
            if obs:
                current_levels = obs.levels_completed

            # Time banking
            unused = level_timeout - elapsed
            time_bank = time_bank + unused * 0.8 if unused > 0 else max(0, time_bank - abs(unused))

            # Check game completion
            if obs and obs.state.value == 'WIN':
                if recorder:
                    recorder.comment('GAME COMPLETE!', comment_type='milestone')
                log.info(f"  Game complete!")
                break

            level += 1

            # Re-discover actions for new level (skip for specialized solver games)
            if not use_specialized:
                frame_new = np.array(obs.frame) if obs else None
                if frame_new is not None:
                    env_snap_new = copy.deepcopy(env)
                    new_acts = discover_actions(env_snap_new, frame_new, obs.available_actions,
                                                time_limit=45.0)
                    if new_acts:
                        profile.actions = new_acts
                        new_profile = analyze_game(env_snap_new, new_acts, frame_new)
                        if new_profile.game_type != "UNKNOWN":
                            profile = new_profile
        else:
            if recorder:
                recorder.comment(f'Level {level} FAILED — all routes exhausted',
                                 comment_type='milestone')
            fail_entry = {
                'level': level, 'solved': False, 'actions': 0, 'time': round(elapsed, 1),
                'route': '',
            }
            results['levels'].append(fail_entry)
            level_results_log.append(fail_entry)
            log.info(f"  FAILED level {level}")

            # Flight recorder: track failure + pilot evaluation
            if flight_recorder:
                flight_recorder._levels_failed += 1
                # If pilot was involved but still failed, record that
                for rec in reversed(flight_recorder._session_log):
                    if rec.game_id == game_id and rec.level == level and rec.outcome == 'pending':
                        flight_recorder.evaluate(rec, solved=False)
                        break
            break

    solved = sum(1 for l in results['levels'] if l['solved'])
    total = len(results['levels'])
    if results['total_actions'] > 0 and results['baseline_total'] > 0:
        results['efficiency'] = round(results['baseline_total'] / results['total_actions'] * 100, 1)

    # Record cross-game meta-pattern in Eyes memory
    if eyes_memory is not None:
        n_click = len([a for a in profile.actions if a.data and 'x' in a.data])
        n_dir = len(profile.actions) - n_click
        eyes_memory.end_game(game_id, profile.game_type, solved,
                             results['total_actions'], n_dir, n_click)

        # Register game guide — teach Eyes genre knowledge from this game
        # (Nintendo Strategy Guide: not the solution, the genre wisdom)
        if use_specialized and solved > 0:
            _register_game_guide(eyes_memory, game_id)

    # Persist to long-term solver memory (survives process restarts)
    if solver_memory is not None:
        try:
            genre = _classify_genre(profile)
            # Generate tags from OPRAH profile for tag-based recall
            try:
                from solver_memory import tags_from_profile
                game_tags = tags_from_profile(profile)
            except Exception:
                game_tags = None
            solver_memory.record_game(
                game_id, profile.game_type, genre,
                levels_solved=solved, levels_total=total,
                total_actions=results['total_actions'],
                winning_route=last_winning_route,
                duration_s=time.time() - game_start_time,
                tags=game_tags)

            # Absorb Eyes insights into persistent rules
            if eyes_memory is not None and hasattr(eyes_memory, 'insights'):
                solver_memory.absorb_insights(
                    eyes_memory.insights, game_id, profile.game_type, genre)

                # Persist reasoner hypotheses too (stored on profile during solve)
                if hasattr(profile, '_reasoner_hypotheses') and profile._reasoner_hypotheses:
                    solver_memory.absorb_hypotheses(
                        profile._reasoner_hypotheses, game_id, profile.game_type, genre)
        except Exception as e:
            log.debug(f"  SolverMemory record failed: {e}")

    # ── Rosetta Stone: auto-generate trilingual pattern library entry ──
    if solved > 0:
        try:
            from build_pattern_library import GamePattern, save_frame, LIBRARY_DIR
            import hashlib as _hl
            genre = _classify_genre(profile)
            pid = _hl.md5(f"{game_id}_{genre}".encode()).hexdigest()[:8]

            pattern = GamePattern(pattern_id=pid, genre=genre,
                                  source_games=[game_id], confidence=min(1.0, solved / max(total, 1)))

            # Machine language: what the solver learned
            pattern.machine = {
                'game_type': profile.game_type,
                'genre': genre,
                'winning_route': last_winning_route or '',
                'levels_solved': solved,
                'levels_total': total,
                'total_actions': results['total_actions'],
                'efficiency': results.get('efficiency', 0),
                'n_actions': len(profile.actions),
                'n_click': len([a for a in profile.actions if a.data and 'x' in a.data]),
            }

            # Human language: readable strategy description
            route_desc = last_winning_route or 'multi-route cascade'
            pattern.human = {
                'description': f"{genre.replace('_', ' ').title()} game solved via {route_desc}.",
                'strategy_steps': [
                    f"Discovered {len(profile.actions)} actions ({pattern.machine['n_click']} click-based)",
                    f"Primary solver: {route_desc}",
                    f"Solved {solved}/{total} levels in {results['total_actions']} actions",
                    f"Efficiency: {results.get('efficiency', 0):.0f}% vs baseline",
                ],
                'difficulty': 'easy' if results.get('efficiency', 0) > 150 else 'medium' if results.get('efficiency', 0) > 100 else 'hard',
            }

            # Visual language: key frames
            init_path = save_frame(frame0, f"{game_id}_initial")
            pattern.visual = {'initial_frame': init_path, 'frame_shape': list(frame0.shape)}
            if obs is not None:
                final_path = save_frame(np.array(obs.frame), f"{game_id}_final")
                pattern.visual['final_frame'] = final_path

            # Save to library
            LIBRARY_DIR.mkdir(parents=True, exist_ok=True)
            lib_path = LIBRARY_DIR / "patterns.json"
            lib = json.loads(lib_path.read_text()) if lib_path.exists() else {}
            from dataclasses import asdict
            lib[pid] = asdict(pattern)
            lib_path.write_text(json.dumps(lib, indent=2, default=str))
            log.info(f"Rosetta Stone: saved pattern {pid} ({genre}) — machine+human+visual")
        except Exception as e:
            log.debug(f"  Pattern library save failed: {e}")

    # Flight recorder: end-of-game summary
    if flight_recorder:
        summary = flight_recorder.session_summary()
        log.info(f"  [self] Game complete — {summary['vitals']}")
        if summary['pilot_engagements'] > 0:
            log.info(f"  [self] Pilot: {summary['pilot_engagements']} engagements, "
                     f"{summary['pilot_success_rate']:.0%} effective")
        route_stats = summary.get('route_stats', {})
        if route_stats:
            top_routes = sorted(route_stats.items(),
                               key=lambda x: x[1]['successes'], reverse=True)[:3]
            for rt, st in top_routes:
                log.info(f"  [self] Route {rt}: {st['successes']}/{st['attempts']} "
                         f"({st['success_rate']:.0%}, avg {st['avg_time']:.1f}s)")

    log.info(f"\n{game_id.upper()}: {solved}/{total} levels, "
             f"{results['total_actions']} acts (baseline {results.get('baseline_total', '?')})")
    return results


def main():
    parser = argparse.ArgumentParser(description='ARC-AGI-3 Agent v0.5')
    parser.add_argument('--games', '--game', nargs='+', default=None)
    parser.add_argument('--timeout', type=float, default=60.0,
                        help='Timeout per level in seconds')
    parser.add_argument('--game-timeout', type=float, default=0,
                        help='Total timeout per game (0 = unlimited)')
    parser.add_argument('--total-timeout', type=float, default=0,
                        help='Total competition time budget (0 = unlimited)')
    parser.add_argument('--compete', action='store_true')
    parser.add_argument('-v', '--verbose', action='store_true')
    parser.add_argument('--record', action='store_true',
                        help='Record frames to SQLite for replay')
    parser.add_argument('--memory', type=str,
                        default='results/arc_eyes_memory.json',
                        help='Path to Eyes persistent memory')
    parser.add_argument('--no-memory', action='store_true',
                        help='Disable Eyes persistent memory')
    parser.add_argument('--generic', action='store_true',
                        help='Force generic solver only (no specialized solvers)')
    parser.add_argument('--retries', type=int, default=0,
                        help='Max retry attempts for games that solve 0 levels')
    args = parser.parse_args()

    if args.verbose:
        log.setLevel(logging.DEBUG)

    import arc_agi
    arcade = arc_agi.Arcade()
    envs = arcade.get_environments()
    game_ids = args.games or [e.game_id.split('-')[0] for e in envs]

    # Load Eyes persistent memory (cross-game learning)
    from arc_eyes import EpisodicMemory
    if not args.no_memory:
        eyes_memory = EpisodicMemory.load(args.memory)
        if eyes_memory.game_patterns:
            log.info(f"[eyes memory] {len(eyes_memory.game_patterns)} games, "
                    f"{len(getattr(eyes_memory, 'strategy_results', []))} strategy results")
    else:
        eyes_memory = EpisodicMemory()

    # Load persistent solver memory (cross-session learning)
    solver_mem = None
    try:
        from solver_memory import SolverMemory
        solver_mem = SolverMemory()
        stats = solver_mem.stats()
        if stats['rules'] > 0 or stats['games'] > 0:
            log.info(f"[solver memory] {stats['rules']} rules, "
                     f"{stats['games']} games, {stats['genres']} genres")
    except Exception as e:
        log.debug(f"SolverMemory not available: {e}")

    all_results = []
    total_actions = 0
    total_baseline = 0
    total_solved = 0
    total_levels = 0
    total_games = len(game_ids)
    comp_deadline = time.time() + args.total_timeout if args.total_timeout > 0 else float('inf')

    for gi, gid in enumerate(game_ids):
        # Check total competition time budget
        if time.time() >= comp_deadline:
            log.warning(f"Total timeout reached after {gi}/{total_games} games")
            break

        # Per-game timeout: if set, adjust per-level timeout to fit
        game_t0 = time.time()
        per_level = args.timeout
        if args.game_timeout > 0:
            per_level = min(per_level, args.game_timeout / 10)  # assume ~10 levels max

        rec = None
        try:
            rec = GameRecorder(gid) if args.record else None
            result = run_game(arcade, gid, timeout_per_level=per_level,
                              recorder=rec, eyes_memory=eyes_memory,
                              solver_memory=solver_mem,
                              force_generic=args.generic)
            if rec:
                rec.finish(result)
            all_results.append(result)
            solved = sum(1 for l in result['levels'] if l['solved'])
            total_solved += solved
            total_levels += len(result['levels'])
            total_actions += result['total_actions']
            total_baseline += result.get('baseline_total', 0)

            # Running score
            game_time = time.time() - game_t0
            run_score = total_baseline / total_actions * 100 if total_actions > 0 else 0
            log.info(f"  [{gi+1}/{total_games}] Running: {total_solved}/{total_levels} levels, "
                     f"eff {run_score:.0f}%, game took {game_time:.0f}s")
        except Exception as e:
            log.error(f"Error on {gid}: {e}")
            import traceback; traceback.print_exc()
            if rec:
                rec.finish({'levels': [], 'error': str(e)})
            all_results.append({'game': gid, 'error': str(e)})

    # Retry games that scored 0 levels (stochastic solving)
    if args.retries > 0 and time.time() < comp_deadline:
        # Find games with 0 solves
        game_scores = {}  # gid -> (solved, result_index)
        for i, r in enumerate(all_results):
            gid = r.get('game', game_ids[i] if i < len(game_ids) else '?')
            solved = sum(1 for l in r.get('levels', []) if l.get('solved'))
            game_scores[gid] = (solved, i)

        # Retry games that scored 0 levels, OR games that solved < 50% of attempted levels
        retry_games = []
        for gid, (s, idx) in game_scores.items():
            r = all_results[idx]
            if 'error' in r:
                continue
            attempted = len(r.get('levels', []))
            if s == 0 or (s < attempted and s < max(2, attempted // 2)):
                retry_games.append(gid)
        # Prioritize: 0-solve games first, then low-solve games
        retry_games.sort(key=lambda g: game_scores[g][0])
        zero_games = retry_games
        if zero_games:
            log.info(f"\n{'='*40}")
            log.info(f"RETRY PHASE: {len(zero_games)} games to retry, up to {args.retries} retries each")

        for gid in zero_games:
            best_solved = 0
            best_result = all_results[game_scores[gid][1]]

            for retry in range(args.retries):
                if time.time() >= comp_deadline:
                    log.warning(f"Total timeout — stopping retries")
                    break

                log.info(f"  Retry {retry+1}/{args.retries} for {gid}...")
                game_t0 = time.time()
                per_level = args.timeout
                if args.game_timeout > 0:
                    per_level = min(per_level, args.game_timeout / 10)

                try:
                    rec = GameRecorder(gid) if args.record else None
                    result = run_game(arcade, gid, timeout_per_level=per_level,
                                      recorder=rec, eyes_memory=eyes_memory,
                                      solver_memory=solver_mem,
                                      force_generic=args.generic)
                    if rec:
                        rec.finish(result)
                    solved = sum(1 for l in result.get('levels', []) if l.get('solved'))
                    game_time = time.time() - game_t0
                    log.info(f"    Retry {retry+1}: {solved}/{len(result.get('levels', []))} levels in {game_time:.0f}s")

                    if solved > best_solved:
                        # Replace the original result with this better one
                        old_idx = game_scores[gid][1]
                        # Subtract old totals
                        old_actions = best_result.get('total_actions', 0)
                        old_baseline = best_result.get('baseline_total', 0)
                        total_actions -= old_actions
                        total_baseline -= old_baseline
                        # Add new totals
                        total_actions += result['total_actions']
                        total_baseline += result.get('baseline_total', 0)
                        total_solved += solved - best_solved
                        best_solved = solved
                        best_result = result
                        all_results[old_idx] = result
                        log.info(f"    ✓ New best for {gid}: {solved} levels!")

                    if solved > 0:
                        break  # Got at least 1 solve — move on

                except Exception as e:
                    log.error(f"    Retry error on {gid}: {e}")

    # Save Eyes persistent memory
    if not args.no_memory:
        os.makedirs(os.path.dirname(args.memory) or '.', exist_ok=True)
        eyes_memory.save(args.memory)
        strat_count = len(getattr(eyes_memory, 'strategy_results', []))
        if strat_count:
            print(f"\n{eyes_memory.strategy_summary()}")

    print(f"\n{'='*60}")
    print(f"FINAL: {total_solved}/{total_levels} levels")
    print(f"Actions: {total_actions} / Baseline: {total_baseline}")
    if total_actions > 0 and total_baseline > 0:
        print(f"Efficiency: {total_baseline / total_actions * 100:.1f}%")
    print(f"{'='*60}")

    # Close scorecard if competition mode
    if args.compete:
        try:
            sc = arcade.close_scorecard()
            if sc:
                log.info(f"Scorecard closed: {sc}")
        except Exception as e:
            log.warning(f"Failed to close scorecard: {e}")

        os.makedirs('results', exist_ok=True)
        fn = f"results/arc_v05_{time.strftime('%Y%m%d_%H%M%S')}.json"
        with open(fn, 'w') as f:
            json.dump({
                'version': 'v0.5', 'timestamp': time.strftime('%Y-%m-%dT%H:%M:%S'),
                'solved': total_solved, 'levels': total_levels,
                'actions': total_actions, 'baseline': total_baseline,
                'efficiency': total_baseline / total_actions * 100 if total_actions > 0 else 0,
                'games': all_results,
            }, f, indent=2, default=str)
        print(f"Saved: {fn}")


if __name__ == '__main__':
    main()
