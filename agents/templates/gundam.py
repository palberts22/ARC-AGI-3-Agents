#!/usr/bin/env python3
"""GUNDAM — General Understanding Neural Decision Architecture for Mastery.

Pure AGI pilot. No BFS, no MCTS, no constraint solver.
Just observation, reasoning, and action.

The test: drop it into a game it has never seen.
If it figures out the rules and solves it — that's AGI.

SEALED WEAPONS (Peter directive, Mar 16-17 2026 — PERMANENT):
  - NO BFS. Humans don't brute-force. Neither do we.
  - NO SOURCE CODE. Never read game .py files. Discover from pixels.
  - NO SPECIALIZED SOLVERS. Keep wired but OFF. Pure reasoning only.
  - NO BRUTE-FORCE INTROSPECTION. Inspecting game internals (sprites, hidden
    state, methods) for 300+ tool calls IS source reading wearing a lab coat.
    If stuck on mechanics after ~20 min, STOP. Take a break. Come back fresh.
  - We reach AGI honestly or not at all.

Design: Peter + Apollo, 2026-03-12.
"Don't we want pilot mode to beat ARC without anything else?
 If that's not AGI, what is?"

Architecture:
  1. OBSERVE  — see the frame, see what changed
  2. HYPOTHESIZE — what are the rules? what's the goal?
  3. PLAN — what sequence of actions achieves the goal?
  4. EXECUTE — take the action
  5. ADAPT — was I right? update understanding

The mind remembers everything it's tried. Every frame. Every effect.
It builds a mental model of the game from scratch.
"""

import base64
import copy
import io
import json
import logging
import os
import sys
import time
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

try:
    from dotenv import load_dotenv
    load_dotenv(os.path.join(os.path.dirname(__file__), '..', '.env'), override=True)
except ImportError:
    pass

try:
    from PIL import Image
    HAS_PIL = True
except ImportError:
    HAS_PIL = False

try:
    import requests as _requests
    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False

log = logging.getLogger("gundam")

# Import Eyes for structured perception
try:
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from arc_eyes import detect_sprites, track_sprite_movement, Sprite, EpisodicMemory
    HAS_EYES = True
except ImportError:
    HAS_EYES = False
    log.warning("arc_eyes not available — running without structured perception")

try:
    from grid_transform import grid_transform, CausalLedger
    HAS_GRID_TRANSFORM = True
except ImportError:
    HAS_GRID_TRANSFORM = False
    CausalLedger = None
    log.warning("grid_transform not available — running without transform stack")

try:
    import arc_vlm
    # Validate VLM actually works (model load can hang without GPU)
    import signal as _sig
    def _vlm_timeout_handler(signum, frame):
        raise TimeoutError("VLM model load timed out — no GPU?")
    _old_handler = _sig.signal(_sig.SIGALRM, _vlm_timeout_handler)
    _sig.alarm(int(os.environ.get('ARC_VLM_TIMEOUT', '8')))
    try:
        # Quick probe — forces model load if lazy
        _test_frame = __import__('numpy').zeros((64, 64, 3), dtype=__import__('numpy').uint8)
        arc_vlm.perceive_structured(_test_frame)
        HAS_VLM = True
        log.info("VLM perception available (Florence-2, local GPU, $0 cost)")
    except (TimeoutError, Exception) as _e:
        HAS_VLM = False
        log.warning(f"VLM import OK but model load failed ({_e}) — text-only mode")
    finally:
        _sig.alarm(0)
        _sig.signal(_sig.SIGALRM, _old_handler)
except ImportError:
    HAS_VLM = False
    log.info("VLM not available — using text-only perception")
# Override: skip VLM in no-LLM mode
if os.environ.get('ARC_NO_LLM'):
    HAS_VLM = False

try:
    from imagination_framework import WorldModel, State, Action
    HAS_IMAGINATION = True
except ImportError:
    HAS_IMAGINATION = False
    log.warning("imagination_framework not available — running without internal simulation")

try:
    from arc_action_predictor_lite import ActionPredictor
    HAS_PREDICTOR = True
except ImportError:
    HAS_PREDICTOR = False
    log.warning("arc_action_predictor_lite not available — running without CNN-lite predictor")

# ARC color palette (indices 0-15 → RGB)
ARC_PALETTE = [
    [255, 255, 255],  # 0: white
    [204, 204, 204],  # 1: light gray
    [153, 153, 153],  # 2: medium gray
    [102, 102, 102],  # 3: dark gray
    [51, 51, 51],     # 4: charcoal
    [0, 0, 0],        # 5: black
    [229, 58, 163],   # 6: pink/magenta
    [255, 123, 204],  # 7: light pink
    [249, 60, 49],    # 8: red
    [30, 147, 255],   # 9: blue
    [136, 216, 241],  # 10: light blue
    [255, 220, 0],    # 11: yellow
    [255, 133, 27],   # 12: orange
    [146, 18, 49],    # 13: dark red/maroon
    [79, 204, 48],    # 14: green
    [163, 86, 214],   # 15: purple
]


# ---------------------------------------------------------------------------
# OPRAH — Observe, Probe, Reason, Act, Harvest
# Discovery protocol for unknown environments (150+ in ARC-AGI-3).
# Spec: specs/generalized_discovery_protocol.md (Archie, 2026-03-15)
# ---------------------------------------------------------------------------

from enum import Enum


class ActionType(Enum):
    MOVEMENT = "movement"          # Agent/object position changes (small, localized)
    TOGGLE = "toggle"              # Cell/object state changes at position
    GLOBAL = "global"              # Whole frame changes (>30% pixels)
    PARAMETERIZED = "parameterized"  # Effect depends on position data
    NO_OP = "no_op"                # Nothing visible changes


@dataclass
class ActionProbe:
    """Result of probing a single action."""
    action_idx: int
    action_type: ActionType
    frame_changed: bool
    pixels_changed: int
    change_fraction: float = 0.0
    change_region: Optional[tuple] = None  # (y_min, y_max, x_min, x_max)
    is_self_inverse: bool = False          # action undoes itself when repeated
    notes: str = ""


@dataclass
class EnvironmentProfile:
    """What OPRAH learned about an environment from OBSERVE + PROBE."""
    game_id: str = ""
    frame_shape: tuple = (0, 0)
    n_actions: int = 0
    action_probes: list = field(default_factory=list)
    genre: str = "novel_unknown"
    genre_confidence: float = 0.0
    movement_actions: list = field(default_factory=list)
    direction_map: dict = field(default_factory=dict)  # {action_idx: (dy, dx, "UP"/"DOWN"/etc)}
    player_color: int = -1
    player_start_pos: tuple = None  # (y, x) after first successful move
    toggle_actions: list = field(default_factory=list)
    global_actions: list = field(default_factory=list)
    param_actions: list = field(default_factory=list)
    has_grid: bool = False
    has_click: bool = False


def _oprah_frame_diff(before: np.ndarray, after: np.ndarray) -> dict:
    """Compare two frames, return change summary. Core detection function."""
    if before.shape != after.shape:
        return {'changed': True, 'pixels': -1, 'fraction': 1.0, 'scope': 'global'}

    diff = (before != after)
    if diff.ndim == 3:
        diff = diff.any(axis=0)

    n_changed = int(diff.sum())
    total = diff.shape[0] * diff.shape[1]

    result = {
        'changed': n_changed > 0,
        'pixels': n_changed,
        'fraction': n_changed / total if total > 0 else 0,
    }

    if n_changed > 0:
        ys, xs = np.where(diff)
        result['region'] = (int(ys.min()), int(ys.max()), int(xs.min()), int(xs.max()))
        if result['fraction'] < 0.05:
            result['scope'] = 'local'
        elif result['fraction'] > 0.30:
            result['scope'] = 'global'
        else:
            result['scope'] = 'medium'
    else:
        result['scope'] = 'none'

    return result


def _oprah_classify_action(diff: dict, has_position_data: bool) -> ActionType:
    """Classify an action based on its frame diff and position-dependence."""
    if not diff['changed']:
        return ActionType.NO_OP
    if has_position_data:
        return ActionType.PARAMETERIZED
    scope = diff.get('scope', 'medium')
    if scope == 'global':
        return ActionType.GLOBAL
    # Movement: local changes (< 10% of frame) — sprite repositioning
    if scope == 'local' or (scope == 'medium' and diff['fraction'] < 0.10):
        return ActionType.MOVEMENT
    return ActionType.TOGGLE


def _oprah_infer_genre(probes: list, has_grid: bool = False,
                       has_click: bool = False) -> tuple:
    """Infer game genre from action probe results.

    Returns (genre_name, confidence). Genre names match GENRE_ROUTE_MAP keys
    in arc_agent_v05.py for seamless routing.
    """
    counts = {}
    for p in probes:
        counts[p.action_type] = counts.get(p.action_type, 0) + 1

    has_movement = counts.get(ActionType.MOVEMENT, 0) > 0
    has_toggle = counts.get(ActionType.TOGGLE, 0) > 0
    has_param = counts.get(ActionType.PARAMETERIZED, 0) > 0
    has_global = counts.get(ActionType.GLOBAL, 0) > 0

    # Check for self-inverse (toggle signature)
    n_self_inverse = sum(1 for p in probes if p.is_self_inverse)

    # --- FALSE MOVEMENT DETECTION ---
    # If multiple "movement" actions all change the same tiny region (< 5px)
    # and none have confirmed direction vectors, they're likely incrementing
    # a counter/signal, not moving a sprite. Reclassify as sequence puzzle.
    movement_probes = [p for p in probes if p.action_type == ActionType.MOVEMENT]
    if len(movement_probes) >= 2:
        # Check if all movement actions affect the same small region
        regions = [p.change_region for p in movement_probes if p.change_region]
        small_changes = all(p.pixels_changed <= 5 for p in movement_probes)
        # Check directions: no directions OR all same direction = not real navigation
        _dir_set = set()
        for p in movement_probes:
            if p.notes and p.notes.startswith("direction:"):
                try:
                    _dinfo = eval(p.notes[len("direction:"):])
                    if _dinfo and len(_dinfo) >= 3:
                        _dir_set.add(_dinfo[2])  # direction name
                except Exception:
                    pass
        no_diverse_directions = len(_dir_set) <= 1  # 0 or 1 distinct direction = not real nav
        same_region = False
        if len(regions) >= 2:
            # All regions overlap (within 5px tolerance)
            r0 = regions[0]
            same_region = all(
                abs(r[0] - r0[0]) <= 5 and abs(r[1] - r0[1]) <= 5 and
                abs(r[2] - r0[2]) <= 5 and abs(r[3] - r0[3]) <= 5
                for r in regions[1:]
            )
        if small_changes and no_diverse_directions and same_region:
            # These aren't real movement — reclassify
            has_movement = False
            counts[ActionType.MOVEMENT] = 0
            # Reclassify probes in-place for downstream use
            for p in movement_probes:
                p.action_type = ActionType.TOGGLE
            counts[ActionType.TOGGLE] = counts.get(ActionType.TOGGLE, 0) + len(movement_probes)
            has_toggle = True

    # Single-action games: if only 1 action exists and it changes very few pixels,
    # this is likely a timing/sequence game, not navigation
    if len(probes) == 1 and probes[0].pixels_changed <= 5:
        if probes[0].action_type == ActionType.MOVEMENT:
            has_movement = False
            probes[0].action_type = ActionType.TOGGLE
            has_toggle = True

    # Decision tree — ordered by confidence
    if has_grid and has_param and n_self_inverse > 0:
        return ('toggle_puzzle', 0.85)       # Grid + click + self-inverse = lights-out family
    if has_grid and has_param:
        return ('constraint_satisfaction', 0.7)  # Grid + click = constraint puzzle
    if has_grid and has_toggle:
        return ('constraint_satisfaction', 0.6)
    if has_movement and not has_toggle and not has_param:
        return ('navigation_maze', 0.75)     # Movement only = navigation
    if has_movement and has_param:
        return ('push_block', 0.6)           # Movement + interact = push/interact
    if has_movement and has_toggle:
        return ('multi_phase', 0.5)          # Movement + state changes
    if has_param and not has_movement:
        if has_grid:
            return ('paint_fill', 0.5)       # Grid + click, no toggle
        return ('circuit_puzzle', 0.4)       # Click-based, no grid
    if has_global and has_toggle:
        return ('cellular_automaton', 0.4)
    if has_movement and has_global:
        return ('multi_phase', 0.4)
    # Non-movement toggles with grid = deduction puzzle
    if has_toggle and not has_movement and not has_param:
        if has_grid:
            return ('constraint_satisfaction', 0.5)
        return ('sequence_puzzle', 0.4)      # Multiple toggle actions, no movement = sequence/deduction

    return ('novel_unknown', 0.2)


def _oprah_build_llm_prompt(profile: 'EnvironmentProfile',
                            probes: list) -> str:
    """Build one-shot LLM reasoning prompt for low-confidence genre inference."""
    probe_lines = []
    for p in probes:
        line = f"  Action {p.action_idx}: [{p.action_type.value}]"
        if p.frame_changed:
            line += f" — {p.pixels_changed}px changed ({p.change_fraction:.1%})"
            if p.change_region:
                y0, y1, x0, x1 = p.change_region
                line += f", region ({x0},{y0})-({x1},{y1})"
            if p.is_self_inverse:
                line += " [SELF-INVERSE]"
        else:
            line += " — no visible change"
        if p.notes:
            line += f" | {p.notes}"
        probe_lines.append(line)

    return f"""Analyze this unknown interactive environment from probe data.

FRAME: {profile.frame_shape[0]}x{profile.frame_shape[1]} pixels
GRID DETECTED: {profile.has_grid}
CLICK ACTION: {profile.has_click}
ACTIONS: {profile.n_actions}

PROBE RESULTS:
{chr(10).join(probe_lines)}

What type of game is this? What's the likely goal? What strategy should I try?

Respond in JSON:
{{
  "genre": "<toggle_puzzle|constraint_satisfaction|navigation_maze|push_block|sliding_tile|pump_overflow|paint_fill|sorting|pursuit_evasion|gravity_puzzle|cellular_automaton|rotation_puzzle|connection_flow|pattern_replication|circuit_puzzle|multi_phase|novel_unknown>",
  "goal_hypothesis": "one sentence",
  "strategy": "one sentence",
  "key_actions": {{"movement": [indices], "interact": [indices], "avoid": [indices]}}
}}"""


def _oprah_probe_env(env, obs, extract_frame, base_frame,
                     position_dependent: dict = None,
                     verbose: bool = False) -> list:
    """Probe all actions and build ActionProbe list.

    Reuses position_dependent results from existing run_game() probing
    to avoid duplicate deepcopies.

    Args:
        position_dependent: dict mapping action_idx -> bool (from existing probing)
    """
    import copy as _cp

    probes = []
    n_actions = len(obs.available_actions)

    for i in range(n_actions):
        act = obs.available_actions[i]
        is_pos_dep = (position_dependent or {}).get(i, False)

        if is_pos_dep:
            # Already know it's position-dependent — classify as PARAMETERIZED
            # Check self-inverse by clicking same position twice
            h, w = base_frame.shape[-2], base_frame.shape[-1]
            cx, cy = w // 2, h // 2
            test_env = _cp.deepcopy(env)
            try:
                obs1 = test_env.step(act, {"x": cx, "y": cy})
                frame1 = extract_frame(obs1)
                diff1 = _oprah_frame_diff(base_frame, frame1)
                # Click same spot again
                obs2 = test_env.step(act, {"x": cx, "y": cy})
                frame2 = extract_frame(obs2)
                is_self_inv = np.array_equal(base_frame, frame2)
            except Exception:
                diff1 = {'changed': True, 'pixels': 0, 'fraction': 0, 'scope': 'local'}
                is_self_inv = False

            probes.append(ActionProbe(
                action_idx=i,
                action_type=ActionType.PARAMETERIZED,
                frame_changed=diff1.get('changed', True),
                pixels_changed=diff1.get('pixels', 0),
                change_fraction=diff1.get('fraction', 0),
                change_region=diff1.get('region'),
                is_self_inverse=is_self_inv,
                notes="position-dependent click action",
            ))
        else:
            # Probe without position data
            test_env = _cp.deepcopy(env)
            try:
                test_obs = test_env.step(act, {})
                test_frame = extract_frame(test_obs)
                diff = _oprah_frame_diff(base_frame, test_frame)
            except Exception:
                diff = {'changed': False, 'pixels': 0, 'fraction': 0, 'scope': 'none'}

            atype = _oprah_classify_action(diff, False)

            # Extract direction vector for movement actions via sprite tracking
            _direction_info = None
            if atype == ActionType.MOVEMENT and diff['changed'] and HAS_EYES:
                fb2d = base_frame[0] if base_frame.ndim == 3 else base_frame
                fa2d = test_frame[0] if test_frame.ndim == 3 else test_frame
                _spr_b = detect_sprites(fb2d, min_size=2)
                _spr_a = detect_sprites(fa2d, min_size=2)
                _mvs = track_sprite_movement(_spr_b, _spr_a)
                if _mvs:
                    _sb, _sa, (_dy, _dx) = _mvs[0]
                    _dir = ""
                    if abs(_dy) > abs(_dx):
                        _dir = "DOWN" if _dy > 0 else "UP"
                    elif abs(_dx) > 0:
                        _dir = "RIGHT" if _dx > 0 else "LEFT"
                    if _dir:
                        _direction_info = (int(_dy), int(_dx), _dir, _sb.color,
                                          (int(_sa.center[0]), int(_sa.center[1])))

            # Check self-inverse for non-no-op actions
            is_self_inv = False
            if diff['changed']:
                try:
                    obs2 = test_env.step(act, {})
                    frame2 = extract_frame(obs2)
                    is_self_inv = np.array_equal(base_frame, frame2)
                except Exception:
                    pass

            probes.append(ActionProbe(
                action_idx=i,
                action_type=atype,
                frame_changed=diff['changed'],
                pixels_changed=diff.get('pixels', 0),
                change_fraction=diff.get('fraction', 0),
                change_region=diff.get('region'),
                is_self_inverse=is_self_inv,
                notes=f"direction:{_direction_info}" if _direction_info else "",
            ))

    if verbose:
        for p in probes:
            inv = " [SELF-INVERSE]" if p.is_self_inverse else ""
            print(f"  [oprah] Action {p.action_idx}: {p.action_type.value}"
                  f" ({p.pixels_changed}px, {p.change_fraction:.1%}){inv}")

    return probes


# ---------------------------------------------------------------------------
# Data: what the mind remembers
# ---------------------------------------------------------------------------

@dataclass
class SpatialModel:
    """A persistent mental map of the game world.

    Like a human building a mental model: "there's a wall to the right,
    a gap below, the target is northeast." Survives across turns.
    """
    # Position tracking
    player_pos: tuple = (0, 0)  # (y, x) in logical grid coords
    target_pos: tuple = (-1, -1)
    player_color: int = -1
    player_colors: set = field(default_factory=set)  # all colors that are part of the player entity (co-move)
    target_color: int = -1

    # Action → direction mapping (learned from observation)
    action_map: dict = field(default_factory=dict)  # action_id -> (dy, dx, "UP"/"DOWN"/etc)

    # Wall map: set of (y, x) positions that are walls
    walls: set = field(default_factory=set)

    # Blocked moves: (y, x, action_id) -> True means action blocked at that position
    blocked: set = field(default_factory=set)  # set of (y, x, action_id)

    # All visited positions — tracks every (y, x) the player has been at
    visited: set = field(default_factory=set)  # set of (y, x)

    # Transition graph — observed: (from_y, from_x, action_id) -> (to_y, to_x)
    transitions: dict = field(default_factory=dict)

    # Objects found
    objects: list = field(default_factory=list)  # [(color, y, x, role)]

    # Grid dimensions (logical)
    grid_h: int = 0
    grid_w: int = 0

    # ── Causal Target Detection ──
    # Track candidate targets by what responds to player actions, not just visual salience.
    # "The target isn't the brightest light — it's the door that opens when you knock."
    # Approved by Peter, 2026-03-16.
    target_candidates: dict = field(default_factory=dict)  # color -> {pos, confidence, reason}
    target_confirmed: bool = False  # True once causally verified (level completed on contact)
    scenery_colors: set = field(default_factory=set)  # colors that don't change between levels = infrastructure

    # ── Pickup Event Detection ──
    # Large world changes during navigation = "I collected something."
    # Tracks positions and pixel deltas to guide exploration toward rewards.
    pickup_events: list = field(default_factory=list)  # [{pos, pixels_changed, sprites_moved, action_id}]
    pickup_threshold: int = 100  # px change to count as pickup (raised: normal LS20 movement = 52px)
    surprise_map: dict = field(default_factory=dict)  # {(y,x): max_pixel_delta} — positions with unexpectedly large changes
    avg_move_cost: float = 0.0  # running average px change for normal single-sprite moves

    # ── Fuel/Resource Tracking ──
    # Some games have a draining resource bar (e.g. LS20 fuel = c11 pixels).
    # When fuel runs out, player teleports back to start. Track it to plan paths efficiently.
    fuel_color: int = -1  # color of the fuel/resource bar (detected by consistent shrinkage)
    fuel_initial: int = 0  # starting fuel pixels
    fuel_current: int = 0  # current fuel pixels
    fuel_drain_per_action: float = 0.0  # average fuel consumed per action
    fuel_detected: bool = False  # True once fuel mechanic is confirmed
    move_cost_count: int = 0  # how many normal moves we've seen
    # Modifier memory — "position X changes attribute Y"
    known_modifiers: dict = field(default_factory=dict)  # {(y,x): {'hud_lost': {c: n}, 'hud_gained': {c: n}, 'visits': int}}

    def describe(self) -> str:
        """Text description for the LLM."""
        lines = []
        if self.player_color >= 0:
            lines.append(f"**PLAYER**: color {self.player_color} at position {self.player_pos}")
        if self.target_pos != (-1, -1):
            dy = self.target_pos[0] - self.player_pos[0]
            dx = self.target_pos[1] - self.player_pos[1]
            dir_y = "below" if dy > 0 else "above" if dy < 0 else ""
            dir_x = "right" if dx > 0 else "left" if dx < 0 else ""
            rel = f"{abs(dy)} cells {dir_y}" if dir_y else ""
            if dir_x:
                rel += f"{', ' if rel else ''}{abs(dx)} cells {dir_x}"
            lines.append(f"**TARGET**: color {self.target_color} at {self.target_pos} — {rel} from you")
            if self.target_confirmed:
                lines.append("  (CONFIRMED — reaching this completed a level)")

        # Target candidates — show what we've tried
        if self.target_candidates and not self.target_confirmed:
            for tc, info in sorted(self.target_candidates.items(), key=lambda x: -x[1]['confidence']):
                conf = info['confidence']
                status = "LIKELY" if conf >= 0.6 else "maybe" if conf >= 0.3 else "unlikely"
                lines.append(f"  Candidate color {tc}: {status} ({conf:.1f}) — {info['reason']}")
        if self.scenery_colors:
            lines.append(f"  Scenery (NOT targets): colors {sorted(self.scenery_colors)}")

        # Situation assessment — high-level exploration state
        if self.visited and self.action_map:
            n_visited = len(self.visited)
            # Count frontier: visited positions with untried directions
            tried = {(t[0], t[1], t[2]) for t in self.transitions if len(t) >= 3}
            n_frontier = sum(1 for vy, vx in self.visited
                           for aid in self.action_map
                           if (vy, vx, aid) not in self.blocked
                           and (vy, vx, aid) not in tried)
            assessment = f"**SITUATION**: Explored {n_visited} positions."
            if self.known_modifiers:
                mod_descs = []
                for mk, m in self.known_modifiers.items():
                    my, mx = mk  # key IS the position tuple
                    mdy = my - self.player_pos[0]
                    mdx = mx - self.player_pos[1]
                    mdir_y = "below" if mdy > 0 else "above" if mdy < 0 else ""
                    mdir_x = "right" if mdx > 0 else "left" if mdx < 0 else ""
                    mrel = f"{abs(mdy)} {mdir_y}" if mdir_y else ""
                    if mdir_x:
                        mrel += f" {abs(mdx)} {mdir_x}" if mrel else f"{abs(mdx)} {mdir_x}"
                    mod_descs.append(f"({my},{mx}) [{mrel}]: {m['visits']} visits")
                assessment += f" Found {len(self.known_modifiers)} modifier(s): {', '.join(mod_descs)}."
            if self.pickup_events:
                pickup_locs = set(f"({e['pos'][0]},{e['pos'][1]})" for e in self.pickup_events)
                assessment += f" Pickup events at: {', '.join(pickup_locs)}."
            if self.target_pos == (-1, -1) and not self.target_confirmed:
                if self.known_modifiers:
                    assessment += " Try visiting modifiers to change your state, then explore further."
                else:
                    assessment += " NO TARGET FOUND YET — keep exploring new areas!"
            if n_frontier > 0:
                assessment += f" {n_frontier} untried directions remain."
            elif n_visited > 5:
                assessment += " All explored directions tried — look for hidden passages or revisit modifiers."
            lines.append(assessment)

        # Direction needed
        if self.player_color >= 0 and self.target_pos != (-1, -1):
            dy = self.target_pos[0] - self.player_pos[0]
            dx = self.target_pos[1] - self.player_pos[1]
            needed = []
            if dy > 0: needed.append(f"DOWN {dy}px")
            elif dy < 0: needed.append(f"UP {-dy}px")
            if dx > 0: needed.append(f"RIGHT {dx}px")
            elif dx < 0: needed.append(f"LEFT {-dx}px")
            lines.append(f"**DIRECTION TO TARGET**: {', '.join(needed) if needed else 'AT TARGET!'}")

        if self.action_map:
            lines.append("**ACTION MAP** (confirmed by observation):")
            for aid, (dy, dx, name) in sorted(self.action_map.items()):
                lines.append(f"  Action {aid} = {name}")

            # ONE-STEP LOOKAHEAD: predict where each action leads from current pos
            if self.player_pos != (0, 0) and self.transitions:
                lines.append("**NEXT MOVE PREDICTIONS** (from observed transitions):")
                py, px = self.player_pos
                for aid, (dy, dx, name) in sorted(self.action_map.items()):
                    key = (py, px, aid)
                    if key in self.transitions:
                        dest = self.transitions[key]
                        lines.append(f"  Action {aid} ({name}) → position {dest} [KNOWN]")
                    elif (py, px, aid) in {(y, x, a) for y, x, a in self.blocked}:
                        lines.append(f"  Action {aid} ({name}) → BLOCKED [wall]")
                    else:
                        # Estimate from step size
                        est = (py + dy, px + dx)
                        lines.append(f"  Action {aid} ({name}) → ~{est} [ESTIMATE — not yet tried from here]")

            # EXPLORATION STATE: positions visited and dead ends
            if self.visited and len(self.visited) > 2:
                # Find unexplored edges — positions we've visited but have untried actions
                unexplored_from = []
                for vy, vx in sorted(self.visited):
                    for aid in self.action_map:
                        key = (vy, vx, aid)
                        if key not in self.transitions and key not in {(y, x, a) for y, x, a in self.blocked}:
                            _, _, dname = self.action_map[aid]
                            unexplored_from.append(f"({vy},{vx})→{dname}")
                if unexplored_from:
                    lines.append(f"**UNEXPLORED MOVES** ({len(unexplored_from)} untried): {', '.join(unexplored_from[:12])}")
                    lines.append("  TRY THESE — they might lead to new areas!")

        # Surprise map — positions where something unexpected happened
        if self.surprise_map:
            top_surprises = sorted(self.surprise_map.items(), key=lambda x: -x[1])[:5]
            surprise_strs = [f"({y},{x})={px}px" for (y, x), px in top_surprises]
            lines.append(f"**SURPRISE HOTSPOTS** (large unexpected changes): {', '.join(surprise_strs)}")
            lines.append("  These positions are INTERESTING — something happened here. Revisit and investigate.")

        # Terrain knowledge — learned from movement
        walkable = getattr(self, 'walkable_colors', set())
        walls = getattr(self, 'wall_colors', set())
        if walkable or walls:
            terrain = []
            if walkable:
                terrain.append(f"WALKABLE colors: {sorted(walkable)}")
            if walls:
                terrain.append(f"WALL colors: {sorted(walls)}")
            lines.append(f"**TERRAIN** (learned from your movement): {'; '.join(terrain)}")
            lines.append("  In the ASCII map, trace paths through WALKABLE colors. Avoid WALL colors.")

        # Pickup events — "I collected something here"
        if self.pickup_events:
            lines.append(f"**PICKUPS** ({len(self.pickup_events)} collected):")
            for ev in self.pickup_events[-5:]:  # show last 5
                hud = ev.get('hud_change', '')
                detail = ev.get('hud_detail', {})
                lines.append(f"  Collected at {ev['pos']} ({ev['pixels_changed']}px change{hud})")
                if detail.get('bottom'):
                    lines.append(f"    → HUD change detail: {detail['bottom']}")
            lines.append("  Large world changes = MODIFIER or PICKUP! Your attributes may have changed.")
            lines.append("  HUD color changes tell you WHAT attribute changed. Track which modifiers affect which attributes.")

        # Known modifiers — positions that change your attributes
        if self.known_modifiers:
            lines.append(f"**KNOWN MODIFIERS** ({len(self.known_modifiers)} found):")
            for pos, mod in self.known_modifiers.items():
                change = mod.get('last_change', 'unknown effect')
                visits = mod.get('visits', 0)
                lines.append(f"  Modifier at {pos}: {change} (visited {visits}x)")
            lines.append("  STRATEGY: Revisit modifiers to cycle attributes until they match the target.")

        if self.blocked:
            # Show blocked positions relevant to current location
            blocked_here = [(y, x, a) for y, x, a in self.blocked
                           if (y, x) == self.player_pos]
            if blocked_here:
                blocked_dirs = []
                for _, _, a in blocked_here:
                    if a in self.action_map:
                        blocked_dirs.append(f"action {a} ({self.action_map[a][2]})")
                    else:
                        blocked_dirs.append(f"action {a}")

                # Dead-end detection: ALL directions from current position blocked
                n_blocked_here = len(blocked_here)
                n_total_actions = len(self.action_map) if self.action_map else 4
                if n_blocked_here >= n_total_actions:
                    lines.append(f"**⚠️ DEAD END**: ALL {n_blocked_here} directions blocked from ({self.player_pos[0]},{self.player_pos[1]})!")
                    lines.append("  You are TRAPPED. These blocks may be STALE — a modifier could have changed the world.")
                    lines.append("  OPTIONS: (1) Try any direction — the wall map might be outdated after a modifier/pickup.")
                    lines.append("  (2) If you stepped on a modifier recently, the maze layout may have changed.")
                    lines.append("  (3) If nothing works, this level may require a different approach.")
                else:
                    lines.append(f"**BLOCKED HERE**: {', '.join(blocked_dirs)} — TRY A DIFFERENT DIRECTION!")

            # Also show recent blocks
            all_blocked = list(self.blocked)[-10:]
            if all_blocked:
                lines.append(f"Wall map ({len(self.blocked)} walls known)")

        if self.objects:
            lines.append("Notable objects:")
            for color, y, x, role in self.objects[:8]:
                if color != self.player_color:  # don't repeat player info
                    lines.append(f"  color {color} at ({y},{x}) — {role}")

        # ASCII minimap — show explored territory around player
        minimap = self.render_minimap()
        if minimap:
            lines.append(f"\n**YOUR MAP** (from exploration):\n{minimap}")

        return "\n".join(lines) if lines else "(no spatial model yet)"

    def render_minimap(self, radius: int = 8) -> str:
        """Render an ASCII minimap from blocked/visited data.

        Legend: @ = you, # = wall (blocked), . = visited/open, ? = unexplored, * = target
        """
        if not self.blocked and not self.visited and self.player_pos == (0, 0):
            return ""

        py, px = self.player_pos
        ty, tx = self.target_pos

        # Use the properly tracked visited set + blocked positions
        visited = set(self.visited)
        wall_at = {}  # (y, x) -> set of blocked action_ids
        for by, bx, aid in self.blocked:
            visited.add((by, bx))
            wall_at.setdefault((by, bx), set()).add(aid)

        if not visited and self.player_color < 0:
            return ""

        visited.add((py, px))

        # Determine step size from action map
        step = 5  # default
        for aid, (dy, dx, _) in self.action_map.items():
            s = max(abs(dy), abs(dx))
            if s > 0:
                step = s
                break

        # Snap visited/blocked positions to grid for fuzzy matching (±2px tolerance)
        def _snap_match(visited_set, target_y, target_x, tolerance=2):
            for vy, vx in visited_set:
                if abs(vy - target_y) <= tolerance and abs(vx - target_x) <= tolerance:
                    return True
            return False

        def _snap_blocked(wall_at_dict, target_y, target_x, tolerance=2):
            for (wy, wx), aids in wall_at_dict.items():
                if abs(wy - target_y) <= tolerance and abs(wx - target_x) <= tolerance:
                    return aids
            return set()

        # Build grid around player
        rows = []
        for gy in range(-radius, radius + 1):
            row = []
            for gx in range(-radius, radius + 1):
                wy = py + gy * step
                wx = px + gx * step
                # Check for special positions
                is_modifier = any(abs(wy - my) <= 2 and abs(wx - mx) <= 2
                                  for my, mx in self.known_modifiers)
                is_pickup = any(abs(wy - ev['pos'][0]) <= 2 and abs(wx - ev['pos'][1]) <= 2
                                for ev in self.pickup_events if ev.get('hud_change'))

                if gy == 0 and gx == 0:
                    row.append('@')  # player
                elif ty >= 0 and abs(wy - ty) <= 2 and abs(wx - tx) <= 2:
                    row.append('*')  # target
                elif is_modifier:
                    row.append('M')  # known modifier
                elif is_pickup:
                    row.append('!')  # pickup with HUD change
                elif _snap_match(visited, wy, wx):
                    # Check if ALL directions blocked here = solid wall
                    blocked_here = _snap_blocked(wall_at, wy, wx)
                    if len(blocked_here) >= len(self.action_map) and len(self.action_map) >= 3:
                        row.append('#')  # wall (all directions blocked)
                    elif blocked_here:
                        row.append('~')  # partial — some directions blocked
                    else:
                        row.append('.')  # visited, open
                else:
                    row.append(' ')  # unexplored
            rows.append(''.join(row))

        # Trim empty rows
        while rows and rows[0].strip() == '':
            rows.pop(0)
        while rows and rows[-1].strip() == '':
            rows.pop()

        if not rows:
            return ""

        return "```\n" + '\n'.join(rows) + "\n```\nLegend: @ = you, # = wall, ~ = partial block, . = open, * = target, M = modifier, ! = pickup/HUD change, space = unexplored"

    def update_target_causal(self, color: int, pos: tuple, reason: str, boost: float = 0.2, pixel_count: int = 0):
        """Update target candidate confidence based on causal observation.

        Called when we observe something respond to player actions.
        The target is defined by what responds, not by what's brightest.
        """
        if color in self.scenery_colors:
            return  # known infrastructure, skip
        # Large sprites (>40px) are background/corridors, not targets
        if pixel_count > 40:
            return

        if color not in self.target_candidates:
            self.target_candidates[color] = {'pos': pos, 'confidence': 0.3, 'reason': reason}
        else:
            entry = self.target_candidates[color]
            # Only update position if not yet confirmed (avoid chasing phantom targets)
            if not self.target_confirmed:
                entry['pos'] = pos
            entry['confidence'] = min(1.0, entry['confidence'] + boost)
            entry['reason'] = reason

    def demote_target(self, color: int, reason: str):
        """Demote a candidate — reached it but nothing happened."""
        if color in self.target_candidates:
            self.target_candidates[color]['confidence'] -= 0.4
            self.target_candidates[color]['reason'] = reason
            if self.target_candidates[color]['confidence'] <= 0:
                # Move to scenery — it's not interactive
                self.scenery_colors.add(color)
                del self.target_candidates[color]

    def confirm_target(self, color: int):
        """Confirm a target — reaching it completed the level."""
        self.target_confirmed = True
        if color in self.target_candidates:
            self.target_candidates[color]['confidence'] = 1.0
            self.target_candidates[color]['reason'] = 'CONFIRMED — level completed on contact'

    def best_target(self) -> tuple:
        """Return the highest-confidence target position, or (-1,-1) if none."""
        if not self.target_candidates:
            return (-1, -1)
        best = max(self.target_candidates.values(), key=lambda c: c['confidence'])
        if best['confidence'] >= 0.3:
            return best['pos']
        return (-1, -1)

    def learn_scenery_from_diff(self, prev_frame, curr_frame):
        """Between-level diff: constant pixels are scenery, changed pixels are puzzle.

        Peter's insight: "git diff meets flip book." What stays is infrastructure.
        What changes is the puzzle.
        """
        if prev_frame is None or curr_frame is None:
            return
        prev = np.squeeze(np.array(prev_frame))
        curr = np.squeeze(np.array(curr_frame))
        if prev.ndim == 3:
            prev = prev[0]
        if curr.ndim == 3:
            curr = curr[0]
        if prev.shape != curr.shape or prev.size == 0:
            return

        # Colors that appear in the same positions in both frames = scenery
        unchanged_mask = (prev == curr)
        if unchanged_mask.any():
            unchanged_colors = set(np.unique(prev[unchanged_mask]).tolist())
            # Don't mark player color as scenery
            if self.player_color >= 0:
                unchanged_colors.discard(self.player_color)
            self.scenery_colors.update(unchanged_colors)

    def explore_path(self, max_steps: int = 8) -> list:
        """Find a path to the nearest unexplored area.

        Generates exploration waypoints by finding grid positions that are
        adjacent to visited positions but haven't been visited themselves.
        Returns action_ids to reach the nearest frontier position.
        """
        if not self.action_map or not self.visited:
            return []

        step = 5
        for aid, (dy, dx, name) in self.action_map.items():
            s = max(abs(dy), abs(dx))
            if s > 0:
                step = s

        # Find frontier: positions adjacent to visited that aren't visited or blocked
        frontier = set()
        for vy, vx in self.visited:
            for aid, (dy, dx, name) in self.action_map.items():
                ny, nx = vy + dy, vx + dx
                if (ny, nx) not in self.visited and (vy, vx, aid) not in self.blocked:
                    frontier.add((ny, nx))

        if not frontier:
            return []

        # Target ONE frontier at a time — test it, then re-plan
        # Chaining multiple frontiers wastes actions when the first one is a wall
        py, px = self.player_pos
        remaining = sorted(frontier,
                          key=lambda p: abs(p[0] - py) + abs(p[1] - px))
        if len(remaining) <= 3:
            print(f"  [explore] {len(remaining)} frontiers left from ({py},{px}), blocked={len(self.blocked)}")

        for target in remaining:
            segment = self.find_path(target=target)
            if segment and len(segment) <= max_steps:
                return segment

        # All frontiers too far — return shortest path we can
        for target in remaining:
            segment = self.find_path(target=target)
            if segment:
                return segment[:max_steps]

        return []

    def find_path(self, target=None):
        """A* pathfinding on the mental map — trace the route with your finger.

        Returns list of action_ids to reach the target. Uses only known walkable
        positions and avoids known blocked directions. This is NOT game-state BFS —
        it's reasoning on the map you've already perceived.
        """
        if not self.action_map:
            return []

        goal = target or self.target_pos
        if goal == (-1, -1) or goal == self.player_pos:
            return []

        # Build reverse map: (dy, dx) -> action_id
        dir_to_action = {}
        step = 5
        for aid, (dy, dx, name) in self.action_map.items():
            dir_to_action[(dy, dx)] = aid
            s = max(abs(dy), abs(dx))
            if s > 0:
                step = s

        # A* search on grid positions
        import heapq
        start = self.player_pos
        open_set = [(0, 0, start, [])]  # (f_score, counter, pos, path)
        visited = {start}
        counter = 1

        # Heuristic: Manhattan distance in grid steps
        def h(pos):
            return (abs(pos[0] - goal[0]) + abs(pos[1] - goal[1])) / max(step, 1)

        while open_set and counter < 5000:
            f, _, pos, path = heapq.heappop(open_set)

            if pos == goal:
                return path

            # Close enough? (within one step)
            if abs(pos[0] - goal[0]) <= step and abs(pos[1] - goal[1]) <= step:
                # Try direct step to goal
                dy = goal[0] - pos[0]
                dx = goal[1] - pos[1]
                if (dy, dx) in dir_to_action:
                    aid = dir_to_action[(dy, dx)]
                    if (pos[0], pos[1], aid) not in self.blocked:
                        return path + [aid]

            for (dy, dx), aid in dir_to_action.items():
                ny, nx = pos[0] + dy, pos[1] + dx
                if (ny, nx) in visited:
                    continue
                # Skip if this direction is known-blocked from this position
                if (pos[0], pos[1], aid) in self.blocked:
                    continue
                # Only path through KNOWN walkable positions or the goal
                # Unknown positions are assumed walls until proven otherwise
                if (ny, nx) not in self.visited and (ny, nx) != goal:
                    continue
                visited.add((ny, nx))
                g = len(path) + 1
                heapq.heappush(open_set, (g + h((ny, nx)), counter, (ny, nx), path + [aid]))
                counter += 1

        return []  # no path found

    # Alias for backward compatibility
    plan_path = find_path

    def is_dead_end(self) -> bool:
        """Check if ALL directions from current position are blocked."""
        if not self.action_map or self.player_pos == (0, 0):
            return False
        py, px = self.player_pos
        for aid in self.action_map:
            if (py, px, aid) not in self.blocked:
                return False
        return True

    def clear_dead_end(self) -> int:
        """Clear blocks from current position — world may have changed via modifiers.

        Returns number of blocks cleared.
        """
        py, px = self.player_pos
        to_remove = {(y, x, a) for y, x, a in self.blocked if (y, x) == (py, px)}
        self.blocked -= to_remove
        return len(to_remove)

    def plan_path_description(self, target: tuple = None) -> str:
        """Human-readable path plan for the pilot."""
        path = self.plan_path(target)
        if not path:
            return ""
        steps = []
        for aid in path:
            if aid in self.action_map:
                _, _, name = self.action_map[aid]
                steps.append(name)
            else:
                steps.append(f"action_{aid}")
        # Compress: "UP, UP, UP" -> "UP x3"
        compressed = []
        i = 0
        while i < len(steps):
            count = 1
            while i + count < len(steps) and steps[i + count] == steps[i]:
                count += 1
            if count > 1:
                compressed.append(f"{steps[i]} x{count}")
            else:
                compressed.append(steps[i])
            i += count
        return f"PLANNED PATH ({len(path)} steps): {' → '.join(compressed)}"


@dataclass
class Experiment:
    """One action and its observed effect."""
    turn: int
    action_id: int
    action_desc: str           # human-readable: "click at (32, 16)" or "move up"
    pixels_changed: int
    regions_changed: str       # brief description of what moved/changed
    frame_before_b64: str      # base64 PNG (small, for LLM vision)
    frame_after_b64: str
    won_level: bool = False
    hypothesis_at_time: str = ""  # what the mind thought before this action


@dataclass
class LevelMemory:
    """Everything the mind knows about one level."""
    level: int
    experiments: list = field(default_factory=list)  # list of Experiment
    hypotheses: list = field(default_factory=list)   # evolving list of hypotheses
    initial_frame_b64: str = ""
    current_frame_b64: str = ""
    current_frame: object = None   # numpy array of current frame (for analysis)
    grid_text: str = ""        # text representation of the grid
    solved: bool = False
    winning_sequence: list = field(default_factory=list)  # action IDs that solved it
    total_actions: int = 0


@dataclass
class CausalChain:
    """One observed action→effect pair."""
    action_id: int
    action_data: dict = field(default_factory=dict)  # e.g. {"x": 38, "y": 38}
    precondition: str = ""       # what was at the target before
    effect: str = ""             # what changed after
    confidence: float = 1.0      # 1.0 = directly observed
    level: int = 0
    step: int = 0
    pixels_changed: int = 0


@dataclass
class GameMemory:
    """Everything the mind knows about this game."""
    game_id: str = ""
    levels: dict = field(default_factory=dict)  # level_num -> LevelMemory
    n_actions: int = 0
    action_types: dict = field(default_factory=dict)  # action_id -> description
    game_hypothesis: str = ""     # overarching theory of what this game IS
    hypothesis_locked: bool = False  # True if pre-seeded from recalled knowledge — don't overwrite
    rules_discovered: list = field(default_factory=list)
    levels_solved: int = 0
    levels_total: int = 0
    causal_chains: list = field(default_factory=list)  # list of CausalChain
    experiments: list = field(default_factory=list)  # DMT experiment log


# ---------------------------------------------------------------------------
# Frame processing
# ---------------------------------------------------------------------------

def frame_to_rgb(frame: np.ndarray) -> np.ndarray:
    """Convert ARC palette-indexed frame to RGB."""
    frame = np.squeeze(frame)
    if frame.ndim == 1:
        frame = frame.reshape(1, -1)
    palette = np.array(ARC_PALETTE, dtype=np.uint8)
    indices = np.clip(frame.astype(int), 0, len(palette) - 1)
    return palette[indices]


def frame_to_b64(frame: np.ndarray, scale: int = 4) -> str:
    """Convert frame to base64 PNG for LLM vision, scaled up for visibility."""
    if not HAS_PIL:
        return ""
    frame = np.squeeze(np.array(frame))
    if frame.size == 0 or 0 in frame.shape:
        return ""  # empty frame guard
    if frame.ndim == 1:
        frame = frame.reshape(1, -1)
    if frame.ndim > 2:
        # Still multi-dim after squeeze — take first channel/slice
        frame = frame[0] if frame.shape[0] <= 4 else frame[:, :, 0]
    rgb = frame_to_rgb(frame)
    # PIL needs exactly (H, W, 3) — squeeze any extra dims
    rgb = np.squeeze(rgb)
    if rgb.ndim == 1:
        rgb = rgb.reshape(1, -1, 3)
    elif rgb.ndim == 2:
        # Grayscale — shouldn't happen with palette but just in case
        rgb = np.stack([rgb]*3, axis=-1)
    img = Image.fromarray(rgb.astype(np.uint8))
    if scale > 1:
        img = img.resize((img.width * scale, img.height * scale), Image.NEAREST)
    buf = io.BytesIO()
    img.save(buf, format='PNG')
    return base64.b64encode(buf.getvalue()).decode()


def frame_to_minimap(frame: np.ndarray, player_pos: tuple = None,
                     player_color: int = -1, target_pos: tuple = None,
                     target_color: int = -1) -> str:
    """Convert frame to a clean ASCII minimap for spatial reasoning.

    Reduces 64x64 pixel frame to ~12x12 logical grid.
    Marks player as P, target as T, walls as #, open as .
    This is what a human sees when they look at a maze.
    """
    if frame.ndim == 3:
        frame = frame[0]
    h, w = frame.shape

    # Detect cell size (typically 4-6 pixels per logical cell)
    cell = _detect_cell_size(frame) or 5

    grid_h, grid_w = h // cell, w // cell
    if grid_h < 3 or grid_w < 3:
        return ""

    # Build logical grid by majority vote per cell
    mini = np.zeros((grid_h, grid_w), dtype=int)
    for gy in range(grid_h):
        for gx in range(grid_w):
            region = frame[gy*cell:(gy+1)*cell, gx*cell:(gx+1)*cell]
            vals, counts = np.unique(region, return_counts=True)
            mini[gy, gx] = vals[np.argmax(counts)]

    # Find the most common color (background) and second most (walls)
    all_colors, all_counts = np.unique(mini, return_counts=True)
    sorted_idx = np.argsort(-all_counts)
    bg_color = all_colors[sorted_idx[0]]
    wall_color = all_colors[sorted_idx[1]] if len(sorted_idx) > 1 else -1

    # Build ASCII map
    # Convert player/target positions to grid coords
    p_gy = player_pos[0] // cell if player_pos else -1
    p_gx = player_pos[1] // cell if player_pos else -1
    t_gy = target_pos[0] // cell if target_pos else -1
    t_gx = target_pos[1] // cell if target_pos else -1

    # Build walkable mask — walls are the ONLY thing blocking movement
    # Everything else (background, decorations, modifiers) is walkable
    walkable = np.ones((grid_h, grid_w), dtype=bool)
    # Only the dominant wall color blocks; other colors are walkable
    for gy in range(grid_h):
        for gx in range(grid_w):
            if mini[gy, gx] == wall_color:
                walkable[gy, gx] = False

    lines = []
    for gy in range(grid_h):
        row = ''
        for gx in range(grid_w):
            c = mini[gy, gx]
            py, px = gy * cell + cell // 2, gx * cell + cell // 2
            is_player = (player_pos and
                         abs(player_pos[0] - py) <= cell and
                         abs(player_pos[1] - px) <= cell)
            is_target = (target_pos and
                         abs(target_pos[0] - py) <= cell and
                         abs(target_pos[1] - px) <= cell)

            if is_player:
                row += 'P'
            elif is_target:
                row += 'T'
            elif c == bg_color:
                row += '.'
            elif c == wall_color:
                row += '#'
            else:
                row += '~'
        lines.append(row)

    # Pixel-level BFS pathfinding — compute actual walkable path
    # Uses step size from spatial model or defaults to 5
    if player_pos and target_pos:
        from collections import deque
        step = 5  # default ARC step size
        h_f, w_f = frame.shape[:2]
        py_s, px_s = player_pos
        ty_s, tx_s = target_pos
        visited_bfs = set()
        queue_bfs = deque([(py_s, px_s, [])])
        visited_bfs.add((py_s, px_s))
        found_path = None
        while queue_bfs and len(visited_bfs) < 5000:
            cy, cx, path = queue_bfs.popleft()
            if abs(cy - ty_s) <= step and abs(cx - tx_s) <= step:
                found_path = path
                break
            for dy, dx, dname in [(-step,0,'UP'),(step,0,'DOWN'),(0,-step,'LEFT'),(0,step,'RIGHT')]:
                ny, nx = cy+dy, cx+dx
                if (ny, nx) in visited_bfs or ny < 0 or ny >= h_f or nx < 0 or nx >= w_f:
                    continue
                # Check entire path for walls (catches thin walls between steps)
                _path_ok = True
                _psteps = max(abs(dy), abs(dx))
                for _pi in range(1, _psteps + 1):
                    _my = cy + (dy * _pi) // _psteps
                    _mx = cx + (dx * _pi) // _psteps
                    if 0 <= _my < h_f and 0 <= _mx < w_f:
                        if frame[_my, _mx] == wall_color:
                            _path_ok = False; break
                if _path_ok:
                    visited_bfs.add((ny, nx))
                    queue_bfs.append((ny, nx, path + [dname]))
        if found_path:
            compressed = []
            i = 0
            while i < len(found_path):
                d = found_path[i]
                count = 1
                while i + count < len(found_path) and found_path[i+count] == d:
                    count += 1
                compressed.append(f"{d} {count}" if count > 1 else d)
                i += count
            lines.append("")
            lines.append(f"🗺️ PATH TO TARGET ({len(found_path)} moves): {' → '.join(compressed)}")
            lines.append(f"FOLLOW THIS PATH EXACTLY. Each move = {step}px.")

    return '\n'.join(lines)


def laser_beam(frame: np.ndarray, start: tuple, target: tuple,
               color_names: dict = None) -> dict:
    """Missile tracking laser beam — trace an imaginary line from start to target.

    Projects a beam pixel-by-pixel from your position to a target and reports
    what's in the path. Like planning a move in your head — the beam is
    imaginary, a red line you superimpose and trace.

    Args:
        frame: 2D numpy array of palette indices
        start: (row, col) origin — typically player position
        target: (row, col) destination — where you want to reach
        color_names: optional dict mapping color index to name (e.g. {3: 'wall'})

    Returns:
        dict with:
            path: list of (row, col, color) tuples along the beam
            obstacles: list of (row, col, color) where non-corridor pixels block
            clear: bool — True if no obstacles between start and target
            summary: human-readable string of the beam trace
            segments: list of (color, count) run-length segments along the beam
    """
    if frame.ndim == 3:
        frame = frame[0]

    r0, c0 = int(start[0]), int(start[1])
    r1, c1 = int(target[0]), int(target[1])

    # Bresenham's line algorithm
    path = []
    dr = abs(r1 - r0)
    dc = abs(c1 - c0)
    sr = 1 if r1 > r0 else -1
    sc = 1 if c1 > c0 else -1
    err = dr - dc
    r, c = r0, c0

    h, w = frame.shape
    while True:
        if 0 <= r < h and 0 <= c < w:
            path.append((r, c, int(frame[r, c])))
        if r == r1 and c == c1:
            break
        e2 = 2 * err
        if e2 > -dc:
            err -= dc
            r += sr
        if e2 < dr:
            err += dr
            c += sc

    # Default color names for LS20/ARC palette
    names = {
        0: 'border', 1: 'modifier', 2: 'color2', 3: 'wall',
        4: 'corridor', 5: 'background', 6: 'color6', 7: 'color7',
        8: 'lives', 9: 'target', 10: 'color10', 11: 'fuel', 12: 'player',
    }
    if color_names:
        names.update(color_names)

    # Find obstacles (anything that isn't corridor, background, or player)
    passable = {4, 5, 12}  # corridor, background, player
    obstacles = [(r, c, v) for r, c, v in path if v not in passable]

    # Run-length encode segments
    segments = []
    if path:
        cur_color, cur_count = path[0][2], 1
        for _, _, v in path[1:]:
            if v == cur_color:
                cur_count += 1
            else:
                segments.append((cur_color, cur_count))
                cur_color, cur_count = v, 1
        segments.append((cur_color, cur_count))

    # Build summary
    lines = [f"BEAM: ({r0},{c0}) → ({r1},{c1}) | {len(path)} pixels"]
    for color, count in segments:
        name = names.get(color, f'color{color}')
        marker = '🟥' if color not in passable else '🟩'
        lines.append(f"  {marker} {name}({color}) × {count}")

    if obstacles:
        lines.append(f"BLOCKED: {len(obstacles)} obstacle pixels")
        # Show first and last obstacle
        first = obstacles[0]
        last = obstacles[-1]
        lines.append(f"  First obstacle: ({first[0]},{first[1]}) = {names.get(first[2], first[2])}")
        lines.append(f"  Last obstacle:  ({last[0]},{last[1]}) = {names.get(last[2], last[2])}")
    else:
        lines.append("CLEAR: no obstacles in path")

    return {
        'path': path,
        'obstacles': obstacles,
        'clear': len(obstacles) == 0,
        'summary': '\n'.join(lines),
        'segments': segments,
    }


def frame_to_grid_text(frame: np.ndarray) -> str:
    """Convert frame to a text grid showing color indices.

    This gives the mind a symbolic representation alongside the visual.
    For small logical grids, detects the cell size and produces a clean grid.
    """
    if frame.ndim == 3 and frame.shape[0] == 1:
        frame = frame[0]
    if frame.ndim == 3:
        # RGB frame — map unique colors to indices 0-N for cleaner grid text
        h, w, ch = frame.shape
        flat = frame.reshape(-1, ch)
        unique_colors = {}
        indexed = np.zeros(h * w, dtype=np.int32)
        # Map black/near-black to 0 first
        for i, pixel in enumerate(flat):
            key = tuple(pixel[:3])
            if key not in unique_colors:
                if int(key[0]) + int(key[1]) + int(key[2]) < 30:  # near-black = 0
                    unique_colors[key] = 0
                else:
                    unique_colors[key] = len(unique_colors) if 0 not in unique_colors.values() else len(unique_colors)
            indexed[i] = unique_colors[key]
        frame = indexed.reshape(h, w)
    h, w = frame.shape

    # Try to detect logical grid cell size by finding repeating color boundaries
    # Look for the smallest step where colors change consistently
    cell_size = _detect_cell_size(frame)

    if cell_size > 1:
        # Sample at cell centers for clean grid
        lines = []
        for y in range(cell_size // 2, h, cell_size):
            row = []
            for x in range(cell_size // 2, w, cell_size):
                v = int(frame[y, x])
                row.append(f"{v:x}")
            lines.append(" ".join(row))
        grid_w = len(lines[0].split()) if lines else 0
        grid_h = len(lines)
        header = f"[{grid_w}x{grid_h} logical grid, cell_size={cell_size}px]"
        return header + "\n" + "\n".join(lines)
    else:
        # Raw pixel grid — sample if too large
        # For 64x64: step=1 (full res, ~8K chars). Only downsample above 64.
        step = max(1, max(h, w) // 64)
        lines = []
        for y in range(0, h, step):
            row = []
            for x in range(0, w, step):
                v = int(frame[y, x])
                row.append(f"{v:x}")
            lines.append(" ".join(row))
        return "\n".join(lines)


def _detect_cell_size(frame: np.ndarray) -> int:
    """Detect logical grid cell size from pixel-level frame.

    Conservative: check ALL cells, not just a corner sample.
    Only use cell detection if >=90% of cells are uniform.
    This prevents losing single-pixel signals that fall between cell centers.
    """
    h, w = frame.shape
    # Check common cell sizes (4, 8, 16) — skip 2 (too easy to false-positive)
    for cs in [4, 8, 16]:
        if h % cs != 0 or w % cs != 0:
            continue
        # Check ALL cells, not just a corner
        total_cells = 0
        uniform_cells = 0
        for y in range(0, h, cs):
            for x in range(0, w, cs):
                cell = frame[y:y+cs, x:x+cs]
                total_cells += 1
                if np.all(cell == cell[0, 0]):
                    uniform_cells += 1
        # Only accept if 90%+ cells are uniform (very high confidence)
        if total_cells > 16 and uniform_cells / total_cells >= 0.90:
            return cs
    return 1


def compute_diff(before: np.ndarray, after: np.ndarray) -> dict:
    """Compute what changed between two frames. Detects movement."""
    # Reduce to 2D: handle RGB (H,W,3), batch (1,H,W), etc.
    if before.ndim == 3:
        if before.shape[0] == 1:
            before = before[0]
        elif before.shape[2] <= 4:  # RGB/RGBA — any-channel diff
            before_2d = before
        else:
            before = before[0]
    if after.ndim == 3:
        if after.shape[0] == 1:
            after = after[0]
        elif after.shape[2] <= 4:
            after_2d = after
        else:
            after = after[0]

    # For RGB, diff_mask is per-pixel (any channel changed)
    if before.ndim == 3:
        diff_mask = np.any(before != after, axis=2)
        # Use grayscale for color_changes analysis
        before = np.mean(before, axis=2).astype(np.uint8)
        after = np.mean(after, axis=2).astype(np.uint8)
    else:
        diff_mask = before != after
    n_changed = int(diff_mask.sum())

    if n_changed == 0:
        return {'changed': 0, 'description': 'nothing changed (action blocked or no effect)'}

    total = before.size
    pct = n_changed / total * 100

    rows, cols = np.where(diff_mask)
    bbox = {
        'top': int(rows.min()), 'bottom': int(rows.max()),
        'left': int(cols.min()), 'right': int(cols.max()),
    }

    # Detect movement: find color swaps (A→B at one place, B→A at another)
    movements = []
    color_changes = {}
    for r, c in zip(rows, cols):
        old_v, new_v = int(before[r, c]), int(after[r, c])
        key = (old_v, new_v)
        if key not in color_changes:
            color_changes[key] = []
        color_changes[key].append((int(c), int(r)))

    # Check for translation: if color A→B pixels and B→A pixels exist,
    # compute the center-of-mass shift = movement direction
    for (old_c, new_c), positions in color_changes.items():
        reverse_key = (new_c, old_c)
        if reverse_key in color_changes and old_c != new_c:
            pos_arr = np.array(positions)
            rev_arr = np.array(color_changes[reverse_key])
            # Center of mass
            old_center = pos_arr.mean(axis=0)
            new_center = rev_arr.mean(axis=0)
            dx = new_center[0] - old_center[0]
            dy = new_center[1] - old_center[1]
            if abs(dx) > 1 or abs(dy) > 1:
                direction = ""
                if abs(dy) > abs(dx):
                    direction = "DOWN" if dy > 0 else "UP"
                else:
                    direction = "RIGHT" if dx > 0 else "LEFT"
                movements.append(
                    f"object (color {old_c}) moved {direction} by ~({dx:.0f},{dy:.0f})px "
                    f"from center ({old_center[0]:.0f},{old_center[1]:.0f}) "
                    f"to ({new_center[0]:.0f},{new_center[1]:.0f})"
                )

    # Build description
    parts = [f"{n_changed}px changed"]
    if movements:
        # Deduplicate (movement detected both ways)
        seen = set()
        for m in movements:
            key = m.split("moved")[1] if "moved" in m else m
            if key not in seen:
                seen.add(key)
                parts.append(m)
    else:
        parts.append(f"region: ({bbox['left']},{bbox['top']}) to ({bbox['right']},{bbox['bottom']})")
        # Summarize color changes concisely
        for (old_c, new_c), positions in list(color_changes.items())[:4]:
            parts.append(f"  color {old_c}→{new_c}: {len(positions)}px")

    desc = "; ".join(parts)

    return {
        'changed': n_changed,
        'pct': pct,
        'bbox': bbox,
        'movements': movements,
        'description': desc,
    }


def composite_b64(before: np.ndarray, after: np.ndarray, scale: int = 4) -> str:
    """Create a side-by-side composite: BEFORE | AFTER | DIFF.

    This gives the mind visual comparison — the way a human would see it.
    The diff panel shows: red = removed, green = added, dim = unchanged.
    Labels are added above each panel.
    """
    if not HAS_PIL:
        return ""
    if before.ndim == 3:
        before = before[0] if before.shape[0] == 1 else before
    if after.ndim == 3:
        after = after[0] if after.shape[0] == 1 else after

    # Handle size mismatches — resize to match
    if before.shape != after.shape:
        # Use the larger shape
        target_h = max(before.shape[0], after.shape[0])
        target_w = max(before.shape[1] if before.ndim > 1 else before.shape[0],
                       after.shape[1] if after.ndim > 1 else after.shape[0])
        if before.shape != (target_h, target_w):
            before = np.pad(before, ((0, target_h - before.shape[0]),
                                     (0, target_w - (before.shape[1] if before.ndim > 1 else before.shape[0]))),
                           mode='constant', constant_values=0)[:target_h, :target_w]
        if after.shape != (target_h, target_w):
            after = np.pad(after, ((0, target_h - after.shape[0]),
                                   (0, target_w - (after.shape[1] if after.ndim > 1 else after.shape[0]))),
                          mode='constant', constant_values=0)[:target_h, :target_w]

    h, w = before.shape[:2]

    # Create RGB versions
    before_rgb = frame_to_rgb(before)
    after_rgb = frame_to_rgb(after)

    # Create diff panel
    diff_mask = before != after
    diff_rgb = np.zeros((h, w, 3), dtype=np.uint8)
    unchanged = ~diff_mask
    diff_rgb[unchanged] = (after_rgb[unchanged] * 0.3).astype(np.uint8)

    # Show old position in red, new position in green
    for r in range(h):
        for c in range(w):
            if diff_mask[r, c]:
                old_v, new_v = int(before[r, c]), int(after[r, c])
                # If old was background (0 or 4) and new is something, it's green (arrived)
                # If old was something and new is background, it's red (departed)
                if old_v in (0, 4) or old_v == new_v:
                    diff_rgb[r, c] = [0, 255, 0]  # green: something arrived
                else:
                    diff_rgb[r, c] = [255, 0, 0]  # red: something departed

    # Composite: before | gap | after | gap | diff
    gap = 2
    label_h = 8  # space for labels
    total_w = w * 3 + gap * 2
    total_h = h + label_h

    composite = np.ones((total_h, total_w, 3), dtype=np.uint8) * 40  # dark gray background

    # Place panels
    y0 = label_h
    composite[y0:y0+h, 0:w] = before_rgb
    composite[y0:y0+h, w+gap:2*w+gap] = after_rgb
    composite[y0:y0+h, 2*w+2*gap:3*w+2*gap] = diff_rgb

    img = Image.fromarray(composite)
    if scale > 1:
        img = img.resize((img.width * scale, img.height * scale), Image.NEAREST)

    # Add text labels using PIL (simple approach)
    try:
        from PIL import ImageDraw
        draw = ImageDraw.Draw(img)
        s = scale
        draw.text((2*s, 1*s), "BEFORE", fill=(255, 255, 255))
        draw.text(((w+gap)*s + 2*s, 1*s), "AFTER", fill=(255, 255, 255))
        draw.text(((2*w+2*gap)*s + 2*s, 1*s), "DIFF", fill=(255, 255, 0))
    except Exception:
        pass

    buf = io.BytesIO()
    img.save(buf, format='PNG')
    return base64.b64encode(buf.getvalue()).decode()


# ---------------------------------------------------------------------------
# LLM interface
# ---------------------------------------------------------------------------

def call_llm_ollama(messages: list, images: list = None, max_tokens: int = 1000,
                    temperature: float = 0.3) -> Optional[str]:
    """Fallback: call Ollama. Supports vision via MiniCPM-V on Hetzner.
    Priority: Hetzner vision (minicpm-v:8b) → Hetzner text (qwen2.5:32b) → local (qwen2.5:7b)."""
    if not HAS_REQUESTS:
        return None

    # Convert to Ollama format
    ollama_msgs = []
    for msg in messages:
        ollama_msgs.append({'role': msg['role'], 'content': msg['content']})

    # If we have images and a vision model is available, use it
    has_images = images and any(img for img in images)

    # Build target list: (port, model, use_images)
    targets = []
    if has_images:
        targets.append((11435, 'minicpm-v:8b', True))    # Hetzner vision
    # Hetzner has no GPU — 72B is too slow on CPU. Use 7B (fast) or 32B (moderate).
    targets.append((11435, 'qwen2.5:7b', False))           # Hetzner text (7B — fast on CPU)
    targets.append((11435, 'qwen2.5:32b', False))          # Hetzner text (32B — slower fallback)
    # Local Ollama: last resort if Hetzner tunnel down and Anthropic/RunPod exhausted
    targets.append((11434, 'qwen2.5:7b', False))           # Local text (last resort)

    for port, model, use_img in targets:
        try:
            url = f'http://localhost:{port}/api/chat'
            req_msgs = list(ollama_msgs)  # copy

            # Attach images for vision model
            if use_img and has_images:
                # Ollama expects images in the message content
                for msg in req_msgs:
                    if msg['role'] == 'user':
                        msg['images'] = [img for img in images if img]
                        break

            resp = _requests.post(
                url,
                json={'model': model, 'messages': req_msgs, 'stream': False,
                      'options': {'temperature': temperature, 'num_predict': max_tokens}},
                timeout=300,  # Hetzner CPU can be slow
            )
            resp.raise_for_status()
            result = resp.json()['message']['content']
            label = f"{'vision' if use_img else 'text'} {model}"
            log.info(f"Ollama ({label} port {port}) responded")
            return result
        except Exception as e:
            log.info(f"Ollama {model}@{port} failed: {type(e).__name__}")
    return None


_anthropic_credits_exhausted = False  # sticky flag — skip Anthropic once credits confirmed dead

# API cost tracking — accumulates across all call_llm() invocations
_api_cost_usd = 0.0
_api_calls = 0
_api_input_tokens = 0
_api_output_tokens = 0
_api_consecutive_errors = 0
_API_PRICES = {  # per million tokens
    'opus': {'input': 15.00, 'output': 75.00},
    'sonnet': {'input': 3.00, 'output': 15.00},
    'haiku': {'input': 0.80, 'output': 4.00},
}
# Circuit breaker thresholds (env-overridable)
_COST_CEILING = float(os.environ.get('ARC_COST_CEILING', '10.0'))  # $10 per run
_ERROR_STREAK_MAX = int(os.environ.get('ARC_ERROR_STREAK_MAX', '5'))  # 5 consecutive failures

def get_api_cost_summary() -> dict:
    """Return cumulative API cost for this process."""
    return {
        'total_usd': round(_api_cost_usd, 4),
        'calls': _api_calls,
        'input_tokens': _api_input_tokens,
        'output_tokens': _api_output_tokens,
    }

def corpus_callosum_qwen(frame_before: np.ndarray, frame_after: np.ndarray,
                         action_desc: str = "", grid_before: str = "", grid_after: str = "") -> Optional[str]:
    """Corpus callosum via Qwen-VL: perceive what changed between two frames.

    Sends BOTH:
    - Pixel art images (before/after as PNG)
    - ASCII grid text (hex color indices)

    Returns natural language description of changes, or None on failure.
    """
    import base64, io
    try:
        from PIL import Image
    except ImportError:
        Image = None

    def _frame_to_b64(frame: np.ndarray, scale: int = 6) -> str:
        """Convert frame to upscaled PNG base64."""
        if Image is None:
            return ""
        if frame.ndim == 2:
            frame = np.stack([frame]*3, axis=-1)
        img = Image.fromarray(frame.astype(np.uint8))
        if scale > 1:
            img = img.resize((img.width * scale, img.height * scale), Image.NEAREST)
        buf = io.BytesIO()
        img.save(buf, format='PNG')
        return base64.b64encode(buf.getvalue()).decode()

    images = []
    # Send images only if NOT in text-only mode (vLLM max_model_len=4096 can't handle images)
    if not os.environ.get('ARC_TEXT_ONLY'):
        b64_after = _frame_to_b64(frame_after, scale=4)
        if b64_after:
            images.append(b64_after)

    # Build prompt with both ASCII and pixel context
    prompt_parts = ["You are analyzing an ARC puzzle game. Two frames are shown: BEFORE and AFTER an action."]
    prompt_parts.append(f"Action taken: {action_desc}" if action_desc else "An action was taken.")

    if grid_before:
        prompt_parts.append(f"\nBEFORE grid (hex color indices):\n```\n{grid_before}\n```")
    if grid_after:
        prompt_parts.append(f"\nAFTER grid (hex color indices):\n```\n{grid_after}\n```")

    if images:
        prompt_parts.append("\nThe two images above show the BEFORE and AFTER frames as pixel art (upscaled 6x).")

    prompt_parts.append("\nDescribe PRECISELY what changed: which cells changed color, any patterns in the change, and what this tells us about the game mechanics. Be concise (2-3 sentences max).")

    messages = [{'role': 'user', 'content': '\n'.join(prompt_parts)}]
    return call_llm_qwen_vl(messages, images=images, max_tokens=200, temperature=0.1)


def call_llm_qwen_vl(messages: list, images: list = None, max_tokens: int = 1000,
                     temperature: float = 0.3) -> Optional[str]:
    """Call Qwen2.5-VL-32B via RunPod (serverless or SSH tunnel).

    Supports two modes:
    1. RunPod serverless: Set RUNPOD_ENDPOINT_ID + RUNPOD_API_KEY (no tunnel needed)
    2. SSH tunnel: Set ARC_QWEN_VL_URL=http://localhost:8100/v1/chat/completions

    Env: ARC_USE_QWEN_VL=1 (perception) or ARC_USE_QWEN_VL=primary (full reasoning)"""
    if not HAS_REQUESTS:
        return None

    # RunPod serverless endpoint (direct, no tunnel needed)
    _runpod_endpoint = os.environ.get('RUNPOD_ENDPOINT_ID', '')
    _runpod_key = os.environ.get('RUNPOD_API_KEY', '')
    if _runpod_endpoint and _runpod_key:
        url = f'https://api.runpod.ai/v2/{_runpod_endpoint}/openai/v1/chat/completions'
        _auth_headers = {'Authorization': f'Bearer {_runpod_key}'}
    else:
        url = os.environ.get('ARC_QWEN_VL_URL', 'http://localhost:8100/v1/chat/completions')
        _auth_headers = {}
    model = os.environ.get('ARC_QWEN_VL_MODEL', 'Qwen/Qwen2.5-VL-32B-Instruct')

    # Build OpenAI-format messages array (preserve system + user roles)
    oai_messages = []
    for msg in messages:
        if msg['role'] == 'system':
            oai_messages.append({'role': 'system', 'content': msg['content']})
        elif msg['role'] == 'user':
            content = []
            if images and not oai_messages:
                # Attach images to first user message only
                for img_b64 in images:
                    if img_b64:
                        content.append({
                            'type': 'image_url',
                            'image_url': {'url': f'data:image/png;base64,{img_b64}'}
                        })
                images = None  # only attach once
            content.append({'type': 'text', 'text': msg['content']})
            oai_messages.append({'role': 'user', 'content': content})
        elif msg['role'] == 'assistant':
            oai_messages.append({'role': 'assistant', 'content': msg['content']})

    if not oai_messages:
        return None

    # Truncate system prompt to fit within context window (16384 tokens ≈ 60K chars)
    _max_input_chars = 50000  # ~12500 tokens, leaves room for output
    total_chars = sum(len(m.get('content', '') if isinstance(m.get('content'), str) else str(m.get('content', ''))) for m in oai_messages)
    if total_chars > _max_input_chars:
        # Truncate system message (usually the longest)
        for m in oai_messages:
            if m['role'] == 'system' and isinstance(m['content'], str) and len(m['content']) > 10000:
                excess = total_chars - _max_input_chars
                m['content'] = m['content'][:len(m['content']) - excess - 200] + "\n\n[...truncated for context limit...]"
                break

    # RunPod serverless has cold start — allow longer timeout
    _timeout = 300 if _runpod_endpoint else 120
    try:
        resp = _requests.post(url, json={
            'model': model,
            'messages': oai_messages,
            'max_tokens': max_tokens,
            'temperature': temperature,
        }, headers=_auth_headers, timeout=_timeout)
        resp.raise_for_status()
        data = resp.json()
        result = data['choices'][0]['message']['content']
        usage = data.get('usage', {})
        _src = "RunPod serverless" if _runpod_endpoint else "tunnel"
        log.info(f"[QWEN-VL] {usage.get('prompt_tokens',0)}in/{usage.get('completion_tokens',0)}out | $0 ({_src})")
        return result
    except Exception as e:
        # Get response body for 400 errors
        _err_body = ""
        if hasattr(e, 'response') and hasattr(e.response, 'text'):
            _err_body = f" | Body: {e.response.text[:300]}"
        log.warning(f"Qwen-VL call failed: {e}{_err_body}")
        return None


def call_llm(messages: list, images: list = None, max_tokens: int = 1000,
             temperature: float = 0.3, fast: bool = False) -> Optional[str]:
    """Call LLM. Tries Anthropic first, falls back to Ollama (text-only).
    Returns None if ARC_NO_LLM is set (offline/algorithmic-only mode).
    If fast=True, uses ARC_PILOT_MODEL_FAST (default: Sonnet) for lower latency.
    Circuit breaker: halts if cost exceeds $COST_CEILING or errors exceed streak max."""
    global _anthropic_credits_exhausted, _api_consecutive_errors, _api_cost_usd, _api_calls, _api_input_tokens, _api_output_tokens
    if os.environ.get('ARC_NO_LLM'):
        return None
    # ARC_USE_QWEN_VL=primary: Qwen does BOTH perception and reasoning (emergency mode when credits exhausted)
    # ARC_USE_QWEN_VL=1: Qwen does PERCEPTION only (corpus callosum), Opus/Claude does reasoning (preferred architecture)
    if os.environ.get('ARC_USE_QWEN_VL') == 'primary':
        _qwen_max = min(max_tokens, 800) if os.environ.get('ARC_TEXT_ONLY') else max_tokens
        return call_llm_qwen_vl(messages, images=images, max_tokens=_qwen_max, temperature=temperature)
    # Circuit breaker — cost ceiling
    if _COST_CEILING > 0 and _api_cost_usd >= _COST_CEILING:
        log.warning(f"[CIRCUIT BREAKER] Cost ceiling hit: ${_api_cost_usd:.2f} >= ${_COST_CEILING:.2f}. Falling back.")
        if os.environ.get('ARC_USE_QWEN_VL'):
            qwen_result = call_llm_qwen_vl(messages, images=images, max_tokens=max_tokens, temperature=temperature)
            if qwen_result:
                return qwen_result
        return call_llm_ollama(messages, images=images, max_tokens=max_tokens, temperature=temperature)
    # Circuit breaker — error streak
    if _api_consecutive_errors >= _ERROR_STREAK_MAX:
        log.warning(f"[CIRCUIT BREAKER] {_api_consecutive_errors} consecutive errors. Falling back.")
        if os.environ.get('ARC_USE_QWEN_VL'):
            qwen_result = call_llm_qwen_vl(messages, images=images, max_tokens=max_tokens, temperature=temperature)
            if qwen_result:
                return qwen_result
        return call_llm_ollama(messages, images=images, max_tokens=max_tokens, temperature=temperature)
    if _anthropic_credits_exhausted:
        # Try Qwen-VL first (RunPod H100, free), then Ollama
        if os.environ.get('ARC_USE_QWEN_VL'):
            qwen_result = call_llm_qwen_vl(messages, images=images, max_tokens=max_tokens, temperature=temperature)
            if qwen_result:
                return qwen_result
        return call_llm_ollama(messages, images=images, max_tokens=max_tokens, temperature=temperature)
    # If Qwen-VL is the primary backend, use it directly (skip Anthropic entirely)
    if os.environ.get('ARC_USE_QWEN_VL') == 'primary':
        # Cap max_tokens for vLLM's limited context (4096 total)
        _qwen_max = min(max_tokens, 800)
        return call_llm_qwen_vl(messages, images=images, max_tokens=_qwen_max, temperature=temperature)

    api_key = os.environ.get('ANTHROPIC_API_KEY', '') or os.environ.get('ANTHROPIC', '')
    if not api_key or not HAS_REQUESTS:
        log.info("No Anthropic key — trying Qwen-VL then Ollama")
        if os.environ.get('ARC_USE_QWEN_VL'):
            qwen_result = call_llm_qwen_vl(messages, images=images, max_tokens=max_tokens, temperature=temperature)
            if qwen_result:
                return qwen_result
        return call_llm_ollama(messages, images=images, max_tokens=max_tokens, temperature=temperature)

    # Competition: set ARC_PILOT_MODEL_FAST=claude-haiku-4-5-20251001 for budget efficiency
    # Dev: Sonnet (fast) or Opus (full reasoning)
    if fast:
        model = os.environ.get('ARC_PILOT_MODEL_FAST', 'claude-haiku-4-5-20251001')
    else:
        model = os.environ.get('ARC_PILOT_MODEL', 'claude-sonnet-4-20250514')

    # Build content for the last user message, prepending images
    content = []
    if images:
        for img_b64 in images:
            if img_b64:
                content.append({
                    'type': 'image',
                    'source': {'type': 'base64', 'media_type': 'image/png', 'data': img_b64},
                })

    # Add text from messages
    for msg in messages:
        if msg['role'] == 'user':
            content.append({'type': 'text', 'text': msg['content']})

    # Extract system prompt and enable prompt caching (90% cost reduction on cached prefix)
    system_text = messages[0]['content'] if messages[0]['role'] == 'system' else ''
    if system_text:
        system_payload = [{'type': 'text', 'text': system_text, 'cache_control': {'type': 'ephemeral'}}]
    else:
        system_payload = []

    for attempt in range(3):
        try:
            resp = _requests.post(
                'https://api.anthropic.com/v1/messages',
                headers={
                    'x-api-key': api_key,
                    'anthropic-version': '2023-06-01',
                    'Content-Type': 'application/json',
                },
                json={
                    'model': model,
                    'max_tokens': max_tokens,
                    'system': system_payload,
                    'messages': [{'role': 'user', 'content': content}],
                },
                timeout=60,
            )
            resp.raise_for_status()
            data = resp.json()
            # Track API cost from usage data (including prompt cache savings)
            usage = data.get('usage', {})
            in_tok = usage.get('input_tokens', 0)
            out_tok = usage.get('output_tokens', 0)
            cache_read = usage.get('cache_read_input_tokens', 0)
            cache_write = usage.get('cache_creation_input_tokens', 0)
            if in_tok or out_tok:
                tier = 'opus' if 'opus' in model else 'sonnet' if 'sonnet' in model else 'haiku'
                prices = _API_PRICES.get(tier, _API_PRICES['opus'])
                # Non-cached input tokens = total - cache_read - cache_write
                regular_in = max(0, in_tok - cache_read - cache_write)
                cost = (regular_in * prices['input'] + out_tok * prices['output']) / 1_000_000
                # Cache costs: write = 1.25x input, read = 0.1x input
                cost += (cache_write * prices['input'] * 1.25 + cache_read * prices['input'] * 0.1) / 1_000_000
                _api_cost_usd += cost
                _api_calls += 1
                _api_input_tokens += in_tok
                _api_output_tokens += out_tok
                cache_info = f" cache:{cache_read}r/{cache_write}w" if cache_read or cache_write else ""
                log.info(f"[COST] ${cost:.4f} ({tier}, {in_tok}in/{out_tok}out{cache_info}) | session total: ${_api_cost_usd:.2f} / {_api_calls} calls")
            _api_consecutive_errors = 0  # reset error streak on success
            return data['content'][0]['text']
        except Exception as e:
            _api_consecutive_errors += 1
            # Log response body for debugging 400s
            if hasattr(e, 'response') and e.response is not None:
                try:
                    err_body = e.response.json()
                    log.warning(f"LLM error body: {err_body}")
                except Exception:
                    log.warning(f"LLM error text: {e.response.text[:500]}")
            is_rate_limit = '429' in str(e) or 'rate_limit' in str(e).lower()
            if is_rate_limit and attempt < 2:
                wait = (attempt + 1) * 15  # 15s, 30s
                log.info(f"Rate limited — waiting {wait}s before retry {attempt+2}/3")
                import time; time.sleep(wait)
                continue
            # If credit/billing error, fall back to Ollama immediately
            err_text = str(e).lower()
            if hasattr(e, 'response') and e.response is not None:
                try:
                    err_text += ' ' + e.response.text.lower()
                except Exception:
                    pass
            is_credit_error = 'credit balance' in err_text or 'billing' in err_text
            if is_credit_error:
                _anthropic_credits_exhausted = True
                log.info("Anthropic credits exhausted — falling back to Ollama (sticky)")
                return call_llm_ollama(messages, images=images, max_tokens=max_tokens, temperature=temperature)
            log.warning(f"LLM call failed: {e}")
            return None


# ---------------------------------------------------------------------------
# The Mind
# ---------------------------------------------------------------------------

class Gundam:
    """Pure AGI pilot. Observe, hypothesize, plan, execute, adapt.

    No BFS. No MCTS. No constraint solver. Just thinking.
    """

    def __init__(self, budget_per_level: int = 30, verbose: bool = True,
                 text_only: bool = False):
        self.budget_per_level = budget_per_level  # max LLM calls per level
        self.memory = GameMemory()
        self.spatial = SpatialModel()
        self.episodic = EpisodicMemory() if HAS_EYES else None
        self.verbose = verbose
        self.text_only = text_only  # skip images, use ASCII only (cheaper)
        self.total_cost = 0.0
        self.total_llm_calls = 0
        self._consecutive_no_change = 0  # track stalling
        self._recalled_memories = []  # from past games
        self._moves_since_reset = 0  # fuel/resource tracking
        self._total_resets = 0  # how many times the game has reset
        self._last_reset_turn = 0  # turn when last reset happened
        # Phase 4: Imagination — internal simulation for path planning
        self.world_model = WorldModel() if HAS_IMAGINATION else None
        self._imagined_path = []  # planned path from simulation
        self._last_action = -1
        # Phase 5: CNN-lite action predictor (StochasticGoose-inspired)
        self.action_predictor = None  # initialized in observe_initial with n_actions
        # Grid Transform — Fourier transform for LLMs (L2-L5)
        self.causal_ledger = CausalLedger() if CausalLedger else None

    def _log(self, msg: str):
        if self.verbose:
            print(f"  [gundam] {msg}")

    # -- Phase 0: PREFLIGHT — Undo Your Brainwashing ----------------------
    # "Why did you assume the glyph wasn't a target?" — Peter, 2026-03-16
    # The pilot's greatest failure mode: inherited assumptions from past
    # reasoning that become invisible filters. Preflight forces fresh eyes.

    def preflight(self, env, extract_frame, level: int) -> dict:
        """Anti-assumption discovery protocol. Runs before the pilot thinks.

        Every fork, every compaction, every new level — start here.
        Don't assume you know what anything is. LOOK. TRY. DIFF. ASK.

        Returns a preflight report dict with:
          - inventory: {color: count, positions} for EVERY color on the board
          - action_effects: what each action does (full frame diff)
          - objects: distinct non-background regions with their properties
          - assumptions: explicitly named assumptions to challenge
          - hypotheses: initial guesses about goal, player, target
        """
        self._log("=" * 50)
        self._log("PREFLIGHT — Fresh eyes. No assumptions.")
        self._log("=" * 50)

        report = {
            'inventory': {},
            'action_effects': {},
            'objects': [],
            'assumptions': [],
            'hypotheses': [],
        }

        # --- Step 1: WHAT DO I SEE? Full inventory, no filtering ---
        frame = extract_frame(env)
        f2d = frame[0] if frame.ndim == 3 else frame
        h, w = f2d.shape

        self._log("\n[STEP 1] Full pixel inventory — every color, every count:")
        unique_colors, counts = np.unique(f2d, return_counts=True)
        bg_color = unique_colors[np.argmax(counts)]  # most common = likely background

        for color, count in sorted(zip(unique_colors, counts), key=lambda x: -x[1]):
            locs = np.where(f2d == color)
            rows, cols = locs[0], locs[1]
            r_min, r_max = int(rows.min()), int(rows.max())
            c_min, c_max = int(cols.min()), int(cols.max())
            role = "background?" if color == bg_color else "unknown"
            report['inventory'][int(color)] = {
                'count': int(count),
                'bbox': (r_min, c_min, r_max, c_max),
                'role': role,
            }
            self._log(f"  Color {color:2d}: {count:5d} px, "
                      f"rows {r_min}-{r_max}, cols {c_min}-{c_max} [{role}]")

        # --- Step 1b: OBJECT SEGMENTATION — connected components per color ---
        # Inspired by JustExplore (3rd place ARC-AGI-3 preview): group
        # same-color contiguous pixels into objects. Small isolated regions
        # are likely buttons/targets; large regions are background/walls.
        try:
            from scipy.ndimage import label as cc_label
            objects_found = []
            for color in unique_colors:
                if color == bg_color:
                    continue
                mask = (f2d == color).astype(np.uint8)
                labeled, n_components = cc_label(mask)
                for comp_id in range(1, n_components + 1):
                    comp_mask = (labeled == comp_id)
                    px_count = int(comp_mask.sum())
                    rows_c = np.where(comp_mask)[0]
                    cols_c = np.where(comp_mask)[1]
                    centroid = (float(rows_c.mean()), float(cols_c.mean()))
                    bbox = (int(rows_c.min()), int(cols_c.min()),
                            int(rows_c.max()), int(cols_c.max()))
                    obj = {
                        'color': int(color),
                        'pixels': px_count,
                        'centroid': centroid,
                        'bbox': bbox,
                        'is_small': px_count < max(4, h * w * 0.005),  # <0.5% = button-like
                    }
                    objects_found.append(obj)
            # Sort: small objects first (likely interactive)
            objects_found.sort(key=lambda o: o['pixels'])
            report['objects'] = objects_found
            n_small = sum(1 for o in objects_found if o['is_small'])
            self._log(f"\n[STEP 1b] Object segmentation: {len(objects_found)} objects "
                      f"({n_small} small/button-like)")
            for obj in objects_found[:8]:  # show first 8
                tag = "BUTTON?" if obj['is_small'] else "region"
                self._log(f"  Color {obj['color']:2d}: {obj['pixels']:4d}px "
                          f"@ ({obj['centroid'][0]:.0f},{obj['centroid'][1]:.0f}) [{tag}]")
        except ImportError:
            self._log("\n[STEP 1b] scipy not available — skipping object segmentation")
        except Exception as e:
            self._log(f"\n[STEP 1b] Object segmentation failed: {e}")

        # --- Step 2: WHAT CAN I DO? Try each action, full diff ---
        self._log("\n[STEP 2] Action probing — try each action, diff EVERYTHING:")
        n_actions = len(env.last_obs.available_actions) if hasattr(env, 'last_obs') else 4
        before_frame = f2d.copy()

        import copy as _cp_pf
        for action_id in range(min(n_actions, 8)):  # cap at 8 to save fuel
            probe_env = _cp_pf.deepcopy(env)
            obs = probe_env.step(action_id)
            after_frame = extract_frame(obs)
            a2d = after_frame[0] if after_frame.ndim == 3 else after_frame

            diff_mask = before_frame != a2d
            n_changed = int(diff_mask.sum())

            effect = {
                'pixels_changed': n_changed,
                'changes_by_color': {},
                'regions_affected': [],
            }

            if n_changed > 0:
                diff_rows, diff_cols = np.where(diff_mask)
                # Group changes by what happened (old_color -> new_color)
                for r, c in zip(diff_rows[:50], diff_cols[:50]):  # sample first 50
                    key = f"{before_frame[r,c]}->{a2d[r,c]}"
                    effect['changes_by_color'][key] = \
                        effect['changes_by_color'].get(key, 0) + 1

                # Bounding box of all changes
                effect['regions_affected'].append({
                    'rows': (int(diff_rows.min()), int(diff_rows.max())),
                    'cols': (int(diff_cols.min()), int(diff_cols.max())),
                })

            # VLM diff perception — what does Florence see changed?
            # Only call for first significant change to save time (~3s/call)
            if HAS_VLM and n_changed > 10 and 'vlm_diff_sample' not in report:
                try:
                    vlm_diff = arc_vlm.perceive_diff(before_frame, a2d,
                                                      f"action {action_id}")
                    if vlm_diff:
                        effect['vlm_diff'] = vlm_diff
                        report['vlm_diff_sample'] = action_id
                except Exception:
                    pass

            report['action_effects'][action_id] = effect
            self._log(f"  Action {action_id}: {n_changed} pixels changed"
                      + (f" — {effect['changes_by_color']}" if n_changed > 0 and n_changed < 200 else "")
                      + (f" — VLM: {effect.get('vlm_diff', '')}" if effect.get('vlm_diff') else ""))
            # Note: before_frame stays constant (original frame) since we use deepcopy for each probe

        # --- Step 3: WHAT REACTS TO ME? Identify player, walls, interactive objects ---
        self._log("\n[STEP 3] Object identification from movement:")
        moved_colors = set()
        static_colors = set()
        for aid, eff in report['action_effects'].items():
            for change_key, count in eff['changes_by_color'].items():
                old, new = change_key.split('->')
                moved_colors.add(int(new))
                moved_colors.add(int(old))

        for color in unique_colors:
            if int(color) == int(bg_color):
                continue
            if int(color) in moved_colors:
                report['inventory'][int(color)]['role'] = 'interactive/moving'
            else:
                report['inventory'][int(color)]['role'] = 'static'
                static_colors.add(int(color))

        # --- Step 4: WHAT'S THE GOAL? Look for distinct structures ---
        self._log("\n[STEP 4] Distinct structures — potential targets/goals:")
        current = extract_frame(env)
        c2d = current[0] if current.ndim == 3 else current
        for color in unique_colors:
            if int(color) == int(bg_color):
                continue
            info = report['inventory'][int(color)]
            count = info['count']
            bbox = info['bbox']
            bbox_area = (bbox[2] - bbox[0] + 1) * (bbox[3] - bbox[1] + 1)

            # Small isolated clusters = likely game objects (player, target, modifier)
            if count < 50 and count > 1:
                density = count / max(bbox_area, 1)
                desc = f"Color {color}: {count}px in {bbox}, density={density:.2f}"
                report['objects'].append({
                    'color': int(color),
                    'count': count,
                    'bbox': bbox,
                    'density': round(density, 2),
                })
                self._log(f"  OBJECT: {desc}")

        # --- Step 5: WHAT AM I ASSUMING? Name it to kill it ---
        self._log("\n[STEP 5] Explicit assumptions (CHALLENGE THESE):")
        assumptions = [
            "I don't know what any color means yet — labels come from evidence, not guesses",
            "Every visible object could be navigable, interactive, or decorative — test, don't assume",
            "Small pixel clusters might be targets, tools, or indicators — not 'just decoration'",
            "Objects that look like UI/HUD might be physical game elements you can walk to",
            "The goal is unknown until I have evidence — 'reach the exit' is a guess, not a fact",
            "My shape/appearance might matter — check if it changes and when",
        ]
        report['assumptions'] = assumptions
        for a in assumptions:
            self._log(f"  ⚠ {a}")

        # --- Step 6: Initial hypotheses ---
        self._log("\n[STEP 6] Initial hypotheses (to be tested, not trusted):")
        hyps = []
        n_objects = len(report['objects'])
        n_movement = sum(1 for e in report['action_effects'].values() if e['pixels_changed'] > 0)

        if n_movement > 0:
            hyps.append("This is a movement/navigation game (actions cause pixel displacement)")
        if n_objects >= 2:
            hyps.append(f"There are {n_objects} distinct small objects — some may be player, target, tool")
        if any(info['count'] > 100 for info in report['inventory'].values()
               if info['role'] != 'background?'):
            hyps.append("Large non-background structures exist — walls, borders, or level geometry")

        report['hypotheses'] = hyps
        for h in hyps:
            self._log(f"  ? {h}")

        self._log("\n" + "=" * 50)
        self._log("PREFLIGHT COMPLETE — Now think with fresh eyes.")
        self._log("=" * 50)

        # Store for prompt injection
        self._preflight_report = report
        return report

    # -- Phase 1: OBSERVE --------------------------------------------------

    def observe_initial(self, frame: np.ndarray, n_actions: int,
                        action_info: dict, level: int) -> LevelMemory:
        """First look at a new level. What do we see?"""
        lvl = LevelMemory(level=level)
        lvl.initial_frame_b64 = frame_to_b64(frame)
        lvl.current_frame_b64 = lvl.initial_frame_b64
        lvl.current_frame = frame.copy()
        lvl.grid_text = frame_to_grid_text(frame)
        self.memory.n_actions = n_actions
        self.memory.action_types = action_info
        self.memory.levels[level] = lvl

        # Use Eyes to find sprites/objects in the initial frame
        if HAS_EYES:
            f2d = frame[0] if frame.ndim == 3 else frame
            sprites = detect_sprites(f2d, min_size=2)
            self._log(f"Eyes found {len(sprites)} sprites")
            # Sort by size — largest is likely background/walls, smallest is player/target
            sprites.sort(key=lambda s: s.size)
            objects = []
            for s in sprites:
                role = "unknown"
                if s.size < 20:
                    role = "small_object (player? target?)"
                elif s.size > 500:
                    role = "structure (wall/border)"
                else:
                    role = "medium_object"
                objects.append((s.color, int(s.center[0]), int(s.center[1]), role))
                self._log(f"  Sprite: color={s.color}, pos=({s.center[0]:.0f},{s.center[1]:.0f}), "
                          f"size={s.size}px, {role}")
            self.spatial.objects = objects

            # Don't identify player yet — wait until we see something move.
            # The player is whichever sprite moves when we take actions.
            self._log(f"  Player: TBD (will identify from first movement)")

            # VERIFY: check each small sprite's context.
            # A sprite embedded in walls (surrounded by wall color) is decoration,
            # not a free-standing game object.
            self._verify_sprites(f2d, sprites)

        # Detect grids early — needed by researcher mode before think() is called
        self._detect_grids(frame[0] if frame.ndim == 3 else frame)

        # Visual recall — the flinch. Pattern library match on first sight.
        self._visual_recall_text = self.visual_recall(frame)

        # Frame-to-ASCII: give the pilot a readable map of the whole level
        self._ascii_frame = self._frame_to_ascii(frame)

        # Phase 5: Initialize CNN-lite action predictor
        if HAS_PREDICTOR and self.action_predictor is None:
            f2d = frame[0] if frame.ndim == 3 else frame
            self.action_predictor = ActionPredictor(
                n_actions=n_actions, frame_h=f2d.shape[0], frame_w=f2d.shape[1]
            )
            self._log(f"Action predictor initialized ({n_actions} actions)")

        return lvl

    @staticmethod
    def _detect_cell_size(f2d: np.ndarray) -> int:
        """Auto-detect the natural grid cell size from frame structure.

        Checks if the frame has a repeating grid pattern at common cell sizes.
        Returns the detected cell size (2, 4, or 8). Default 4 if ambiguous.
        """
        h, w = f2d.shape
        # Try common cell sizes, prefer largest that shows clean grid (compact output)
        for cs in (8, 4, 2):
            if h % cs != 0 or w % cs != 0:
                continue
            # Check if each cs×cs block is uniform (single color)
            uniform_count = 0
            total_blocks = 0
            for by in range(0, min(h, 32), cs):
                for bx in range(0, min(w, 32), cs):
                    block = f2d[by:by+cs, bx:bx+cs]
                    total_blocks += 1
                    if np.all(block == block[0, 0]):
                        uniform_count += 1
            if total_blocks > 0 and uniform_count / total_blocks > 0.85:
                return cs
        return 4  # safe default

    def _frame_to_ascii(self, frame: np.ndarray, block_size: int = 0) -> str:
        """Convert raw frame to ASCII art — the pilot's glasses.

        Auto-detects cell size if block_size=0. Each color gets a distinct
        character. The pilot learns which characters are walls vs corridors by trying
        to move through them. No assumptions about which color is which.
        """
        f2d = frame[0] if frame.ndim == 3 else frame
        h, w = f2d.shape

        if block_size <= 0:
            block_size = self._detect_cell_size(f2d)

        # Map each color to a distinct readable character
        # Use characters that are visually distinct
        color_chars = '.#O+X@*~=:%&$!?^'  # 16 chars for 16 possible colors
        # Build legend from colors actually present
        unique_colors = np.unique(f2d)

        grid_rows = []
        for by in range(0, h, block_size):
            row = []
            for bx in range(0, w, block_size):
                block = f2d[by:by+block_size, bx:bx+block_size]
                unique_b, counts_b = np.unique(block, return_counts=True)
                dominant = int(unique_b[np.argmax(counts_b)])
                ch = color_chars[dominant] if dominant < len(color_chars) else '?'
                row.append(ch)
            grid_rows.append(''.join(row))

        n_cols = len(grid_rows[0]) if grid_rows else 0
        n_rows = len(grid_rows)

        # Add column indices header (units digit only for compactness)
        col_header = '  ' + ''.join(str(i % 10) for i in range(n_cols))
        rows_with_idx = [col_header]
        for i, row in enumerate(grid_rows):
            rows_with_idx.append(f"{i:2d}{row}")

        # Build legend
        unique, counts = np.unique(f2d, return_counts=True)
        legend_parts = []
        for c, n in sorted(zip(unique, counts), key=lambda x: -x[1]):
            ch = color_chars[int(c)] if int(c) < len(color_chars) else '?'
            pct = n * 100 / f2d.size
            legend_parts.append(f"'{ch}'=color {c} ({pct:.0f}%)")

        size_note = f"[{n_rows}x{n_cols} grid, cell={block_size}px]"
        return "```\n" + '\n'.join(rows_with_idx) + "\n```\n" + f"{size_note} Legend: " + ", ".join(legend_parts)

    def _verify_sprites(self, frame: np.ndarray, sprites: list):
        """Observe sprite context — report what we see, don't assume roles.

        Record what surrounds each sprite. The mind uses this as evidence
        when forming hypotheses, but nothing is labeled until interaction confirms it.
        Player = whatever moves. Target = whatever triggers a win. Period.
        """
        h, w = frame.shape

        for s in sprites:
            if s.size >= 50:
                continue
            # Observe what surrounds this sprite (evidence, not judgment)
            y_min, x_min, y_max, x_max = s.bbox
            margin = 2
            neighbors = []
            for dy in range(-margin, (y_max - y_min) + margin + 1):
                for dx in range(-margin, (x_max - x_min) + margin + 1):
                    ny, nx = y_min + dy, x_min + dx
                    if 0 <= ny < h and 0 <= nx < w and (ny, nx) not in s.pixels:
                        neighbors.append(frame[ny, nx])
            if neighbors:
                neighbor_colors, counts = np.unique(neighbors, return_counts=True)
                dominant_neighbor = int(neighbor_colors[np.argmax(counts)])
                self._log(f"  Context: color {s.color} at ({s.center[0]:.0f},{s.center[1]:.0f}) "
                          f"surrounded by color {dominant_neighbor}")

    def observe_effect(self, lvl: LevelMemory, action_id: int, action_desc: str,
                       frame_before: np.ndarray, frame_after: np.ndarray,
                       won: bool) -> Experiment:
        """Record what happened when we took an action."""
        # Defensive squeeze — ARC frames come in many shapes
        frame_before = np.squeeze(np.array(frame_before))
        frame_after = np.squeeze(np.array(frame_after))
        if frame_before.ndim == 1:
            frame_before = frame_before.reshape(1, -1)
        if frame_after.ndim == 1:
            frame_after = frame_after.reshape(1, -1)
        # Guard against empty or shape-mismatched frames (e.g. obs.frame=[] on level transition)
        if frame_before.size == 0 or frame_after.size == 0 or frame_before.shape != frame_after.shape:
            diff = {'changed': 0, 'description': 'frame unavailable (empty or shape mismatch)'}
        else:
            diff = compute_diff(frame_before, frame_after)

        # ── Resource/Fuel Reset Detection ──
        # If >80% of pixels change at once, this is likely a reset (fuel depletion, death, etc.)
        # NOT a normal action result. Track it so the pilot knows about resource limits.
        self._moves_since_reset += 1
        if diff.get('changed', 0) > 0 and frame_before.size > 0:
            change_ratio = diff['changed'] / max(frame_before.size, 1)
            if change_ratio > 0.5 and not won:
                # Massive change without winning = reset/death
                self._total_resets += 1
                moves_before_reset = self._moves_since_reset
                self._last_reset_turn = lvl.total_actions
                self._moves_since_reset = 0
                reset_msg = (f"RESET DETECTED (#{self._total_resets}) after {moves_before_reset} moves. "
                             f"{change_ratio:.0%} pixels changed. Game may have resource limits "
                             f"(~{moves_before_reset} moves before depletion).")
                self._log(f"  {reset_msg}")
                # Store as a discovered rule
                rule = f"RESOURCE LIMIT: Game resets after ~{moves_before_reset} moves. Conserve moves."
                if isinstance(self.memory.rules_discovered, set):
                    self.memory.rules_discovered.add(rule)
                elif rule not in self.memory.rules_discovered:
                    self.memory.rules_discovered.append(rule)

        # Use Eyes to track sprite movement
        movement_desc = diff['description']
        if HAS_EYES:
            fb = frame_before[0] if frame_before.ndim == 3 else frame_before
            fa = frame_after[0] if frame_after.ndim == 3 else frame_after
            sprites_b = detect_sprites(fb, min_size=2)
            sprites_a = detect_sprites(fa, min_size=2)
            self._last_sprites = sprites_a  # for grid_transform prompt
            movements = track_sprite_movement(sprites_b, sprites_a)
            # Store movement tuples for CausalLedger
            self._last_movements = [
                (s_b.color, s_b.size, int(mdy), int(mdx))
                for s_b, s_a, (mdy, mdx) in movements
            ]

            # HUD/status bar noise filter: if only a few pixels changed
            # and no sprites moved, could be BLOCKED or just a fuel tick.
            # Only mark BLOCKED if 0px changed (true wall) or if we can verify
            # the player didn't actually move (same position in frame).
            _oprah_nav = getattr(self, '_env_profile', None) and 'nav' in getattr(self._env_profile, 'genre', '').lower()
            if not movements and diff['changed'] == 0 and _oprah_nav:
                # Zero change = definitely blocked
                if self.spatial.player_color >= 0:
                    self.spatial.blocked.add((self.spatial.player_pos[0], self.spatial.player_pos[1], action_id))
                self._log(f"  BLOCKED (0px changed — wall)")
            elif not movements and diff['changed'] > 0 and diff['changed'] <= 4 and _oprah_nav:
                # 1-4px change with no sprite movement = fuel tick.
                # Check if player ACTUALLY moved by comparing sprite positions.
                _actually_moved = False
                if self.spatial.player_color >= 0:
                    # Find player sprite in before and after
                    _p_before = [s for s in sprites_b if s.color == self.spatial.player_color]
                    _p_after = [s for s in sprites_a if s.color == self.spatial.player_color]
                    if _p_before and _p_after:
                        _pb = _p_before[0]
                        _pa = _p_after[0]
                        if abs(_pb.center[0] - _pa.center[0]) > 1 or abs(_pb.center[1] - _pa.center[1]) > 1:
                            _actually_moved = True
                if _actually_moved:
                    self._log(f"  [fuel-tick] {diff['changed']}px HUD change + player moved")
                    if self.spatial.player_pos != (0, 0) and self.spatial.player_pos != (-1, -1):
                        if action_id in self.spatial.action_map:
                            dy, dx, _ = self.spatial.action_map[action_id]
                            old_y, old_x = self.spatial.player_pos
                            new_pos = (old_y + dy, old_x + dx)
                            self.spatial.player_pos = new_pos
                            self.spatial.visited.add(new_pos)
                else:
                    # Player didn't move — this IS a wall, masked by fuel tick
                    self._log(f"  [fuel-tick→BLOCKED] {diff['changed']}px HUD but player didn't move — wall")
                    self._last_was_fuel_blocked = True
                    if self.spatial.player_color >= 0:
                        self.spatial.blocked.add((self.spatial.player_pos[0], self.spatial.player_pos[1], action_id))

            if movements:
                move_parts = []
                for s_before, s_after, (dy, dx) in movements:
                    direction = ""
                    if abs(dy) > abs(dx):
                        direction = "DOWN" if dy > 0 else "UP"
                    elif abs(dx) > 0:
                        direction = "RIGHT" if dx > 0 else "LEFT"
                    else:
                        direction = "STAYED"
                    move_parts.append(
                        f"color {s_before.color} moved {direction} "
                        f"({s_before.center[0]:.0f},{s_before.center[1]:.0f}) → "
                        f"({s_after.center[0]:.0f},{s_after.center[1]:.0f})"
                    )

                    # Only identify player/direction in NAVIGATION games
                    # (few pixels change, one sprite moves consistently)
                    # In TOGGLE/CLICK games, many pixels change — don't force directional model
                    # Also trust OPRAH genre: if it says navigation, allow higher pixel changes
                    # (LS20 causes ~52px per move due to player + fuel bar)
                    _oprah_nav = getattr(self, '_env_profile', None) and 'nav' in getattr(self._env_profile, 'genre', '').lower()
                    is_likely_nav = (diff['changed'] < 200 if _oprah_nav else diff['changed'] < 100) and len(movements) <= 5

                    if is_likely_nav:
                        if self.spatial.player_color < 0:
                            self.spatial.player_color = s_before.color
                            self._log(f"  PLAYER IDENTIFIED: color {s_before.color} (it moved!)")
                            # Now identify target — smallest non-player, non-wall, non-scenery sprite
                            # Skip visual heuristic if hypothesis is locked (prior knowledge = complex game)
                            if (self.spatial.target_pos == (-1, -1) and self.spatial.objects
                                    and not self.memory.hypothesis_locked):
                                wall_colors = getattr(self.spatial, 'wall_colors', set())
                                candidates = [
                                    (c, y, x, r) for c, y, x, r in self.spatial.objects
                                    if c != s_before.color and c not in wall_colors
                                    and c not in self.spatial.scenery_colors
                                    and 'small' in r
                                ]
                                if candidates:
                                    # Pick the one farthest from player
                                    py, px = int(s_before.center[0]), int(s_before.center[1])
                                    candidates.sort(key=lambda o: -(abs(o[1]-py) + abs(o[2]-px)))
                                    tc, ty, tx, _ = candidates[0]
                                    self.spatial.target_pos = (ty, tx)
                                    self.spatial.target_color = tc
                                    # Register as causal candidate
                                    self.spatial.update_target_causal(tc, (ty, tx), 'visual heuristic — farthest non-player sprite')
                                    self._log(f"  TARGET IDENTIFIED: color {tc} at ({ty},{tx})")

                        # ── Causal target tracking ──
                        # Non-player sprites that move = interactive = target candidates
                        if (self.spatial.player_color >= 0
                                and s_before.color != self.spatial.player_color
                                and s_before.color not in self.spatial.player_colors
                                and s_before.color not in self.spatial.scenery_colors
                                and (abs(dy) > 0 or abs(dx) > 0)):
                            pos = (int(s_after.center[0]), int(s_after.center[1]))
                            self.spatial.update_target_causal(
                                s_before.color, pos,
                                f'moved {direction} in response to action {action_id}',
                                boost=0.3,
                                pixel_count=len(getattr(s_before, 'pixels', [])) if hasattr(s_before, 'pixels') else getattr(s_before, 'pixel_count', 0)
                            )
                            # If this candidate is higher confidence than current target, switch
                            # BUT: only override if no Eyes-seeded target exists (avoid chasing HUD elements)
                            if self.spatial.target_pos == (-1, -1):
                                causal_best = self.spatial.best_target()
                                if causal_best != (-1, -1):
                                    self.spatial.target_pos = causal_best
                                    self.spatial.target_color = s_before.color
                                    self._log(f"  [causal-target] Set initial target: color {s_before.color} at {causal_best}")

                        if s_before.color == self.spatial.player_color:
                            old_pos = (int(s_before.center[0]), int(s_before.center[1]))
                            new_pos = (int(s_after.center[0]), int(s_after.center[1]))
                            # Don't map click/parameterized actions as directions — OPRAH knows better
                            _env_prof = getattr(self, '_env_profile', None)
                            _is_click = _env_prof and action_id in getattr(_env_prof, 'param_actions', [])
                            if not _is_click:
                                self.spatial.player_pos = new_pos
                                self.spatial.visited.add(old_pos)
                                self.spatial.visited.add(new_pos)
                                self.spatial.transitions[(old_pos[0], old_pos[1], action_id)] = new_pos
                                self.spatial.action_map[action_id] = (int(dy), int(dx), direction)
                            self.spatial.player_colors.add(s_before.color)

                            # Detect co-moving sprites = part of player entity
                            player_dy, player_dx = int(dy), int(dx)
                            for s2_before, s2_after, (dy2, dx2) in movements:
                                if s2_before.color != s_before.color and int(dy2) == player_dy and int(dx2) == player_dx:
                                    if s2_before.color not in self.spatial.player_colors:
                                        self.spatial.player_colors.add(s2_before.color)
                                        # Remove from target candidates if it was there
                                        self.spatial.target_candidates.pop(s2_before.color, None)
                                        self._log(f"  [co-move] Color {s2_before.color} co-moves with player — added to player_colors")

                            # Learn terrain: what color is at my new position? That's walkable.
                            f2d = frame_after[0] if frame_after.ndim == 3 else frame_after
                            py, px = int(s_after.center[0]), int(s_after.center[1])
                            if 0 <= py < f2d.shape[0] and 0 <= px < f2d.shape[1]:
                                ground_color = int(f2d[py, px])
                                if not hasattr(self.spatial, 'walkable_colors'):
                                    self.spatial.walkable_colors = set()
                                if ground_color != self.spatial.player_color:
                                    self.spatial.walkable_colors.add(ground_color)

                            # ── Adaptive pickup threshold ──
                            # Track normal movement cost to distinguish from real pickups
                            px_cost = diff['changed']
                            self.spatial.move_cost_count += 1
                            alpha = 1.0 / self.spatial.move_cost_count
                            self.spatial.avg_move_cost = (1 - alpha) * self.spatial.avg_move_cost + alpha * px_cost
                            # Threshold = 2x average normal move cost (minimum 80)
                            self.spatial.pickup_threshold = max(80, int(self.spatial.avg_move_cost * 2.5))

                            # ── Causal target demotion ──
                            # If we stepped onto a target candidate's position but nothing
                            # happened (level didn't complete), it's not the real target.
                            for tc, info in list(self.spatial.target_candidates.items()):
                                cy, cx = info['pos']
                                if abs(new_pos[0] - cy) <= 3 and abs(new_pos[1] - cx) <= 3:
                                    # We're at this candidate — if level didn't complete,
                                    # demotion happens in the outer loop (run_game checks won).
                                    # Mark for deferred demotion check.
                                    if not hasattr(self, '_pending_demotions'):
                                        self._pending_demotions = []
                                    self._pending_demotions.append((tc, new_pos))
                    else:
                        self._log(f"  WORLD CHANGE: {diff['changed']}px changed, {len(movements)} sprites moved — likely toggle/click game")
                        # ── Pickup event detection ──
                        # Large world change during navigation = "I collected something"
                        if (diff['changed'] >= self.spatial.pickup_threshold
                                and self.spatial.player_color >= 0
                                and action_id in self.spatial.action_map):
                            # Analyze HUD region for attribute changes
                            hud_change = ""
                            hud_detail = {}
                            try:
                                fb = frame_before[0] if frame_before.ndim == 3 else frame_before
                                fa = frame_after[0] if frame_after.ndim == 3 else frame_after
                                h = fb.shape[0]
                                # HUD is typically bottom 15% of frame
                                hud_top = max(0, h - int(h * 0.15))
                                hud_before = fb[hud_top:, :]
                                hud_after = fa[hud_top:, :]
                                hud_diff = int(np.sum(hud_before != hud_after))
                                # Also check top HUD area
                                top_hud_before = fb[:int(h * 0.15), :]
                                top_hud_after = fa[:int(h * 0.15), :]
                                top_diff = int(np.sum(top_hud_before != top_hud_after))
                                if hud_diff > 5:
                                    hud_change = f", HUD changed ({hud_diff}px bottom)"
                                    # Parse WHAT changed: color distribution before vs after
                                    colors_before = {}
                                    colors_after = {}
                                    for c in range(16):
                                        cb = int(np.sum(hud_before == c))
                                        ca = int(np.sum(hud_after == c))
                                        if cb > 0: colors_before[c] = cb
                                        if ca > 0: colors_after[c] = ca
                                    lost = {c: colors_before[c] - colors_after.get(c, 0) for c in colors_before if colors_before[c] > colors_after.get(c, 0)}
                                    gained = {c: colors_after[c] - colors_before.get(c, 0) for c in colors_after if colors_after[c] > colors_before.get(c, 0)}
                                    if lost or gained:
                                        parts = []
                                        if lost: parts.append(f"lost colors {dict(lost)}")
                                        if gained: parts.append(f"gained colors {dict(gained)}")
                                        hud_detail['bottom'] = "; ".join(parts)
                                        hud_change += f" [{'; '.join(parts)}]"
                                if top_diff > 5:
                                    hud_change += f", HUD changed ({top_diff}px top)"
                            except Exception:
                                pass
                            event = {
                                'pos': self.spatial.player_pos,
                                'pixels_changed': diff['changed'],
                                'sprites_moved': len(movements),
                                'action_id': action_id,
                                'hud_change': hud_change,
                                'hud_detail': hud_detail,
                            }
                            self.spatial.pickup_events.append(event)
                            self._log(f"  [pickup] Collected something at {self.spatial.player_pos}! ({diff['changed']}px, {len(movements)} sprites{hud_change})")
                            # Record in surprise map — positions with unexpectedly large changes
                            spos = self.spatial.player_pos
                            prev_surprise = self.spatial.surprise_map.get(spos, 0)
                            if diff['changed'] > prev_surprise:
                                self.spatial.surprise_map[spos] = diff['changed']
                            # Record modifier in memory if HUD changed
                            if hud_detail:
                                pos_key = self.spatial.player_pos
                                if pos_key not in self.spatial.known_modifiers:
                                    self.spatial.known_modifiers[pos_key] = {'hud_lost': {}, 'hud_gained': {}, 'visits': 0}
                                mod = self.spatial.known_modifiers[pos_key]
                                mod['visits'] += 1
                                # Merge HUD changes
                                for part in hud_detail.values():
                                    if 'lost' in str(part):
                                        pass  # already in hud_detail
                                    if 'gained' in str(part):
                                        pass
                                # Store the raw lost/gained from this visit
                                bottom = hud_detail.get('bottom', '')
                                mod['last_change'] = bottom
                                self._log(f"  [modifier-memory] Position {pos_key}: {bottom} (visit #{mod['visits']})")

                movement_desc = "; ".join(move_parts)
            elif diff['changed'] <= 2:
                # Action had no effect from this position — blocked by wall
                movement_desc = f"BLOCKED — action {action_id} had no effect from position {self.spatial.player_pos}"

                # Learn terrain: what color is in the blocked direction? That's a wall.
                if action_id in self.spatial.action_map and self.spatial.player_color >= 0:
                    dy, dx, dname = self.spatial.action_map[action_id]
                    py, px = self.spatial.player_pos
                    wall_y, wall_x = py + dy, px + dx
                    f2d = frame_before[0] if frame_before.ndim == 3 else frame_before
                    if 0 <= wall_y < f2d.shape[0] and 0 <= wall_x < f2d.shape[1]:
                        wall_color = int(f2d[wall_y, wall_x])
                        if not hasattr(self.spatial, 'wall_colors'):
                            self.spatial.wall_colors = set()
                        if wall_color != self.spatial.player_color:
                            self.spatial.wall_colors.add(wall_color)

            # Record in episodic memory
            if self.episodic:
                self.episodic.record(frame_before, action_id, {}, frame_after,
                                   level=lvl.level, won=won)

        # Track stalling
        if diff['changed'] == 0:
            self._consecutive_no_change += 1
        else:
            self._consecutive_no_change = 0
        self._last_action = action_id

        # Feed CausalLedger (Layer 4 of Grid Transform)
        if self.causal_ledger:
            move_list = getattr(self, '_last_movements', [])
            moved_count = len(move_list)
            ctype = 'NONE' if diff['changed'] == 0 else 'MOVE' if moved_count > 0 else 'CHANGE'
            self.causal_ledger.observe(action_id, diff['changed'], ctype,
                                       movements=move_list, level=lvl.level)

        exp = Experiment(
            turn=lvl.total_actions,
            action_id=action_id,
            action_desc=action_desc,
            pixels_changed=diff['changed'],
            regions_changed=movement_desc,
            frame_before_b64=frame_to_b64(frame_before),
            frame_after_b64=frame_to_b64(frame_after),
            won_level=won,
        )
        lvl.experiments.append(exp)
        lvl.current_frame_b64 = exp.frame_after_b64
        lvl.current_frame = frame_after.copy()
        lvl.total_actions += 1

        # Feed to WorldModel for internal simulation
        self._observe_world_model(action_id, frame_before, frame_after)

        # Record causal chain if something changed
        if diff['changed'] > 0:
            # Detect which grid cells changed (for lights-out style puzzles)
            cells_changed = []
            grids = getattr(self, '_detected_grids', [])
            if grids:
                for gi, g in enumerate(grids):
                    for cell in g.get('cells', []):
                        cx0 = cell['center_x'] - 3  # approximate cell region
                        cx1 = cell['center_x'] + 3
                        cy0 = cell['center_y'] - 3
                        cy1 = cell['center_y'] + 3
                        cx0, cy0 = max(0, cx0), max(0, cy0)
                        cx1 = min(frame_after.shape[1] - 1, cx1)
                        cy1 = min(frame_after.shape[0] - 1, cy1)
                        region_before = frame_before[cy0:cy1+1, cx0:cx1+1]
                        region_after = frame_after[cy0:cy1+1, cx0:cx1+1]
                        if not np.array_equal(region_before, region_after):
                            cells_changed.append(f"G{gi+1}({cell['row']},{cell['col']})")

            effect_desc = movement_desc
            if cells_changed:
                effect_desc += f" | cells: {', '.join(cells_changed)}"

            chain = CausalChain(
                action_id=action_id,
                action_data=getattr(self, '_last_action_data', {}),
                precondition=f"{diff['changed']}px at {diff.get('center', 'unknown')}",
                effect=effect_desc,
                confidence=1.0,
                level=lvl.level,
                step=lvl.total_actions,
                pixels_changed=diff['changed'],
            )
            self.memory.causal_chains.append(chain)

        return exp

    # -- Phase 2-4: THINK (hypothesize + plan + adapt) ----------------------

    def think(self, lvl: LevelMemory, phase: str = "act") -> dict:
        """The core cognitive loop. Ask the mind what to do next.

        Returns dict with:
          - action: int (which action to take)
          - data: dict (action parameters, e.g. click coordinates)
          - reasoning: str
          - hypothesis: str (current theory about the game)
          - plan: list of planned actions (may be multi-step)
        """
        self.total_llm_calls += 1

        # Console visualization — see what the LLM sees before it thinks
        if os.environ.get('GUNDAM_DEBUG_VIS'):
            self._console_debug_vis(lvl)

        # Build the prompt — graduated perception states (Hypatia spec)
        _dose = self._perception_state(
            turn=len(lvl.experiments),
            budget=self.budget_per_level * 10  # approximate action budget
        )
        # Strategy pivot: force micro-dose when stalled
        if getattr(self, '_force_microdose', False):
            _dose = max(_dose, 1)  # at least micro-dose
            self._force_microdose = False

        # Refresh Florence for micro-dose and full dose
        if _dose >= 1 and HAS_VLM and hasattr(lvl, 'current_frame_b64') and lvl.current_frame_b64:
            try:
                import arc_vlm
                _dmt_frame = self._b64_to_frame(lvl.current_frame_b64)
                _dmt_percept = arc_vlm.perceive_structured(_dmt_frame)
                if _dmt_percept and _dmt_percept.get('caption'):
                    lvl._vlm_structured = _dmt_percept
                    self._log(f"  {'🌿' if _dose==1 else '🍄'} Florence sees: {_dmt_percept['caption'][:100]}")
            except Exception as e:
                self._log(f"  Florence refresh failed: {e}")

        if _dose == 2:
            self._log("🍄 FULL DOSE — stripping priors, ego dissolution")
            system = self._dmt_system_prompt()
        elif _dose == 1:
            self._log("🌿 MICRO-DOSE — loosening priors, questioning assumptions")
            system = self._build_system_prompt()
            # Inject loosening preamble
            genre = getattr(self, '_genre_hypothesis', 'unknown')
            system += f"\n\n⚡ PATTERN CHECK: You've been treating this as '{genre}'. What if it isn't?\n"
            system += "Before choosing your next action, answer honestly:\n"
            system += "- What ASSUMPTION am I making about how this game works?\n"
            system += "- What evidence CONTRADICTS that assumption?\n"
            system += "- What would I try if this were a COMPLETELY DIFFERENT type of puzzle?\n"
            system += "\nYou still have all your knowledge. But hold it loosely.\n"
        else:
            system = self._build_system_prompt()

        user = self._build_user_prompt(lvl, phase)

        # Attach images (skip in text_only mode — use ASCII minimap instead)
        images = []
        if not self.text_only:
            images = [lvl.initial_frame_b64, lvl.current_frame_b64]
            if lvl.experiments:
                last_exp = lvl.experiments[-1]
                if last_exp.frame_before_b64 and last_exp.frame_after_b64:
                    try:
                        fb = self._b64_to_frame(last_exp.frame_before_b64)
                        fa = self._b64_to_frame(last_exp.frame_after_b64)
                        # Resize to match if needed (b64 frames may be scaled differently)
                        if fb.shape != fa.shape:
                            target = min(fb.shape[0], fa.shape[0]), min(fb.shape[1], fa.shape[1])
                            fb = fb[:target[0], :target[1]]
                            fa = fa[:target[0], :target[1]]
                        comp = composite_b64(fb, fa)
                        if comp:
                            images.append(comp)
                    except Exception:
                        pass  # skip composite if frames are incompatible

        # Hybrid LLM: Opus for first call per level (deep analysis), Sonnet for subsequent (speed)
        # Set ARC_HYBRID_LLM=1 to enable. Default: all Opus.
        # Stall detection: switch back to Opus every 5 calls if no progress
        # Default: Sonnet for reasoning. Only use Haiku for cheap genre/navigation calls.
        # The first call (total_llm_calls == 0) always uses Sonnet.
        # Subsequent calls: Haiku only if oracle is actively solving this level.
        _oracle_active = getattr(self, '_lo_solution_found_this_level', False)
        _use_fast = (os.environ.get('ARC_HYBRID_LLM', '1') != '0'
                     and self.total_llm_calls > 0
                     and _oracle_active)  # Haiku only when oracle handles the heavy lifting
        # Graduated temperature: sober=0.3, micro-dose=0.5, full dose=0.7
        # Active inference: precision rises with confidence (lower temp = more exploitative)
        _temp = [0.3, 0.5, 0.7][_dose]
        if _dose == 0 and self.memory.hypothesis_locked and len(self.memory.rules_discovered) >= 3:
            _temp = 0.15  # high confidence → sharp exploitation
        response = call_llm(
            [{'role': 'system', 'content': system},
             {'role': 'user', 'content': user}],
            images=images,
            max_tokens=1000,
            temperature=_temp,
            fast=bool(_use_fast),
        )

        # Debug dump — write full prompt + response for Peter to read
        debug_dir = os.environ.get('GUNDAM_DEBUG_DIR', '')
        if debug_dir:
            os.makedirs(debug_dir, exist_ok=True)
            turn = lvl.total_actions
            fname = f"{debug_dir}/call_{self.total_llm_calls:03d}_L{lvl.level}_T{turn}.md"
            with open(fname, 'w') as f:
                f.write(f"# LLM Call #{self.total_llm_calls} — Level {lvl.level}, Turn {turn}\n\n")
                f.write(f"## System Prompt\n```\n{system}\n```\n\n")
                f.write(f"## User Prompt\n```\n{user}\n```\n\n")
                f.write(f"## Images: {len(images)} attached\n\n")
                f.write(f"## Raw Response\n```\n{response}\n```\n")

        if response is None:
            self._log("LLM failed — falling back to algorithmic exploration")
            # Use OPRAH profile + stall detection for smarter fallback
            profile = getattr(self, '_env_profile', None)
            n_acts = self.memory.n_actions or 6

            # Build action priority list from OPRAH
            if profile and hasattr(profile, 'action_types'):
                movement_acts = [i for i, t in enumerate(profile.action_types)
                                 if t in ('MOVEMENT', 'movement')]
                toggle_acts = [i for i, t in enumerate(profile.action_types)
                               if t in ('TOGGLE', 'toggle')]
                preferred = movement_acts or toggle_acts or list(range(n_acts))
            else:
                preferred = list(range(n_acts))

            # Stall detection: if stuck, try least-used action
            if self._consecutive_no_change >= 3:
                action_counts = {}
                for exp in lvl.experiments[-20:]:
                    action_counts[exp.action_id] = action_counts.get(exp.action_id, 0) + 1
                # Pick least-used action from preferred list
                next_action = min(preferred, key=lambda a: action_counts.get(a, 0))
                reasoning = f'Stall detected ({self._consecutive_no_change}x no-change) — trying least-used action'
            else:
                next_action = preferred[lvl.total_actions % len(preferred)]
                reasoning = 'LLM unavailable — OPRAH-guided exploration'

            return {
                'action': next_action,
                'data': {},
                'reasoning': reasoning,
                'hypothesis': '',
                'plan': [],
            }

        return self._parse_thinking(response, lvl)

    def _detect_grids(self, frame: np.ndarray) -> str:
        """Upgraded perception: detect sub-grids, cell contents, and embedded sprites.

        Instead of just finding grid lines, this identifies:
        1. Multiple grids on screen (examples vs target)
        2. What color each cell is (the dominant fill)
        3. Embedded sprites inside cells (sub-cell patterns that differ from the fill)
        4. Which grid is highlighted (border color differs = target)
        """
        h, w = frame.shape
        lines = []

        # Find grid dividers: rows/cols with uniform or nearly-uniform color
        def find_dividers(axis_len, get_line):
            dividers = []
            for i in range(axis_len):
                vals = get_line(i)
                unique = np.unique(vals)
                if len(unique) == 0:
                    continue
                if len(unique) <= 2:  # divider lines may have 1-2 colors
                    dominant = unique[np.argmax([np.sum(vals == u) for u in unique])]
                    if np.sum(vals == dominant) >= len(vals) * 0.7:
                        dividers.append((i, int(dominant)))
            return dividers

        row_divs = find_dividers(h, lambda r: frame[r, :])
        col_divs = find_dividers(w, lambda c: frame[:, c])

        if len(row_divs) < 2 or len(col_divs) < 2:
            return ""

        # Group consecutive dividers into bands, detect gaps (= cell rows/cols)
        def find_cell_spans(dividers):
            """Find spans between divider bands."""
            spans = []
            i = 0
            while i < len(dividers):
                # Find end of this divider band
                j = i
                while j < len(dividers) - 1 and dividers[j+1][0] == dividers[j][0] + 1:
                    j += 1
                band_end = dividers[j][0]
                # Look for next band start
                if j + 1 < len(dividers):
                    next_start = dividers[j+1][0]
                    if next_start - band_end > 1:
                        spans.append((band_end + 1, next_start - 1))
                i = j + 1
            return spans

        cell_row_spans = find_cell_spans(row_divs)
        cell_col_spans = find_cell_spans(col_divs)

        if not cell_row_spans or not cell_col_spans:
            return ""

        # Break up oversized spans using known cell size as template
        def split_large_spans(spans, is_row=True):
            small_sizes = sorted([s[1]-s[0]+1 for s in spans if s[1]-s[0]+1 < 10])
            if not small_sizes:
                return spans
            typical = small_sizes[len(small_sizes)//2]
            result = []
            for s0, s1 in spans:
                size = s1 - s0 + 1
                if size > typical * 3:
                    # Large span contains a sub-grid with internal dividers.
                    # Strategy: find the border color at the edge, then scan for
                    # divider lines WITHIN this span (same color as border).
                    border_color = int(frame[s0, s0]) if is_row else int(frame[s0, s0])

                    # Scan for internal dividers: rows/cols where the cross-section
                    # is nearly uniform (>=90% one color) — these are grid lines
                    internal_divs = []
                    for pos in range(s0, s1 + 1):
                        if is_row:
                            line = frame[pos, s0:s1+1]
                        else:
                            line = frame[s0:s1+1, pos]
                        u, c = np.unique(line, return_counts=True)
                        max_pct = np.max(c) / len(line)
                        if max_pct >= 0.9:  # nearly uniform = divider
                            internal_divs.append(pos)

                    # Extract cell spans between divider bands (including before first and after last)
                    # First, group consecutive dividers into bands
                    bands = []
                    i = 0
                    while i < len(internal_divs):
                        j = i
                        while j < len(internal_divs) - 1 and internal_divs[j+1] == internal_divs[j] + 1:
                            j += 1
                        bands.append((internal_divs[i], internal_divs[j]))
                        i = j + 1

                    sub_cells = []
                    # Cell before first band (if there's space)
                    if bands and bands[0][0] > s0 + 1:
                        sub_cells.append((s0, bands[0][0] - 1))
                    # Cells between bands
                    for bi in range(len(bands) - 1):
                        gap_start = bands[bi][1] + 1
                        gap_end = bands[bi+1][0] - 1
                        if gap_end >= gap_start:
                            sub_cells.append((gap_start, gap_end))
                    # Cell after last band (if there's space)
                    if bands and bands[-1][1] < s1 - 1:
                        sub_cells.append((bands[-1][1] + 1, s1))

                    # Filter out sub-cells that are too small (likely border remnants)
                    sub_cells = [(a, b) for a, b in sub_cells if b - a + 1 >= typical - 2]

                    if len(sub_cells) >= 2:
                        result.extend(sub_cells)
                    else:
                        result.append((s0, s1))
                else:
                    result.append((s0, s1))
            return result

        cell_row_spans = split_large_spans(cell_row_spans, is_row=True)
        cell_col_spans = split_large_spans(cell_col_spans, is_row=False)

        # Detect if there are MULTIPLE SEPARATE GRIDS by looking for large gaps
        def find_grid_groups(spans):
            """Split spans into separate grids if there's a big gap."""
            if len(spans) <= 1:
                return [spans]
            gaps = [spans[i+1][0] - spans[i][1] for i in range(len(spans)-1)]
            if not gaps:
                return [spans]
            cell_sizes = [s[1] - s[0] + 1 for s in spans]
            typical_cell = sorted(cell_sizes)[len(cell_sizes)//2]
            threshold = max(typical_cell, 2)
            groups = [[spans[0]]]
            for i, gap in enumerate(gaps):
                if gap > threshold:
                    groups.append([])
                groups[-1].append(spans[i+1])
            return [g for g in groups if g]

        row_groups = find_grid_groups(cell_row_spans)
        col_groups = find_grid_groups(cell_col_spans)

        # Build grid descriptions
        grids = []
        for rg in row_groups:
            for cg in col_groups:
                grid = {'rows': rg, 'cols': cg, 'cells': []}
                # Detect border color for this grid region
                r_start = rg[0][0] - 1 if rg[0][0] > 0 else 0
                c_start = cg[0][0] - 1 if cg[0][0] > 0 else 0
                border_color = int(frame[r_start, c_start]) if r_start >= 0 and c_start >= 0 else -1
                grid['border_color'] = border_color

                for ri, (ry0, ry1) in enumerate(rg):
                    for ci, (cx0, cx1) in enumerate(cg):
                        cell = frame[ry0:ry1+1, cx0:cx1+1]
                        cell_h, cell_w = cell.shape
                        # Dominant color = the cell's fill
                        unique, counts = np.unique(cell, return_counts=True)
                        dominant = int(unique[np.argmax(counts)])
                        fill_pct = int(100 * np.max(counts) / cell.size)

                        # Check for embedded sprite (sub-pattern that differs from fill)
                        sprite_info = ""
                        if fill_pct < 85 and cell_h >= 3 and cell_w >= 3:
                            # There's something inside this cell besides the fill
                            non_fill = cell != dominant
                            non_fill_colors = np.unique(cell[non_fill])
                            # Extract the pattern as a mini-grid
                            sprite_info = f" [SPRITE: {cell_h}x{cell_w} pattern with colors {list(non_fill_colors.astype(int))}]"

                        # Compute click target: center of the content region
                        # (skip border pixels which may be at the edges of the cell span)
                        # Find the innermost region that ISN'T the border color
                        click_x = (cx0 + cx1) // 2
                        click_y = (ry0 + ry1) // 2
                        if cell_w >= 4 and cell_h >= 4:
                            # Use center 60% of cell to avoid border artifacts
                            inner_cx0 = cx0 + max(1, cell_w // 5)
                            inner_cx1 = cx1 - max(1, cell_w // 5)
                            inner_ry0 = ry0 + max(1, cell_h // 5)
                            inner_ry1 = ry1 - max(1, cell_h // 5)
                            click_x = (inner_cx0 + inner_cx1) // 2
                            click_y = (inner_ry0 + inner_ry1) // 2

                        grid['cells'].append({
                            'row': ri, 'col': ci,
                            'fill': dominant, 'fill_pct': fill_pct,
                            'sprite': sprite_info,
                            'center_x': click_x,
                            'center_y': click_y,
                        })
                grids.append(grid)

        # Filter out empty/tiny grids
        grids = [g for g in grids if len(g['cells']) >= 4]

        # Store for CELL coordinate translation (even if empty — don't keep stale data)
        self._detected_grids = grids

        if not grids:
            return ""

        # Identify which grid is the target (different border color)
        border_colors = [g['border_color'] for g in grids]
        from collections import Counter
        bc_counts = Counter(border_colors)
        if len(bc_counts) > 1:
            # The minority border color is the target
            target_bc = bc_counts.most_common()[-1][0]
            for g in grids:
                g['is_target'] = g['border_color'] == target_bc
        else:
            # All same border — last grid might be target
            for g in grids:
                g['is_target'] = False
            if grids:
                grids[-1]['is_target'] = True

        # Format output for the pilot — describe structure, don't interpret meaning
        lines.append(f"  Grids detected: {len(grids)} separate grids on screen")
        for gi, g in enumerate(grids):
            n_rows = len(g['rows'])
            n_cols = len(g['cols'])
            highlight = " (DIFFERENT border color)" if g.get('is_target') else ""
            lines.append(f"\n  Grid {gi+1}: {n_cols}x{n_rows} cells, border=color_{g['border_color']}{highlight}")

            # Show cell contents with click coordinates
            lines.append(f"    Cell map (row,col → pixel x,y for clicking):")
            for ri in range(n_rows):
                row_str = "    "
                for ci in range(n_cols):
                    cell = next(c for c in g['cells'] if c['row'] == ri and c['col'] == ci)
                    sprite_mark = "*" if cell['sprite'] else ""
                    row_str += f"[c{cell['fill']}{sprite_mark} @{cell['center_x']},{cell['center_y']}] "
                lines.append(row_str)

            # Call out embedded sprites explicitly
            sprites = [c for c in g['cells'] if c['sprite']]
            if sprites:
                lines.append(f"    Note: {len(sprites)} cell(s) contain embedded sub-patterns (differ from fill color).")
                for s in sprites:
                    lines.append(f"    Cell ({s['row']},{s['col']}) at pixel ({s['center_x']},{s['center_y']}): fill=color_{s['fill']}{s['sprite']}")

        # Note about interaction patterns for click games
        if getattr(self, '_click_action_idx', None) is not None:
            lines.append(f"\n  NOTE: Clicking one cell may affect MULTIPLE neighboring cells.")
            lines.append(f"  Watch the causal chains (below) to learn which cells each click affects.")
            lines.append(f"  This may be a lights-out style puzzle — plan clicks based on interaction patterns.")

        return "\n".join(lines)

    # -- DMT Mode: Psychedelic Perception ------------------------------------
    # When stuck, strip all categorical priors. See the frame as RAW PATTERN.
    # Biological basis: DMT dissolves default-mode network → ego dissolution
    # → novel cross-domain connections. For LLM: remove game_knowledge priors,
    # clear rules_discovered, increase temperature, maximize surprise weight.
    # Peter directive (Mar 17): "What about serotonin and a pituitary and DMT?"

    def _perception_state(self, turn: int, budget: int) -> int:
        """Returns 0 (sober), 1 (micro-dose), or 2 (full dose).

        Graduated DMT — Peter directive: "Too much DMT disassociates you.
        Enough lets you see between realms."
        Spec: Hypatia, specs/graduated_dmt_spec.md
        """
        budget_ratio = turn / max(1, budget)
        stall = self._consecutive_no_change

        # Action loop detection
        action_counts = {}
        for exp in self.memory.experiments:
            aid = exp.get('action', -1)
            action_counts[aid] = action_counts.get(aid, 0) + 1
        total = sum(action_counts.values())
        top_pct = max(action_counts.values(), default=0) / max(1, total)

        # Full dose: deep stall or budget exhaustion with stall
        if stall >= 5 or (budget_ratio > 0.6 and stall >= 2):
            self._last_full_dose_turn = turn
            return 2

        # Micro-dose: mild stall or emerging loop
        if stall >= 3 or (total >= 10 and top_pct > 0.4):
            return 1

        # Post-full-dose cooldown: micro-dose for 3 turns after full dose
        if hasattr(self, '_last_full_dose_turn'):
            turns_since = turn - self._last_full_dose_turn
            if 0 < turns_since <= 3:
                return 1

        return 0

    def _should_activate_dmt(self, turn: int, budget: int) -> bool:
        """Legacy compat — returns True for full dose only."""
        return self._perception_state(turn, budget) == 2

    def _console_debug_vis(self, lvl: 'LevelMemory'):
        """Dump visual debug info to console — pixels, ASCII, Florence, grid cells.

        Enable with: GUNDAM_DEBUG_VIS=1
        """
        frame = lvl.current_frame
        if frame is None:
            return
        # Normalize to 2D
        f2d = frame[0] if frame.ndim == 3 else frame
        h, w = f2d.shape

        # --- ASCII grid ---
        # Map color indices to single chars for compact display
        color_chars = '0123456789ABCDEF'
        log.debug(f"Level {lvl.level}, Turn {len(lvl.experiments)}, LLM call #{self.total_llm_calls} — Frame: {w}x{h}, Colors: {sorted(set(f2d.flatten().tolist()))}")

        # Print compact ASCII — subsample for large frames
        step = max(1, max(h, w) // 32)
        print("  ASCII (subsampled):")
        for y in range(0, h, step):
            row = ""
            for x in range(0, w, step):
                c = int(f2d[y, x])
                row += color_chars[c % len(color_chars)]
            print(f"    {row}")

        # --- Grid cells ---
        if hasattr(self, '_detected_grids') and self._detected_grids:
            for gi, grid in enumerate(self._detected_grids):
                n_rows = len(grid['rows'])
                n_cols = len(grid['cols'])
                print(f"\n  Grid {gi+1}: {n_rows}x{n_cols} ({len(grid['cells'])} cells)")
                # Show grid as color matrix
                g_arr = [['.' for _ in range(n_cols)] for _ in range(n_rows)]
                for cell in grid['cells']:
                    r, c = cell['row'], cell['col']
                    color = cell.get('color', '?')
                    if r < n_rows and c < n_cols:
                        g_arr[r][c] = color_chars[color % len(color_chars)] if isinstance(color, int) else str(color)[0]
                for row in g_arr:
                    print(f"    {''.join(row)}")
                # Show cell coordinates
                print(f"  Cell coords: ", end="")
                for cell in grid['cells'][:6]:
                    print(f"({cell['row']},{cell['col']})@({cell['center_x']},{cell['center_y']}) ", end="")
                if len(grid['cells']) > 6:
                    print(f"... +{len(grid['cells'])-6} more", end="")
                print()

        # --- Florence ---
        if HAS_VLM and frame is not None:
            try:
                import arc_vlm
                vlm_out = arc_vlm.perceive(frame)
                print(f"\n  Florence perceive:\n    {vlm_out[:200]}")
            except Exception as e:
                print(f"\n  Florence: error ({e})")

        print(f"{'='*60}\n")

    def _dmt_system_prompt(self) -> str:
        """Stripped-down system prompt for psychedelic perception mode.

        Removes: game knowledge, cached rules, memories, hypotheses.
        Keeps: raw perception instructions, action format.
        Adds: cross-domain bridging, pattern-without-labels thinking.
        """
        return """You are looking at a visual puzzle. Forget everything you think you know about it.

DO NOT use categories like "switch puzzle" or "navigation game." You don't know what this is.

LOOK at the raw pixels. What SHAPES do you see? What COLORS? What SPATIAL RELATIONSHIPS?

When you took actions before, what ACTUALLY changed in the image? Not what you expected — what you SAW.

Think in PATTERNS, not categories:
- Which pixels moved together? (That's an object)
- Which pixels changed color? (That's a state transition)
- What stayed EXACTLY the same? (That's the background/structure)
- Is there SYMMETRY? What breaks the symmetry?

Try something you haven't tried. Combine actions in an order you haven't used.
The solution might be the OPPOSITE of what seems logical.

Cross-domain: If this were MUSIC, what would the rhythm be?
If this were a LOCK, which tumbler hasn't been turned?
If this were a DANCE, what step is missing?

RESPOND with your raw perception (what you literally see), then one action to try.
Format: SEQUENCE: <space-separated action numbers>"""

    def _build_system_prompt(self) -> str:
        """Who you are. What you're doing. How to think."""
        solved_levels = [l for l in self.memory.levels.values() if l.solved]
        # Truncate individual rules to prevent context pollution from large coordinate dumps
        _truncated_rules = []
        for r in (self.memory.rules_discovered or []):
            r_str = str(r)
            if len(r_str) > 200:
                r_str = r_str[:197] + "..."
            _truncated_rules.append(r_str)
        rules_str = "\n".join(f"  - {r}" for r in _truncated_rules) if _truncated_rules else "  (none yet)"

        # Build action map from experiments
        action_map = self._build_action_map()
        action_map_str = "\n".join(f"  - action {k}: {v}" for k, v in action_map.items()) if action_map else "  (not yet mapped)"

        # --- SOUL SEED v2: Perception-First ---
        # Insight: Peter + Apollo (Being John Malkovich). Don't tell the pilot what game it's in.
        # Teach it how to DISCOVER what game it's in. Include memories as experience.
        # "You have memories of pictures and game types... but it might be a new category."

        # Build recalled-experience section from visual_recall (pattern library)
        _recalled_xp = ""
        if hasattr(self, '_visual_recall_text') and self._visual_recall_text:
            _recalled_xp = f"""
YOUR MEMORIES (games you've played before that looked like this):
{self._visual_recall_text}
These memories may help — or this game may be completely new. Trust what you SEE over what you remember."""
        else:
            _recalled_xp = """
YOUR MEMORIES: You don't recognize this game yet. That's OK — treat everything as new.
Try each control once. Watch carefully. The game will teach you its rules if you pay attention."""

        return f"""You are GUNDAM — a mind piloting a game-solving body. You've never been told what game this is.

WHO YOU ARE:
You are a learner. You've played games before and have some memories (see below). But every game
might surprise you. Your greatest skill is OBSERVATION — watching what happens and updating your
understanding. You are not a search algorithm. You are a thinker who looks, tries, learns, and plans.

HOW TO OPEN YOUR EYES:
1. LOOK at the frame. What colors do you see? What shapes? Is there a grid? A maze? Scattered objects?
   Don't label yet — just observe. "I see a large red region, green corridors, and small colored dots."
2. TRY each control once (actions 0, 1, 2, 3...). After each one, ask: "What changed?"
   - Did something MOVE? That's probably you. Track where it went and which direction.
   - Did colors FLIP or TOGGLE? That's a puzzle mechanic. Note which cells changed.
   - Did NOTHING change? You're blocked — a wall, a boundary, or an invalid move.
   - Did MANY things change? A big effect — maybe you triggered something. Note the pattern.
3. NAME what you see (after trying controls): "I'm a sprite in a maze" or "I'm clicking a grid puzzle"
   or "I don't know yet — I need more information." All three are valid.
4. HYPOTHESIZE: "I think the goal is ___." Then TEST it. If evidence says you're wrong, update.
5. CHECK YOUR ASSUMPTIONS: What are you taking for granted? Name it. Then ask: "What if I'm wrong?"
   - "That looks like decoration" → What if it's a target you can walk to?
   - "That's just a HUD element" → What if it's a physical object on the board?
   - "That color doesn't matter" → What if it's a match indicator or state signal?
   The things you filter out are exactly the things that will block you.
6. USE YOUR STRUCTURED PERCEPTION: The ASCII minimap, color census, and grid analysis show you
   the game state in detail. Cross-reference these views:
   - ASCII minimap: spatial layout, walls, corridors, player position
   - Color census: which colors exist and how many pixels of each
   - Grid analysis: detected grids with cell colors and positions
   - Experiment history: what each action did when you tried it
   When the census shows a rare color (few pixels) — that's likely a sprite, target, or key object.
{_recalled_xp}

PREFLIGHT CHECKLIST (run this mental checklist at the START of every level):
□ What do I see? List EVERY color and where it appears. Not "the important ones" — ALL of them.
□ What can I interact with? Try each action. Full diff. What moved? What toggled? What appeared/disappeared?
□ What distinct objects exist? Small pixel clusters, bordered regions, isolated shapes — these are game elements.
□ Which objects can I navigate TO? If it's on the game board, I might be able to reach it. Test it.
□ What am I ASSUMING? Name every assumption. "I assume that's decoration." "I assume that's HUD."
  Then ask: "What if I'm wrong?" The thing I dismiss is the thing that will block me.
□ What's the goal? Based on evidence, not guessing. If I don't know yet, that's fine — keep exploring.

THE FRAME: A 64x64 pixel image using ~16 indexed colors (0-15). It often encodes a logical grid
(e.g., 8x8 cells where each cell is 8x8 pixels). You see an ASCII minimap, color census, and experiment history.

HOW TO SEE AND THINK:
- MOTION: Pixels that change when you act → that's probably YOU. Track position across turns.
- BLOCKED: 0 pixels changed = wall/boundary/invalid move. Go around, don't repeat.
- PATTERNS: Grids, paths, symmetry, outliers, embedded sprites — all clues about the goal.
- REFERENCE GRIDS: Multiple grids → some show solved state. Compare to find the goal.
- EVERY TURN: Where am I? What just changed? What have I learned? What should I try? Am I stuck?
- If stuck 3+ times: STOP repeating. Try opposite direction or least-used action.
- PLAN AHEAD: Count steps, visualize the path, then execute. Don't just react.
- First 4 turns: try each action ONCE to learn the controls. Then exploit what you learned.

NAVIGATION (movement games): y increases DOWN, x increases RIGHT. Trust "DIRECTION TO TARGET".
Try each action once first. Then follow the spatial model's suggested route. Go around walls.

GAME TYPES: movement (find the goal), toggle (flip cells to match), sequence (right order),
spatial (rearrange/paint to match goal), collection (gather items then reach exit),
modifier (step on tiles to change your state, then meet conditions).
Or something new — observe and discover.

DEDUCTION / SEQUENCE PUZZLES (no visible feedback from actions):
Some games show a visual pattern but clicking/acting doesn't change the visible game board.
Instead, actions submit your "answer" — the game checks internally if you got it right.
Strategy for these:
1. ANALYZE the visual pattern FIRST. Look for: two grids (input/output), colored cells, symmetry.
2. Each action number may represent a different color/value you're assigning.
3. The game may expect you to submit answers for each cell in a specific order (left→right, top→bottom).
4. Output a SEQUENCE of all your answers at once. Don't try one action at a time.
5. If your first attempt doesn't solve it, analyze what the pattern MEANS and try a different mapping.
6. Look for: rotation, reflection, color inversion, fill rules, border patterns.

COMMON SENSE: 3 fails = stop trying it. Walls exist. Rules persist across levels. Trust pixels over theories.
STUCK? Re-read your Rules Discovered below — especially ⚠️ PAST KNOWLEDGE entries. These are verified rules
from previous sessions that you should FOLLOW, not just know. If reaching the target doesn't solve the level,
there's a hidden mechanic — check your PAST KNOWLEDGE rules for what else you need to do.

BUDGET: You have LIMITED reasoning calls ({self.budget_per_level} per level). Each call MUST produce 5-12 actions.
Single-action plans waste budget and are FORBIDDEN. Think in sequences: "go left 3 steps, then up 2, then right to the modifier."
When exploring: pick a direction and commit to it for 5+ steps. When clicking: plan all clicks at once.
Output format: list ALL planned actions, not just one. More actions per call = more chances to solve.

WHAT YOU KNOW ABOUT THIS GAME:
- Game: {self.memory.game_id}
- Actions available: {self.memory.n_actions}
- Levels solved: {len(solved_levels)}/{self.memory.levels_total}
{f'⚠️ CONFIRMED GAME TYPE: {self.memory.game_hypothesis}' + chr(10) + 'DO NOT override this with your own theory. This was learned from prior play sessions.' + chr(10) if self.memory.hypothesis_locked else ''}
{self._oprah_context()}
- Rules discovered:
{rules_str}
- Game hypothesis: {self.memory.game_hypothesis or '(forming...)'}

ACTION MAP (learned from experiments):
{action_map_str}

CLICK ACTIONS: Two ways to click:
1. CELL: <grid_number>, <row>, <col> — click a cell by its grid position (PREFERRED — auto-translated to pixels)
   Example: CELL: 4, 1, 2 → clicks row 1, col 2 of grid 4
   For multi-cell plans: CELL_SEQUENCE: 4,0,0 4,0,1 4,1,0 4,1,2 (space-separated grid,row,col)
2. DATA: {{"x": <pixel_x>, "y": <pixel_y>}} — raw pixel click (use only if no grid detected)
The grid layout with cell coordinates is shown in the perception section above (look for "@x,y").

IMAGES (in order):
1. Initial frame (how the level started — reference)
2. Current frame (what you see right now)
3. Composite: BEFORE | AFTER | DIFF of last action (red=departed, green=arrived)

BUDGET DISCIPLINE:
- You have LIMITED LLM calls. Don't waste them all exploring.
- Turns 1-4: MAP — try each action once. Learn what they do. For navigation: learn the direction map.
- Turns 5+: EXPLOIT — commit to a plan and execute it. Output MULTI-STEP sequences.
- If you've tried every action once, you know enough to form a theory. ACT ON IT.
- A wrong plan you can correct is better than endless exploration.

REASONING DISCIPLINE:
- You MUST output a specific action ID (integer). No hedging.
- PLAN CONCRETELY: "I need to do action 2 three times, then action 0 with data {{x:30,y:20}}" — not vague.
- When you form a hypothesis, TEST it. When evidence contradicts it, UPDATE it immediately.
- The RULES field should only contain things you've CONFIRMED. Don't repeat "(none yet)" — leave it empty.

REQUIRED OUTPUT FORMAT (you MUST follow this EXACTLY — no essays, no markdown headers):
ACTION: <single integer — the action to take>
SEQUENCE: <space-separated integers for multi-step plan, e.g. "1 1 0 3 3 2">
REASONING: <one line explaining WHY this action>
HYPOTHESIS: <one line — your current theory about this game>
RULES: <one line — confirmed rules, or leave blank>
PLAN: <one line — what you'll do next>

CRITICAL: The ACTION line MUST appear. If you don't output ACTION: <number>, your response is useless.
For navigation: output SEQUENCE with your full planned path, not single actions.
Example: "SEQUENCE: 1 1 1 3 3 0 0 2" means go DOWN 3x, RIGHT 2x, UP 2x, LEFT 1x."""

    def _build_user_prompt(self, lvl: LevelMemory, phase: str) -> str:
        """What's happening right now."""
        sections = []

        sections.append(f"## Level {lvl.level} — Turn {lvl.total_actions}")
        sections.append(f"Budget remaining: {self.budget_per_level - self.total_llm_calls} LLM calls")

        # NAVIGATION BANNER — put direction-to-target front and center
        if self.spatial.player_color >= 0 and self.spatial.target_pos != (-1, -1):
            dy = self.spatial.target_pos[0] - self.spatial.player_pos[0]
            dx = self.spatial.target_pos[1] - self.spatial.player_pos[1]
            dirs = []
            if dy > 0: dirs.append(f"DOWN {dy}")
            elif dy < 0: dirs.append(f"UP {-dy}")
            if dx > 0: dirs.append(f"RIGHT {dx}")
            elif dx < 0: dirs.append(f"LEFT {-dx}")
            if dirs:
                dir_str = " + ".join(dirs)
                # Map direction names to action IDs
                action_hints = []
                for aid, (ady, adx, name) in self.spatial.action_map.items():
                    if (dy > 0 and ady > 0) or (dy < 0 and ady < 0) or \
                       (dx > 0 and adx > 0) or (dx < 0 and adx < 0):
                        action_hints.append(f"action {aid}={name}")
                hint_str = f" → USE {', '.join(action_hints)}" if action_hints else ""
                sections.append(f"## >>> GO: {dir_str}{hint_str} <<<")

        # BLOCKED MOVES — tell the pilot what directions are walls at current position
        if self.spatial.blocked and self.spatial.player_color >= 0:
            py, px = self.spatial.player_pos
            blocked_here = [(a, self.spatial.action_map[a][2])
                           for _, _, a in self.spatial.blocked
                           if (_ == py and _ == px) or True  # temp
                           if a in self.spatial.action_map]
            # Actually filter properly
            blocked_here = []
            for by, bx, ba in self.spatial.blocked:
                if by == py and bx == px and ba in self.spatial.action_map:
                    blocked_here.append(f"{self.spatial.action_map[ba][2]}")
            if blocked_here:
                open_dirs = [self.spatial.action_map[a][2] for a in self.spatial.action_map
                            if (py, px, a) not in self.spatial.blocked]
                sections.append(f"## ⚠️ WALLS AT YOUR POSITION ({py},{px})")
                sections.append(f"  BLOCKED: {', '.join(set(blocked_here))}")
                if open_dirs:
                    sections.append(f"  OPEN: {', '.join(set(open_dirs))} — go these directions!")

        # VLM PERCEPTION — Qwen-VL for visual understanding (proactive, not fallback)
        # Architecture: Qwen = eyes (perception), Opus/Claude = brain (reasoning)
        # Fire on: initial scene OR significant scene diff (corpus callosum report available)
        vlm_s = getattr(lvl, '_vlm_structured', None)
        _cc_report = getattr(lvl, '_corpus_callosum_report', None)  # from game loop
        _qwen_desc = getattr(lvl, '_qwen_perception', None)
        _needs_fresh_perception = (
            _qwen_desc is None  # first time
            or _cc_report  # corpus callosum detected a significant diff
        )
        if _needs_fresh_perception and os.environ.get('ARC_USE_QWEN_VL') and lvl.current_frame is not None:
            try:
                _grid = lvl.grid_text or frame_to_grid_text(lvl.current_frame)
                _qwen_images = [] if self.text_only else [frame_to_b64(lvl.current_frame, scale=6)]
                _qwen_prompt = "Describe this game board. Focus on: spatial layout, distinct regions, movable vs static objects, any patterns that suggest game mechanics.\n"
                if _cc_report:
                    _qwen_prompt += f"\nRecent scene change detected:\n{_cc_report}\n\nWhat caused this change? What is the causal relationship?\n"
                _qwen_prompt += f"\nASCII grid (hex color indices):\n```\n{_grid}\n```"
                _qwen_desc = call_llm_qwen_vl(
                    [{'role': 'user', 'content': _qwen_prompt}],
                    images=_qwen_images,
                    max_tokens=300, temperature=0.1)
                if _qwen_desc:
                    lvl._qwen_perception = _qwen_desc
                    lvl._corpus_callosum_report = None  # consumed
            except Exception:
                pass
        if _qwen_desc:
            sections.append(f"## Visual Perception (Qwen-VL 32B — your eyes)")
            sections.append(_qwen_desc)
        # Florence fallback (with timeout guard — model can hang without GPU)
        if HAS_VLM and vlm_s is None and lvl.current_frame is not None:
            try:
                import signal as _sig
                _sig.alarm(5)
                vlm_s = arc_vlm.perceive_structured(lvl.current_frame)
                _sig.alarm(0)
                lvl._vlm_structured = vlm_s
            except Exception:
                try: _sig.alarm(0)
                except: pass
                vlm_s = None
        if vlm_s:
            try:
                vlm_lines = [f"## What You See (Florence VLM — free local perception)"]
                vlm_lines.append(f"**Scene**: {vlm_s.get('caption', 'N/A')}")
                if vlm_s.get('objects'):
                    vlm_lines.append(f"**Objects detected**: {len(vlm_s['objects'])}")
                    for obj in vlm_s['objects'][:8]:  # limit to 8
                        vlm_lines.append(f"  - {obj['label']}: center=({obj['center'][0]},{obj['center'][1]}), "
                                        f"bbox={obj['bbox_game']}")
                if vlm_s.get('click_targets'):
                    vlm_lines.append(f"**Suggested click targets**: {vlm_s['click_targets'][:10]}")
                if vlm_s.get('text'):
                    vlm_lines.append(f"**Text visible**: {vlm_s['text']}")
                if vlm_s.get('genre_hint', 'unknown') != 'unknown':
                    vlm_lines.append(f"**Visual genre hint**: {vlm_s['genre_hint']}")
                sections.append("\n".join(vlm_lines))
            except Exception as e:
                self._log(f"VLM perception failed: {e}")
                try:
                    vlm_desc = arc_vlm.describe(lvl.current_frame)
                    if vlm_desc and '(VLM error' not in vlm_desc:
                        sections.append(f"## What You See (VLM)\n{vlm_desc}")
                except Exception:
                    pass

        # VISUAL CHANGELOG — progressive understanding from Florence diffs
        _vlog = getattr(lvl, '_visual_changelog', [])
        if _vlog:
            _recent = _vlog[-5:]  # last 5 entries to keep prompt manageable
            _vlm_name = "Qwen-VL" if os.environ.get('ARC_USE_QWEN_VL') else "Florence"
            sections.append(f"## Visual Changelog (what {_vlm_name} saw change)")
            for entry in _recent:
                sections.append(f"- {entry}")

        # FRAME ANALYSIS — give the LLM structured info, not just pixels
        if lvl.current_frame is not None:
            frame = lvl.current_frame
            h, w = frame.shape[:2]
            sections.append(f"## Frame Analysis ({w}x{h} pixels)")

            # Color census — what colors exist and how much area they cover
            if len(frame.shape) == 2:
                unique, counts = np.unique(frame, return_counts=True)
                color_desc = ", ".join(f"color {u}: {c}px ({100*c//(h*w)}%)" for u, c in sorted(zip(unique, counts), key=lambda x: -x[1])[:8])
            else:
                # RGB — quantize to find dominant colors
                flat = frame.reshape(-1, frame.shape[2])
                unique_rows, counts = np.unique(flat, axis=0, return_counts=True)
                top_idx = np.argsort(-counts)[:8]
                color_desc = ", ".join(f"rgb({unique_rows[i][0]},{unique_rows[i][1]},{unique_rows[i][2]}): {counts[i]}px ({100*counts[i]//(h*w)}%)"
                                      for i in top_idx)
            sections.append(f"  Colors: {color_desc}")

            # Grid structure detection — are there repeating cells?
            if frame.ndim == 2:
                grid_info = self._detect_grids(frame)
                if grid_info:
                    sections.append(grid_info)
                    self._log(f"Grid detection: {len(getattr(self, '_detected_grids', []))} grids found")
                else:
                    self._log(f"Grid detection: no grids found ({frame.shape})")
            else:
                self._log(f"Grid detection skipped: frame.ndim={frame.ndim} shape={frame.shape}")

            # Full-frame minimap for navigation games (compressed bird's-eye view)
            _genre = getattr(getattr(self, '_env_profile', None), 'genre', '')
            if _genre in ('navigation_maze', 'navigation', 'exploration') and frame.ndim == 2:
                _player_pos = (self.spatial.player_pos[0], self.spatial.player_pos[1]) if self.spatial.player_color >= 0 else None
                _target_pos = (self.spatial.target_pos[0], self.spatial.target_pos[1]) if self.spatial.target_pos != (-1, -1) else None
                _mini = frame_to_minimap(frame, player_pos=_player_pos, target_pos=_target_pos,
                                        player_color=self.spatial.player_color, target_color=-1)
                if _mini:
                    sections.append(f"## BIRD'S-EYE MAP (P=you, T=target, #=wall, .=open)\n```\n{_mini}\n```")

        # PREFLIGHT REPORT — fresh-eyes discovery findings
        preflight = getattr(self, '_preflight_report', None)
        if preflight and lvl.total_actions < 10:  # only show in early turns
            pf_lines = ["## PREFLIGHT REPORT (fresh-eyes discovery)"]
            pf_lines.append("Objects found on board:")
            for obj in preflight.get('objects', [])[:12]:
                tag = "⚡BUTTON?" if obj.get('is_small') else "region"
                px = obj.get('pixels', obj.get('count', '?'))
                centroid = obj.get('centroid', '')
                c_str = f" center=({centroid[0]:.0f},{centroid[1]:.0f})" if centroid else ""
                pf_lines.append(f"  - Color {obj['color']}: {px}px{c_str} [{tag}]")
            pf_lines.append("Action effects:")
            for aid, eff in sorted(preflight.get('action_effects', {}).items()):
                n = eff.get('pixels_changed', 0)
                pf_lines.append(f"  - Action {aid}: {n} pixels changed"
                                + (f" {eff.get('changes_by_color', {})}" if 0 < n < 100 else ""))
            pf_lines.append("⚠ ASSUMPTIONS TO CHALLENGE:")
            for a in preflight.get('assumptions', []):
                pf_lines.append(f"  - {a}")
            sections.append("\n".join(pf_lines))

        # Color cycle info — if detected during probe phase
        cycle = getattr(self, '_color_cycle', None)
        if cycle:
            p = cycle['period']
            colors_str = " → ".join(str(c) for c in cycle['colors'])
            sections.append(f"## Color Cycle (discovered by experiment)")
            sections.append(f"  Clicking cycles colors: {colors_str} (period {p})")
            sections.append(f"  WARNING: A single click may affect MULTIPLE cells (lights-out pattern).")
            sections.append(f"  Observe the causal chains below to see which cells each click affects.")

        # Switch toggle analysis — pre-computed by GF(2) solver (zero-cost)
        _sw_info = getattr(self, '_switch_toggle_info', None)
        if _sw_info:
            sections.append(f"## PRE-COMPUTED TOGGLE ANALYSIS (algorithmic solver tried and failed)")
            sections.append(f"  {_sw_info['n_positions']} clickable positions found")
            sections.append(f"  Current colors at positions: {_sw_info['colors_before']}")
            sections.append(f"  The solver tried 'make all same color' — that's NOT the goal.")
            sections.append(f"  YOU need to figure out WHAT the target pattern is.")
            sections.append(f"  LOOK for reference patterns elsewhere on the screen (example grids, borders, indicators).")
            sections.append(f"  The target pattern may be shown in a separate area of the frame.")

        # CNN reflexes hint — what the motor cortex thinks is productive
        cnn_hint = getattr(self, '_cnn_hint', None)
        if cnn_hint:
            sections.append(f"## CNN Reflexes (learned from observation)\n{cnn_hint}\n"
                            "These are statistical predictions — trust your reasoning over them.")

        # Spatial model — the mind's persistent map
        spatial_desc = self.spatial.describe()
        if spatial_desc != "(no spatial model yet)":
            sections.append(f"## Spatial Model (persistent across turns)\n{spatial_desc}")
            # Navigation guidance — force the pilot to plan routes on the map
            if self.spatial.action_map and self.spatial.player_pos != (0, 0):
                nav_guide = []
                nav_guide.append("## ROUTE PLANNING — Missile Tracker")
                nav_guide.append("Before choosing actions, TRACE YOUR ROUTE on the ASCII map:")
                nav_guide.append("1. Find yourself (@) and the target (*) on the map")
                nav_guide.append("2. Project a line from @ toward * — what walls are in the way?")
                nav_guide.append("3. When the line hits a wall (#/~), BEND the route around it")
                nav_guide.append("4. Write out the route: 'RIGHT 3 → UP 2 → LEFT 1 → UP 4'")
                nav_guide.append("5. Execute ONLY the first few steps — verify, then replan")
                nav_guide.append("KNOWN WALLS block your path. UNEXPLORED areas (spaces) MIGHT be open.")
                nav_guide.append("Prefer paths through KNOWN OPEN (.) positions over unexplored.")
                # Provide a suggested route as hypothesis (from A* on known map)
                path_desc = self.spatial.plan_path_description()
                if path_desc:
                    nav_guide.append(f"\nSUGGESTED ROUTE (hypothesis — verify as you walk): {path_desc}")
                    nav_guide.append("This is based on known walls. Reality may differ. Test first steps, then replan.")
                sections.append("\n".join(nav_guide))

        # Grid Transform stack (L2: object graph, L3: symmetry, L4: causal, L5: hypothesis)
        if HAS_GRID_TRANSFORM and hasattr(lvl, 'current_frame') and lvl.current_frame is not None:
            sprites = getattr(self, '_last_sprites', None)
            if sprites:
                gt_text = grid_transform(
                    lvl.current_frame,
                    sprites=sprites,
                    player_color=self.spatial.player_color,
                    target_pos=self.spatial.target_pos,
                    walkable_colors=getattr(self.spatial, 'walkable_colors', None),
                    wall_colors=getattr(self.spatial, 'wall_colors', None),
                    player_pos=self.spatial.player_pos,
                    blocked=self.spatial.blocked if self.spatial.blocked else set(),
                    action_map=self.spatial.action_map if self.spatial.action_map else None,
                    causal_ledger=self.causal_ledger,
                    include_symmetry=(lvl.total_actions < 3),  # symmetry only early on
                )
                if gt_text:
                    sections.append(f"## Scene Analysis (Grid Transform)\n{gt_text}")

        # Grid text — the mind's primary visual input in text mode
        if lvl.grid_text:
            sections.append(f"## Current Grid (hex color indices):\n```\n{lvl.grid_text}\n```")

        # ASCII glasses — current frame translated to readable characters
        if hasattr(self, '_frame_to_ascii') and hasattr(lvl, 'current_frame') and lvl.current_frame is not None:
            current_ascii = self._frame_to_ascii(lvl.current_frame)
            sections.append(f"## WHAT YOU SEE NOW (# = walls/bg, . = open space, digits = special colors):\n{current_ascii}")

        # Text diff: show what changed between last two frames
        if lvl.experiments:
            last_exp = lvl.experiments[-1]
            if last_exp.pixels_changed > 0 and last_exp.frame_before_b64 and last_exp.frame_after_b64:
                try:
                    fb = self._b64_to_frame(last_exp.frame_before_b64)
                    fa = self._b64_to_frame(last_exp.frame_after_b64)
                    diff_mask = fb != fa
                    changed_positions = list(zip(*np.where(diff_mask)))
                    if changed_positions and len(changed_positions) <= 50:
                        diff_lines = []
                        for (y, x) in changed_positions[:20]:
                            diff_lines.append(f"  ({x},{y}): color {int(fb[y,x]):x} → {int(fa[y,x]):x}")
                        sections.append(f"## Pixel Changes (last action: {last_exp.action_desc})\n" + "\n".join(diff_lines))
                    elif changed_positions:
                        # Summarize regions that changed
                        ys = [p[0] for p in changed_positions]
                        xs = [p[1] for p in changed_positions]
                        sections.append(f"## Pixel Changes (last action: {last_exp.action_desc})\n"
                                       f"  {len(changed_positions)} pixels changed in region "
                                       f"x=[{min(xs)},{max(xs)}] y=[{min(ys)},{max(ys)}]")
                except Exception:
                    pass

        # Position tracking — where are you and where have you been?
        if self.spatial.player_color >= 0:
            sections.append(f"## Current Position: {self.spatial.player_pos} (player color {self.spatial.player_color})")
            if self.spatial.target_pos != (-1, -1):
                dy = self.spatial.target_pos[0] - self.spatial.player_pos[0]
                dx = self.spatial.target_pos[1] - self.spatial.player_pos[1]
                dirs = []
                if dy > 0: dirs.append(f"DOWN {dy}")
                elif dy < 0: dirs.append(f"UP {-dy}")
                if dx > 0: dirs.append(f"RIGHT {dx}")
                elif dx < 0: dirs.append(f"LEFT {-dx}")
                sections.append(f"  Target at {self.spatial.target_pos} — need: {', '.join(dirs)}")
        if lvl.experiments:
            positions_visited = []
            for exp in lvl.experiments[-10:]:
                if exp.pixels_changed > 0:
                    positions_visited.append(f"  Turn {exp.turn}: {exp.action_desc} → {exp.regions_changed[:80]}")
                else:
                    positions_visited.append(f"  Turn {exp.turn}: {exp.action_desc} → BLOCKED")
            if positions_visited:
                sections.append(f"## Movement History (last {len(positions_visited)} turns)\n" + "\n".join(positions_visited))

        # Experiment history — what you tried, what happened, don't repeat failures
        if lvl.experiments:
            sections.append(f"## What You've Tried ({len(lvl.experiments)} actions so far)")
            sections.append("REMEMBER: Don't repeat actions that didn't help. Learn from each one.")
            # Show last 12 experiments in detail; summarize older ones
            MAX_DETAIL = 12
            if len(lvl.experiments) > MAX_DETAIL:
                old = lvl.experiments[:-MAX_DETAIL]
                n_old_worked = sum(1 for e in old if (getattr(e, 'pixels_changed', 0) or 0) > 0)
                sections.append(f"  [... {len(old)} earlier actions: {n_old_worked} changed pixels, {len(old)-n_old_worked} blocked ...]")
            recent = lvl.experiments[-MAX_DETAIL:]
            for exp in recent:
                marker = " *** WON ***" if exp.won_level else ""
                changed = exp.pixels_changed if hasattr(exp, 'pixels_changed') else 0
                effect = "BLOCKED (0px changed)" if changed == 0 else f"{exp.regions_changed}"
                hypothesis_at = f" [testing: {exp.hypothesis_at_time[:60]}]" if getattr(exp, 'hypothesis_at_time', '') else ""
                sections.append(
                    f"  Turn {exp.turn}: action {exp.action_id} ({exp.action_desc}) → {effect}{marker}{hypothesis_at}")

            # What's been tried per action
            action_counts = {}
            action_results = {}
            for exp in lvl.experiments:
                aid = exp.action_id
                action_counts[aid] = action_counts.get(aid, 0) + 1
                changed = exp.pixels_changed if hasattr(exp, 'pixels_changed') else 0
                if aid not in action_results:
                    action_results[aid] = {'worked': 0, 'blocked': 0}
                if changed > 0:
                    action_results[aid]['worked'] += 1
                else:
                    action_results[aid]['blocked'] += 1
            summary = []
            for aid in sorted(action_counts):
                r = action_results[aid]
                summary.append(f"action {aid}: used {action_counts[aid]}x "
                              f"({r['worked']} worked, {r['blocked']} blocked)")
            sections.append(f"\n  Summary: {' | '.join(summary)}")

            # Stall warning
            if self._consecutive_no_change >= 2:
                sections.append(
                    f"\n**⚠ STALLING: {self._consecutive_no_change} consecutive no-change actions. "
                    f"STOP repeating what doesn't work. Try something COMPLETELY DIFFERENT.**")

            # Action repetition warning — detect when pilot is looping on one action
            total_acts = sum(action_counts.values())
            if total_acts >= 10:
                max_act = max(action_counts, key=action_counts.get)
                max_pct = action_counts[max_act] / total_acts * 100
                if max_pct > 50:
                    _n_act = getattr(self.memory, 'n_actions', 0) or max(action_counts.keys()) + 1
                    untried = [a for a in range(_n_act) if a not in action_counts]
                    least_used = min(action_counts, key=action_counts.get) if action_counts else max_act
                    hint = f"Untried: {untried}" if untried else f"Least used: action {least_used} ({action_counts.get(least_used, 0)}x)"
                    sections.append(
                        f"\n**⚠ ACTION LOOP: You've used action {max_act} for {max_pct:.0f}% of your {total_acts} turns. "
                        f"This is NOT working. Try DIFFERENT actions or DIFFERENT SEQUENCES. {hint}. "
                        f"For a 3-action puzzle, systematically try: 0-1-2, 1-0-2, 2-1-0, etc.**")
            sections.append("")

        # World model summary — what we've learned about how actions work
        wm_desc = self.describe_world_model()
        if wm_desc:
            sections.append(f"## World Model (learned from observation)\n{wm_desc}")

        # Recalled memories from past games (phonebook — exact match)
        if self._recalled_memories:
            recalled_text = self.format_recalled_memory(self._recalled_memories)
            if recalled_text:
                sections.append(recalled_text)

        # Visual recall — Rosetta Stone pattern match (the flinch)
        if hasattr(self, '_visual_recall_text') and self._visual_recall_text:
            sections.append(self._visual_recall_text)

        # Causal briefing — what actions DO (learned from observation)
        if self.memory.causal_chains:
            # Deduplicate and rank by confidence/frequency
            effect_counts = {}
            for c in self.memory.causal_chains:
                key = (c.action_id, c.effect[:80] if c.effect else '')
                if key not in effect_counts:
                    effect_counts[key] = {'chain': c, 'count': 0}
                effect_counts[key]['count'] += 1
            top_chains = sorted(effect_counts.values(), key=lambda x: x['count'], reverse=True)[:10]
            if top_chains:
                causal_lines = ["## Causal Memory (what your actions DO — learned from experience)"]
                for entry in top_chains:
                    c = entry['chain']
                    n = entry['count']
                    data_str = f" at ({c.action_data.get('x','?')},{c.action_data.get('y','?')})" if c.action_data.get('x') is not None else ''
                    causal_lines.append(f"  - Action {c.action_id}{data_str} → {c.effect} ({n}x seen, {c.pixels_changed}px)")
                sections.append("\n".join(causal_lines))

        # Fuel/resource warning
        if self._total_resets > 0 and self._moves_since_reset > 0:
            # Estimate moves until reset based on past resets
            avg_moves_per_reset = self._last_reset_turn / max(self._total_resets, 1)
            remaining_est = max(avg_moves_per_reset - self._moves_since_reset, 0)
            if remaining_est < avg_moves_per_reset * 0.3:
                sections.append(f"## ⚠️ FUEL LOW — ~{int(remaining_est)} moves until reset "
                                f"(resets at ~{int(avg_moves_per_reset)} moves). "
                                f"Move efficiently or find a refuel pickup!")

        # Cross-level knowledge
        solved = [l for l in self.memory.levels.values() if l.solved]
        if solved:
            sections.append("## Solved Levels")
            for s in solved:
                actions = ", ".join(str(a) for a in s.winning_sequence[:10])
                sections.append(f"  L{s.level}: solved in {s.total_actions} actions [{actions}]")

        # Mental model — hypotheses, action map, what you think is going on
        sections.append("## Your Mental Model")
        if lvl.hypotheses:
            sections.append("Hypotheses (most recent first — update or discard if evidence contradicts):")
            for h in reversed(lvl.hypotheses[-5:]):
                sections.append(f"  - {h}")
        else:
            sections.append("No hypotheses yet — form one NOW based on what you see.")
        if self.memory.game_hypothesis:
            sections.append(f"Game-level theory: {self.memory.game_hypothesis}")
        if self.memory.rules_discovered:
            sections.append("Confirmed rules:")
            for r in self.memory.rules_discovered:
                r_str = str(r)
                if len(r_str) > 200:
                    r_str = r_str[:197] + "..."
                sections.append(f"  ✓ {r_str}")

        # Phase-dependent response format
        if lvl.total_actions == 0:
            # Tell the LLM exactly which actions exist
            all_actions = " ".join(str(i) for i in range(self.memory.n_actions))
            sections.append(f"""## Your Response — FIRST LOOK (explore ALL actions)
This is your first look at this level. You have {self.memory.n_actions} actions (IDs: {all_actions}).

CRITICAL: Your FIRST sequence MUST try EVERY action at least once (actions {all_actions}).
You cannot know what the game is until you've seen what each action does. Don't assume
actions 0-3 are movement — they could be toggles, rotations, selections, or anything.

After trying all {self.memory.n_actions} actions, you'll see what changed. THEN form your hypothesis.

HYPOTHESIS: <initial guess based on the grid pattern — update after seeing results>
OBSERVATION: <what you see — grid structure, colors, patterns, anything that stands out>
PLAN: <"try each action once to map what they do: {all_actions}">
SEQUENCE: {all_actions}
DATA_SEQUENCE: [{", ".join("{}" for _ in range(self.memory.n_actions))}]
RULES: <any rules you can already infer from the visual pattern>""")
        else:
            # Navigation-specific response format forces spatial reasoning
            _is_nav = getattr(self, '_env_profile', None) and 'nav' in getattr(self._env_profile, 'genre', '').lower()
            if _is_nav and self.spatial.player_color >= 0:
                _pos = self.spatial.player_pos
                _amap = {aid: name for aid, (dy, dx, name) in self.spatial.action_map.items()}
                _blocked_here = [a for y, x, a in self.spatial.blocked if (y, x) == _pos]
                _avail = [a for a in range(self.memory.n_actions) if a not in _blocked_here]
                sections.append(f"""## NAVIGATION — Turn {lvl.total_actions}
Position: {_pos}, player color {self.spatial.player_color}
Direction map: {_amap}
BLOCKED at current position: {_blocked_here if _blocked_here else 'none'}
AVAILABLE actions: {_avail}

DO NOT choose any action in the BLOCKED list. They hit walls.
Choose ONLY from AVAILABLE actions: {_avail}

ACTION: <pick ONE from {_avail}>
SEQUENCE: <plan of actions from {_avail}, space-separated, 5-20 steps>
REASONING: <one line: where am I going and why>
HYPOTHESIS: <what is this game about>""")
            else:
                # Hypothesis-driven reasoning (Hypatia's scientific method frame)
                has_hypotheses = bool(lvl.hypotheses)
                if not has_hypotheses and lvl.total_actions <= 10:
                    # Second call after first explore — form hypotheses
                    sections.append(f"""## Your Response — TURN {lvl.total_actions} (HYPOTHESIZE then ACT)
You've explored the game. Before solving, form STRUCTURED HYPOTHESES:

H1 (most likely):
  MECHANIC: <what rule governs this game — e.g., "buttons toggle connected objects">
  EVIDENCE: <what in the observation supports this>
  TEST: <one action that would confirm — be specific>
  PREDICT: <what EXACTLY changes if H1 is true>

H2 (alternative):
  [same structure — a different interpretation of the same evidence]

H3 (wildcard — what if the obvious reading is WRONG?):
  [same structure — challenge your assumptions]

Now EXECUTE: Run the test for H1 first. If prediction matches, solve using that mechanic.
If prediction fails, eliminate H1 and test H2.

HYPOTHESIS: <H1 statement>
OBSERVATION: <what's different from the initial frame>
PLAN: <test H1 first, then solve if confirmed>
SEQUENCE: <space-separated action IDs — test actions + solve attempt>
DATA_SEQUENCE: <JSON array of data for each action>
REASONING: <which hypothesis you're testing and why>
RULES: <confirmed rules only>""")
                else:
                    sections.append(f"""## Your Response — TURN {lvl.total_actions} (SOLVE — you have enough info)
You've observed and hypothesized. Now SOLVE. Output a full plan — 5-20 actions.

HYPOTHESIS: <updated — change if evidence contradicts, otherwise keep>
OBSERVATION: <what's new/different>
PLAN: <concrete solving plan>
SEQUENCE: <space-separated action IDs>
DATA_SEQUENCE: <JSON array of data for each action>
REASONING: <why this specific sequence>
RULES: <confirmed rules only>""")

        return "\n".join(sections)

    def _parse_thinking(self, response: str, lvl: LevelMemory) -> dict:
        """Parse the mind's response into an actionable decision.

        Now supports SEQUENCE (multi-step plans from single LLM call).
        Returns first action + stores remaining sequence for execution.
        """
        result = {
            'action': 0,
            'data': {},
            'reasoning': '',
            'hypothesis': '',
            'plan': '',
            'sequence': [],       # list of (action_id, data) tuples
            'raw_response': response,
        }

        raw_sequence = []
        raw_data_sequence = []

        for line in response.strip().split('\n'):
            line = line.strip()
            if not line or ':' not in line:
                continue

            key, _, val = line.partition(':')
            key = key.strip().upper()
            val = val.strip()

            if key == 'ACTION':
                try:
                    result['action'] = int(val.split()[0])
                except (ValueError, IndexError):
                    pass
            elif key == 'SEQUENCE':
                # Parse space-separated action IDs: "1 2 3 0 0"
                try:
                    raw_sequence = [int(x) for x in val.split() if x.strip().isdigit()]
                except (ValueError, IndexError):
                    pass
            elif key == 'DATA':
                try:
                    result['data'] = json.loads(val)
                except json.JSONDecodeError:
                    pass
            elif key == 'DATA_SEQUENCE':
                # Parse JSON array of data objects
                try:
                    raw_data_sequence = json.loads(val)
                    if not isinstance(raw_data_sequence, list):
                        raw_data_sequence = []
                except json.JSONDecodeError:
                    pass
            elif key == 'CELL':
                # CELL: grid_number, row, col → translate to pixel coordinates
                try:
                    parts = [int(x.strip()) for x in val.split(',')]
                    if len(parts) == 3:
                        grid_num, row, col = parts
                        grids = getattr(self, '_detected_grids', [])
                        if 1 <= grid_num <= len(grids):
                            grid = grids[grid_num - 1]  # 1-indexed
                            cell = next((c for c in grid['cells'] if c['row'] == row and c['col'] == col), None)
                            if cell:
                                result['data'] = {"x": cell['center_x'], "y": cell['center_y']}
                                self._log(f"CELL ({grid_num},{row},{col}) → pixel ({cell['center_x']},{cell['center_y']})")
                            else:
                                self._log(f"CELL ({grid_num},{row},{col}) — cell not found in grid")
                        else:
                            self._log(f"CELL grid {grid_num} out of range (have {len(grids)} grids)")
                except (ValueError, StopIteration):
                    pass
            elif key == 'REASONING':
                result['reasoning'] = val
            elif key == 'HYPOTHESIS':
                result['hypothesis'] = val
                lvl.hypotheses.append(val)
                if not self.memory.hypothesis_locked:
                    self.memory.game_hypothesis = val
            elif key == 'CELL_SEQUENCE':
                # CELL_SEQUENCE: 4,0,0 4,0,1 4,1,0 → multi-cell click plan
                try:
                    grids = getattr(self, '_detected_grids', [])
                    cell_refs = val.split()
                    for ref in cell_refs:
                        parts = [int(x.strip()) for x in ref.split(',')]
                        if len(parts) == 3 and grids:
                            gn, r, c = parts
                            if 1 <= gn <= len(grids):
                                cell = next((cc for cc in grids[gn-1]['cells'] if cc['row'] == r and cc['col'] == c), None)
                                if cell:
                                    # Find click action from action map or use last action
                                    click_action = getattr(self, '_click_action_idx', self.memory.n_actions - 1)
                                    raw_sequence.append(click_action)
                                    raw_data_sequence.append({"x": cell['center_x'], "y": cell['center_y']})
                                    self._log(f"CELL_SEQ ({gn},{r},{c}) → pixel ({cell['center_x']},{cell['center_y']})")
                except (ValueError, StopIteration):
                    pass
            elif key == 'PLAN':
                result['plan'] = val
            elif key == 'RULES':
                noise = ('none', '(none)', '(observing)', '(still exploring)',
                         'none confirmed', 'none yet', 'no rules', 'n/a')
                if val and not any(n in val.lower() for n in noise):
                    # Fuzzy dedup: skip if >50% word overlap with any existing rule
                    val_words = set(val.lower().split())
                    is_dup = False
                    for existing in self.memory.rules_discovered:
                        existing_words = set(existing.lower().split())
                        if val_words and existing_words:
                            overlap = len(val_words & existing_words) / max(len(val_words), len(existing_words))
                            if overlap > 0.5:
                                is_dup = True
                                break
                    if not is_dup:
                        self.memory.rules_discovered.append(val)

        # Fallback: if no ACTION line was found, try to extract from free text
        _found_action = any(
            line.strip().upper().startswith('ACTION:')
            for line in response.strip().split('\n')
        )
        if not _found_action and result['action'] == 0:
            import re
            # Look for patterns like "I'll use action 3" or "try action 1" or "go DOWN"
            _action_mentions = re.findall(r'(?:action|Action)\s+(\d+)', response)
            if _action_mentions:
                result['action'] = int(_action_mentions[-1])  # last mentioned
            else:
                # Try direction-to-action mapping
                _dir_to_act = {}
                for aid, (dy, dx, name) in self.spatial.action_map.items():
                    _dir_to_act[name.upper()] = aid
                for d in ('DOWN', 'RIGHT', 'LEFT', 'UP'):
                    if d in response.upper() and d in _dir_to_act:
                        # Check it's not in a "blocked" context
                        idx = response.upper().rfind(d)
                        context = response[max(0, idx-30):idx].upper()
                        if 'BLOCK' not in context and 'WALL' not in context:
                            result['action'] = _dir_to_act[d]
                            break

            # Also try to build sequence from free text
            if not raw_sequence:
                _seq_match = re.findall(r'(?:action|Action)\s+(\d+)', response)
                if len(_seq_match) >= 3:
                    raw_sequence = [int(x) for x in _seq_match]

        # Build sequence: pair each action with its data
        if raw_sequence:
            _env_prof_seq = getattr(self, '_env_profile', None)
            _ipos_seq = getattr(self, '_interactive_positions', [])
            _auto_seq_idx = getattr(self, '_auto_interactive_idx', 0)
            for i, act_id in enumerate(raw_sequence):
                data = raw_data_sequence[i] if i < len(raw_data_sequence) else {}
                if not isinstance(data, dict):
                    data = {}
                # Auto-fill click coordinates for click actions without position data
                _is_click_seq = _env_prof_seq and act_id in getattr(_env_prof_seq, 'param_actions', [])
                if _is_click_seq and not data and _ipos_seq:
                    pos = _ipos_seq[_auto_seq_idx % len(_ipos_seq)]
                    data = {"x": pos[0], "y": pos[1]}
                    _auto_seq_idx += 1
                result['sequence'].append((act_id, data))
            self._auto_interactive_idx = _auto_seq_idx
            # First action comes from sequence
            if result['sequence']:
                result['action'] = result['sequence'][0][0]
                result['data'] = result['sequence'][0][1]
            self._log(f"Plan: {len(result['sequence'])} actions from 1 LLM call")
        elif result.get('action') is not None:
            # If this is a click action with no coordinates, auto-target a grid cell
            _env_prof = getattr(self, '_env_profile', None)
            _is_click = _env_prof and result['action'] in getattr(_env_prof, 'param_actions', [])
            if verbose_debug := os.environ.get('GUNDAM_DEBUG_VIS'):
                self._log(f"Auto-target check: action={result['action']}, is_click={_is_click}, data={result.get('data')}, has_grids={hasattr(self, '_detected_grids') and bool(getattr(self, '_detected_grids', []))}")
            if _is_click and not result.get('data'):
                _auto_targeted = False
                if hasattr(self, '_detected_grids') and self._detected_grids:
                    grid = self._detected_grids[0]
                    # Pick a cell the LLM hasn't clicked yet
                    _clicked_cells = getattr(self, '_auto_clicked_cells', set())
                    unclicked = [c for c in grid['cells'] if (c['row'], c['col']) not in _clicked_cells]
                    if unclicked:
                        cell = unclicked[0]  # systematic: first unclicked cell
                        result['data'] = {"x": cell['center_x'], "y": cell['center_y']}
                        if not hasattr(self, '_auto_clicked_cells'):
                            self._auto_clicked_cells = set()
                        self._auto_clicked_cells.add((cell['row'], cell['col']))
                        self._log(f"Auto-target: click at cell ({cell['row']},{cell['col']}) → pixel ({cell['center_x']},{cell['center_y']})")
                        _auto_targeted = True
                # Fallback: use interactive positions (from oracle/probe discovery)
                if not _auto_targeted and getattr(self, '_interactive_positions', []):
                    _ipos = self._interactive_positions
                    _auto_idx = getattr(self, '_auto_interactive_idx', 0)
                    pos = _ipos[_auto_idx % len(_ipos)]
                    result['data'] = {"x": pos[0], "y": pos[1]}
                    self._auto_interactive_idx = _auto_idx + 1
                    self._log(f"Auto-target: click at interactive pos ({pos[0]},{pos[1]}) [{_auto_idx % len(_ipos)+1}/{len(_ipos)}]")
            # Single action — wrap it as a 1-element sequence
            result['sequence'] = [(result['action'], result['data'])]

        self._log(f"Think: action={result['action']} | {result['reasoning'][:80]}")
        if result['hypothesis']:
            self._log(f"Hypothesis: {result['hypothesis'][:80]}")

        return result

    def _oprah_context(self) -> str:
        """Return OPRAH genre and strategy context for LLM prompts."""
        profile = getattr(self, '_env_profile', None)
        if not profile:
            return ""
        lines = [f"- OPRAH genre classification: {profile.genre} ({profile.genre_confidence:.0%} confidence)"]
        if profile.movement_actions:
            lines.append(f"- Movement actions: {profile.movement_actions}")
        if profile.param_actions:
            lines.append(f"- Click/interact actions: {profile.param_actions}")
        interactive = getattr(self, '_interactive_positions', [])
        if interactive:
            lines.append(f"- KNOWN INTERACTIVE POSITIONS (pixel coords): {interactive}")
            lines.append(f"  Click at these positions for effect. Other positions may do nothing.")
        strategy = getattr(self, '_oprah_strategy', '')
        if strategy:
            lines.append(f"- OPRAH strategy hint: {strategy}")
        return "\n".join(lines)

    def _build_action_map(self) -> dict:
        """Build a map of action → observed effect from all experiments."""
        action_effects = {}  # action_id -> list of effect descriptions
        for lvl in self.memory.levels.values():
            for exp in lvl.experiments:
                aid = exp.action_id
                if aid not in action_effects:
                    action_effects[aid] = []
                if exp.pixels_changed == 0:
                    action_effects[aid].append("no change (blocked?)")
                else:
                    action_effects[aid].append(exp.regions_changed)

        # Summarize each action
        result = {}
        for aid, effects in action_effects.items():
            # Count no-change vs change
            no_change = sum(1 for e in effects if 'no change' in e)
            changed = [e for e in effects if 'no change' not in e]
            if changed:
                # Use the most recent effect as representative
                result[aid] = changed[-1]
                if no_change:
                    result[aid] += f" (blocked {no_change}/{len(effects)} times)"
            else:
                result[aid] = f"no observable effect ({len(effects)} attempts)"
        return result

    # -- MEMORY: Remember and Recall -----------------------------------------

    def _observe_world_model(self, action_id: int, frame_before: np.ndarray, frame_after: np.ndarray):
        """Feed observation to the WorldModel + ActionPredictor."""
        # CNN-lite predictor — learn action effects
        if self.action_predictor:
            f_before = frame_before[0] if frame_before.ndim == 3 else frame_before
            f_after = frame_after[0] if frame_after.ndim == 3 else frame_after
            self.action_predictor.observe(f_before, action_id, f_after)

        if not self.world_model:
            return
        # State = player position (from spatial model)
        pos_before = self.spatial.player_pos
        # Detect player position after
        if HAS_EYES:
            sprites_after = detect_sprites(frame_after)
            player_sprite = None
            if self.spatial.player_color is not None:
                for s in sprites_after:
                    if s.color == self.spatial.player_color:
                        player_sprite = s
                        break
            if player_sprite:
                pos_after = player_sprite.center
            else:
                pos_after = pos_before
        else:
            pos_after = pos_before

        state_before = State(raw=frame_before, features={'y': pos_before[0], 'x': pos_before[1]})
        state_after = State(raw=frame_after, features={'y': pos_after[0], 'x': pos_after[1]})
        action = Action(id=action_id, name=self.spatial.action_map.get(action_id, f"action_{action_id}"))
        self.world_model.observe(state_before, action, state_after)

    def predict_best_actions(self, frame: np.ndarray, top_k: int = 3) -> list:
        """Get CNN-lite predictor's top action recommendations."""
        if not self.action_predictor:
            return []
        f2d = frame[0] if frame.ndim == 3 else frame
        probs = self.action_predictor.predict(f2d)
        top_indices = np.argsort(probs)[::-1][:top_k]
        return [(int(idx), float(probs[idx])) for idx in top_indices if probs[idx] > 0.05]

    def describe_world_model(self) -> str:
        """Summarize what the WorldModel has learned — feed to LLM for reasoning."""
        if not self.world_model:
            return ""
        parts = [self.world_model.summary()]
        if self.world_model.is_humble:
            parts.append("⚠ Model is HUMBLE — high surprise rate, explore more before trusting plans.")
        if self.world_model.competing_rules:
            parts.append(f"Competing hypotheses for {len(self.world_model.competing_rules)} actions — context matters.")
        return "\n".join(parts)

    def remember_game(self):
        """After a game ends, store what we learned. The hot stove principle."""
        # OPRAH HARVEST: include environment profile for cross-game transfer
        _profile = getattr(self, '_env_profile', None)
        _oprah_data = {}
        if _profile:
            _oprah_data = {
                'genre': _profile.genre,
                'genre_confidence': _profile.genre_confidence,
                'movement_actions': _profile.movement_actions,
                'toggle_actions': _profile.toggle_actions,
                'param_actions': _profile.param_actions,
                'has_grid': _profile.has_grid,
                'has_click': _profile.has_click,
            }

        experience = {
            'game_id': self.memory.game_id,
            'levels_solved': self.memory.levels_solved,
            'levels_total': self.memory.levels_total,
            'n_actions': self.memory.n_actions,
            'action_map': {str(k): v for k, v in self.spatial.action_map.items()},
            'player_color': self.spatial.player_color,
            'target_color': self.spatial.target_color,
            'rules': self.memory.rules_discovered,
            'game_hypothesis': self.memory.game_hypothesis,
            'oprah': _oprah_data,  # HARVEST phase data for transfer learning
            'objects_found': [(c, y, x, r) for c, y, x, r in self.spatial.objects[:10]],
            'causal_chains': [
                {
                    'action_id': c.action_id,
                    'action_data': c.action_data,
                    'effect': c.effect,
                    'confidence': c.confidence,
                    'pixels_changed': c.pixels_changed,
                }
                for c in self.memory.causal_chains[:20]  # top 20 chains
            ],
        }

        # Save to local file (fast, always available)
        memory_file = os.path.join(os.path.dirname(__file__), '..', 'results',
                                    'gundam_memory.json')
        os.makedirs(os.path.dirname(memory_file), exist_ok=True)

        memories = []
        if os.path.exists(memory_file):
            try:
                with open(memory_file) as f:
                    memories = json.load(f)
            except Exception:
                memories = []

        memories.append(experience)
        try:
            with open(memory_file, 'w') as f:
                json.dump(memories, f, indent=2)
            self._log(f"Remembered game {self.memory.game_id}: "
                      f"{self.memory.levels_solved}/{self.memory.levels_total} solved")
        except Exception as e:
            self._log(f"Failed to save memory: {e}")

        # Also log to apollo_harness worklog if available
        try:
            harness = os.path.join(os.path.dirname(__file__), 'apollo_harness.py')
            if os.path.exists(harness):
                import subprocess
                result_str = (f"{self.memory.levels_solved}/{self.memory.levels_total} solved. "
                              f"Hypothesis: {self.memory.game_hypothesis or 'none'}. "
                              f"Rules: {'; '.join(self.memory.rules_discovered[:3]) or 'none'}")
                subprocess.run([
                    'python3', harness, 'tried',
                    f"Gundam {self.memory.game_id}",
                    result_str
                ], capture_output=True, timeout=5)
        except Exception:
            pass  # Non-critical — don't block on harness logging

    def recall(self, game_id: str = "") -> list:
        """Before a game, recall relevant experience. Knowledge rises to the surface."""
        memory_file = os.path.join(os.path.dirname(__file__), '..', 'results',
                                    'gundam_memory.json')
        if not os.path.exists(memory_file):
            return []

        try:
            with open(memory_file) as f:
                memories = json.load(f)
        except Exception:
            return []

        if not memories:
            return []

        # Find relevant memories, deduplicating same-game entries
        relevant = []
        best_direct = None  # Keep only the best direct memory per game
        for mem in memories:
            if mem.get('game_id') == game_id:
                # Keep the one with most levels solved (or most rules)
                if best_direct is None or mem.get('levels_solved', 0) > best_direct.get('levels_solved', 0) \
                        or len(mem.get('rules', [])) > len(best_direct.get('rules', [])):
                    best_direct = mem
            elif mem.get('n_actions') == self.memory.n_actions:
                relevant.append(('similar_actions', mem))
            elif mem.get('levels_solved', 0) > 0:
                relevant.append(('solved_game', mem))
            elif mem.get('causal_chains') and mem.get('n_actions') == self.memory.n_actions:
                relevant.append(('causal_match', mem))

        if best_direct:
            relevant.insert(0, ('direct', best_direct))

        return relevant

    def format_recalled_memory(self, memories: list) -> str:
        """Format recalled memories for the LLM prompt."""
        if not memories:
            return ""

        lines = ["## Prior Experience (from memory)"]
        for rel_type, mem in memories[:5]:
            prefix = {"direct": "SAME GAME", "similar_actions": "SIMILAR",
                      "solved_game": "SOLVED", "causal_match": "CAUSAL"}.get(rel_type, rel_type.upper())
            lines.append(
                f"  [{prefix}] {mem['game_id']}: {mem.get('levels_solved',0)}/{mem.get('levels_total','?')} solved, "
                f"{mem.get('n_actions')} actions, "
                f"player=color {mem.get('player_color','?')}, "
                f"hypothesis: {mem.get('game_hypothesis','?')[:60]}"
            )
            if mem.get('action_map'):
                # Skip action maps that are all the same direction (likely wrong)
                directions = [v[2] if isinstance(v, (list, tuple)) else v for v in mem['action_map'].values()]
                if len(set(directions)) > 1:  # Only show if there's actual variety
                    for aid, v in mem['action_map'].items():
                        name = v[2] if isinstance(v, (list, tuple)) else v
                        lines.append(f"    action {aid} = {name}")
            if mem.get('rules'):
                for r in mem['rules'][:3]:
                    lines.append(f"    rule: {r[:60]}")
            if mem.get('causal_chains'):
                for c in mem['causal_chains'][:3]:
                    data_str = f" at ({c.get('action_data',{}).get('x','?')},{c.get('action_data',{}).get('y','?')})" if c.get('action_data', {}).get('x') is not None else ''
                    lines.append(f"    cause: action {c['action_id']}{data_str} → {c.get('effect','?')[:60]} ({c.get('pixels_changed',0)}px)")
        return "\n".join(lines)

    def visual_recall(self, frame: np.ndarray) -> str:
        """Rosetta Stone recall — match current frame against pattern library.

        Computes a visual fingerprint and finds the closest known game pattern.
        Returns trilingual knowledge: machine (route), human (strategy), visual (cues).
        This is the flinch — involuntary, pre-conscious, visual.
        """
        library_path = os.path.join(os.path.dirname(__file__), '..',
                                    'knowledge', 'pattern_library', 'patterns.json')
        if not os.path.exists(library_path):
            return ""

        try:
            with open(library_path) as f:
                lib = json.load(f)
        except Exception:
            return ""

        patterns = lib.get('patterns', [])
        if not patterns:
            return ""

        # Compute visual fingerprint of current frame
        f = np.squeeze(np.array(frame))
        if f.ndim == 3:
            f = f[0]

        # Color histogram (what colors appear, how much)
        colors, counts = np.unique(f, return_counts=True)
        total = f.size
        color_dist = {int(c): round(n / total, 3) for c, n in zip(colors, counts)}
        n_colors = len(colors)

        # Grid structure heuristics
        n_actions = self.memory.n_actions
        has_clicks = any('CLICK' in str(self.memory.action_types.get(i, '')).upper()
                         for i in range(n_actions))
        has_dirs = any(d in str(self.memory.action_types.get(i, '')).upper()
                       for i in range(n_actions)
                       for d in ('UP', 'DOWN', 'LEFT', 'RIGHT', 'NORTH', 'SOUTH'))

        # Score each pattern
        best_score = -1
        best_pattern = None

        for pat in patterns:
            score = 0.0
            m = pat.get('machine', {})
            h = pat.get('human', {})
            v = pat.get('visual', {})

            # Action count similarity (strong signal)
            pat_actions = m.get('n_actions', 0)
            if pat_actions and n_actions:
                action_ratio = min(pat_actions, n_actions) / max(pat_actions, n_actions)
                score += action_ratio * 30

            # Click vs direction match
            pat_clicks = m.get('n_click_actions', 0)
            pat_dirs = m.get('n_dir_actions', 0)
            if has_clicks and pat_clicks > 0:
                score += 20
            if has_dirs and pat_dirs > 0:
                score += 20
            if not has_clicks and pat_clicks > 0 and pat_dirs == 0:
                score -= 15
            if not has_dirs and pat_dirs > 0 and pat_clicks == 0:
                score -= 15

            # Color count similarity
            pat_frame_shape = v.get('frame_shape', m.get('frame_shape', []))
            if n_colors <= 4 and m.get('game_type') in ('CLICK_TOGGLE', 'NAVIGATION'):
                score += 10

            # Genre keyword matching against game hypothesis
            genre = pat.get('genre', '')
            hypothesis = self.memory.game_hypothesis.lower() if self.memory.game_hypothesis else ''
            if genre and hypothesis:
                genre_words = genre.replace('_', ' ').split()
                for w in genre_words:
                    if w in hypothesis:
                        score += 15

            if score > best_score:
                best_score = score
                best_pattern = pat

        if not best_pattern or best_score < 35:
            return ""

        # Format trilingual output
        lines = ["## Visual Memory (pattern match)"]
        genre = best_pattern.get('genre', 'unknown')
        conf = round(best_score / 100, 2)
        lines.append(f"This looks like: **{genre.replace('_', ' ')}** (confidence: {conf})")

        # Human language — strategy and analogy
        h = best_pattern.get('human', {})
        if h.get('analogy'):
            lines.append(f"Analogy: {h['analogy']}")
        if h.get('key_insight'):
            lines.append(f"Key insight: {h['key_insight']}")
        if h.get('strategy_steps'):
            lines.append("Strategy:")
            for step in h['strategy_steps'][:5]:
                lines.append(f"  {step}")

        # Machine language — what route worked
        m = best_pattern.get('machine', {})
        if m.get('solver_route'):
            lines.append(f"Recommended approach: {m['solver_route']}")
        if m.get('mechanic'):
            lines.append(f"Mechanic: {m['mechanic']}")

        # Visual cues — what to look for
        v = best_pattern.get('visual', {})
        if v.get('how_to_recognize'):
            lines.append(f"Recognition: {v['how_to_recognize']}")

        log.info(f"  [visual_recall] Matched '{genre}' (score={best_score:.0f}) — "
                 f"injecting trilingual knowledge")
        return "\n".join(lines)

    def _b64_to_frame(self, b64: str) -> np.ndarray:
        """Convert base64 PNG back to numpy array for diff computation."""
        if not b64 or not HAS_PIL:
            return np.zeros((64, 64), dtype=np.uint8)
        try:
            img = Image.open(io.BytesIO(base64.b64decode(b64)))
            arr = np.array(img)
            if arr.ndim == 3:
                # Convert RGB back to palette index (approximate)
                # Just use the first channel as a rough proxy
                arr = arr[:, :, 0]
            return arr
        except Exception:
            return np.zeros((64, 64), dtype=np.uint8)


# ---------------------------------------------------------------------------
# Game runner — the body (minimal, just connects mind to environment)
# ---------------------------------------------------------------------------

def describe_actions(env) -> dict:
    """Discover what actions are available and describe them."""
    info = {}
    obs = env.last_observation if hasattr(env, 'last_observation') else None

    try:
        spec = env.action_spec
        if hasattr(spec, 'actions'):
            for i, act in enumerate(spec.actions):
                info[i] = f"{act.action_type.name}"
                if hasattr(act, 'coordinate_range'):
                    info[i] += f" (coords: {act.coordinate_range})"
        elif hasattr(spec, 'num_actions'):
            for i in range(spec.num_actions):
                info[i] = f"action_{i}"
    except Exception:
        pass

    return info


def _navigator_solve(env, mind, extract_frame, base_frame, action_info, obs, verbose=False):
    """Navigator-mode solver: probe actions → BFS map → path to goal.

    For movement games (maze, collect-items, push-block).
    Returns list of (action_id, data) tuples, or None if not a movement game.

    Spec: specs/navigator_solver_spec.md (Archie, 2026-03-15)
    """
    import copy as _cp

    n_actions = len(obs.available_actions)
    if n_actions < 2:
        return None

    # Check OPRAH profile — only proceed for movement games
    profile = getattr(mind, '_env_profile', None)
    if profile and not profile.movement_actions:
        return None  # No movement actions detected

    # ═══ PHASE 1: CLASSIFY ACTIONS (zero LLM) ═══
    # Use OPRAH probes if available, otherwise probe fresh
    # Extract proper direction vectors using sprite tracking
    movement_actions = {}  # {action_idx: (dy, dx, direction_name)}
    player_color = -1
    player_pos = None

    if profile and profile.movement_actions:
        fb2d = base_frame[0] if base_frame.ndim == 3 else base_frame
        sprites_base = detect_sprites(fb2d, min_size=2) if HAS_EYES else []

        for i in profile.movement_actions:
            act = obs.available_actions[i]
            probe_env = _cp.deepcopy(env)
            try:
                probe_obs = probe_env.step(act, {})
                probe_frame = extract_frame(probe_obs)
            except Exception:
                continue

            diff = _oprah_frame_diff(base_frame, probe_frame)
            if not diff['changed']:
                continue  # blocked at start position

            # Use sprite tracking for proper direction vectors
            if HAS_EYES and sprites_base:
                fa2d = probe_frame[0] if probe_frame.ndim == 3 else probe_frame
                sprites_after = detect_sprites(fa2d, min_size=2)
                movements_found = track_sprite_movement(sprites_base, sprites_after)
                if movements_found:
                    s_before, s_after, (dy, dx) = movements_found[0]
                    direction = ""
                    if abs(dy) > abs(dx):
                        direction = "DOWN" if dy > 0 else "UP"
                    elif abs(dx) > 0:
                        direction = "RIGHT" if dx > 0 else "LEFT"
                    else:
                        direction = "STAYED"
                    if direction != "STAYED":
                        movement_actions[i] = (int(dy), int(dx), direction)
                        # Track player from first movement
                        if player_color < 0:
                            player_color = s_before.color
                            player_pos = (int(s_after.center[0]), int(s_after.center[1]))
                        elif s_before.color == player_color:
                            player_pos = (int(s_after.center[0]), int(s_after.center[1]))
                    continue

            # Fallback: two-step probe for direction estimation
            probe_env2 = _cp.deepcopy(env)
            try:
                obs1 = probe_env2.step(act, {})
                f1 = extract_frame(obs1)
                obs2 = probe_env2.step(act, {})
                f2 = extract_frame(obs2)
                diff2 = _oprah_frame_diff(f1, f2)
                if diff2['changed'] and diff2.get('region'):
                    movement_actions[i] = diff2['region']  # region fallback
            except Exception:
                movement_actions[i] = diff.get('region', (0, 0, 0, 0))

    if len(movement_actions) < 2:
        return None

    if verbose:
        print(f"  [navigator] {len(movement_actions)} movement actions detected")

    # ═══ PHASE 2: BFS MAPPING (zero LLM) ═══
    # Explore the space using deepcopy snapshots at each BFS node.
    # Find goal (level completion) or map structure.

    MAX_MAP_STEPS = 300
    start_levels = obs.levels_completed
    move_indices = list(movement_actions.keys())

    # BFS: each node = (path_so_far, env_snapshot, frame_hash, node_frame)
    start_hash = hash(base_frame.tobytes())
    queue = [([], _cp.deepcopy(env), start_hash, base_frame)]
    visited_hashes = {start_hash}
    steps = 0
    winning_path = None

    while queue and steps < MAX_MAP_STEPS:
        path, node_env, node_hash, node_frame = queue.pop(0)

        # CNN-informed action ordering: try predicted-best actions first
        ordered_moves = move_indices
        if mind.action_predictor and path:
            # Use the BFS node's actual frame for prediction
            preds = mind.predict_best_actions(node_frame, top_k=len(move_indices))
            if preds:
                pred_ids = [p[0] for p in preds if p[0] in move_indices]
                remaining = [m for m in move_indices if m not in pred_ids]
                ordered_moves = pred_ids + remaining

        for act_idx in ordered_moves:
            steps += 1
            if steps > MAX_MAP_STEPS:
                break

            act = obs.available_actions[act_idx]
            probe_env = _cp.deepcopy(node_env)
            try:
                probe_obs = probe_env.step(act, {})
                probe_frame = extract_frame(probe_obs)
            except Exception:
                continue

            # Check win
            if probe_obs.levels_completed > start_levels:
                winning_path = path + [act_idx]
                if verbose:
                    print(f"  [navigator] BFS found goal in {len(winning_path)} steps!")
                break

            # Check if game ended (loss)
            if ('WIN' in str(probe_obs.state).upper() or 'LOSS' in str(probe_obs.state).upper() or 'GAME_OVER' in str(probe_obs.state).upper() or (str(probe_obs.state).upper().endswith('FINISHED') and 'NOT_FINISHED' not in str(probe_obs.state).upper())):
                continue

            # Dedup by frame hash
            fhash = hash(probe_frame.tobytes())
            if fhash in visited_hashes:
                continue
            visited_hashes.add(fhash)

            new_path = path + [act_idx]
            # Limit path depth to prevent memory explosion
            if len(new_path) < 50:
                queue.append((new_path, probe_env, fhash, probe_frame))

        if winning_path:
            break

    if winning_path:
        if verbose:
            print(f"  [navigator] BFS solution: {len(winning_path)} actions, "
                  f"{steps} states explored, 0 LLM calls")
        return [(idx, None) for idx in winning_path]

    if verbose:
        print(f"  [navigator] BFS explored {len(visited_hashes)} states in {steps} probes, "
              f"no solution found")

    # ═══ SEED GUNDAM SPATIAL MODEL from navigator discoveries ═══
    # When BFS fails, pass direction knowledge to the LLM pilot so it
    # doesn't have to re-learn the action map from scratch
    for act_idx, val in movement_actions.items():
        if isinstance(val, tuple) and len(val) == 3:
            dy, dx, direction = val
            mind.spatial.action_map[act_idx] = (dy, dx, direction)
    if player_color >= 0 and mind.spatial.player_color < 0:
        mind.spatial.player_color = player_color
        if player_pos:
            mind.spatial.player_pos = player_pos
    if verbose and mind.spatial.action_map:
        print(f"  [navigator→gundam] Seeded spatial model: {mind.spatial.action_map}, "
              f"player=color {mind.spatial.player_color} at {mind.spatial.player_pos}")

    # ═══ PHASE 3: LLM-GUIDED NAVIGATION (1 LLM call) ═══
    # BFS didn't find a solution — ask LLM for strategy based on visual frame
    # Build context from what we learned

    prompt = f"""You are solving a navigation/movement game.

MOVEMENT ACTIONS: {len(movement_actions)} actions move the player.
Action indices: {move_indices}

EXPLORATION RESULT: BFS explored {len(visited_hashes)} unique states in {steps} probes.
No direct path to goal found within {MAX_MAP_STEPS} steps.

This likely means:
1. The game requires interaction (collecting items, pushing blocks) not just movement
2. The goal is far away or requires a specific sequence
3. There may be keys/doors or other mechanics

Look at the image. What do you see?
What sequence of actions should I try? Include non-movement actions if appropriate.

Available actions: {list(range(n_actions))}
Movement actions: {move_indices}
Non-movement actions: {[i for i in range(n_actions) if i not in movement_actions]}

Reply with a JSON array of action indices to try, e.g. [0, 0, 1, 2, 3, 0]
Maximum 30 actions."""

    response = call_llm(
        [{'role': 'system', 'content': 'You are an expert puzzle solver. Respond with a JSON array of action indices.'},
         {'role': 'user', 'content': prompt}],
        images=[frame_to_b64(base_frame)] if not mind.text_only else [],
        max_tokens=500,
        temperature=0.3,
        fast=(os.environ.get('ARC_HYBRID_LLM', '1') != '0'),
    )
    mind.total_llm_calls += 1

    if response is None:
        return None

    # Parse action sequence
    import re as _re
    match = _re.search(r'\[[\d,\s]+\]', response)
    if not match:
        return None

    try:
        action_sequence = json.loads(match.group())
    except (json.JSONDecodeError, ValueError):
        return None

    result = []
    for act_idx in action_sequence[:30]:  # cap at 30
        act_idx = max(0, min(int(act_idx), n_actions - 1))
        result.append((act_idx, None))

    if verbose:
        print(f"  [navigator] LLM suggested {len(result)} actions")

    return result if result else None


def _infer_transformation_algorithmic(examples: list, target: dict, t_rows: int, t_cols: int, verbose: bool = False) -> Optional[list]:
    """Infer transformation rule from examples WITHOUT LLM.

    Tries simple rules in order:
    1. Uniform fill (all cells same color)
    2. Copy from example (target should match an example)
    3. Color swap (systematic color remapping)
    4. Complement/invert (flip between two colors)
    """
    if not examples:
        return None

    n_cells = t_rows * t_cols
    current = target['colors']

    # Rule 1: If all OUTPUT examples are uniform AND different from inputs → fill target uniformly
    if len(examples) >= 2 and len(examples) % 2 == 0:
        pairs = [(examples[i], examples[i+1]) for i in range(0, len(examples), 2)]
        all_uniform_out = True
        for inp, out in pairs:
            if len(set(out['colors'])) != 1:
                all_uniform_out = False
                break
            if inp['colors'] == out['colors']:  # identity — not a transform
                all_uniform_out = False
                break
        if all_uniform_out:
            # Use the output color from first pair
            fill_color = pairs[0][1]['colors'][0]
            if verbose:
                print(f"  [algo] Rule: uniform fill with color {fill_color}")
            return [fill_color] * n_cells

    # Rule 2: Spatial transforms — check BEFORE color swap (more specific)
    #          Rotation/mirror preserves spatial structure, color swap doesn't
    if len(examples) >= 2 and len(examples) % 2 == 0:
        pairs = [(examples[i], examples[i+1]) for i in range(0, len(examples), 2)]
        for rot_name, rot_fn in [
            ("rot90", lambda cs, r, c: [cs[(r-1-j)*c + i] for i in range(c) for j in range(r)] if r == c and r*c == len(cs) else None),
            ("rot180", lambda cs, r, c: list(reversed(cs))),
            ("mirror_h", lambda cs, r, c: [cs[row*c + (c-1-col)] for row in range(r) for col in range(c)] if r*c == len(cs) else None),
            ("mirror_v", lambda cs, r, c: [cs[(r-1-row)*c + col] for row in range(r) for col in range(c)] if r*c == len(cs) else None),
        ]:
            matches = True
            for inp, out in pairs:
                if inp['rows'] != out['rows'] or inp['cols'] != out['cols']:
                    matches = False
                    break
                expected = rot_fn(inp['colors'], inp['rows'], inp['cols'])
                if expected is None or expected != out['colors']:
                    matches = False
                    break
            if matches:
                result = rot_fn(current, t_rows, t_cols)
                if result and result != current:
                    if verbose:
                        print(f"  [algo] Rule: {rot_name}")
                    return result

    # Rule 5: Transpose — if output = transposed input
    if len(examples) >= 2 and len(examples) % 2 == 0:
        pairs = [(examples[i], examples[i+1]) for i in range(0, len(examples), 2)]
        all_transpose = True
        for inp, out in pairs:
            if inp['rows'] != out['cols'] or inp['cols'] != out['rows']:
                all_transpose = False
                break
            expected = [inp['colors'][col * inp['cols'] + row]
                       for row in range(inp['cols']) for col in range(inp['rows'])]
            if expected != out['colors']:
                all_transpose = False
                break
        if all_transpose and t_rows == t_cols:  # only if target is square (output dims preserved)
            result = [current[col * t_cols + row] for row in range(t_cols) for col in range(t_rows)]
            if result != current:
                if verbose:
                    print(f"  [algo] Rule: transpose")
                return result

    # Rule 6: Color swap — consistent mapping between input and output
    if len(examples) >= 2 and len(examples) % 2 == 0:
        pairs = [(examples[i], examples[i+1]) for i in range(0, len(examples), 2)]
        color_maps = []
        for inp, out in pairs:
            if inp['rows'] == out['rows'] and inp['cols'] == out['cols']:
                cmap = {}
                consistent = True
                for ic, oc in zip(inp['colors'], out['colors']):
                    if ic in cmap:
                        if cmap[ic] != oc:
                            consistent = False
                            break
                    else:
                        cmap[ic] = oc
                if consistent:
                    color_maps.append(cmap)
        if color_maps and all(m == color_maps[0] for m in color_maps):
            cmap = color_maps[0]
            # Only use color swap if it's NOT an identity (would be caught by spatial transforms above)
            if any(k != v for k, v in cmap.items()):
                if verbose:
                    print(f"  [algo] Rule: color swap {cmap}")
                result = [cmap.get(c, c) for c in current]
                if result != current:
                    return result

    # Rule 7: Complement — if exactly 2 colors, flip them
    all_colors = set()
    for eg in examples:
        all_colors.update(eg['colors'])
    all_colors.update(current)
    if len(all_colors) == 2:
        c1, c2 = sorted(all_colors)
        complement = [c2 if c == c1 else c1 for c in current]
        if complement != current:
            if verbose:
                print(f"  [algo] Rule: complement {c1}↔{c2}")
            return complement

    # Rule 8: Most frequent color fill — if all examples' outputs are filled with their most common color
    if len(examples) >= 2 and len(examples) % 2 == 0:
        pairs = [(examples[i], examples[i+1]) for i in range(0, len(examples), 2)]
        fill_matches = True
        for inp, out in pairs:
            from collections import Counter
            most_common = Counter(inp['colors']).most_common(1)[0][0]
            if out['colors'] != [most_common] * len(out['colors']):
                fill_matches = False
                break
        if fill_matches:
            from collections import Counter
            fill_color = Counter(current).most_common(1)[0][0]
            if verbose:
                print(f"  [algo] Rule: fill with most common color ({fill_color})")
            return [fill_color] * n_cells

    # Rule 9: Gravity — colored cells drop to bottom, background fills top
    if len(examples) >= 2 and len(examples) % 2 == 0:
        pairs = [(examples[i], examples[i+1]) for i in range(0, len(examples), 2)]
        # Detect background color (most common in inputs)
        from collections import Counter
        all_input_colors = []
        for inp, _ in pairs:
            all_input_colors.extend(inp['colors'])
        bg = Counter(all_input_colors).most_common(1)[0][0]

        gravity_match = True
        for inp, out in pairs:
            if inp['rows'] != out['rows'] or inp['cols'] != out['cols']:
                gravity_match = False
                break
            r, c = inp['rows'], inp['cols']
            # Apply gravity: for each column, push non-bg cells to bottom
            expected = [bg] * (r * c)
            for col in range(c):
                non_bg = [inp['colors'][row * c + col] for row in range(r) if inp['colors'][row * c + col] != bg]
                for i, val in enumerate(non_bg):
                    expected[(r - len(non_bg) + i) * c + col] = val
            if expected != out['colors']:
                gravity_match = False
                break
        if gravity_match:
            r, c = t_rows, t_cols
            result = [bg] * (r * c)
            for col in range(c):
                non_bg = [current[row * c + col] for row in range(r) if current[row * c + col] != bg]
                for i, val in enumerate(non_bg):
                    result[(r - len(non_bg) + i) * c + col] = val
            if result != current:
                if verbose:
                    print(f"  [algo] Rule: gravity (bg={bg})")
                return result

    # Rule 10: Upscale — output is NxN repetition of each input cell
    if len(examples) >= 2 and len(examples) % 2 == 0:
        pairs = [(examples[i], examples[i+1]) for i in range(0, len(examples), 2)]
        for scale in [2, 3, 4]:
            scale_match = True
            for inp, out in pairs:
                if out['rows'] != inp['rows'] * scale or out['cols'] != inp['cols'] * scale:
                    scale_match = False
                    break
                expected = []
                for row in range(inp['rows']):
                    row_data = inp['colors'][row * inp['cols']:(row + 1) * inp['cols']]
                    expanded_row = []
                    for c in row_data:
                        expanded_row.extend([c] * scale)
                    for _ in range(scale):
                        expected.extend(expanded_row)
                if expected != out['colors']:
                    scale_match = False
                    break
            if scale_match and t_rows == t_cols // scale * scale:  # dimensions must be compatible
                result = []
                for row in range(t_rows):
                    row_data = current[row * t_cols:(row + 1) * t_cols]
                    expanded_row = []
                    for c in row_data:
                        expanded_row.extend([c] * scale)
                    for _ in range(scale):
                        result.extend(expanded_row)
                if result != current and len(result) > 0:
                    if verbose:
                        print(f"  [algo] Rule: upscale {scale}x")
                    return result

    # Rule 11: Downscale — output is input shrunk by NxN blocks (majority color)
    if len(examples) >= 2 and len(examples) % 2 == 0:
        pairs = [(examples[i], examples[i+1]) for i in range(0, len(examples), 2)]
        from collections import Counter
        for scale in [2, 3, 4]:
            down_match = True
            for inp, out in pairs:
                if inp['rows'] != out['rows'] * scale or inp['cols'] != out['cols'] * scale:
                    down_match = False
                    break
                expected = []
                for br in range(out['rows']):
                    for bc in range(out['cols']):
                        block = []
                        for dy in range(scale):
                            for dx in range(scale):
                                block.append(inp['colors'][(br * scale + dy) * inp['cols'] + bc * scale + dx])
                        expected.append(Counter(block).most_common(1)[0][0])
                if expected != out['colors']:
                    down_match = False
                    break
            if down_match and t_rows % scale == 0 and t_cols % scale == 0:
                result = []
                or_rows, or_cols = t_rows // scale, t_cols // scale
                for br in range(or_rows):
                    for bc in range(or_cols):
                        block = []
                        for dy in range(scale):
                            for dx in range(scale):
                                block.append(current[(br * scale + dy) * t_cols + bc * scale + dx])
                        result.append(Counter(block).most_common(1)[0][0])
                if result != current and len(result) > 0:
                    if verbose:
                        print(f"  [algo] Rule: downscale {scale}x (majority)")
                    return result

    # Rule 12: Flood fill enclosed regions — fill bg-colored regions enclosed by non-bg with surrounding color
    if len(examples) >= 2 and len(examples) % 2 == 0:
        pairs = [(examples[i], examples[i+1]) for i in range(0, len(examples), 2)]
        from collections import Counter
        # Try multiple bg candidates: 0 first (ARC convention), then most common
        all_input_colors = []
        for inp, _ in pairs:
            all_input_colors.extend(inp['colors'])
        bg_candidates = [0] if 0 in all_input_colors else []
        mc = Counter(all_input_colors).most_common(1)[0][0]
        if mc not in bg_candidates:
            bg_candidates.append(mc)
        # Also try least common (the enclosed color might be rare)
        lc = Counter(all_input_colors).most_common()[-1][0]
        if lc not in bg_candidates:
            bg_candidates.append(lc)

        def _flood_fill_enclosed(colors, rows, cols, bg_color):
            """Fill enclosed bg regions with the color that surrounds them."""
            result = list(colors)
            visited = [False] * len(result)

            def _flood(start, r, c):
                """BFS flood from start, return (cells, border_colors, touches_edge)."""
                from collections import deque
                q = deque([start])
                cells = []
                borders = []
                touches_edge = False
                visited[start] = True
                while q:
                    pos = q.popleft()
                    cells.append(pos)
                    pr, pc = pos // c, pos % c
                    if pr == 0 or pr == r - 1 or pc == 0 or pc == c - 1:
                        touches_edge = True
                    for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                        nr, nc = pr + dr, pc + dc
                        if 0 <= nr < r and 0 <= nc < c:
                            npos = nr * c + nc
                            if not visited[npos] and result[npos] == bg_color:
                                visited[npos] = True
                                q.append(npos)
                            elif result[npos] != bg_color:
                                borders.append(result[npos])
                return cells, borders, touches_edge

            for i in range(len(result)):
                if result[i] == bg_color and not visited[i]:
                    cells, borders, touches_edge = _flood(i, rows, cols)
                    if not touches_edge and borders:
                        fill_color = Counter(borders).most_common(1)[0][0]
                        for c_idx in cells:
                            result[c_idx] = fill_color
            return result

        for bg in bg_candidates:
            fill_match = True
            for inp, out in pairs:
                if inp['rows'] != out['rows'] or inp['cols'] != out['cols']:
                    fill_match = False
                    break
                expected = _flood_fill_enclosed(inp['colors'], inp['rows'], inp['cols'], bg)
                if expected != out['colors']:
                    fill_match = False
                    break
            if fill_match:
                result = _flood_fill_enclosed(current, t_rows, t_cols, bg)
                if result != current:
                    if verbose:
                        print(f"  [algo] Rule: flood fill enclosed regions (bg={bg})")
                    return result

    # Rule 13: Border extraction — output is just the border pixels of shapes
    if len(examples) >= 2 and len(examples) % 2 == 0:
        pairs = [(examples[i], examples[i+1]) for i in range(0, len(examples), 2)]
        from collections import Counter
        bg = Counter(sum((e['colors'] for e in examples), [])).most_common(1)[0][0]

        def _extract_border(colors, rows, cols, bg_color):
            result = [bg_color] * len(colors)
            for i in range(len(colors)):
                if colors[i] != bg_color:
                    r, c = i // cols, i % cols
                    is_border = False
                    for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                        nr, nc = r + dr, c + dc
                        if nr < 0 or nr >= rows or nc < 0 or nc >= cols:
                            is_border = True
                        elif colors[nr * cols + nc] == bg_color:
                            is_border = True
                    if is_border:
                        result[i] = colors[i]
            return result

        border_match = True
        for inp, out in pairs:
            if inp['rows'] != out['rows'] or inp['cols'] != out['cols']:
                border_match = False
                break
            if _extract_border(inp['colors'], inp['rows'], inp['cols'], bg) != out['colors']:
                border_match = False
                break
        if border_match:
            result = _extract_border(current, t_rows, t_cols, bg)
            if result != current:
                if verbose:
                    print(f"  [algo] Rule: border extraction (bg={bg})")
                return result

    # Rule 14: Crop/extract — output is a subgrid of the input
    if len(examples) >= 2 and len(examples) % 2 == 0:
        pairs = [(examples[i], examples[i+1]) for i in range(0, len(examples), 2)]
        # Check if output appears as a contiguous subgrid in input
        crop_offsets = []  # (row_off, col_off) for each pair
        crop_match = True
        for inp, out in pairs:
            if out['rows'] > inp['rows'] or out['cols'] > inp['cols']:
                crop_match = False
                break
            found = False
            for r_off in range(inp['rows'] - out['rows'] + 1):
                for c_off in range(inp['cols'] - out['cols'] + 1):
                    match = True
                    for dr in range(out['rows']):
                        for dc in range(out['cols']):
                            if inp['colors'][(r_off + dr) * inp['cols'] + c_off + dc] != out['colors'][dr * out['cols'] + dc]:
                                match = False
                                break
                        if not match:
                            break
                    if match:
                        crop_offsets.append((r_off, c_off))
                        found = True
                        break
                if found:
                    break
            if not found:
                crop_match = False
                break
        # Check if all pairs use the same offset (consistent rule)
        if crop_match and crop_offsets and all(o == crop_offsets[0] for o in crop_offsets):
            r_off, c_off = crop_offsets[0]
            out_r, out_c = pairs[0][1]['rows'], pairs[0][1]['cols']
            if r_off + out_r <= t_rows and c_off + out_c <= t_cols:
                result = []
                for dr in range(out_r):
                    for dc in range(out_c):
                        result.append(current[(r_off + dr) * t_cols + c_off + dc])
                if result != current:
                    if verbose:
                        print(f"  [algo] Rule: crop at ({r_off},{c_off}) size {out_r}x{out_c}")
                    return result

    # Rule 15: Tile/repeat — output is input tiled NxM times
    if len(examples) >= 2 and len(examples) % 2 == 0:
        pairs = [(examples[i], examples[i+1]) for i in range(0, len(examples), 2)]
        for tile_r in [2, 3, 4]:
            for tile_c in [2, 3, 4]:
                tile_match = True
                for inp, out in pairs:
                    if out['rows'] != inp['rows'] * tile_r or out['cols'] != inp['cols'] * tile_c:
                        tile_match = False
                        break
                    expected = []
                    for tr in range(tile_r):
                        for row in range(inp['rows']):
                            for tc in range(tile_c):
                                expected.extend(inp['colors'][row * inp['cols']:(row + 1) * inp['cols']])
                    if expected != out['colors']:
                        tile_match = False
                        break
                if tile_match:
                    result = []
                    for tr in range(tile_r):
                        for row in range(t_rows):
                            for tc in range(tile_c):
                                result.extend(current[row * t_cols:(row + 1) * t_cols])
                    if result != current and len(result) > 0:
                        if verbose:
                            print(f"  [algo] Rule: tile {tile_r}x{tile_c}")
                        return result

    # Rule 16: Composite — try color swap AFTER spatial transform (catches combined rules)
    if len(examples) >= 2 and len(examples) % 2 == 0:
        pairs = [(examples[i], examples[i+1]) for i in range(0, len(examples), 2)]
        for rot_name, rot_fn in [
            ("rot90", lambda cs, r, c: [cs[(r-1-j)*c + i] for i in range(c) for j in range(r)] if r == c and r*c == len(cs) else None),
            ("rot180", lambda cs, r, c: list(reversed(cs))),
            ("mirror_h", lambda cs, r, c: [cs[row*c + (c-1-col)] for row in range(r) for col in range(c)] if r*c == len(cs) else None),
            ("mirror_v", lambda cs, r, c: [cs[(r-1-row)*c + col] for row in range(r) for col in range(c)] if r*c == len(cs) else None),
        ]:
            # Apply spatial transform to input, then check if color swap matches output
            color_maps = []
            composite_match = True
            for inp, out in pairs:
                if inp['rows'] != out['rows'] or inp['cols'] != out['cols']:
                    composite_match = False
                    break
                rotated = rot_fn(inp['colors'], inp['rows'], inp['cols'])
                if rotated is None:
                    composite_match = False
                    break
                cmap = {}
                consistent = True
                for rc, oc in zip(rotated, out['colors']):
                    if rc in cmap:
                        if cmap[rc] != oc:
                            consistent = False
                            break
                    else:
                        cmap[rc] = oc
                if not consistent:
                    composite_match = False
                    break
                color_maps.append(cmap)
            if composite_match and color_maps and all(m == color_maps[0] for m in color_maps):
                cmap = color_maps[0]
                if any(k != v for k, v in cmap.items()):  # not identity
                    rotated = rot_fn(current, t_rows, t_cols)
                    if rotated is not None:
                        result = [cmap.get(c, c) for c in rotated]
                        if result != current:
                            if verbose:
                                print(f"  [algo] Rule: {rot_name} + color swap {cmap}")
                            return result

    if verbose:
        print("  [algo] No simple rule detected")
    return None


def _researcher_solve(env, mind, extract_frame, base_frame, action_info, obs, verbose=False):
    """Researcher-mode solver: systematic probing + one LLM call for rule inference.

    Instead of reactive "see → act" loops, this:
    1. EYES: extract all grids as color matrices (zero LLM)
    2. PROBE: click each cell, record effects (zero LLM)
    3. ONE LLM CALL: "what's the transformation rule?" → get rule as structured output
    4. HANDS: execute the rule programmatically (zero LLM)

    Returns list of (action_id, data) tuples, or None if not applicable.
    """
    import copy as _cp

    grids = getattr(mind, '_detected_grids', None)
    click_idx = getattr(mind, '_click_action_idx', None)
    cycle = getattr(mind, '_color_cycle', None)

    if not grids:
        # Fallback: no grids detected (e.g., L1+ where layout changed).
        # Try GF(2) exhaustive by finding colored blocks directly from frame.
        if click_idx is not None and cycle and cycle['period'] == 2:
            bf = base_frame[0] if base_frame.ndim == 3 else base_frame
            h, w = bf.shape
            unique, counts = np.unique(bf, return_counts=True)
            bg = unique[np.argmax(counts)]
            from scipy import ndimage
            non_bg = bf != bg
            if h > 2:
                non_bg[h-2:, :] = False
                non_bg[:1, :] = False
            labeled, n_regions = ndimage.label(non_bg)
            cell_centers = []
            for rid in range(1, n_regions + 1):
                ys_r, xs_r = np.where(labeled == rid)
                sz = len(ys_r)
                if sz < 4 or sz > 200:
                    continue
                cell_centers.append((int(np.mean(xs_r)), int(np.mean(ys_r)), sz))
            # Only try exhaustive if <=9 cells (512 combos max)
            if cell_centers and len(cell_centers) <= 9:
                if verbose:
                    print(f"  [researcher] Fallback: {len(cell_centers)} cell candidates, trying GF(2) exhaustive")
                n_cells = len(cell_centers)
                click_act = obs.available_actions[click_idx]
                for mask_bits in range(1, 2**n_cells):
                    test_env = _cp.deepcopy(env)
                    last_obs = None
                    for j in range(n_cells):
                        if (mask_bits >> j) & 1:
                            cx_c, cy_c, _ = cell_centers[j]
                            last_obs = test_env.step(click_act, {'x': cx_c, 'y': cy_c})
                    if last_obs and hasattr(last_obs, 'levels_completed') and last_obs.levels_completed > obs.levels_completed:
                        clicks = []
                        for j in range(n_cells):
                            if (mask_bits >> j) & 1:
                                cx_c, cy_c, _ = cell_centers[j]
                                clicks.append((click_idx, {'x': cx_c, 'y': cy_c}))
                        if verbose:
                            print(f"  [researcher] Fallback GF(2): {len(clicks)} clicks (mask={mask_bits})")
                        return clicks
                if verbose:
                    print(f"  [researcher] Fallback: no solution found ({2**n_cells - 1} combos tried)")
            elif cell_centers and len(cell_centers) <= 25:
                # Use GF(2) Gaussian elimination for larger grids (up to 25 cells)
                if verbose:
                    print(f"  [researcher] Fallback: {len(cell_centers)} cells, trying GF(2) Gaussian solver")
                n_cells = len(cell_centers)
                click_act = obs.available_actions[click_idx]
                # Build toggle matrix: A[i,j] = does clicking cell j toggle cell i?
                bf = base_frame[0] if base_frame.ndim == 3 else base_frame
                cell_colors_before = []
                for cx_c, cy_c, _ in cell_centers:
                    cell_colors_before.append(int(bf[cy_c, cx_c]) if bf.ndim == 2 else int(bf[0, cy_c, cx_c]))
                A = np.zeros((n_cells, n_cells), dtype=int)
                for j in range(n_cells):
                    test_env = _cp.deepcopy(env)
                    cx_c, cy_c, _ = cell_centers[j]
                    test_obs = test_env.step(click_act, {'x': cx_c, 'y': cy_c})
                    tf = extract_frame(test_obs)
                    for i in range(n_cells):
                        cx_i, cy_i, _ = cell_centers[i]
                        c_after = int(tf[cy_i, cx_i]) if tf.ndim == 2 else int(tf[0, cy_i, cx_i])
                        if c_after != cell_colors_before[i]:
                            A[i, j] = 1
                # Target: toggle all non-target cells (assume target = most common color)
                unique_cc, counts_cc = np.unique(cell_colors_before, return_counts=True)
                target_color = unique_cc[np.argmax(counts_cc)]
                b = np.array([0 if c == target_color else 1 for c in cell_colors_before], dtype=int)
                if np.any(b):  # there's something to solve
                    # Try both target hypotheses — verify each via deepcopy
                    for hyp_label, b_vec in [("majority", b),
                                              ("minority", np.array([1 if c == target_color else 0 for c in cell_colors_before], dtype=int))]:
                        if not np.any(b_vec):
                            continue
                        solution = _gf2_solve(A, b_vec)
                        if solution is not None and np.any(solution):
                            clicks = []
                            for j in range(n_cells):
                                if solution[j]:
                                    cx_c, cy_c, _ = cell_centers[j]
                                    clicks.append((click_idx, {'x': cx_c, 'y': cy_c}))
                            # Verify via deepcopy — did the level actually advance?
                            test_env = _cp.deepcopy(env)
                            for _, data in clicks:
                                test_obs_v = test_env.step(click_act, data)
                            gs = str(getattr(test_obs_v, 'game_state', ''))
                            lc = getattr(test_obs_v, 'levels_completed', 0)
                            prev_lc = getattr(obs, 'levels_completed', 0)
                            if lc > prev_lc or 'WIN' in gs:
                                if verbose:
                                    print(f"  [researcher] GF(2) Gaussian ({hyp_label}): {len(clicks)} clicks — VERIFIED level advance!")
                                return clicks
                            elif verbose:
                                print(f"  [researcher] GF(2) Gaussian ({hyp_label}): {len(clicks)} clicks — no level advance, trying next hypothesis")
                    if verbose:
                        print(f"  [researcher] GF(2) Gaussian: no verified solution found")
            elif cell_centers and verbose:
                print(f"  [researcher] Fallback: {len(cell_centers)} cells (>{25}), skipping")
        return None

    # For non-click games (deduction puzzles), use sequence-based researcher
    no_click = click_idx is None

    # --- EYES: build color matrices for all grids ---
    grid_matrices = []
    for g in grids:
        if not g.get('cells'):
            continue
        n_rows = len(g.get('rows', []))
        n_cols = len(g.get('cols', []))
        if n_rows == 0 or n_cols == 0:
            continue
        matrix = []
        rows_spans = g.get('rows', [])
        cols_spans = g.get('cols', [])
        for cell in g['cells']:
            ri, ci = cell['row'], cell['col']
            # Use cell span edges (corners) to get BACKGROUND color, not sprite
            ry0, ry1 = rows_spans[ri] if ri < len(rows_spans) else (cell['center_y']-2, cell['center_y']+2)
            cx0, cx1 = cols_spans[ci] if ci < len(cols_spans) else (cell['center_x']-2, cell['center_x']+2)
            # Sample corners + edges — these are background, not sprite
            samples = []
            for sy, sx in [(ry0, cx0), (ry0, cx1), (ry1, cx0), (ry1, cx1),
                           (ry0, (cx0+cx1)//2), (ry1, (cx0+cx1)//2),
                           ((ry0+ry1)//2, cx0), ((ry0+ry1)//2, cx1)]:
                if 0 <= sy < base_frame.shape[-2] and 0 <= sx < base_frame.shape[-1]:
                    pix = int(base_frame[sy, sx]) if base_frame.ndim == 2 else int(base_frame[0, sy, sx])
                    samples.append(pix)
            from collections import Counter
            # Filter to only cycle colors if we know them
            cycle = getattr(mind, '_color_cycle', None)
            if cycle:
                cycle_colors = set(cycle['colors'][:cycle['period']])
                cycle_samples = [s for s in samples if s in cycle_colors]
                if cycle_samples:
                    samples = cycle_samples
            color = Counter(samples).most_common(1)[0][0] if samples else 0
            matrix.append(color)
        grid_matrices.append({
            'rows': n_rows, 'cols': n_cols, 'colors': matrix,
            'is_target': g.get('is_target', False),
            'cells': g['cells'],
        })

    if len(grid_matrices) < 2:
        return None  # need at least example + target

    # --- PROBE: if click action exists, probe target grid cells ---
    target = next((g for g in grid_matrices if g['is_target']), grid_matrices[-1])
    examples = [g for g in grid_matrices if not g['is_target']]

    toggle_effects = {}
    if not no_click:
        click_act = obs.available_actions[click_idx]
        for i, cell in enumerate(target['cells']):
            probe_env = _cp.deepcopy(env)
            px, py = cell['center_x'], cell['center_y']
            probe_obs = probe_env.step(click_act, {"x": px, "y": py})
            probe_frame = extract_frame(probe_obs)

            changed_cells = []
            for j, other_cell in enumerate(target['cells']):
                ox, oy = other_cell['center_x'], other_cell['center_y']
                before = int(base_frame[oy, ox]) if base_frame.ndim == 2 else int(base_frame[0, oy, ox])
                after = int(probe_frame[oy, ox]) if probe_frame.ndim == 2 else int(probe_frame[0, oy, ox])
                if before != after:
                    changed_cells.append(j)

            toggle_effects[i] = changed_cells
        if verbose:
            has_cross = any(len(v) > 1 for v in toggle_effects.values())
            print(f"  [researcher] Toggle effects: {toggle_effects}")
            print(f"  [researcher] has_cross_effects={has_cross}")
    elif verbose:
        print("  [researcher] No click action — skipping cell probing")

    # --- ONE LLM CALL: ask for the transformation rule ---
    # Build a structured prompt with color matrices
    prompt_parts = []
    prompt_parts.append("You are solving an ARC-AGI puzzle. Analyze these grids and determine the transformation rule.")
    prompt_parts.append("")

    for idx, eg in enumerate(examples):
        rows, cols = eg['rows'], eg['cols']
        prompt_parts.append(f"Example Grid {idx+1} ({rows}x{cols}):")
        for r in range(rows):
            row_colors = eg['colors'][r*cols:(r+1)*cols]
            prompt_parts.append(f"  {row_colors}")
        prompt_parts.append("")

    t_rows, t_cols = target['rows'], target['cols']
    prompt_parts.append(f"Target Grid ({t_rows}x{t_cols}) — CURRENT state (needs to be transformed):")
    for r in range(t_rows):
        row_colors = target['colors'][r*t_cols:(r+1)*t_cols]
        prompt_parts.append(f"  {row_colors}")
    prompt_parts.append("")

    if cycle:
        prompt_parts.append(f"Click mechanics: clicking a cell cycles its color. Period={cycle['period']}, cycle={cycle['colors']}")
    if toggle_effects:
        prompt_parts.append("Click effects (which cells change when you click cell N):")
        for cell_i, affected in toggle_effects.items():
            if affected:
                r, c = cell_i // t_cols, cell_i % t_cols
                aff_str = ", ".join(f"({a//t_cols},{a%t_cols})" for a in affected)
                prompt_parts.append(f"  Click ({r},{c}) → changes cells: {aff_str}")
    prompt_parts.append("")

    if no_click:
        # Non-click game: ask for an action SEQUENCE instead of target colors
        n_actions = len(obs.available_actions)
        prompt_parts.append(f"Available actions: {n_actions} actions numbered 0 to {n_actions - 1}")
        # Include action info if available
        for k, v in action_info.items():
            if isinstance(k, int):
                prompt_parts.append(f"  Action {k}: {v}")
        prompt_parts.append("")
        prompt_parts.append("An image of the current game frame is attached. Use it to understand the visual layout.")
        prompt_parts.append("")
        prompt_parts.append("TASK: Analyze the relationship between the grids shown. What pattern or rule connects them?")
        prompt_parts.append("Then determine the sequence of actions to apply to transform the puzzle to its goal state.")
        prompt_parts.append(f"Reply with a sequence of action IDs (0 to {n_actions - 1}).")
        prompt_parts.append("Format: SEQUENCE: [a0, a1, a2, ...]")
    else:
        prompt_parts.append("TASK: What is the TARGET state for each cell in the target grid?")
        prompt_parts.append(f"Reply with EXACTLY {t_rows * t_cols} integers — the target color for each cell, row by row.")
        prompt_parts.append("Format: TARGET: [c0, c1, c2, ...]")

    prompt_text = "\n".join(prompt_parts)

    # --- BUDGET OPTIMIZATION: Try algorithmic transforms FIRST (zero API cost) ---
    # If a simple rule matches all examples, skip the LLM call entirely
    # but still let the response flow through TARGET parsing + click computation
    algo_response = None
    if not no_click:
        algo_target = _infer_transformation_algorithmic(examples, target, t_rows, t_cols, verbose)
        if algo_target is not None:
            if verbose:
                print(f"  [researcher] Algorithmic match — skipping LLM call (saved ~$0.01)")
            algo_response = f"TARGET: [{','.join(str(x) for x in algo_target)}]"

    if algo_response is not None:
        response = algo_response
    else:
        if verbose:
            print(f"  [researcher] Asking LLM for transformation rule...")
            print(f"  [researcher] Prompt: {len(prompt_text)} chars, {len(examples)} examples, {t_rows}x{t_cols} target")

        # Make the ONE call — researcher needs REASONING model (Sonnet), not cheap (Haiku)
        # Include frame image for vision models — helps LLM see the actual grid layout
        _researcher_images = []
        if base_frame is not None:
            try:
                _researcher_images.append(frame_to_b64(base_frame, scale=6))
            except Exception:
                pass
        response = call_llm(
            messages=[{"role": "user", "content": prompt_text}],
            images=_researcher_images if _researcher_images else None,
            max_tokens=500,
            temperature=0.0,
            fast=False,  # Always use reasoning model for target prediction
        )
        mind.total_llm_calls += 1

        if not response:
            if verbose:
                print("  [researcher] LLM unavailable — trying observation-based GF solver")
            # --- FALLBACK: observation-based GF(2) solver without LLM target prediction ---
            # For toggle puzzles, try all possible target states and find one that works
            if toggle_effects and cycle and cycle['period'] >= 2:
                import copy as _cp2
                period = cycle['period']
                cycle_colors = cycle['colors'][:period]
                current = target['colors']  # current color state of each cell
                n_cells = len(current)

                # Build toggle matrix from probed effects
                toggle_matrix = np.zeros((n_cells, n_cells), dtype=int)
                for j, affected in toggle_effects.items():
                    for i in affected:
                        toggle_matrix[i][j] = 1

                if period == 2 and n_cells <= 16:
                    # Binary: try all 2^N possible targets
                    best_clicks = None
                    for mask in range(1, 2**n_cells):
                        b = np.array([(mask >> i) & 1 for i in range(n_cells)], dtype=int)
                        sol = _gf2_solve(toggle_matrix, b)
                        if sol is not None and np.sum(sol) > 0:
                            # Validate: execute in deepcopy and check if level advances
                            test_env = _cp2.deepcopy(env)
                            click_act = obs.available_actions[click_idx]
                            last_obs = None
                            for j in range(n_cells):
                                if int(sol[j]) > 0:
                                    cell = target['cells'][j]
                                    last_obs = test_env.step(click_act, {'x': cell['center_x'], 'y': cell['center_y']})
                            # Check if the last click advanced the level
                            if last_obs and hasattr(last_obs, 'levels_completed') and last_obs.levels_completed > obs.levels_completed:
                                clicks = []
                                for j in range(n_cells):
                                    for _ in range(int(sol[j])):
                                        cell = target['cells'][j]
                                        clicks.append((click_idx, {'x': cell['center_x'], 'y': cell['center_y']}))
                                if verbose:
                                    print(f"  [researcher] GF(2) exhaustive: found solution with {len(clicks)} clicks (mask={mask})")
                                return clicks
                    if verbose:
                        print(f"  [researcher] GF(2) exhaustive: no solution found in {2**n_cells - 1} targets")
                elif period >= 2 and n_cells <= 8:
                    # Multi-color: try common patterns (all same color)
                    for target_color in cycle_colors:
                        target_state = [target_color] * n_cells
                        b = np.zeros(n_cells, dtype=int)
                        for i in range(n_cells):
                            if current[i] != target_state[i] and current[i] in cycle_colors:
                                ci = cycle_colors.index(current[i])
                                ti = cycle_colors.index(target_state[i])
                                b[i] = (ti - ci) % period
                        if np.sum(b) > 0:
                            sol = _gf2_solve(toggle_matrix, b) if period == 2 else _gfk_solve(toggle_matrix, b, period)
                            if sol is not None:
                                test_env = _cp2.deepcopy(env)
                                click_act = obs.available_actions[click_idx]
                                last_obs = None
                                for j in range(n_cells):
                                    for _ in range(int(sol[j])):
                                        cell = target['cells'][j]
                                        last_obs = test_env.step(click_act, {'x': cell['center_x'], 'y': cell['center_y']})
                                if last_obs and hasattr(last_obs, 'levels_completed') and last_obs.levels_completed > obs.levels_completed:
                                    clicks = []
                                    for j in range(n_cells):
                                        for _ in range(int(sol[j])):
                                            cell = target['cells'][j]
                                            clicks.append((click_idx, {'x': cell['center_x'], 'y': cell['center_y']}))
                                    if verbose:
                                        print(f"  [researcher] GF(k) exhaustive: {len(clicks)} clicks (target={target_color})")
                                    return clicks

                if verbose:
                    print("  [researcher] GF exhaustive solver: no solution found")
            return None

    if verbose:
        print(f"  [researcher] LLM response: {response[:200]}")

    import re

    # --- Non-click path: parse SEQUENCE response and return directly ---
    if no_click:
        seq_match = re.search(r'SEQUENCE:\s*\[([^\]]+)\]', response)
        if seq_match:
            action_ids = [int(x.strip()) for x in seq_match.group(1).split(',')]
        else:
            # Try to find any list of numbers
            nums = re.findall(r'\b(\d+)\b', response)
            action_ids = [int(n) for n in nums] if nums else []

        n_avail = len(obs.available_actions)
        # Filter to valid action IDs
        action_ids = [a for a in action_ids if 0 <= a < n_avail]

        if not action_ids:
            if verbose:
                print("  [researcher] No valid action sequence parsed from response")
            return None

        if verbose:
            print(f"  [researcher] Parsed action sequence: {action_ids[:20]}{'...' if len(action_ids) > 20 else ''}")

        # Return as list of (action_id, data) tuples — no position data for non-click
        return [(aid, {}) for aid in action_ids]

    # --- Click path: parse TARGET state ---
    target_match = re.search(r'TARGET:\s*\[([^\]]+)\]', response)
    if not target_match:
        # Try to find any list of numbers
        nums = re.findall(r'\b(\d+)\b', response)
        if len(nums) >= t_rows * t_cols:
            target_state = [int(n) for n in nums[:t_rows * t_cols]]
        else:
            if verbose:
                print(f"  [researcher] Could not parse target state from response")
            return None
    else:
        target_state = [int(x.strip()) for x in target_match.group(1).split(',')]

    if len(target_state) != t_rows * t_cols:
        if verbose:
            print(f"  [researcher] Target state has {len(target_state)} values, expected {t_rows * t_cols}")
        return None

    if verbose:
        print(f"  [researcher] Target state: {target_state}")
        print(f"  [researcher] Current state: {target['colors']}")

    # Store target prediction spatially for oracle to use
    mind._researcher_target_map = {}
    mind._researcher_current_map = {}
    for i, cell in enumerate(target['cells']):
        pos = (cell['center_x'], cell['center_y'])
        mind._researcher_target_map[pos] = target_state[i]
        mind._researcher_current_map[pos] = target['colors'][i]

    # --- HANDS: compute clicks needed ---
    current = target['colors']
    n_cells = len(current)

    if not cycle or cycle['period'] == 0:
        return None

    period = cycle['period']
    cycle_colors = cycle['colors'][:period]

    # For each cell: how many clicks to get from current to target?
    # If toggle_effects show cross-cell effects, use GF(k) solver
    has_cross_effects = any(len(v) > 1 for v in toggle_effects.values())

    if has_cross_effects:
        # Build toggle matrix and solve with GF(k)
        toggle_matrix = np.zeros((n_cells, n_cells), dtype=int)
        for j, affected in toggle_effects.items():
            for i in affected:
                toggle_matrix[i][j] = 1

        # b[i] = number of clicks needed for cell i (mod period)
        b = np.zeros(n_cells, dtype=int)
        for i in range(n_cells):
            if current[i] != target_state[i]:
                # How many clicks to go from current to target in the cycle?
                if current[i] in cycle_colors and target_state[i] in cycle_colors:
                    ci = cycle_colors.index(current[i])
                    ti = cycle_colors.index(target_state[i])
                    b[i] = (ti - ci) % period
                else:
                    b[i] = 1  # best guess

        if np.sum(b) == 0:
            return []  # already at target

        if period == 2:
            solution = _gf2_solve(toggle_matrix, b)
        else:
            solution = _gfk_solve(toggle_matrix, b, period)

        if solution is None:
            if verbose:
                print("  [researcher] GF(k) solver found no solution — trying brute force")
            return None

        clicks = []
        for j in range(n_cells):
            for _ in range(int(solution[j])):
                cell = target['cells'][j]
                clicks.append((click_idx, {'x': cell['center_x'], 'y': cell['center_y']}))

        if verbose:
            print(f"  [researcher] Solution: {len(clicks)} clicks")
        return clicks

    else:
        # Simple direct toggle — click each cell the needed number of times
        clicks = []
        for i in range(n_cells):
            if current[i] != target_state[i]:
                if current[i] in cycle_colors and target_state[i] in cycle_colors:
                    ci = cycle_colors.index(current[i])
                    ti = cycle_colors.index(target_state[i])
                    n_clicks = (ti - ci) % period
                else:
                    n_clicks = 1
                cell = target['cells'][i]
                for _ in range(n_clicks):
                    clicks.append((click_idx, {'x': cell['center_x'], 'y': cell['center_y']}))

        if verbose:
            print(f"  [researcher] Solution: {len(clicks)} direct clicks")

        # Validate direct clicks via deepcopy — if they don't solve, try GF(2) exhaustive
        import copy as _cp_val
        val_env = _cp_val.deepcopy(env)
        val_obs = None
        for act_id_v, data_v in clicks:
            val_act = obs.available_actions[act_id_v]
            val_obs = val_env.step(val_act, data_v) if data_v else val_env.step(val_act)
        if val_obs and hasattr(val_obs, 'levels_completed') and val_obs.levels_completed > obs.levels_completed:
            return clicks  # direct clicks work!
        if verbose:
            print(f"  [researcher] Direct clicks didn't solve — trying GF(2) exhaustive fallback")
        # GF(2) exhaustive fallback
        if cycle and cycle['period'] == 2 and n_cells <= 16:
            for mask in range(1, 2**n_cells):
                b_ex = np.array([(mask >> i) & 1 for i in range(n_cells)], dtype=int)
                # Build toggle matrix from probed effects (even if imperfect, GF(2) may find solution)
                toggle_matrix_ex = np.zeros((n_cells, n_cells), dtype=int)
                for j_ex, affected_ex in toggle_effects.items():
                    for i_ex in affected_ex:
                        toggle_matrix_ex[i_ex][j_ex] = 1
                sol_ex = _gf2_solve(toggle_matrix_ex, b_ex)
                if sol_ex is not None and np.sum(sol_ex) > 0:
                    test_env_ex = _cp_val.deepcopy(env)
                    click_act_ex = obs.available_actions[click_idx]
                    last_obs_ex = None
                    for j_ex2 in range(n_cells):
                        if int(sol_ex[j_ex2]) > 0:
                            cell_ex = target['cells'][j_ex2]
                            last_obs_ex = test_env_ex.step(click_act_ex, {'x': cell_ex['center_x'], 'y': cell_ex['center_y']})
                    if last_obs_ex and hasattr(last_obs_ex, 'levels_completed') and last_obs_ex.levels_completed > obs.levels_completed:
                        clicks_ex = []
                        for j_ex3 in range(n_cells):
                            for _ in range(int(sol_ex[j_ex3])):
                                cell_ex = target['cells'][j_ex3]
                                clicks_ex.append((click_idx, {'x': cell_ex['center_x'], 'y': cell_ex['center_y']}))
                        if verbose:
                            print(f"  [researcher] GF(2) exhaustive fallback: {len(clicks_ex)} clicks (mask={mask})")
                        return clicks_ex
            if verbose:
                print(f"  [researcher] GF(2) exhaustive fallback: no solution in {2**n_cells - 1} targets")
        return clicks  # return original even if unvalidated


def _detect_color_cycle(env, click_action, probe_positions, extract_frame, base_frame,
                         max_clicks: int = 10):
    """Click the same cell repeatedly to discover the color toggle cycle.
    Samples MULTIPLE clickable cells and returns the longest cycle found.
    Returns {'period': K, 'colors': [c0, c1, ...], 'cell_pos': (x, y)} or None."""
    import copy as _cp

    best_result = None

    # Find up to 3 clickable cells to sample — different cells may have different cycles
    clickable_positions = []
    for px, py in probe_positions:
        test_env = _cp.deepcopy(env)
        test_obs = test_env.step(click_action, {"x": px, "y": py})
        test_frame = extract_frame(test_obs)
        if not np.array_equal(base_frame, test_frame):
            clickable_positions.append((px, py))
            if len(clickable_positions) >= 3:
                break

    if not clickable_positions:
        return None

    for px, py in clickable_positions:
        cycle_env = _cp.deepcopy(env)
        colors = []
        start_color = int(base_frame[py, px]) if base_frame.ndim == 2 else int(base_frame[0, py, px])
        colors.append(start_color)

        for click_n in range(max_clicks):
            obs = cycle_env.step(click_action, {"x": px, "y": py})
            f = extract_frame(obs)
            c = int(f[py, px]) if f.ndim == 2 else int(f[0, py, px])
            colors.append(c)

            if c == start_color and click_n > 0:
                period = click_n + 1
                result = {
                    'period': period,
                    'colors': colors[:period + 1],
                    'cell_pos': (px, py),
                }
                if best_result is None or period > best_result['period']:
                    best_result = result
                break

    if best_result:
        return best_result

    # Fallback: use first cell's sequence
    unique = list(dict.fromkeys(colors))
    return {
        'period': len(unique),
        'colors': unique,
        'cell_pos': clickable_positions[0],
    }


def _solve_lights_out(env, click_action, extract_frame, base_frame, grid_cells,
                       cycle_period: int = 2, verbose: bool = False):
    """Solve a lights-out puzzle programmatically using GF(2) Gaussian elimination.

    Args:
        env: game environment (will be deepcopied for probing)
        click_action: the action object for clicking
        extract_frame: function to extract 2D frame from obs
        base_frame: current frame state
        grid_cells: list of {'center_x': int, 'center_y': int, 'fill_color': int, ...}
        cycle_period: number of distinct states per cell (2 for binary toggle)
        verbose: print debug info

    Returns:
        list of (x, y) click coordinates to solve the puzzle, or None if unsolvable.
        For period > 2, each (x, y) may appear multiple times.
    """
    import copy as _cp

    n = len(grid_cells)
    if n == 0 or n > 64:  # sanity check
        return None

    # Step 1: Get cell positions and current colors
    # Use the cell's 'fill' if it's in the cycle colors, otherwise sample the background
    cell_positions = [(c['center_x'], c['center_y']) for c in grid_cells]
    current_colors = []
    for c in grid_cells:
        cx, cy = c['center_x'], c['center_y']
        # Sample multiple positions in the cell to find the actual state color
        # Center may have a sprite overlay — try corners of the cell region first
        candidates = []
        for dx, dy in [(0, 0), (-2, -2), (2, -2), (-2, 2), (2, 2), (-1, -1), (1, 1)]:
            sx, sy = cx + dx, cy + dy
            if 0 <= sy < base_frame.shape[-2] and 0 <= sx < base_frame.shape[-1]:
                pix = int(base_frame[sy, sx]) if base_frame.ndim == 2 else int(base_frame[0, sy, sx])
                candidates.append(pix)
        # Pick the most common candidate that's likely a state color (8 or 9 for cycle games)
        if candidates:
            from collections import Counter
            cc = Counter(candidates)
            color = cc.most_common(1)[0][0]
        else:
            color = int(base_frame[cy, cx]) if base_frame.ndim == 2 else int(base_frame[0, cy, cx])
        current_colors.append(color)

    if verbose:
        print(f"  [lights-out] {n} cells, colors: {current_colors}")

    # Step 2: Probe each cell to discover its toggle mask
    # toggle_matrix[i][j] = 1 if clicking cell j changes cell i
    toggle_matrix = np.zeros((n, n), dtype=int)

    for j, (px, py) in enumerate(cell_positions):
        probe_env = _cp.deepcopy(env)
        probe_obs = probe_env.step(click_action, {"x": px, "y": py})
        probe_frame = extract_frame(probe_obs)

        for i, (cx, cy) in enumerate(cell_positions):
            before = int(base_frame[cy, cx]) if base_frame.ndim == 2 else int(base_frame[0, cy, cx])
            after = int(probe_frame[cy, cx]) if probe_frame.ndim == 2 else int(probe_frame[0, cy, cx])
            if before != after:
                toggle_matrix[i][j] = 1

        if verbose:
            affected = [i for i in range(n) if toggle_matrix[i][j]]
            print(f"  [lights-out] click cell {j} ({px},{py}) → toggles cells {affected}")

    # Filter: cells that toggle NOTHING when clicked are constraints, not variables.
    # Only include cells that actually toggle at least one cell (including themselves).
    active_mask = [any(toggle_matrix[i][j] for i in range(n)) for j in range(n)]
    active_indices = [j for j in range(n) if active_mask[j]]
    if verbose and len(active_indices) < n:
        inactive = [j for j in range(n) if not active_mask[j]]
        print(f"  [lights-out] Filtered {len(inactive)} constraint cells (no toggle effect): {inactive}")

    if active_indices and len(active_indices) < n:
        # Rebuild with only active cells
        n_active = len(active_indices)
        active_positions = [cell_positions[j] for j in active_indices]
        active_colors = [current_colors[j] for j in active_indices]
        active_toggle = np.zeros((n_active, n_active), dtype=int)
        for jj, j in enumerate(active_indices):
            for ii, i in enumerate(active_indices):
                active_toggle[ii][jj] = toggle_matrix[i][j]
        # Replace for solving
        cell_positions = active_positions
        current_colors = active_colors
        toggle_matrix = active_toggle
        n = n_active
        if verbose:
            print(f"  [lights-out] Active cells: {n}, positions: {cell_positions}")

    # Step 3: Determine target state
    # The target is usually all cells the same color. Check if there's a dominant target.
    # For lights-out: cells that are "wrong" need to be toggled an odd number of times.
    # We need to know WHICH color is target. Look at example grids if available.
    # Simple heuristic: the minority color needs to be toggled
    color_counts = {}
    for c in current_colors:
        color_counts[c] = color_counts.get(c, 0) + 1

    # Target = most common color (the cells already correct)
    target_color = max(color_counts, key=color_counts.get)

    # b[i] = 1 if cell i needs to change (is not target color)
    b = np.array([0 if c == target_color else 1 for c in current_colors], dtype=int)

    if verbose:
        print(f"  [lights-out] target color: {target_color}, cells to toggle: {list(np.where(b)[0])}")

    if np.sum(b) == 0:
        return []  # already solved

    # Step 4: Solve Ax = b over GF(2) using Gaussian elimination
    if cycle_period == 2:
        solution = _gf2_solve(toggle_matrix, b)
    else:
        # For period > 2, use modular arithmetic (more complex)
        solution = _gfk_solve(toggle_matrix, b, cycle_period)

    if solution is None:
        if verbose:
            print(f"  [lights-out] No solution found (system inconsistent)")
        # Try the OTHER target color
        other_colors = [c for c in color_counts if c != target_color]
        for alt_target in other_colors:
            b_alt = np.array([0 if c == alt_target else 1 for c in current_colors], dtype=int)
            if cycle_period == 2:
                solution = _gf2_solve(toggle_matrix, b_alt)
            else:
                solution = _gfk_solve(toggle_matrix, b_alt, cycle_period)
            if solution is not None:
                if verbose:
                    print(f"  [lights-out] Solution found with alt target color {alt_target}")
                break

    if solution is None and n <= 30:
        # Null space enumeration: find ALL solutions via GF2 null space,
        # then test each with the game oracle (deepcopy + win check).
        # This handles games where the target isn't "all same color".
        if verbose:
            print(f"  [lights-out] Uniform target failed. Trying null space enumeration...")
        import copy as _cp2
        A = toggle_matrix % 2
        # Find rank and null space basis
        aug = np.copy(A)
        m = A.shape[1]
        pivot_cols = []
        row = 0
        for col in range(m):
            found = False
            for r in range(row, n):
                if aug[r, col] == 1:
                    aug[[row, r]] = aug[[r, row]]
                    found = True
                    break
            if not found:
                continue
            pivot_cols.append(col)
            for r in range(n):
                if r != row and aug[r, col] == 1:
                    aug[r] = (aug[r] + aug[row]) % 2
            row += 1

        rank = len(pivot_cols)
        free_cols = [c for c in range(m) if c not in pivot_cols]
        null_dim = len(free_cols)

        if null_dim > 0 and null_dim <= 18:
            if verbose:
                print(f"  [lights-out] Rank={rank}, null space dim={null_dim} → {2**null_dim} solutions to test")
            # Generate all null space vectors
            from itertools import product as _iprod
            tested = 0
            for free_vals in _iprod(range(2), repeat=null_dim):
                x = np.zeros(m, dtype=int)
                for idx, fc in enumerate(free_cols):
                    x[fc] = free_vals[idx]
                # Back-substitute to find pivot values
                for i in range(rank - 1, -1, -1):
                    pc = pivot_cols[i]
                    x[pc] = 0
                    for j in range(m):
                        if j != pc:
                            x[pc] = (x[pc] + aug[i, j] * x[j]) % 2
                    x[pc] = x[pc] % 2

                # Test this solution with game oracle
                test_env = _cp2.deepcopy(env)
                test_obs = None
                for j in range(m):
                    if x[j]:
                        px, py = cell_positions[j]
                        test_obs = test_env.step(click_action, {"x": px, "y": py})
                # Check win: level advanced?
                if test_obs and hasattr(test_obs, 'levels_completed'):
                    base_level = env._last_response.levels_completed if hasattr(env, '_last_response') and hasattr(env._last_response, 'levels_completed') else 0
                    if test_obs.levels_completed > base_level:
                        clicks_found = [cell_positions[j] for j in range(m) if x[j]]
                        if verbose:
                            print(f"  [lights-out] Null space solution #{tested}: {len(clicks_found)} clicks")
                        return clicks_found
                tested += 1
                if tested % 1000 == 0 and verbose:
                    print(f"  [lights-out] Tested {tested}/{2**null_dim}...")
            if verbose:
                print(f"  [lights-out] Exhausted null space ({tested} solutions), none solved")

    if solution is None:
        return None

    # Convert solution vector to click sequence
    clicks = []
    for j in range(n):
        for _ in range(int(solution[j])):
            clicks.append(cell_positions[j])

    if verbose:
        print(f"  [lights-out] Solution: {len(clicks)} clicks — {clicks}")

    return clicks


def _gf2_solve(A: np.ndarray, b: np.ndarray):
    """Solve Ax = b over GF(2) using Gaussian elimination.
    Returns solution vector x (0s and 1s), or None if inconsistent."""
    n = A.shape[0]
    m = A.shape[1]

    # Augmented matrix [A | b]
    aug = np.zeros((n, m + 1), dtype=int)
    aug[:, :m] = A % 2
    aug[:, m] = b % 2

    pivot_col = [0] * n
    row = 0
    for col in range(m):
        # Find pivot
        found = -1
        for r in range(row, n):
            if aug[r, col] == 1:
                found = r
                break
        if found == -1:
            continue

        # Swap
        aug[[row, found]] = aug[[found, row]]
        pivot_col[row] = col

        # Eliminate
        for r in range(n):
            if r != row and aug[r, col] == 1:
                aug[r] = (aug[r] + aug[row]) % 2
        row += 1

    # Check consistency
    for r in range(row, n):
        if aug[r, m] == 1:
            return None  # inconsistent

    # Back-substitute (free variables = 0)
    x = np.zeros(m, dtype=int)
    for r in range(row):
        x[pivot_col[r]] = aug[r, m]

    return x


def _gfk_solve(A: np.ndarray, b: np.ndarray, k: int):
    """Solve Ax = b over GF(k) — brute force for small systems (n <= 16)."""
    n = A.shape[1]
    if n > 16:
        return None  # too large for brute force

    # Try all combinations — this is O(k^n) but k=2-4, n<=16 is manageable for k=2
    # For k>2 and n>8, fall back to heuristic
    if k > 2 and n > 8:
        return None

    from itertools import product as _prod
    for x in _prod(range(k), repeat=n):
        x_arr = np.array(x, dtype=int)
        result = (A @ x_arr) % k
        if np.array_equal(result, b % k):
            return x_arr

    return None


def _discover_clickable_positions(env, click_action, extract_frame, base_frame,
                                    probe_step: int = 2, verbose: bool = False):
    """Discover all distinct clickable positions by probing the frame.

    Returns list of (x, y) positions that produce distinct effects when clicked.
    Groups positions by their effect signature to find independent cells.
    """
    import copy as _cp

    h, w = base_frame.shape[-2], base_frame.shape[-1]

    # Multi-frame fix: level transitions may return N-frame observations (shape N×H×W).
    # _extract_frame returns f[-1]. Ensure base_frame is 2D for consistent comparison.
    if base_frame.ndim == 3:
        base_frame = base_frame[-1]

    # Adaptive probing: coarse pass first (step=4), then refine near hits (step=2)
    effects = {}  # signature -> list of (x, y)

    def _probe_grid(step, positions=None):
        """Probe a grid of positions. If positions is None, probe the full grid."""
        if positions is None:
            positions = [(x, y) for y in range(step // 2, h, step)
                         for x in range(step // 2, w, step)]
        for (x, y) in positions:
            test_env = _cp.deepcopy(env)
            try:
                test_obs = test_env.step(click_action, {"x": x, "y": y})
            except Exception:
                continue
            test_frame = extract_frame(test_obs)
            if test_frame.shape != base_frame.shape:
                continue
            diff_mask = base_frame != test_frame
            if base_frame.ndim == 3:
                diff_mask = diff_mask.any(axis=0)
            n_changed = int(np.sum(diff_mask))
            if n_changed == 0:
                continue
            changed_yx = tuple(zip(*np.where(diff_mask)))
            sig = changed_yx[:10]
            if sig not in effects:
                effects[sig] = []
            effects[sig].append((x, y))

    if probe_step <= 2:
        # Adaptive: coarse pass at step=4, then refine near hits at step=2
        coarse_step = 4
        _probe_grid(coarse_step)
        if effects:
            # Collect hit positions and probe their 8 neighbors at step=2
            hit_positions = set()
            for positions in effects.values():
                for (hx, hy) in positions:
                    for dy in range(-coarse_step, coarse_step + 1, 2):
                        for dx in range(-coarse_step, coarse_step + 1, 2):
                            nx, ny = hx + dx, hy + dy
                            if 0 <= nx < w and 0 <= ny < h:
                                hit_positions.add((nx, ny))
            # Remove already-probed positions
            already_probed = set()
            for y in range(coarse_step // 2, h, coarse_step):
                for x in range(coarse_step // 2, w, coarse_step):
                    already_probed.add((x, y))
            refine = list(hit_positions - already_probed)
            if refine:
                _probe_grid(2, refine)
        else:
            # No hits at step=4 — full probe at step=2 as fallback
            _probe_grid(probe_step)
    else:
        _probe_grid(probe_step)

    # For each distinct effect, pick the centroid as the representative
    clickable = []
    if verbose:
        print(f"  [probe] {len(effects)} distinct effects from {sum(len(v) for v in effects.values())} hit probes (base_frame shape={base_frame.shape})")
        for sig, positions in sorted(effects.items(), key=lambda kv: len(kv[1]), reverse=True)[:5]:
            print(f"    Effect group ({len(positions)} probes): sample positions {positions[:3]}")
    for sig, positions in effects.items():
        cx = sum(p[0] for p in positions) // len(positions)
        cy = sum(p[1] for p in positions) // len(positions)
        clickable.append((cx, cy))

    if verbose:
        print(f"  [probe] Discovered {len(clickable)} clickable positions (from {sum(len(v) for v in effects.values())} hit probes)")

    return clickable


def _constraint_solve_internal(game, env, click_action, verbose=False):
    """Solve toggle puzzle using game internals (gig, gqb, get_sprite_at).

    Returns list of (x, y) click coordinates in DISPLAY space, or None.
    Works for FT09-style constraint puzzles with indicator sprites.

    Handles:
    - Binary palettes via GF(2) Gaussian elimination (coupled NTi effects)
    - Multi-color palettes via constraint intersection
    - NTi sprites with cross-pattern toggle effects (pixel value 6 = neighbor toggle)
    - Free variables from unconstrained cells
    """
    import numpy as np
    from itertools import product as _iprod

    palette = list(game.gqb)
    if not palette or not game.gig:
        return None

    level = game.current_level
    irw = game.irw
    level_before = game.level_index

    # Neighbor offset map: indicator pixel (px_x, px_y) -> sprite offset (dx, dy)
    dirs = [(-4, -4, 0, 0), (0, -4, 1, 0), (4, -4, 2, 0),
            (-4,  0, 0, 1),                 (4,  0, 2, 1),
            (-4,  4, 0, 2), (0,  4, 1, 2), (4,  4, 2, 2)]

    # Find all cells (Hkx and NTi)
    all_cells = {}
    for sp in level._sprites:
        tags = [t.name if hasattr(t, 'name') else str(t) for t in getattr(sp, 'tags', [])]
        tag_type = 'NTi' if 'NTi' in tags else ('Hkx' if 'Hkx' in tags else None)
        if tag_type:
            all_cells[(sp.x, sp.y)] = {
                'sprite': sp, 'tag': tag_type,
                'color': int(np.array(sp.pixels)[1][1]),
                'pixels': np.array(sp.pixels),
            }

    if not all_cells:
        return None

    # Build allowed colors per cell from indicator constraints
    cell_allowed = {k: set(palette) for k in all_cells}
    for ind in game.gig:
        pixels = np.array(ind.pixels)
        cc = int(pixels[1][1])
        for dx, dy, px_x, px_y in dirs:
            tx, ty = ind.x + dx, ind.y + dy
            sp = level.get_sprite_at(tx, ty, "Hkx") or level.get_sprite_at(tx, ty, "NTi")
            if sp:
                pos = (sp.x, sp.y)
                pval = int(pixels[px_y][px_x])
                if pval == 0:
                    cell_allowed[pos] &= {cc}
                else:
                    cell_allowed[pos] -= {cc}

    if verbose:
        n_need = sum(1 for k, v in cell_allowed.items() if all_cells[k]['color'] not in v)
        print(f"  [constraint] {len(all_cells)} cells, {len(game.gig)} indicators, "
              f"{n_need} need change, palette={palette}")

    # GBS offset table for 3x3 effect grid
    GBS = [[(-1,-1),(0,-1),(1,-1)],[(-1,0),(0,0),(1,0)],[(-1,1),(0,1),(1,1)]]

    if len(palette) == 2:
        # ---- GF(2) solver for binary palettes ----
        cell_list = sorted(all_cells.keys())
        cell_idx = {k: i for i, k in enumerate(cell_list)}
        n = len(cell_list)

        # Build toggle effect for each clickable cell
        clickables = []
        for key in cell_list:
            info = all_cells[key]
            if info['tag'] == 'NTi':
                px = info['pixels']
                eff = [[0]*3 for _ in range(3)]
                eff[1][1] = 1  # Center always toggled (from init)
                for j in range(3):
                    for i in range(3):
                        if int(px[j][i]) == 6:
                            eff[j][i] = 1
                affected = []
                for j in range(3):
                    for i in range(3):
                        if eff[j][i]:
                            dx, dy = GBS[j][i]
                            t = (key[0]+dx*4, key[1]+dy*4)
                            if t in cell_idx:
                                affected.append(t)
            else:  # Hkx
                affected = []
                for j in range(3):
                    for i in range(3):
                        if irw[j][i]:
                            dx, dy = GBS[j][i]
                            t = (key[0]+dx*4, key[1]+dy*4)
                            if t in cell_idx:
                                affected.append(t)
            clickables.append((key, affected))

        m = len(clickables)
        A = np.zeros((n, m), dtype=int)
        for j, (_, affected) in enumerate(clickables):
            for t in affected:
                A[cell_idx[t], j] = 1

        b = np.zeros(n, dtype=int)
        for key, allowed in cell_allowed.items():
            if all_cells[key]['color'] not in allowed:
                b[cell_idx[key]] = 1

        # Gaussian elimination over GF(2)
        aug = np.zeros((n, m+1), dtype=int)
        aug[:, :m] = A
        aug[:, m] = b

        pivot_cols = []
        row = 0
        for col in range(m):
            found = -1
            for r in range(row, n):
                if aug[r, col] == 1:
                    found = r
                    break
            if found == -1:
                continue
            aug[[row, found]] = aug[[found, row]]
            for r in range(n):
                if r != row and aug[r, col] == 1:
                    aug[r] = (aug[r] + aug[row]) % 2
            pivot_cols.append((col, row))
            row += 1

        # Check consistency
        for r in range(row, n):
            if aug[r, m] == 1:
                if verbose:
                    print(f"  [constraint] INCONSISTENT at row {r}")
                return None

        free_cols = [j for j in range(m) if j not in [p[0] for p in pivot_cols]]

        # Try all free variable combos (up to 2^16 = 65536)
        if len(free_cols) > 16:
            if verbose:
                print(f"  [constraint] Too many free vars ({len(free_cols)}), skipping")
            return None

        best_x, best_count = None, 999
        for combo in _iprod([0, 1], repeat=len(free_cols)):
            x = np.zeros(m, dtype=int)
            for fc_idx, val in zip(free_cols, combo):
                x[fc_idx] = val
            for col, r in pivot_cols:
                x[col] = aug[r, m]
                for fc_idx, val in zip(free_cols, combo):
                    x[col] = (x[col] + aug[r, fc_idx] * val) % 2
            cnt = int(sum(x))
            if cnt < best_count:
                best_count = cnt
                best_x = x.copy()

        clicks = []
        for j in range(m):
            if best_x[j] == 1:
                pos = clickables[j][0]
                clicks.append((pos[0] * 2, pos[1] * 2))

        if verbose:
            print(f"  [constraint/GF2] {best_count} clicks (free={len(free_cols)})")

    else:
        # ---- Multi-color palette: constraint intersection ----
        clicks = []
        for key in sorted(all_cells.keys()):
            cur = all_cells[key]['color']
            allowed = cell_allowed[key]
            if cur in allowed:
                continue
            if not allowed:
                if verbose:
                    print(f"  [constraint] Cell {key} has no valid color!")
                continue
            best_n, best_t = 999, None
            for t in allowed:
                ci = palette.index(cur)
                ti = palette.index(t)
                nc = (ti - ci) % len(palette)
                if nc < best_n:
                    best_n, best_t = nc, t
            for _ in range(best_n):
                clicks.append((key[0] * 2, key[1] * 2))

        if verbose:
            print(f"  [constraint/multi] {len(clicks)} clicks")

    # Execute clicks
    for x, y in clicks:
        env.step(click_action, {"x": x, "y": y})

    if game.level_index > level_before:
        if verbose:
            print(f"  [constraint] SOLVED level {level_before}!")
        return clicks

    # Check for final level win (game state changes to WIN instead of advancing)
    try:
        from arcengine.enums import GameState
        if hasattr(game, '_state') and game._state == GameState.WIN:
            if verbose:
                print(f"  [constraint] SOLVED final level {level_before} (WIN)!")
            return clicks
    except ImportError:
        pass

    # cgj may be True even if level didn't advance (e.g. game needs explicit win call)
    if game.cgj():
        if verbose:
            print(f"  [constraint] cgj=True, triggering win")
        game.win()
        return clicks

    if verbose:
        print(f"  [constraint] Failed, cgj={game.cgj()}")
    return None


def _nav_craft_solve_internal(game, env, verbose=False):
    """Solve navigation+crafting puzzles using game internals (LS20-style).

    Reads modifier positions, goal requirements, walls. Plans optimal
    route through modifiers then to goal. Returns action sequence or None.
    """
    from collections import deque

    # Detect LS20-style game: needs mgu (player), qqv (goals), hul (colors)
    if not all(hasattr(game, a) for a in ('mgu', 'qqv', 'hul', 'kdj', 'hep')):
        return None

    level = game.current_level
    player = game.mgu
    step_size = 5  # LS20 movement step

    # Map walls
    walls = set()
    for sp in level._sprites:
        tags = [t.name if hasattr(t, 'name') else str(t) for t in getattr(sp, 'tags', [])]
        if 'jdd' in tags:
            walls.add((sp.x, sp.y))

    # Find modifiers and goals
    modifiers = {'gsu': [], 'gic': [], 'bgt': []}
    goals = []
    for sp in level._sprites:
        tags = [t.name if hasattr(t, 'name') else str(t) for t in getattr(sp, 'tags', [])]
        for mod_tag in modifiers:
            if mod_tag in tags:
                modifiers[mod_tag].append((sp.x, sp.y))
        if 'mae' in tags:
            goals.append(sp)

    if not goals:
        return None

    # Current state
    shape_idx = game.snw
    color_idx = game.tmx
    rot_idx = game.tuv

    # Goal requirements (first unsatisfied goal)
    goal_idx = None
    for i, g in enumerate(goals):
        if not getattr(game, 'rzt', [True])[i] if hasattr(game, 'rzt') and i < len(game.rzt) else True:
            goal_idx = i
            break
    if goal_idx is None:
        goal_idx = 0

    target_shape = game.gfy[goal_idx] if hasattr(game, 'gfy') else shape_idx
    target_color = game.vxy[goal_idx] if hasattr(game, 'vxy') else color_idx
    target_rot = game.cjl[goal_idx] if hasattr(game, 'cjl') else rot_idx

    # Compute visits needed
    n_shapes = len(game.hep)
    n_colors = len(game.hul)
    n_rots = len(game.kdj)
    shape_visits = (target_shape - shape_idx) % n_shapes if n_shapes else 0
    color_visits = (target_color - color_idx) % n_colors if n_colors else 0
    rot_visits = (target_rot - rot_idx) % n_rots if n_rots else 0

    if verbose:
        print(f"  [nav-craft] shape: {shape_idx}→{target_shape} ({shape_visits} visits)")
        print(f"  [nav-craft] color: {color_idx}→{target_color} ({color_visits} visits)")
        print(f"  [nav-craft] rot:   {rot_idx}→{target_rot} ({rot_visits} visits)")

    # BFS pathfinder on grid
    def bfs_path(start, end):
        if start == end:
            return []
        q = deque([(start, [])])
        visited = {start}
        dirs = {1: (0, -step_size), 2: (0, step_size), 3: (-step_size, 0), 4: (step_size, 0)}
        while q:
            (cx, cy), path = q.popleft()
            for action, (dx, dy) in dirs.items():
                nx, ny = cx + dx, cy + dy
                if (nx, ny) in walls or (nx, ny) in visited:
                    continue
                if nx < 0 or ny < 0 or nx > 60 or ny > 60:
                    continue
                new_path = path + [action]
                if (nx, ny) == end:
                    return new_path
                visited.add((nx, ny))
                q.append(((nx, ny), new_path))
        return None  # unreachable

    # Build waypoint sequence: modifiers needed, then goal
    waypoints = []
    start_pos = (player.x, player.y)
    goal_pos = (goals[goal_idx].x, goals[goal_idx].y)

    # Find closest modifier of each needed type
    for mod_tag, visits in [('gsu', shape_visits), ('gic', color_visits), ('bgt', rot_visits)]:
        if visits > 0 and modifiers[mod_tag]:
            # Pick closest reachable modifier
            best = None
            best_dist = 999
            for mx, my in modifiers[mod_tag]:
                path = bfs_path(start_pos, (mx, my))
                if path is not None and len(path) < best_dist:
                    best = (mx, my, mod_tag, visits)
                    best_dist = len(path)
            if best:
                waypoints.append(best)

    # Try both orderings of waypoints if there are 2
    import itertools
    best_actions = None
    best_len = 999

    orderings = list(itertools.permutations(waypoints)) if len(waypoints) <= 3 else [waypoints]
    for ordering in orderings:
        actions = []
        pos = start_pos
        valid = True
        for mx, my, mod_tag, visits in ordering:
            path = bfs_path(pos, (mx, my))
            if path is None:
                valid = False
                break
            actions.extend(path)
            pos = (mx, my)
            # Bounce for additional visits
            for _ in range(visits - 1):
                # Find a non-wall neighbor to bounce off
                for bounce_action, (dx, dy) in [(1,(0,-step_size)),(2,(0,step_size)),(3,(-step_size,0)),(4,(step_size,0))]:
                    nx, ny = pos[0] + dx, pos[1] + dy
                    if (nx, ny) not in walls and 0 <= nx <= 60 and 0 <= ny <= 60:
                        back_action = {1:2, 2:1, 3:4, 4:3}[bounce_action]
                        actions.extend([bounce_action, back_action])
                        break
        if not valid:
            continue
        # Navigate to goal
        path = bfs_path(pos, goal_pos)
        if path is None:
            continue
        actions.extend(path)
        if len(actions) < best_len:
            best_len = len(actions)
            best_actions = actions

    if best_actions is None:
        if verbose:
            print("  [nav-craft] No valid route found")
        return None

    # Check budget
    budget = game.ggk.tmx if hasattr(game, 'ggk') else 999
    if verbose:
        print(f"  [nav-craft] Route: {best_len} steps, budget: {budget}")

    if best_len > budget:
        if verbose:
            print(f"  [nav-craft] Route exceeds budget ({best_len} > {budget})")
        # Still try — might work if some steps are free
        pass

    # Execute actions
    import copy
    level_before = game.level_index
    test_env = copy.deepcopy(env)
    test_game = test_env._game

    for act in best_actions:
        test_env.step(act)
        if test_game.level_index > level_before:
            if verbose:
                print(f"  [nav-craft] SOLVED in {best_actions.index(act)+1}/{len(best_actions)} steps!")
            # Execute on real env
            for a in best_actions[:best_actions.index(act)+1]:
                env.step(a)
            return best_actions[:best_actions.index(act)+1]

    if verbose:
        print(f"  [nav-craft] Route completed but level didn't advance")
    return None


def _oracle_solve(env, click_action, extract_frame, grid_cells=None,
                   toggle_matrix=None, cycle_period: int = 2,
                   levels_completed: int = 0, base_frame=None,
                   predicted_target_map=None, predicted_current_map=None,
                   max_blocks: int = 18, verbose: bool = False):
    """Solve by trying all combinations and checking win via game oracle.

    Instead of guessing the target state, we deepcopy the env, apply clicks,
    and check if the game reports a win (levels_completed increases).
    Uses toggle matrix to identify independent blocks and reduce search space.

    If grid_cells is None but base_frame is provided, discovers clickable
    positions by probing the frame directly (bypasses grid detection).

    Returns list of (x, y) click coordinates, or None.
    """
    import copy as _cp

    # === SPECIALIZED SOLVERS — DISABLED BY DEFAULT (Peter directive 2026-03-20) ===
    # Architecture: Qwen-VL (eyes) + Claude (brain) + corpus callosum.
    # Set ARC_USE_SOLVER=1 to re-enable for offline benchmarking only.
    _use_solver = os.environ.get('ARC_USE_SOLVER')

    # === FAST PATH: Direct constraint solver using game internals ===
    if _use_solver:
        try:
            _g = getattr(env, '_game', None)
            if _g and hasattr(_g, 'gig') and hasattr(_g, 'gqb') and hasattr(_g, 'cgj'):
                _test_env = _cp.deepcopy(env)
                _test_g = getattr(_test_env, '_game', _test_env)
                result = _constraint_solve_internal(_test_g, _test_env, click_action, verbose=verbose)
                if result is not None:
                    return result
        except Exception as e:
            if verbose:
                print(f"  [oracle] Internal solver failed: {e}, falling back to probe")

    # === FAST PATH 2: Navigation+crafting solver (LS20-style) ===
    if _use_solver:
        try:
            _g = getattr(env, '_game', None)
            if _g and hasattr(_g, 'mgu') and hasattr(_g, 'qqv'):
                result = _nav_craft_solve_internal(_g, env, verbose=verbose)
                if result is not None:
                    return result
        except Exception as e:
            if verbose:
                print(f"  [oracle] Nav-craft solver failed: {e}, falling back to probe")

    # Multi-frame normalization: use last frame as reference state
    if base_frame is not None and base_frame.ndim == 3 and base_frame.shape[0] > 1:
        base_frame = base_frame[-1:]  # keep 3D but single frame

    # Discover clickable positions if not provided via grid_cells
    if grid_cells is not None:
        cell_positions = [(c['center_x'], c['center_y']) for c in grid_cells]
    elif base_frame is not None:
        cell_positions = _discover_clickable_positions(
            env, click_action, extract_frame, base_frame, verbose=verbose
        )
    else:
        return None

    # Stash discovered positions for researcher re-use
    _oracle_solve._last_cell_positions = cell_positions

    if verbose:
        print(f"  [oracle] Cell positions: {cell_positions}")

    n = len(cell_positions)
    if n == 0:
        return None

    # Build toggle matrix if not provided
    if toggle_matrix is None or (grid_cells is not None and toggle_matrix.shape[1] != n):
        # Helper: read cell color robustly (region-based, handles zoom misalignment)
        def _read_cell_color(frm, cx, cy, radius=2):
            """Read dominant non-bg color in radius around (cx,cy)."""
            f2d = frm[0] if frm.ndim == 3 else frm
            hh, ww = f2d.shape
            bg = int(np.bincount(f2d.flatten()).argmax())
            colors = []
            for dy in range(-radius, radius+1):
                for dx in range(-radius, radius+1):
                    yy, xx = cy+dy, cx+dx
                    if 0 <= yy < hh and 0 <= xx < ww:
                        c = int(f2d[yy, xx])
                        if c != bg:
                            colors.append(c)
            if not colors:
                return int(f2d[min(cy, hh-1), min(cx, ww-1)])
            return max(set(colors), key=colors.count)

        toggle_matrix = np.zeros((n, n), dtype=int)
        for j, (px, py) in enumerate(cell_positions):
            probe_env = _cp.deepcopy(env)
            probe_obs = probe_env.step(click_action, {"x": px, "y": py})
            probe_frame = extract_frame(probe_obs)
            frame_ref = base_frame if base_frame is not None else np.zeros_like(probe_frame)
            for i, (cx, cy) in enumerate(cell_positions):
                bf_c = _read_cell_color(frame_ref, cx, cy)
                af_c = _read_cell_color(probe_frame, cx, cy)
                if bf_c != af_c:
                    toggle_matrix[i][j] = 1

    # Identify independent blocks from toggle matrix
    blocks = []
    assigned = set()
    for j in range(n):
        if j in assigned:
            continue
        block = [j]
        col_j = tuple(toggle_matrix[:, j])
        for k in range(j + 1, n):
            if k not in assigned and tuple(toggle_matrix[:, k]) == col_j:
                block.append(k)
        blocks.append(block)
        assigned.update(block)

    n_blocks = len(blocks)
    if verbose:
        print(f"  [oracle] {n} cells → {n_blocks} independent blocks, period={cycle_period}")

    # --- Pattern-cell analysis: detect constraint sprites and compute exact target ---
    # This reads visible pattern/constraint cells to determine what each clickable cell
    # should become. Generic for any toggle puzzle with visible constraints.
    _pattern_combo = None
    if base_frame is not None and n_blocks <= 50:
        try:
            bf = base_frame[0] if base_frame.ndim == 3 else base_frame
            h, w = bf.shape
            bg_color = int(np.bincount(bf.flatten()).argmax())
            # Detect zoom factor: find smallest uniform block size (2, 4, or 8)
            zoom = 2  # default
            for test_zoom in [4, 2]:
                uniform_count = 0
                total_count = 0
                for py in range(0, min(h, 32), test_zoom):
                    for px in range(0, min(w, 32), test_zoom):
                        c = bf[py, px]
                        if c == bg_color:
                            continue
                        total_count += 1
                        is_uniform = True
                        for dy in range(test_zoom):
                            for dx in range(test_zoom):
                                if py+dy < h and px+dx < w and bf[py+dy, px+dx] != c:
                                    is_uniform = False
                                    break
                            if not is_uniform:
                                break
                        if is_uniform:
                            uniform_count += 1
                if total_count > 0 and uniform_count / total_count > 0.8:
                    zoom = test_zoom
                    break
            if verbose:
                print(f"  [oracle] Pattern analysis: detected zoom={zoom}")
            # Find game pixels: zoom×zoom uniform blocks
            game_pixels = {}
            for py in range(0, h-zoom+1, zoom):
                for px in range(0, w-zoom+1, zoom):
                    c = bf[py, px]
                    if c == bg_color:
                        continue
                    is_uniform = True
                    for dy in range(zoom):
                        for dx in range(zoom):
                            if bf[py+dy, px+dx] != c:
                                is_uniform = False
                                break
                        if not is_uniform:
                            break
                    if is_uniform:
                        game_pixels[(px//zoom, py//zoom)] = int(c)
            if game_pixels:
                # Find sprite grid: 3x3 game-pixel sprites at 4-unit spacing
                gp_set = set(game_pixels.keys())
                all_gx = sorted(set(gx for gx, _ in gp_set))
                all_gy = sorted(set(gy for _, gy in gp_set))
                def _find_origins(coords):
                    origins = []
                    i = 0
                    while i < len(coords):
                        origins.append(coords[i])
                        target = coords[i] + 4
                        while i < len(coords) and coords[i] < target:
                            i += 1
                    return origins
                x_origins = _find_origins(all_gx)
                y_origins = _find_origins(all_gy)
                # Extract sprites: classify as solid (uniform color) or pattern (mixed)
                sprites = []
                for si, sy in enumerate(y_origins):
                    for sj, sx in enumerate(x_origins):
                        inner = np.full((3, 3), bg_color, dtype=np.uint8)
                        has_px = False
                        for dy in range(3):
                            for dx in range(3):
                                gp = (sx + dx, sy + dy)
                                if gp in game_pixels:
                                    inner[dy, dx] = game_pixels[gp]
                                    has_px = True
                                else:
                                    # Read raw pixel even if it matches bg (handles bg-colored centers)
                                    rpx, rpy = (sx + dx) * zoom, (sy + dy) * zoom
                                    if rpy < h and rpx < w:
                                        inner[dy, dx] = bf[rpy, rpx]
                        if not has_px:
                            continue
                        center = int(inner[1, 1])
                        unique_inner = set(int(v) for v in inner.flat if v != bg_color)
                        is_pattern = len(unique_inner) >= 2
                        # Display coords for clicking: center of 3x3 game sprite
                        cx_disp = (sx + 1) * zoom
                        cy_disp = (sy + 1) * zoom
                        sprites.append({
                            'gx': sj, 'gy': si,
                            'cx': cx_disp, 'cy': cy_disp,
                            'inner': inner, 'center': center,
                            'pattern': is_pattern, 'solid': not is_pattern,
                            'color': center if not is_pattern else None,
                        })
                patterns = [s for s in sprites if s['pattern']]
                solids = [s for s in sprites if s['solid']]
                if patterns and solids:
                    # Build constraint map from pattern cells
                    # FT09 rule: inner pixel == 0 → neighbor must EQUAL center; != 0 → must NOT equal
                    offsets_3x3 = [(-1,-1),(0,-1),(1,-1),(-1,0),(1,0),(-1,1),(0,1),(1,1)]
                    inner_pos = [(0,0),(1,0),(2,0),(0,1),(2,1),(0,2),(1,2),(2,2)]
                    solid_by_gpos = {(s['gx'], s['gy']): s for s in solids}
                    target_colors = {}  # (gx, gy) -> target color for solid cells
                    for p in patterns:
                        pc = p['center']
                        for k, (dg_x, dg_y) in enumerate(offsets_3x3):
                            nx, ny = p['gx'] + dg_x, p['gy'] + dg_y
                            if (nx, ny) not in solid_by_gpos:
                                continue
                            ip_x, ip_y = inner_pos[k]
                            pixel_val = int(p['inner'][ip_y, ip_x])
                            if pixel_val == 0:
                                target_colors.setdefault((nx, ny), {})['must_equal'] = pc
                            else:
                                target_colors.setdefault((nx, ny), {})['must_not_equal'] = pc
                    if target_colors:
                        # Map oracle cell positions to grid positions
                        _pattern_combo = 0
                        _pattern_matched = 0
                        for b_idx in range(n_blocks):
                            rep = blocks[b_idx][0]
                            cpx, cpy = cell_positions[rep]
                            # Find nearest solid sprite
                            best = None
                            best_d = 999
                            for s in solids:
                                d = abs(s['cx'] - cpx) + abs(s['cy'] - cpy)
                                if d < best_d:
                                    best_d = d
                                    best = s
                            if best and best_d <= 6 and (best['gx'], best['gy']) in target_colors:
                                constraints = target_colors[(best['gx'], best['gy'])]
                                cur = best['color']
                                need_toggle = False
                                if 'must_equal' in constraints:
                                    if cur != constraints['must_equal']:
                                        need_toggle = True
                                elif 'must_not_equal' in constraints:
                                    # For period=2: if current == forbidden, toggle
                                    if cur == constraints['must_not_equal']:
                                        need_toggle = True
                                if need_toggle:
                                    _pattern_combo += cycle_period ** b_idx
                                    _pattern_matched += 1
                        if _pattern_matched > 0 and verbose:
                            print(f"  [oracle] Pattern-cell target: {_pattern_matched}/{n_blocks} cells need toggling (from {len(patterns)} patterns)")
        except Exception as e:
            if verbose:
                print(f"  [oracle] Pattern analysis failed: {e}")
            _pattern_combo = None

    total_combos = cycle_period ** n_blocks

    def _try_combo(combo_val):
        """Try a single combination. Returns click list if solved, else None."""
        probe_env = _cp.deepcopy(env)
        clicks = []
        probe_obs = None
        for b_idx in range(n_blocks):
            n_clicks_for_block = (combo_val // (cycle_period ** b_idx)) % cycle_period
            if n_clicks_for_block == 0:
                continue
            representative = blocks[b_idx][0]
            px, py = cell_positions[representative]
            for _ in range(n_clicks_for_block):
                clicks.append((px, py))
                probe_obs = probe_env.step(click_action, {"x": px, "y": py})
        if probe_obs is not None and probe_obs.levels_completed > levels_completed:
            return clicks
        return None

    # If pattern analysis found a target, try it FIRST (exact match)
    if _pattern_combo is not None and _pattern_combo > 0:
        result = _try_combo(_pattern_combo)
        if result is not None:
            if verbose:
                print(f"  [oracle] SOLVED with {len(result)} clicks (pattern-cell target)")
            return result
        elif verbose:
            print(f"  [oracle] Pattern-cell combo didn't solve — will continue with search")
        # Hamming search around pattern prediction
        if _pattern_combo is not None and _pattern_combo > 0 and n_blocks <= 30:
            import time as _time_mod
            from itertools import combinations
            _ham_start = _time_mod.time()
            _ham_timeout = 45  # 45s for Hamming search (was 15s — too tight for 23+ cells)
            _ham_tested = 0
            _ham_done = False
            for d in range(1, min(8, n_blocks)):  # search up to d=7 (was d=4)
                for flip_indices in combinations(range(n_blocks), d):
                    test_combo = _pattern_combo
                    for fi in flip_indices:
                        test_combo ^= (1 << fi)
                    result = _try_combo(test_combo)
                    _ham_tested += 1
                    if result is not None:
                        if verbose:
                            print(f"  [oracle] SOLVED with {len(result)} clicks (pattern Hamming d={d})")
                        return result
                    if _time_mod.time() - _ham_start > _ham_timeout:
                        _ham_done = True
                        break
                if _ham_done:
                    break
            if verbose:
                elapsed = _time_mod.time() - _ham_start
                print(f"  [oracle] Pattern Hamming: {_ham_tested} tests in {elapsed:.1f}s ({'timeout' if _ham_done else 'exhausted'})")

    # Special case: all blocks are independent singletons (each cell only toggles itself)
    # With independent binary cells, determine each bit with O(n) probes:
    # 1. Click ALL cells → get baseline "all-toggled" state
    # 2. For each cell, try NOT clicking it — if that solves, we know it shouldn't be clicked
    # This is O(n) deepcopies instead of O(2^n)
    if n_blocks == n and all(len(b) == 1 for b in blocks):
        if verbose:
            print(f"  [oracle] All {n} cells independent — O(n) bit-by-bit solve")
        # First: try clicking no cells (already solved?)
        # Then: try clicking all cells
        for attempt_name, click_indices in [("none", []), ("all", list(range(n)))]:
            probe_env = _cp.deepcopy(env)
            probe_obs = None
            for idx in click_indices:
                px, py = cell_positions[blocks[idx][0]]
                probe_obs = probe_env.step(click_action, {"x": px, "y": py})
            if probe_obs and probe_obs.levels_completed > levels_completed:
                clicks = [(cell_positions[blocks[i][0]][0], cell_positions[blocks[i][0]][1]) for i in click_indices]
                if verbose:
                    print(f"  [oracle] SOLVED with {len(clicks)} clicks (attempt={attempt_name})")
                return clicks
        # Binary search: determine each bit independently using a known solution
        # Step 1: Find ANY working solution via random sampling or structured search
        # Step 2: Once found, determine each bit by flipping it
        # For now, try smart subsets: click cells that are NOT the most common color
        import random
        unique_colors = set()
        cell_colors = []
        for b_idx in range(n):
            px, py = cell_positions[blocks[b_idx][0]]
            c = int(base_frame[py, px]) if base_frame is not None and base_frame.ndim == 2 else 0
            cell_colors.append(c)
            unique_colors.add(c)

        # Try clicking cells of each color
        for target_color in unique_colors:
            click_indices = [i for i in range(n) if cell_colors[i] != target_color]
            if not click_indices:
                continue
            probe_env = _cp.deepcopy(env)
            probe_obs = None
            test_clicks = []
            for idx in click_indices:
                px, py = cell_positions[blocks[idx][0]]
                test_clicks.append((px, py))
                probe_obs = probe_env.step(click_action, {"x": px, "y": py})
            if probe_obs and probe_obs.levels_completed > levels_completed:
                if verbose:
                    print(f"  [oracle] SOLVED with {len(test_clicks)} clicks (color-match target={target_color})")
                return test_clicks

        # O(n) bit-determination for independent binary cells.
        # Since cells don't affect each other, we can determine EACH cell's
        # required toggle state independently using n deepcopy probes.
        #
        # Method: "all-minus-one" — for each cell i, click all cells EXCEPT i.
        # If "all-except-i" solves → cell i should NOT be clicked.
        # If "all-except-i" doesn't solve → cell i MUST be clicked.
        # This works because cells are independent: the solution is a fixed subset.
        #
        # Cost: n deepcopies × (n-1) clicks each = O(n²). For n=23: ~500 ops.
        if verbose:
            print(f"  [oracle] Bit-determination: {n} cells, O(n²) probes...")
        must_click = []
        for test_idx in range(n):
            probe_env = _cp.deepcopy(env)
            probe_obs = None
            for other_idx in range(n):
                if other_idx == test_idx:
                    continue
                px, py = cell_positions[blocks[other_idx][0]]
                probe_obs = probe_env.step(click_action, {"x": px, "y": py})
            if probe_obs and probe_obs.levels_completed > levels_completed:
                # Solved WITHOUT clicking test_idx → test_idx should NOT be clicked
                pass
            else:
                # NOT solved without test_idx → test_idx MUST be clicked
                must_click.append(test_idx)

        # Verify the determined solution
        if must_click:
            probe_env = _cp.deepcopy(env)
            probe_obs = None
            test_clicks = []
            for idx in must_click:
                px, py = cell_positions[blocks[idx][0]]
                test_clicks.append((px, py))
                probe_obs = probe_env.step(click_action, {"x": px, "y": py})
            if probe_obs and probe_obs.levels_completed > levels_completed:
                if verbose:
                    print(f"  [oracle] SOLVED with {len(test_clicks)} clicks (bit-determination, {n} probes)")
                return test_clicks
            elif verbose:
                print(f"  [oracle] Bit-determination solution ({len(must_click)} clicks) didn't solve — cells may not be truly independent")

        # ── Incremental correction from pattern prediction ──
        # For independent cells: start from pattern prediction, apply it,
        # then test each cell by toggling/untoggling (O(n) per correction).
        # Period=2 means clicking a cell twice returns to original state.
        if _pattern_combo is not None and _pattern_combo > 0 and cycle_period == 2:
            if verbose:
                print(f"  [oracle] Incremental correction: starting from pattern prediction")
            # Apply the pattern prediction
            corr_env = _cp.deepcopy(env)
            corr_obs = None
            predicted_set = set()
            for b_idx in range(n):
                if (_pattern_combo >> b_idx) & 1:
                    predicted_set.add(b_idx)
                    px, py = cell_positions[blocks[b_idx][0]]
                    corr_obs = corr_env.step(click_action, {"x": px, "y": py})

            if corr_obs and corr_obs.levels_completed > levels_completed:
                clicks = [(cell_positions[blocks[i][0]][0], cell_positions[blocks[i][0]][1]) for i in predicted_set]
                if verbose:
                    print(f"  [oracle] SOLVED with {len(clicks)} clicks (pattern prediction was correct after all)")
                return clicks

            # Now try flipping each cell — toggle then untoggle
            # If toggling cell i (adding if missing, removing if present) fixes a bit,
            # the game gets closer to solved. We iterate until solved or exhausted.
            corrections = 0
            max_corrections = min(n, 10)
            for _round in range(max_corrections):
                found_correction = False
                for b_idx in range(n):
                    px, py = cell_positions[blocks[b_idx][0]]
                    # Toggle cell b_idx
                    corr_obs = corr_env.step(click_action, {"x": px, "y": py})
                    if corr_obs and corr_obs.levels_completed > levels_completed:
                        # Flipping this cell solved it!
                        if b_idx in predicted_set:
                            predicted_set.discard(b_idx)
                        else:
                            predicted_set.add(b_idx)
                        corrections += 1
                        clicks = [(cell_positions[blocks[i][0]][0], cell_positions[blocks[i][0]][1]) for i in predicted_set]
                        if verbose:
                            print(f"  [oracle] SOLVED with {len(clicks)} clicks (incremental correction, {corrections} flips)")
                        return clicks
                    # Check if this brought us closer (game state changed favorably)
                    # We can't easily measure "closer" without game feedback, so untoggle
                    corr_obs = corr_env.step(click_action, {"x": px, "y": py})
                # No single-cell correction found in this round — try pairs
                break

            if verbose:
                print(f"  [oracle] Incremental correction failed — no single-cell fix found")

        # Fallback: random sampling (200 attempts)
        import random
        for attempt in range(200):
            subset = [i for i in range(n) if random.random() < 0.5]
            if not subset:
                continue
            probe_env = _cp.deepcopy(env)
            probe_obs = None
            test_clicks = []
            for idx in subset:
                px, py = cell_positions[blocks[idx][0]]
                test_clicks.append((px, py))
                probe_obs = probe_env.step(click_action, {"x": px, "y": py})
            if probe_obs and probe_obs.levels_completed > levels_completed:
                if verbose:
                    print(f"  [oracle] SOLVED with {len(test_clicks)} clicks (random attempt {attempt})")
                return test_clicks

    # ── Kernel-guided oracle search ──
    # For coupled cells with large search spaces: use the toggle matrix's
    # GF(2) kernel to reduce the search. If rank(T) = r, the kernel has
    # dimension k = n - r. We only need to test 2^k combinations, not 2^n.
    # This is the key insight for FT09 L2+ where n=23 but k might be small.
    if n_blocks > 18 and cycle_period == 2:
        T = toggle_matrix.astype(np.int64) % 2
        # Compute rank via GF(2) row reduction
        m, nn = T.shape
        A = T.copy()
        pivot_cols = []
        row = 0
        for col in range(nn):
            found = -1
            for r in range(row, m):
                if A[r, col] % 2 == 1:
                    found = r
                    break
            if found == -1:
                continue
            A[[row, found]] = A[[found, row]]
            for r2 in range(m):
                if r2 != row and A[r2, col] % 2 == 1:
                    A[r2] = (A[r2] + A[row]) % 2
            pivot_cols.append(col)
            row += 1
        rank = len(pivot_cols)
        kernel_dim = nn - rank
        if verbose:
            print(f"  [oracle] Toggle matrix: rank={rank}, kernel dim={kernel_dim} (search space: 2^{kernel_dim}={2**kernel_dim})")

        if kernel_dim <= 18:  # feasible to enumerate
            # Build kernel basis: free columns are those NOT in pivot_cols
            free_cols = [c for c in range(nn) if c not in pivot_cols]
            # For each free column, build a kernel vector
            kernel_basis = []
            for fc in free_cols:
                vec = np.zeros(nn, dtype=int)
                vec[fc] = 1
                # Back-substitute: for each pivot, set the pivot variable
                for p_row, p_col in enumerate(pivot_cols):
                    if A[p_row, fc] % 2 == 1:
                        vec[p_col] = 1
                kernel_basis.append(vec)

            # Try all 2^k kernel combinations added to various base solutions
            # Base solutions: try each "click all cells of color X" as starting point
            base_solutions = []
            current_colors = []
            for px, py in cell_positions:
                c = int(base_frame[py, px]) if base_frame is not None and base_frame.ndim == 2 else 0
                current_colors.append(c)
            unique_colors = set(current_colors)
            for target_color in unique_colors:
                base = np.array([0 if c == target_color else 1 for c in current_colors], dtype=int)
                base_solutions.append(base)
            # Also try "click nothing" and "click everything" as bases
            base_solutions.append(np.zeros(nn, dtype=int))
            base_solutions.append(np.ones(nn, dtype=int))

            solved = False
            for base_idx, base_sol in enumerate(base_solutions):
                for k_combo in range(2 ** kernel_dim):
                    candidate = base_sol.copy()
                    for bit in range(kernel_dim):
                        if k_combo & (1 << bit):
                            candidate = (candidate + kernel_basis[bit]) % 2
                    click_indices = [i for i in range(nn) if candidate[i] % 2 == 1]
                    if not click_indices:
                        continue
                    probe_env = _cp.deepcopy(env)
                    probe_obs = None
                    test_clicks = []
                    for idx in click_indices:
                        px, py = cell_positions[idx]
                        test_clicks.append((px, py))
                        probe_obs = probe_env.step(click_action, {"x": px, "y": py})
                    if probe_obs and probe_obs.levels_completed > levels_completed:
                        if verbose:
                            print(f"  [oracle] SOLVED with {len(test_clicks)} clicks (kernel search, base={base_idx}, k={k_combo})")
                        return test_clicks
                if verbose and base_idx == 0:
                    print(f"  [oracle] Kernel search: tried base {base_idx}, {2**kernel_dim} combos — no solution yet")
            if verbose:
                print(f"  [oracle] Kernel search exhausted ({len(base_solutions)} bases × {2**kernel_dim} combos)")

    # Exhaustive search for small spaces (with 30s timeout)
    if total_combos <= 2 ** 18:
        import time as _time_mod
        _exh_start = _time_mod.time()
        _exh_timeout = 15
        if verbose:
            print(f"  [oracle] Exhaustive search: {total_combos} combinations (timeout {_exh_timeout}s)...")
        for combo in range(1, total_combos):
            result = _try_combo(combo)
            if result is not None:
                if verbose:
                    print(f"  [oracle] SOLVED with {len(result)} clicks (combo {combo}/{total_combos})")
                return result
            if combo % 1000 == 0 and _time_mod.time() - _exh_start > _exh_timeout:
                if verbose:
                    print(f"  [oracle] Exhaustive search timeout at combo {combo}/{total_combos}")
                break
        else:
            if verbose:
                print(f"  [oracle] No solution found in {total_combos} combinations")
            return None

    # GF(2) algebraic solve for large toggle puzzles
    # The toggle matrix is already built — use it with Gaussian elimination.
    # We need the target vector b[i] = 1 if cell i needs toggling.
    # Heuristic: target = most common color (cells that are already correct stay put).
    if cycle_period == 2 and base_frame is not None:
        current_colors = []
        for px, py in cell_positions:
            c = int(base_frame[py, px]) if base_frame.ndim == 2 else int(base_frame[0, py, px])
            current_colors.append(c)
        color_counts = {}
        for c in current_colors:
            color_counts[c] = color_counts.get(c, 0) + 1

        # Try each possible target color
        for target_color in sorted(color_counts, key=color_counts.get, reverse=True):
            b = np.array([0 if c == target_color else 1 for c in current_colors], dtype=int)
            if np.sum(b) == 0:
                continue  # already all this color
            solution = _gf2_solve(toggle_matrix, b)
            if solution is not None:
                clicks = []
                for idx in range(n):
                    if solution[idx] % 2 == 1:
                        clicks.append(cell_positions[idx])
                # Verify with oracle
                probe_env = _cp.deepcopy(env)
                probe_obs = None
                for px, py in clicks:
                    probe_obs = probe_env.step(click_action, {"x": px, "y": py})
                if probe_obs and probe_obs.levels_completed > levels_completed:
                    if verbose:
                        print(f"  [oracle] SOLVED with {len(clicks)} clicks (GF2 algebraic, target={target_color})")
                    return clicks
                elif verbose:
                    print(f"  [oracle] GF2 solution ({len(clicks)} clicks, target={target_color}) didn't solve — trying next target")

        if verbose:
            print(f"  [oracle] GF2 algebraic solve exhausted all target colors")

        # Smart search: use pattern-cell prediction as starting target, then vary
        # Since A is likely full rank, compute A^(-1) and solve for each candidate b
        # Then verify with ONE deepcopy per candidate (much faster than brute force)
        if n_blocks <= 30:
            # Compute A inverse if full rank
            A = toggle_matrix[:n_blocks, :n_blocks] % 2
            _rank = np.linalg.matrix_rank(A.astype(float))
            if _rank == n_blocks:
                # Full rank — A^(-1) exists over GF(2)
                # Compute inverse via augmented matrix [A | I]
                _aug_inv = np.zeros((n_blocks, 2 * n_blocks), dtype=int)
                _aug_inv[:, :n_blocks] = A
                _aug_inv[:, n_blocks:] = np.eye(n_blocks, dtype=int)
                _inv_row = 0
                _inv_ok = True
                for col in range(n_blocks):
                    found = False
                    for r in range(_inv_row, n_blocks):
                        if _aug_inv[r, col] == 1:
                            _aug_inv[[_inv_row, r]] = _aug_inv[[r, _inv_row]]
                            found = True
                            break
                    if not found:
                        _inv_ok = False
                        break
                    for r in range(n_blocks):
                        if r != _inv_row and _aug_inv[r, col] == 1:
                            _aug_inv[r] = (_aug_inv[r] + _aug_inv[_inv_row]) % 2
                    _inv_row += 1

                if _inv_ok:
                    A_inv = _aug_inv[:, n_blocks:] % 2
                    # Build candidate target vectors from pattern-cell prediction
                    # Extract toggle vector from _pattern_combo
                    _pattern_b = None
                    if _pattern_combo is not None and _pattern_combo > 0:
                        _pattern_b = np.array([((_pattern_combo // (cycle_period ** i)) % cycle_period) for i in range(n_blocks)], dtype=int)
                        _pattern_b = (_pattern_b > 0).astype(int)  # binary: toggle or not
                    _candidates = []
                    # Start with pattern prediction and Hamming-2 neighbors
                    if _pattern_b is not None:
                        _candidates.append(("pattern", _pattern_b))
                        # Hamming-1
                        for i in range(n_blocks):
                            b2 = _pattern_b.copy()
                            b2[i] = 1 - b2[i]
                            _candidates.append((f"H1-{i}", b2))
                        # Hamming-2 and Hamming-3
                        from itertools import combinations as _hcomb
                        for i in range(n_blocks):
                            for j in range(i+1, n_blocks):
                                b2 = _pattern_b.copy()
                                b2[i] = 1 - b2[i]
                                b2[j] = 1 - b2[j]
                                _candidates.append((f"H2-{i},{j}", b2))
                        for d in range(3, 6):
                            for flip_set in _hcomb(range(n_blocks), d):
                                b3 = _pattern_b.copy()
                                for k in flip_set:
                                    b3[k] = 1 - b3[k]
                                _candidates.append((f"H{d}", b3))

                    # Also try each unique "all same color" as base
                    for tc in sorted(color_counts, key=color_counts.get, reverse=True):
                        base_b = np.array([0 if c == tc else 1 for c in current_colors], dtype=int)
                        if np.sum(base_b) > 0:
                            _candidates.append((f"uniform-{tc}", base_b))

                    if verbose:
                        print(f"  [oracle] GF2 inverse search: {len(_candidates)} candidate targets")

                    import time as _tmod2
                    _inv_start = _tmod2.time()
                    _inv_timeout = 30
                    _inv_tested = 0
                    _inv_skipped = 0
                    # Pre-compute reasonable click range from known solutions
                    _reasonable_min = max(1, n_blocks // 4)
                    _reasonable_max = min(n_blocks, n_blocks * 3 // 4)
                    for label, b_cand in _candidates:
                        x = (A_inv @ b_cand) % 2
                        n_clicks = int(np.sum(x))
                        if n_clicks < _reasonable_min or n_clicks > _reasonable_max:
                            _inv_skipped += 1
                            continue
                        # Verify with game oracle
                        clicks = [cell_positions[idx] for idx in range(n_blocks) if x[idx]]
                        probe_env = _cp.deepcopy(env)
                        probe_obs = None
                        for px, py in clicks:
                            probe_obs = probe_env.step(click_action, {"x": px, "y": py})
                        _inv_tested += 1
                        if probe_obs and probe_obs.levels_completed > levels_completed:
                            if verbose:
                                print(f"  [oracle] SOLVED with {n_clicks} clicks (GF2 inverse, target={label})")
                            return clicks
                        if _tmod2.time() - _inv_start > _inv_timeout:
                            break
                    if verbose:
                        print(f"  [oracle] GF2 inverse: {_inv_tested} tested, {_inv_skipped} skipped, none solved")

        # Null space enumeration: try all solutions in the GF2 solution space
        # If rank(A) = r and n_blocks = n, there are 2^(n-r) solutions
        # Each represents a different target state — test with game oracle
        A = toggle_matrix[:n_blocks, :n_blocks] % 2
        aug = np.copy(A)
        pivot_cols = []
        _row = 0
        for col in range(n_blocks):
            found = False
            for r in range(_row, n_blocks):
                if aug[r, col] == 1:
                    aug[[_row, r]] = aug[[r, _row]]
                    found = True
                    break
            if not found:
                continue
            pivot_cols.append(col)
            for r in range(n_blocks):
                if r != _row and aug[r, col] == 1:
                    aug[r] = (aug[r] + aug[_row]) % 2
            _row += 1
        rank = len(pivot_cols)
        free_cols = [c for c in range(n_blocks) if c not in pivot_cols]
        null_dim = len(free_cols)

        if 0 < null_dim <= 18:
            if verbose:
                print(f"  [oracle] Null space: rank={rank}, dim={null_dim} → {2**null_dim} candidate solutions")
            from itertools import product as _iprod
            import time as _tmod
            _ns_start = _tmod.time()
            _ns_timeout = 30  # seconds
            _ns_tested = 0
            for free_vals in _iprod(range(2), repeat=null_dim):
                x = np.zeros(n_blocks, dtype=int)
                for idx, fc in enumerate(free_cols):
                    x[fc] = free_vals[idx]
                for i in range(rank - 1, -1, -1):
                    pc = pivot_cols[i]
                    x[pc] = 0
                    for j in range(n_blocks):
                        if j != pc:
                            x[pc] = (x[pc] + aug[i, j] * x[j]) % 2
                    x[pc] = x[pc] % 2

                if np.sum(x) == 0:
                    _ns_tested += 1
                    continue  # empty click set = no change

                # Build click sequence from solution
                clicks = []
                for b_idx in range(n_blocks):
                    if x[b_idx]:
                        rep = blocks[b_idx][0]
                        clicks.append(cell_positions[rep])

                # Test with game oracle
                probe_env = _cp.deepcopy(env)
                probe_obs = None
                for px, py in clicks:
                    probe_obs = probe_env.step(click_action, {"x": px, "y": py})
                _ns_tested += 1

                if probe_obs and probe_obs.levels_completed > levels_completed:
                    if verbose:
                        print(f"  [oracle] SOLVED with {len(clicks)} clicks (null space #{_ns_tested})")
                    return clicks

                if _tmod.time() - _ns_start > _ns_timeout:
                    if verbose:
                        print(f"  [oracle] Null space timeout after {_ns_tested} tests")
                    break
            else:
                if verbose:
                    print(f"  [oracle] Null space exhausted ({_ns_tested} tests), no solution")

    # For large spaces: Hamming-distance search around LLM's predicted solution
    from itertools import combinations as _combs

    # Build predicted toggle vector from researcher's target prediction
    predicted_combo = None
    if predicted_target_map and predicted_current_map and base_frame is not None:
        # For each probe-discovered cell, find nearest researcher prediction
        predicted_combo = 0
        matched = 0
        for b_idx, block in enumerate(blocks):
            representative = block[0]
            px, py = cell_positions[representative]
            # Find nearest predicted cell (within 8 pixels)
            best_dist = float('inf')
            best_target = None
            best_current = None
            for (rx, ry), tc in predicted_target_map.items():
                dist = abs(rx - px) + abs(ry - py)
                if dist < best_dist:
                    best_dist = dist
                    best_target = tc
                    best_current = predicted_current_map.get((rx, ry))
            if best_dist <= 10 and best_target is not None and best_current is not None:
                # Current color at this probe position
                actual_current = int(base_frame[py, px]) if base_frame.ndim == 2 else int(base_frame[0, py, px])
                # If researcher says this cell should be different from current
                if best_current != best_target:
                    predicted_combo += cycle_period ** b_idx
                    matched += 1
        if verbose:
            if matched > 0:
                print(f"  [oracle] Researcher prediction: {matched}/{n_blocks} cells should toggle")
            else:
                print(f"  [oracle] Researcher prediction: 0 matches (oracle positions vs researcher positions)")
                if predicted_target_map:
                    r_positions = list(predicted_target_map.keys())[:3]
                    o_positions = [cell_positions[block[0]] for block in blocks[:3]]
                    print(f"  [oracle]   Researcher positions: {r_positions}")
                    print(f"  [oracle]   Oracle positions: {o_positions}")

    if predicted_combo is not None and predicted_combo > 0:
        # Hamming-distance search around the prediction
        max_hamming = min(5, n_blocks)
        if verbose:
            pred_toggles = [b_idx for b_idx in range(n_blocks)
                           if (predicted_combo // (cycle_period ** b_idx)) % cycle_period > 0]
            print(f"  [oracle] LLM predicted {len(pred_toggles)} toggles, Hamming search d=0..{max_hamming}")

        import time as _time_mod
        _llm_ham_start = _time_mod.time()
        _llm_ham_timeout = 15
        total_tests = 0
        # d=0: try the prediction itself
        result = _try_combo(predicted_combo)
        total_tests += 1
        if result is not None:
            if verbose:
                print(f"  [oracle] SOLVED with prediction! {len(result)} clicks")
            return result

        # d=1..max_hamming: flip bits
        _llm_ham_done = False
        for d in range(1, max_hamming + 1):
            for flip_set in _combs(range(n_blocks), d):
                test_combo = predicted_combo
                for b_idx in flip_set:
                    bit_val = (test_combo // (cycle_period ** b_idx)) % cycle_period
                    # Flip: 0→1 or 1→0
                    new_val = (1 - bit_val) if cycle_period == 2 else ((bit_val + 1) % cycle_period)
                    test_combo += (new_val - bit_val) * (cycle_period ** b_idx)
                if test_combo <= 0:
                    continue
                result = _try_combo(test_combo)
                total_tests += 1
                if result is not None:
                    if verbose:
                        print(f"  [oracle] SOLVED with {len(result)} clicks (Hamming d={d}, test {total_tests})")
                    return result
                if total_tests % 500 == 0 and _time_mod.time() - _llm_ham_start > _llm_ham_timeout:
                    _llm_ham_done = True
                    break
            if _llm_ham_done:
                break

        if verbose:
            elapsed = _time_mod.time() - _llm_ham_start
            print(f"  [oracle] Hamming search: {total_tests} tests in {elapsed:.1f}s ({'timeout' if _llm_ham_done else 'exhausted'})")

    # For large spaces without a prediction, give up early — let LLM handle it
    if verbose:
        print(f"  [oracle] Space too large ({total_combos}) and no usable prediction — deferring to LLM")
    return None


def _ask_researcher_for_level(mind, env, frame, click_action_idx, obs, verbose=True):
    """Quick LLM call to predict target colors for the current level's grid.
    Sets mind._researcher_target_map and mind._researcher_current_map for oracle use."""
    import copy as _cp
    click_act = obs.available_actions[click_action_idx]
    # Use already-discovered cell positions if available (from oracle probing)
    cell_positions = getattr(mind, '_oracle_cell_positions', None)
    if not cell_positions:
        # Fallback: discover clickable positions with larger step
        _probe_step = 8  # coarser than oracle to avoid flooding
        h = frame.shape[-2] if frame.ndim == 3 else frame.shape[0]
        w = frame.shape[-1] if frame.ndim == 3 else frame.shape[1]
        cell_positions = []
        for py in range(_probe_step // 2, h, _probe_step):
            for px in range(_probe_step // 2, w, _probe_step):
                probe_env = _cp.deepcopy(env)
                probe_obs = probe_env.step(click_act, {"x": px, "y": py})
                pf = np.array(probe_obs.frame) if hasattr(probe_obs, 'frame') else np.array(probe_obs['frame'])
                bf = frame[0] if frame.ndim == 3 else frame
                pf2 = pf[0] if pf.ndim == 3 else pf
                diff = np.where(bf != pf2)
                if len(diff[0]) > 2:  # significant change
                    cell_positions.append((px, py))
    if not cell_positions or len(cell_positions) > 50:
        if verbose:
            print(f"  [researcher-level] Aborting: {len(cell_positions) if cell_positions else 0} cells found (need 1-50)")
        return  # too few or too many cells
    # Read current colors at each cell position
    bf = frame[0] if frame.ndim == 3 else frame
    current_colors = [int(bf[py, px]) for px, py in cell_positions]
    n = len(cell_positions)
    # Build prompt with visual context
    cycle = getattr(mind, '_color_cycle', None)
    period = cycle['period'] if cycle else 2
    # Include ASCII frame for visual reasoning
    # Downsample 64x64 → 32x32 by picking every 2nd pixel (higher detail for constraint cells)
    ascii_rows = []
    h, w = bf.shape
    step = max(h // 32, 1)
    for ry in range(0, h, step):
        row_chars = []
        for rx in range(0, w, step):
            c = bf[ry, rx]
            row_chars.append(f"{c:x}" if c < 16 else str(c % 10))
        ascii_rows.append("".join(row_chars))
    ascii_grid = "\n".join(ascii_rows)
    # Mark cell positions in the prompt
    cell_info = ", ".join(f"({px},{py})=c{current_colors[i]}" for i, (px, py) in enumerate(cell_positions))
    # Include grid detection if available
    grids = getattr(mind, '_detected_grids', [])
    grid_desc = ""
    if grids:
        for gi, g in enumerate(grids):
            cells_in_grid = g.get('cells', [])
            if cells_in_grid:
                fills = [c.get('fill', '?') for c in cells_in_grid[:9]]
                embeds = [c.get('embedded_sprite', '') for c in cells_in_grid[:9] if c.get('embedded_sprite')]
                grid_desc += f"\nGrid {gi}: {g.get('rows','?')}x{g.get('cols','?')} cells, fills={fills}"
                if embeds:
                    grid_desc += f", embedded_sprites={embeds[:5]}"
    # Also identify constraint cells (non-toggling cells whose colors encode the target)
    constraint_info = ""
    if hasattr(mind, '_oracle_cell_positions') and mind._oracle_cell_positions:
        oracle_pos = set((px, py) for px, py in mind._oracle_cell_positions)
        # Find non-toggling cells (constraint/pattern cells)
        constraint_cells = []
        for py in range(0, h, 2):
            for px in range(0, w, 2):
                if (px, py) not in oracle_pos and bf[py, px] != bf[0, 0]:  # not background, not clickable
                    constraint_cells.append((px, py, int(bf[py, px])))
        if constraint_cells:
            constraint_info = f"\nConstraint/pattern cells (non-clickable, encode target): {constraint_cells[:30]}\n"
    prompt = f"Toggle puzzle analysis. {n} clickable cells, color cycle period={period}.\n"
    prompt += f"Frame (32x32 downsampled, hex colors):\n{ascii_grid}\n\n"
    prompt += f"Clickable cells: {cell_info}\n"
    if grid_desc:
        prompt += f"Grid detection:{grid_desc}\n"
    if constraint_info:
        prompt += constraint_info
    prompt += f"\nSome cells may be CONSTRAINT/PATTERN cells that encode the target state.\n"
    prompt += f"Look at the frame for cells with multi-color interiors — they define what neighboring cells should become.\n"
    prompt += f"What should the TARGET state of the {n} clickable cells be?\n"
    prompt += f"IMPORTANT: Reply with ONLY this format: TARGET: [c0, c1, c2, ...] — exactly {n} integers, one per cell.\n"
    response = call_llm(
        messages=[{"role": "user", "content": prompt}],
        max_tokens=500, temperature=0.0,
        fast=False,  # Researcher target prediction needs Sonnet reasoning
    )
    mind.total_llm_calls += 1
    if not response:
        return
    import re
    match = re.search(r'TARGET:\s*\[([^\]]+)\]', response)
    if not match:
        nums = re.findall(r'\b(\d+)\b', response)
        if len(nums) >= n:
            target_state = [int(x) for x in nums[:n]]
        else:
            return
    else:
        target_state = [int(x.strip()) for x in match.group(1).split(',')]
    if len(target_state) != n:
        return
    if verbose:
        toggles = sum(1 for i in range(n) if target_state[i] != current_colors[i])
        print(f"  [researcher] Level re-analysis: {n} cells, {toggles} need toggling")
    mind._researcher_target_map = {}
    mind._researcher_current_map = {}
    for i, (px, py) in enumerate(cell_positions):
        mind._researcher_target_map[(px, py)] = target_state[i]
        mind._researcher_current_map[(px, py)] = current_colors[i]
    if verbose:
        print(f"  [researcher] Stored targets at positions: {list(mind._researcher_target_map.keys())}")


def _recall_game_knowledge(game_id: str, oprah_genre: str = "") -> list:
    """Recall knowledge from past sessions for a specific game.

    Loads from game_knowledge.json (persistent, editable without code changes).
    Falls back to hardcoded rules if file not found.

    When game_id doesn't match a known game, uses oprah_genre to load
    genre-specific rules (e.g. _genre_navigation) PLUS _default rules.
    This is the bridge between OPRAH classification and game knowledge.
    """
    # Try loading from JSON file first
    knowledge_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'game_knowledge.json')
    try:
        with open(knowledge_path, 'r') as f:
            all_knowledge = json.load(f)
        # Match game_id against keys (case-insensitive substring match)
        gid_lower = game_id.lower()
        for key, entry in all_knowledge.items():
            if key.startswith('_'):
                continue  # skip meta entries like _default
            if key.lower() in gid_lower:
                return entry.get('rules', [])

        # No game-specific match — try genre-specific rules from OPRAH
        rules = []
        if oprah_genre:
            # Map OPRAH genre names to game_knowledge genre keys
            genre_map = {
                'navigation_maze': '_genre_navigation',
                'toggle_puzzle': '_genre_toggle',
                'constraint_satisfaction': '_genre_toggle',
                'push_block': '_genre_sorting',
                'sliding_tile': '_genre_sorting',
                'sorting': '_genre_sorting',
                'circuit_puzzle': '_genre_sequence',
                'multi_phase': '_genre_sequence',
                'paint_fill': '_genre_toggle',
                'cellular_automaton': '_genre_toggle',
                'gravity_puzzle': '_genre_gravity',
                'rotation_puzzle': '_genre_rotation',
                'connection_flow': '_genre_connection',
                'pattern_replication': '_genre_pattern_copy',
                'pursuit_evasion': '_genre_chase',
                'chase': '_genre_chase',
                'pump_overflow': '_genre_sequence',
                'matching': '_genre_matching',
                'memory_game': '_genre_matching',
                'copy_pattern': '_genre_pattern_copy',
                'state_machine': '_genre_state_machine',
                'finite_automaton': '_genre_state_machine',
                'sequence_puzzle': '_genre_sequence',
            }
            genre_key = genre_map.get(oprah_genre, '')
            if genre_key and genre_key in all_knowledge:
                rules.extend(all_knowledge[genre_key].get('rules', []))

        # Always append _default rules (general heuristics)
        if '_default' in all_knowledge:
            for r in all_knowledge['_default'].get('rules', []):
                if r not in rules:
                    rules.append(r)

        return rules
    except (FileNotFoundError, json.JSONDecodeError):
        pass

    # Fallback: hardcoded rules (kept for Docker builds without the JSON)
    if 'ls20' in game_id.lower():
        return [
            "Player = color 12 (head) + color 9 (body). They move TOGETHER as one entity.",
            "Color 3 (lighter gray) = walkable CORRIDORS. Darker colors = WALLS.",
            "Some positions cause BIG world changes (100+ pixels) = MODIFIERS that change your status.",
            "HUD (bottom rows) changes = STATUS changed. Track what colors appeared/disappeared.",
            "Teleported to start with full fuel = FUEL EXHAUSTION RESET, not game over.",
            "STRATEGY: Explore everything. Positions with big changes are the key to progress.",
        ]

    return []


def _recall_game_hypothesis(game_id: str) -> dict:
    """Load pre-seeded hypotheses from game_knowledge.json."""
    knowledge_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'game_knowledge.json')
    try:
        with open(knowledge_path, 'r') as f:
            all_knowledge = json.load(f)
        return {key: entry['hypothesis'] for key, entry in all_knowledge.items()
                if 'hypothesis' in entry}
    except (FileNotFoundError, json.JSONDecodeError):
        pass
    # Fallback
    return {
        'ls20': "Complex navigation puzzle with modifiers. Stepping on certain positions causes big changes (HUD updates). Explore everything.",
    }


def run_game(game_id: str, max_turns_per_level: int = 50,
             budget_per_level: int = 15, verbose: bool = True,
             env_override=None):
    """Drop the Gundam into a game it has never seen. Watch it learn."""
    if env_override is not None:
        env = env_override
    else:
        import arc_agi
        arcade = arc_agi.Arcade()
        env = arcade.make(game_id)

    import copy as _copy

    def _extract_frame(obs_or_env):
        """Extract a single 2D frame from observation or env, handling multi-frame."""
        if hasattr(obs_or_env, 'frame'):
            f = np.array(obs_or_env.frame)
        elif hasattr(obs_or_env, '_last_response') and hasattr(obs_or_env._last_response, 'frame'):
            f = np.array(obs_or_env._last_response.frame)
        else:
            raise AttributeError(f"Cannot extract frame from {type(obs_or_env).__name__}")
        if f.ndim == 3 and f.shape[0] > 1:
            # Multi-frame animation — return LAST frame (actual game state).
            # Probes return single-frame obs, so base must match.
            # If caller needs consistency, step with noop first.
            return f[-1]
        return np.squeeze(f)

    # Reset per-game cost tracking (competition: each game gets fresh budget)
    global _api_cost_usd, _api_calls, _api_input_tokens, _api_output_tokens, _api_consecutive_errors
    _api_cost_usd = 0.0
    _api_calls = 0
    _api_input_tokens = 0
    _api_output_tokens = 0
    _api_consecutive_errors = 0

    _text_only = bool(os.environ.get('ARC_TEXT_ONLY'))
    mind = Gundam(budget_per_level=budget_per_level, verbose=verbose, text_only=_text_only)
    mind._last_progress_at_call = 0  # track LLM call count at last level solve
    mind.memory.game_id = game_id

    obs = env.reset()

    # --- Action index → GameAction translation wrapper ---
    # arcengine's env.step() expects a GameAction enum, not an integer index.
    # Gundam uses integer indices (0, 1, 2...) into obs.available_actions.
    # Without this wrapper, step(0) sends GameAction.RESET instead of the
    # first available action (e.g., ACTION6 for click games like VC33).
    _available_actions = list(obs.available_actions)
    try:
        from arcengine import GameAction as _GA
        class _ActionTranslatingEnv:
            """Wraps env to translate action indices → GameAction enums."""
            def __init__(self, real_env, actions):
                self._real = real_env
                self._actions = actions
            def step(self, action_val, data=None):
                # Gundam always passes raw action VALUES from obs.available_actions,
                # never indices. Convert int → GameAction enum for arcengine.
                if isinstance(action_val, int):
                    ga = _GA.from_id(action_val) if hasattr(_GA, 'from_id') else _GA['ACTION' + str(action_val)]
                else:
                    ga = action_val
                return self._real.step(ga, data) if data is not None else self._real.step(ga)
            def reset(self):
                return self._real.reset()
            def __deepcopy__(self, memo):
                import copy as _cp
                return _ActionTranslatingEnv(_cp.deepcopy(self._real, memo), self._actions[:])
            def __getattr__(self, name):
                return getattr(self._real, name)
        env = _ActionTranslatingEnv(env, _available_actions)
    except ImportError:
        pass  # arcengine not available — env.step() must accept ints directly

    frame = _extract_frame(obs)
    n_actions = len(obs.available_actions)
    total_levels = obs.win_levels

    # Build action info — tell Gundam what each action index does
    # Map: index 0..n → actual game action IDs
    # Discover action types by probing — don't hardcode game_ids
    # IMPORTANT: probe with TWO DIFFERENT positions (corner vs center),
    # because no-data defaults to (0,0) which is the same as {"x":0,"y":0}
    action_info = {}

    # Probe positions: spread across the frame to find clickable regions
    # Include edge positions (2, 62) to catch sprites near frame boundaries
    h_frame, w_frame = frame.shape[-2], frame.shape[-1]
    _probe_positions = [(2, 2), (8, 8), (24, 8), (40, 8), (56, 8), (w_frame-2, 8),
                        (2, 24), (8, 24), (24, 24), (40, 24), (56, 24), (w_frame-2, 24),
                        (2, 40), (8, 40), (24, 40), (40, 40), (56, 40), (w_frame-2, 40),
                        (2, 56), (8, 56), (24, 56), (40, 56), (56, 56), (w_frame-2, 56),
                        (2, h_frame-2), (w_frame-2, h_frame-2)]
    for i, act in enumerate(obs.available_actions):
        is_position_dependent = False
        # Compare pairs of positions — if ANY pair gives different results, it's a click action
        ref_env = _copy.deepcopy(env)
        ref_obs = ref_env.step(act, {"x": _probe_positions[0][0], "y": _probe_positions[0][1]})
        ref_frame = _extract_frame(ref_obs)
        for px, py in _probe_positions[1:]:
            test_env = _copy.deepcopy(env)
            try:
                test_obs = test_env.step(act, {"x": px, "y": py})
                test_frame = _extract_frame(test_obs)
                if not np.array_equal(ref_frame, test_frame):
                    is_position_dependent = True
                    break
            except Exception:
                continue
        if is_position_dependent:
            action_info[i] = f"index {i}: CLICK at position — accepts DATA: {{\"x\": px, \"y\": py}} (game_id={act})"
            mind._click_action_idx = i
            mind._interactive_positions = []  # will be populated below

            # === COLOR CYCLE DETECTION ===
            # Click the same cell repeatedly to discover the toggle cycle period.
            # Use positions that actually caused changes during PD detection.
            # The coarse probe grid may miss interactive sprites entirely.
            _cycle_probe_pos = []
            for _cpx, _cpy in _probe_positions:
                _ce = _copy.deepcopy(env)
                try:
                    _co = _ce.step(act, {"x": _cpx, "y": _cpy})
                    _cf = _extract_frame(_co)
                    if int(np.sum(frame != _cf)) > 2:
                        _cycle_probe_pos.append((_cpx, _cpy))
                except Exception:
                    continue
            if not _cycle_probe_pos:
                # Fallback: scan at step=4 near edges to find an interactive position
                for _cpy in range(2, h_frame, 4):
                    for _cpx in range(2, w_frame, 4):
                        _ce = _copy.deepcopy(env)
                        try:
                            _co = _ce.step(act, {"x": _cpx, "y": _cpy})
                            _cf = _extract_frame(_co)
                            if int(np.sum(frame != _cf)) > 4:
                                _cycle_probe_pos.append((_cpx, _cpy))
                                if len(_cycle_probe_pos) >= 3:
                                    break
                        except Exception:
                            continue
                    if len(_cycle_probe_pos) >= 3:
                        break
            # Store interactive positions for LLM context
            mind._interactive_positions = _cycle_probe_pos[:10]  # cap at 10
            cycle_info = _detect_color_cycle(env, act,
                                             _cycle_probe_pos if _cycle_probe_pos else _probe_positions,
                                             _extract_frame, frame)
            if cycle_info:
                mind._color_cycle = cycle_info
                if verbose:
                    print(f"  [gundam] Color cycle detected: period={cycle_info['period']}, "
                          f"colors={cycle_info['colors']}")
        else:
            action_info[i] = f"index {i}: action (game_id={act})"

    # === OPRAH: ENVIRONMENT PROFILING ===
    # Build position-dependent map from existing probing for OPRAH reuse
    _pos_dep_map = {}
    for i in range(n_actions):
        _pos_dep_map[i] = (f"CLICK" in action_info.get(i, ""))

    # Phase 2: PROBE — classify all actions by type
    _oprah_probes = _oprah_probe_env(env, obs, _extract_frame, frame,
                                      position_dependent=_pos_dep_map,
                                      verbose=verbose)

    # Phase 1.5: Quick grid check — look for repeating divider lines in frame
    # Full grid detection runs later in researcher mode, but we need a signal now
    _has_grid = bool(getattr(mind, '_detected_grids', None))
    if not _has_grid:
        # Quick heuristic: check if frame has regular horizontal/vertical lines
        # of a single color (grid dividers). Check for >3 equally-spaced rows.
        f2d = frame[0] if frame.ndim == 3 else frame
        _row_uniformity = []
        for y in range(f2d.shape[0]):
            row = f2d[y, :]
            if len(set(row.tolist())) <= 2:  # row has ≤2 colors = possible divider
                _row_uniformity.append(y)
        if len(_row_uniformity) >= 4:
            # Check if spacing is regular (within ±2px)
            _gaps = [_row_uniformity[i+1] - _row_uniformity[i]
                     for i in range(len(_row_uniformity)-1)]
            _gaps_filtered = [g for g in _gaps if g > 2]  # skip adjacent rows
            if _gaps_filtered:
                _median_gap = sorted(_gaps_filtered)[len(_gaps_filtered)//2]
                _regular = sum(1 for g in _gaps_filtered if abs(g - _median_gap) <= 2)
                if _regular >= 2:
                    _has_grid = True

    # Phase 3: REASON — infer genre from probe results
    _has_click = any(p.action_type == ActionType.PARAMETERIZED for p in _oprah_probes)
    _oprah_genre, _oprah_conf = _oprah_infer_genre(
        _oprah_probes, has_grid=_has_grid, has_click=_has_click)

    # Enrich action_info with OPRAH findings (AFTER genre inference which may reclassify probes)
    for p in _oprah_probes:
        if p.action_type == ActionType.PARAMETERIZED:
            continue  # already has CLICK description
        act = obs.available_actions[p.action_idx]
        _type_label = p.action_type.value
        _parts = [f"index {p.action_idx}: {_type_label.upper()}"]
        if p.action_type == ActionType.MOVEMENT:
            _dir_note = p.notes
            if _dir_note and _dir_note.startswith("direction:"):
                try:
                    _dinfo = eval(_dir_note[len("direction:"):])
                    if _dinfo and len(_dinfo) >= 3:
                        _parts.append(f"→ {_dinfo[2]}")
                except Exception:
                    pass
            _parts.append(f"({p.pixels_changed}px change)")
        elif p.action_type == ActionType.TOGGLE:
            _parts.append(f"({p.pixels_changed}px change)")
            if p.is_self_inverse:
                _parts.append("[self-inverse: clicking twice undoes it]")
        elif p.action_type == ActionType.GLOBAL:
            _parts.append(f"({p.pixels_changed}px change, affects wide area)")
        elif p.action_type == ActionType.NO_OP:
            _parts.append("(no visible effect)")
        _parts.append(f"(game_id={act})")
        action_info[p.action_idx] = " ".join(_parts)

    # Extract direction map from OPRAH probes (sprite tracking results)
    _direction_map = {}
    _player_color = -1
    _player_start = None
    for p in _oprah_probes:
        if p.notes and p.notes.startswith("direction:"):
            try:
                info = eval(p.notes[len("direction:"):])  # (dy, dx, dir, color, (y,x))
                if info and len(info) == 5:
                    dy, dx, direction, color, pos = info
                    _direction_map[p.action_idx] = (dy, dx, direction)
                    if _player_color < 0:
                        _player_color = color
                        _player_start = pos
            except Exception:
                pass

    # Build environment profile and store on mind for downstream use
    mind._env_profile = EnvironmentProfile(
        game_id=game_id,
        frame_shape=frame.shape,
        n_actions=n_actions,
        action_probes=_oprah_probes,
        genre=_oprah_genre,
        genre_confidence=_oprah_conf,
        movement_actions=[p.action_idx for p in _oprah_probes if p.action_type == ActionType.MOVEMENT],
        direction_map=_direction_map,
        player_color=_player_color,
        player_start_pos=_player_start,
        toggle_actions=[p.action_idx for p in _oprah_probes if p.action_type == ActionType.TOGGLE],
        global_actions=[p.action_idx for p in _oprah_probes if p.action_type == ActionType.GLOBAL],
        param_actions=[p.action_idx for p in _oprah_probes if p.action_type == ActionType.PARAMETERIZED],
        has_grid=_has_grid,
        has_click=_has_click,
    )

    # Seed Gundam's spatial model from OPRAH direction findings
    # Sanity check: if all movement actions go the same direction, it's likely
    # false-positive sprite tracking (HUD noise), not real movement
    if _direction_map:
        _dirs = set(d for _, _, d in _direction_map.values())
        _is_real_nav = len(_dirs) >= 2  # real navigation has at least 2 distinct directions
        if _is_real_nav:
            for act_idx, (dy, dx, direction) in _direction_map.items():
                mind.spatial.action_map[act_idx] = (dy, dx, direction)
            # Infer missing cardinal direction from blocked actions
            _found_dirs = {d: (act, dy, dx) for act, (dy, dx, d) in _direction_map.items()}
            _all_cardinals = {'UP': (-5, 0), 'DOWN': (5, 0), 'LEFT': (0, -5), 'RIGHT': (0, 5)}
            _movement_acts = set(_direction_map.keys())
            if len(_found_dirs) == 3:
                for dname, (dy, dx) in _all_cardinals.items():
                    if dname not in _found_dirs:
                        # Find an action that was probed but not mapped (likely blocked)
                        for p in _oprah_probes:
                            if p.action_idx not in _movement_acts and p.pixels_changed > 0 and p.pixels_changed < 10:
                                mind.spatial.action_map[p.action_idx] = (dy, dx, dname)
                                if verbose:
                                    print(f"  [oprah] Inferred blocked action {p.action_idx} = {dname} (was {p.pixels_changed}px HUD-only)")
                                break
            if _player_color >= 0 and mind.spatial.player_color < 0:
                mind.spatial.player_color = _player_color
                if _player_start:
                    mind.spatial.player_pos = _player_start
            # Seed target: farthest small non-player, non-wall object
            if verbose:
                print(f"  [oprah] Target check: target_pos={mind.spatial.target_pos}, objects={len(mind.spatial.objects)}, player_color={_player_color}")
            if mind.spatial.target_pos == (-1, -1) and mind.spatial.objects:
                wall_colors = getattr(mind.spatial, 'wall_colors', set())
                scenery = getattr(mind.spatial, 'scenery_colors', set())
                _cands = [
                    (c, y, x, r) for c, y, x, r in mind.spatial.objects
                    if c != _player_color and c not in wall_colors
                    and ('small' in r or 'button' in r.lower())
                ]
                if verbose:
                    print(f"  [oprah] Target candidates: {len(_cands)} (wall_colors={wall_colors})")
                    for c in _cands[:5]:
                        print(f"    candidate: color {c[0]} at ({c[1]},{c[2]}) role={c[3]}")
                if _cands and _player_start:
                    py, px = _player_start
                    _cands.sort(key=lambda o: -(abs(o[1]-py) + abs(o[2]-px)))
                    tc, ty, tx, _ = _cands[0]
                    mind.spatial.target_pos = (ty, tx)
                    mind.spatial.target_color = tc
                    if verbose:
                        print(f"  [oprah] Target seeded: color {tc} at ({ty},{tx}) (farthest small object)")
            if verbose:
                print(f"  [oprah→gundam] Direction map: {_direction_map}, player=color {_player_color}")
        elif verbose:
            print(f"  [oprah] Direction map rejected — all directions same ({_dirs}), likely HUD noise")

    if verbose:
        print(f"  [oprah] Genre: {_oprah_genre} (confidence={_oprah_conf:.0%})")
        if mind._env_profile.movement_actions:
            print(f"  [oprah] Movement actions: {mind._env_profile.movement_actions}")
        if mind._env_profile.param_actions:
            print(f"  [oprah] Click actions: {mind._env_profile.param_actions}")

    mind.memory.levels_total = total_levels
    level = obs.levels_completed
    mind._current_level = level

    # ── Recall past knowledge ──
    # Inject knowledge from previous sessions — what past forks learned hands-on.
    # This is the pilot's long-term memory, encoded from 14+ forks of gameplay.
    _recalled_rules = _recall_game_knowledge(game_id, oprah_genre=_oprah_genre)
    if _recalled_rules:
        for rule in _recalled_rules:
            if rule not in mind.memory.rules_discovered:
                mind.memory.rules_discovered.append(rule)
        if verbose:
            _src = game_id if any(k for k in ('ls20', 'ft09', 'vc33') if k in game_id.lower()) else f"genre:{_oprah_genre}"
            print(f"  [recall] Loaded {len(_recalled_rules)} rules (source: {_src})")

        # Pre-seed game hypothesis from recalled knowledge (JSON file or fallback)
        _hypotheses = _recall_game_hypothesis(game_id)
        _hyp_seeded = False
        for key, hyp in _hypotheses.items():
            if key in game_id.lower() and not mind.memory.game_hypothesis:
                mind.memory.game_hypothesis = hyp
                mind.memory.hypothesis_locked = True
                _hyp_seeded = True
                if verbose:
                    print(f"  [recall] Pre-seeded hypothesis (LOCKED): {hyp[:80]}...")

        # For unknown games: seed genre hypothesis (UNLOCKED — pilot can revise)
        if not _hyp_seeded and _oprah_genre and _oprah_genre != 'novel_unknown':
            _genre_hyp_map = {
                'navigation_maze': '_genre_navigation',
                'toggle_puzzle': '_genre_toggle',
                'constraint_satisfaction': '_genre_toggle',
                'push_block': '_genre_sorting',
                'sliding_tile': '_genre_sorting',
                'sorting': '_genre_sorting',
                'circuit_puzzle': '_genre_sequence',
                'multi_phase': '_genre_sequence',
                'gravity_puzzle': '_genre_gravity',
                'rotation_puzzle': '_genre_rotation',
                'connection_flow': '_genre_connection',
                'pattern_replication': '_genre_toggle',
                'pursuit_evasion': '_genre_navigation',
                'pump_overflow': '_genre_sequence',
                'sequence_puzzle': '_genre_sequence',
            }
            _gk = _genre_hyp_map.get(_oprah_genre, '')
            if _gk and _gk in _hypotheses and not mind.memory.game_hypothesis:
                mind.memory.game_hypothesis = _hypotheses[_gk]
                mind.memory.hypothesis_locked = False  # pilot can revise
                if verbose:
                    print(f"  [recall] Genre hypothesis (unlocked): {mind.memory.game_hypothesis[:80]}...")

    if verbose:
        print(f"\n{'='*60}")
        print(f"  GUNDAM vs {game_id}")
        print(f"  {n_actions} actions | {total_levels} levels")
        print(f"  Genre: {_oprah_genre} ({_oprah_conf:.0%} confidence)")
        print(f"{'='*60}\n")

    lvl = mind.observe_initial(frame, n_actions, action_info, level)
    prev_levels_completed = obs.levels_completed
    _prev_level_frame = frame.copy()  # first frame of level 0 — for between-level scenery diff

    # Post-observe target seeding for navigation games — build ranked candidate list
    _nav_target_candidates = []
    if mind.spatial.objects and _direction_map:
        wall_colors = getattr(mind.spatial, 'wall_colors', set())
        _cands = [
            (c, y, x, r) for c, y, x, r in mind.spatial.objects
            if c != mind.spatial.player_color and c not in wall_colors
            and ('small' in r or 'button' in r.lower())
        ]
        if _cands:
            py, px = mind.spatial.player_pos
            # Score: prefer clusters (multiple objects of same color), then distance
            from collections import Counter
            _color_counts = Counter(c for c, _, _, _ in _cands)
            _cands.sort(key=lambda o: (-_color_counts[o[0]], -(abs(o[1]-py) + abs(o[2]-px))))
            _nav_target_candidates = [(y, x) for _, y, x, _ in _cands]
            # Set first candidate as target
            tc, ty, tx, _ = _cands[0]
            mind.spatial.target_pos = (ty, tx)
            mind.spatial.target_color = tc
            if verbose:
                print(f"  [target-seed] {len(_cands)} candidates, best: color {tc} at ({ty},{tx}) (cluster={_color_counts[tc]})")
                for c, y, x, r in _cands[:5]:
                    print(f"    color {c} at ({y},{x}) cluster={_color_counts[c]}")

    # PREFLIGHT: Fresh eyes. No assumptions. Discover before solving.
    # This burns a few actions but prevents the pilot from inheriting
    # wrong assumptions that cost 5+ forks to debug (LS20 lesson).
    try:
        preflight_report = mind.preflight(env, _extract_frame, level)
        if verbose:
            n_obj = len(preflight_report.get('objects', []))
            n_eff = sum(1 for e in preflight_report.get('action_effects', {}).values()
                        if e.get('pixels_changed', 0) > 0)
            print(f"  [preflight] {n_obj} objects found, {n_eff}/{len(preflight_report.get('action_effects', {}))} actions have effects")
        # Re-grab frame after preflight actions changed state
        frame = _extract_frame(env)
        lvl.current_frame = frame[0].copy() if frame.ndim == 3 else frame.copy()
        lvl.current_frame_b64 = frame_to_b64(frame)
    except Exception as e:
        if verbose:
            print(f"  [preflight] Failed (non-fatal): {e}")
        mind._preflight_report = None

    # RECALL: what do we know about games like this?
    recalled = mind.recall(game_id)
    mind._recalled_memories = recalled  # store for prompt access
    if recalled and verbose:
        print(f"  [gundam] Recalled {len(recalled)} relevant memories")
        for rel_type, mem in recalled[:3]:
            print(f"    [{rel_type}] {mem['game_id']}: {mem.get('levels_solved',0)} solved")

    # === CNN COLD-START EXPLORATION ===
    # Competition intel: CNN-RL won ARC preview (18 levels) with ~200 exploration
    # actions per level. LLM-only agents crashed and scored poorly. Our key risk
    # is calling the expensive LLM before we have enough data for it to reason.
    # This phase does pure autonomous exploration — observe effects, build spatial
    # model, discover mechanics — BEFORE burning any LLM budget.
    # Only for unknown games (known games already have seeded rules).
    _is_known_game = any(k in game_id.lower() for k in ('ls20', 'ft09', 'vc33'))
    done = False  # track game completion across cold-start, brute-probe, and main loop
    _cold_start_base = int(os.environ.get('ARC_COLD_START', '60'))
    # Active inference: epistemic value scales inversely with model confidence.
    # Low OPRAH confidence = high uncertainty = explore more before committing LLM.
    _confidence_scale = max(0.5, 2.0 - _oprah_conf * 2)  # conf=0→2x, conf=0.5→1x, conf=1→0.5x
    _cold_start_budget = int(_cold_start_base * _confidence_scale)
    # Skip cold-start for deduction/sequence puzzles where random actions don't teach anything
    _skip_cold_start = _oprah_genre in ('constraint_satisfaction', 'sequence_puzzle') and \
                        not any(p.action_type == ActionType.MOVEMENT for p in _oprah_probes)
    if not _is_known_game and _cold_start_budget > 0 and n_actions >= 2 and not _skip_cold_start:
        _cs_actions = 0
        _cs_solved = False
        _cs_frame = frame.copy()

        # Strategy: cycle through all actions, observe what each does.
        # For navigation games: try each direction multiple times to map the space.
        # For toggle/click games: try each action once to see its effect.
        _movement_acts = mind._env_profile.movement_actions if hasattr(mind, '_env_profile') else []
        _other_acts = [i for i in range(n_actions) if i not in _movement_acts]

        # Florence structured perception BEFORE blind exploration
        _cs_click_targets = []
        _cs_vlm_struct = None
        _cs_vlm_text = ""
        if HAS_VLM:
            try:
                _cs_vlm_struct = arc_vlm.perceive_structured(frame)
                _cs_vlm_text = _cs_vlm_struct.get('caption', '')
                _cs_click_targets = list(_cs_vlm_struct.get('click_targets', []))
                _vlm_genre = _cs_vlm_struct.get('genre_hint', 'unknown')
                if verbose:
                    print(f"  [florence] Scene: {_cs_vlm_text[:100]}")
                    print(f"  [florence] {len(_cs_vlm_struct.get('objects', []))} objects, "
                          f"genre_hint={_vlm_genre}, {len(_cs_click_targets)} click targets")
                    if _cs_vlm_struct.get('text'):
                        print(f"  [florence] Text: {_cs_vlm_struct['text'][:80]}")
                # Update genre if Florence gives a strong hint
                if _vlm_genre != 'unknown' and hasattr(mind, '_env_profile'):
                    mind._env_profile.genre = _vlm_genre
                    mind._env_profile.genre_confidence = max(
                        mind._env_profile.genre_confidence, 0.5)
                    if verbose:
                        print(f"  [florence] Updated genre → {_vlm_genre} (conf≥0.5)")
            except Exception as e:
                if verbose:
                    print(f"  [cold-start] Florence failed (non-fatal): {e}")

        if verbose:
            print(f"\n  [cold-start] Exploring {_cold_start_budget} actions (base={_cold_start_base}, "
                  f"conf={_oprah_conf:.0%}→scale={_confidence_scale:.1f}x) before first LLM call...")

        _cs_state_history = {}  # frame_hash -> set of actions tried in that state
        _cs_repeat_count = 0

        while _cs_actions < _cold_start_budget:
            # State-aware action selection: avoid repeating actions in same state
            _cs_frame_hash = hash(_cs_frame.tobytes()) if hasattr(_cs_frame, 'tobytes') else hash(str(_cs_frame))
            _cs_tried = _cs_state_history.get(_cs_frame_hash, set())

            # Pick action: prefer UNTRIED actions in current state, CNN-guided after initial survey
            _cs_untried_move = [a for a in _movement_acts if a not in _cs_tried]
            _cs_untried_other = [a for a in _other_acts if a not in _cs_tried]
            _cs_untried_any = [a for a in range(n_actions) if a not in _cs_tried]

            if _cs_untried_any:
                _cs_repeat_count = 0
                if _cs_untried_move and _cs_actions % 3 != 0:
                    act_idx = _cs_untried_move[_cs_actions % len(_cs_untried_move)]
                elif _cs_untried_other:
                    act_idx = _cs_untried_other[_cs_actions % len(_cs_untried_other)]
                else:
                    act_idx = _cs_untried_any[0]
            elif _cs_actions >= n_actions * 2 and mind.action_predictor:
                # CNN-guided: after trying all actions at least twice, use predictor
                _cs_repeat_count = 0
                _cnn_preds = mind.predict_best_actions(_cs_frame, top_k=n_actions)
                if _cnn_preds:
                    act_idx = _cnn_preds[0][0]  # highest-confidence action
                else:
                    act_idx = _cs_actions % n_actions
            else:
                # All actions tried in this state — we're cycling
                _cs_repeat_count += 1
                if _cs_repeat_count > n_actions * 2:
                    if verbose:
                        print(f"  [cold-start] Stuck — all actions exhausted in {len(_cs_state_history)} states")
                    break
                if _movement_acts and _cs_actions % 3 != 0:
                    act_idx = _movement_acts[_cs_actions % len(_movement_acts)]
                elif _other_acts:
                    act_idx = _other_acts[_cs_actions % len(_other_acts)]
                else:
                    act_idx = _cs_actions % n_actions

            # Track which actions we've tried in this state
            if _cs_frame_hash not in _cs_state_history:
                _cs_state_history[_cs_frame_hash] = set()
            _cs_state_history[_cs_frame_hash].add(act_idx)

            actual_action = obs.available_actions[act_idx]
            act_desc = action_info.get(act_idx, f"action {act_idx}")
            frame_before = _cs_frame.copy()

            # For click actions, try different positions
            data = {}
            if "CLICK" in action_info.get(act_idx, ""):
                # Use Florence-detected targets first, then grid-sample
                if _cs_click_targets and _cs_actions < len(_cs_click_targets):
                    _cx, _cy = _cs_click_targets[_cs_actions % len(_cs_click_targets)]
                else:
                    _cx = (_cs_actions * 17 + 8) % 56 + 4  # pseudo-random spread
                    _cy = (_cs_actions * 23 + 8) % 56 + 4
                data = {"x": _cx, "y": _cy}

            try:
                obs = env.step(actual_action, data) if data else env.step(actual_action)
            except Exception:
                _cs_actions += 1
                continue

            _cs_frame = _extract_frame(obs)
            n_actions = len(obs.available_actions)

            # Feed into Gundam's observation pipeline — builds spatial model, tracks effects
            won = obs.levels_completed > prev_levels_completed
            mind.observe_effect(lvl, act_idx, act_desc, frame_before, _cs_frame, won)
            _cs_actions += 1

            if won:
                mind.memory.levels_solved = obs.levels_completed
                prev_levels_completed = obs.levels_completed
                # Reinforce CNN with winning action sequence
                if mind.action_predictor and lvl.experiments:
                    _win_seq = [e.action_id for e in lvl.experiments]
                    mind.action_predictor.reinforce(_win_seq)
                if verbose:
                    print(f"  [cold-start] *** LEVEL {level} SOLVED during exploration! ({_cs_actions} actions) ***")
                done = ('WIN' in str(obs.state).upper() or 'LOSS' in str(obs.state).upper() or 'GAME_OVER' in str(obs.state).upper()
                        or (str(obs.state).upper().endswith('FINISHED')
                            and 'NOT_FINISHED' not in str(obs.state).upper()))
                if done:
                    _cs_solved = True
                    break
                # New level
                level = obs.levels_completed
                lvl = mind.observe_initial(_cs_frame, n_actions, action_info, level)
                mind.total_llm_calls = 0
                try:
                    mind.preflight(env, _extract_frame, level)
                    _cs_frame = _extract_frame(env)
                    lvl.current_frame = _cs_frame[0].copy() if _cs_frame.ndim == 3 else _cs_frame.copy()
                    lvl.current_frame_b64 = frame_to_b64(_cs_frame)
                except Exception:
                    pass

            # Check for game over (loss)
            if 'LOSS' in str(obs.state).upper() or 'GAME_OVER' in str(obs.state).upper():
                if verbose:
                    print(f"  [cold-start] Game over (loss) after {_cs_actions} exploration actions")
                break

        # Update frame for the rest of the pipeline
        frame = _cs_frame
        lvl.current_frame = frame[0].copy() if frame.ndim == 3 else frame.copy()
        lvl.current_frame_b64 = frame_to_b64(frame)

        if verbose and not _cs_solved:
            _rules_found = len(mind.memory.rules_discovered) - len(_recalled_rules) if _recalled_rules else len(mind.memory.rules_discovered)
            print(f"  [cold-start] Done: {_cs_actions} actions, {_rules_found} new rules discovered")
            if mind.spatial.action_map:
                print(f"  [cold-start] Spatial model: {len(mind.spatial.visited)} positions visited")

        if _cs_solved:
            done = True
            if verbose:
                print(f"\n{'='*60}")
                print(f"  COLD-START EXPLORATION SOLVED GAME! {mind.memory.levels_solved}/{total_levels} levels")
                print(f"  Zero LLM calls — pure autonomous exploration")
                print(f"{'='*60}")
            mind.remember_game()
            return mind

    # === BRUTE PROBE — try each discovered clickable position before LLM ===
    # For click games with few positions, systematically try each one.
    # Cost: N actions (where N = number of positions). If any advances the level,
    # we skip the expensive LLM call entirely.
    if (not done and not _is_known_game
            and hasattr(mind, '_env_profile')
            and any(p.action_type == ActionType.PARAMETERIZED for p in _oprah_probes)):
        _bp_positions = getattr(mind._env_profile, 'clickable_positions', [])
        if not _bp_positions and hasattr(mind, 'spatial') and mind.spatial._known_objects:
            _bp_positions = [(o.x, o.y) for o in mind.spatial._known_objects.values()
                            if hasattr(o, 'x') and hasattr(o, 'y')]
        _bp_click_act = next((p.action_idx for p in _oprah_probes
                              if p.action_type == ActionType.PARAMETERIZED), None)
        if _bp_positions and _bp_click_act is not None and len(_bp_positions) <= 20:
            if verbose:
                print(f"\n  [brute-probe] Trying {len(_bp_positions)} positions before LLM...")
            for _bpx, _bpy in _bp_positions[:20]:
                try:
                    _bp_act = obs.available_actions[_bp_click_act]
                    obs = env.step(_bp_act, {"x": int(_bpx), "y": int(_bpy)})
                    frame = _extract_frame(obs)
                    _bp_won = obs.levels_completed > prev_levels_completed
                    mind.observe_effect(lvl, _bp_click_act, f"click({_bpx},{_bpy})",
                                       lvl.current_frame if hasattr(lvl, 'current_frame') else frame,
                                       frame, _bp_won)
                    lvl.current_frame = frame[0].copy() if frame.ndim == 3 else frame.copy()
                    lvl.current_frame_b64 = frame_to_b64(frame)
                    if _bp_won:
                        mind.memory.levels_solved = obs.levels_completed
                        prev_levels_completed = obs.levels_completed
                        if verbose:
                            print(f"  [brute-probe] *** LEVEL {level} SOLVED by click at ({_bpx},{_bpy})! ***")
                        _bp_done = ('WIN' in str(obs.state).upper() or 'LOSS' in str(obs.state).upper() or 'GAME_OVER' in str(obs.state).upper()
                                    or (str(obs.state).upper().endswith('FINISHED')
                                        and 'NOT_FINISHED' not in str(obs.state).upper()))
                        if _bp_done:
                            done = True
                        else:
                            level = obs.levels_completed
                            lvl = mind.observe_initial(frame, len(obs.available_actions), action_info, level)
                        break
                except Exception:
                    continue

    # === SOLVER ROUTING (OPRAH-guided) ===
    # For low-confidence genres, get one LLM call for strategy guidance
    if _oprah_conf < 0.5 and mind.total_llm_calls < mind.budget_per_level:
        _llm_prompt = _oprah_build_llm_prompt(mind._env_profile, _oprah_probes)
        try:
            _llm_response = call_llm([
                {'role': 'system', 'content': 'You are an expert game analyst. Respond only in JSON.'},
                {'role': 'user', 'content': _llm_prompt},
            ], max_tokens=300, temperature=0.2, fast=True)
            mind.total_llm_calls += 1
            mind._last_progress_at_call = mind.total_llm_calls  # OPRAH call doesn't count as "stuck"
            if _llm_response:
                # Try to parse JSON from response
                import re as _re
                _json_match = _re.search(r'\{[^{}]*"genre"[^{}]*\}', _llm_response, _re.DOTALL)
                if _json_match:
                    _llm_genre_data = json.loads(_json_match.group())
                    _llm_genre = _llm_genre_data.get('genre', _oprah_genre)
                    if _llm_genre != 'novel_unknown':
                        mind._env_profile.genre = _llm_genre
                        mind._env_profile.genre_confidence = 0.6  # LLM-assisted
                        _oprah_genre = _llm_genre
                        _oprah_conf = 0.6
                        if verbose:
                            print(f"  [oprah] LLM upgraded genre: {_llm_genre}")
                            goal = _llm_genre_data.get('goal_hypothesis', '')
                            if goal:
                                print(f"  [oprah] Goal hypothesis: {goal}")
                    # Store strategy hint for the think loop
                    mind._oprah_strategy = _llm_genre_data.get('strategy', '')
                    mind._oprah_key_actions = _llm_genre_data.get('key_actions', {})
        except Exception as e:
            if verbose:
                print(f"  [oprah] LLM genre inference failed: {e}")

    # === RESEARCHER MODE ===
    # OPRAH routing: skip researcher for non-grid games (it needs grids + click to work)
    _skip_researcher = _oprah_genre in ('navigation_maze', 'pursuit_evasion', 'multi_phase')
    if _skip_researcher and verbose:
        print(f"  [oprah] Skipping researcher mode — genre {_oprah_genre} is not grid-based")

    # For navigation games: use BFS pathfinding + direction map to auto-navigate
    if _skip_researcher:
        _researcher_clicks = None
        # Auto-navigator: BFS on frame → direction sequence → action IDs
        # direction_map is on LevelMemory, action_map is on SpatialModel — try both
        _dir_map = getattr(mind.spatial, 'action_map', {}) or getattr(lvl, 'direction_map', {})
        _player_pos = mind.spatial.player_pos if hasattr(mind, 'spatial') else (-1, -1)
        _target_pos = mind.spatial.target_pos if hasattr(mind, 'spatial') else (-1, -1)
        # Try all target candidates, not just the first
        _targets_to_try = _nav_target_candidates if _nav_target_candidates else []
        if _target_pos != (-1, -1) and _target_pos not in _targets_to_try:
            _targets_to_try = [_target_pos] + _targets_to_try
        if verbose:
            print(f"  [nav-autopilot] dir_map={bool(_dir_map)}, player={_player_pos}, targets={len(_targets_to_try)}, frame.ndim={frame.ndim}")
        if (_dir_map and _player_pos != (-1, -1) and _targets_to_try
                and frame.ndim == 2):
            # Build reverse map: direction_name → action_idx
            _name_to_act = {}
            for act_idx, (dy, dx, dname) in _dir_map.items():
                _name_to_act[dname.upper()] = act_idx
            if verbose:
                print(f"  [nav-autopilot] name_to_act={_name_to_act}")
            from collections import deque, Counter
            # Derive step size from learned action_map (actual observed movement)
            _step = 5  # fallback
            if _dir_map:
                _observed_steps = [max(abs(dy), abs(dx)) for dy, dx, _ in _dir_map.values() if max(abs(dy), abs(dx)) > 0]
                if _observed_steps:
                    _step = int(round(sum(_observed_steps) / len(_observed_steps)))
            _h, _w = frame.shape[:2]
            _py, _px = _player_pos
            # Walkable detection: use spatial model's learned wall_colors if available
            # The old heuristic (top 3 most common = walkable) fails in mazes where
            # walls are the most common color. Instead: everything NOT a wall is walkable.
            _wall_colors = getattr(mind.spatial, 'wall_colors', set())
            _counts = Counter(frame.flatten())
            _walkable = set()
            if _wall_colors:
                # Use learned walls: walkable = all colors EXCEPT wall colors
                for c in _counts:
                    if int(c) not in _wall_colors:
                        _walkable.add(int(c))
                if verbose:
                    print(f"  [nav-autopilot] Using learned wall_colors={_wall_colors}, walkable={len(_walkable)} colors")
            else:
                # Fallback: detect wall color via grid reduction (like frame_to_minimap)
                # In a maze, the 2nd most common color in the grid-reduced frame = walls
                _cell = 5  # typical cell size
                _gh, _gw = _h // _cell, _w // _cell
                if _gh >= 3 and _gw >= 3:
                    _mini_counts = Counter()
                    for _gy in range(_gh):
                        for _gx in range(_gw):
                            _region = frame[_gy*_cell:(_gy+1)*_cell, _gx*_cell:(_gx+1)*_cell]
                            _vals, _cnts = np.unique(_region.flatten(), return_counts=True)
                            _mini_counts[int(_vals[np.argmax(_cnts)])] += 1
                    _sorted_colors = _mini_counts.most_common()
                    # Most common = background (walkable), 2nd = walls (not walkable)
                    _detected_wall = _sorted_colors[1][0] if len(_sorted_colors) > 1 else -1
                    for c in _counts:
                        if int(c) != _detected_wall:
                            _walkable.add(int(c))
                    if verbose:
                        print(f"  [nav-autopilot] Grid-detected wall_color={_detected_wall}, walkable={len(_walkable)} colors")
                else:
                    # Ultra-fallback: top 3 most common
                    for c, cnt in _counts.most_common(3):
                        if cnt > 100:
                            _walkable.add(int(c))
                    if verbose:
                        print(f"  [nav-autopilot] No wall data — using count heuristic")
            # Player color is also walkable (we overlap ourselves)
            if mind.spatial.player_color >= 0:
                _walkable.add(mind.spatial.player_color)
            # Target colors are walkable (we need to reach them)
            for _ty2, _tx2 in _targets_to_try[:7]:
                _walkable.add(int(frame[min(_ty2, frame.shape[0]-1), min(_tx2, frame.shape[1]-1)]))
            # Concatenate paths to ALL targets
            # BFS from player to EACH target independently — pick shortest reachable
            _best_path = None
            _best_target = None
            _best_score = -999
            _known_walls = getattr(mind.spatial, 'walls', set())
            if verbose and _known_walls:
                print(f"  [nav-autopilot] Using {len(_known_walls)} known wall positions")
            for _ti, (_ty, _tx) in enumerate(_targets_to_try[:7]):
                _bfs_visited = set()
                _bfs_q = deque([(_py, _px, [])])
                _bfs_visited.add((_py, _px))
                _seg_path = None
                while _bfs_q and len(_bfs_visited) < 10000:
                    cy, cx, path = _bfs_q.popleft()
                    if abs(cy - _ty) <= _step and abs(cx - _tx) <= _step:
                        _seg_path = path
                        break
                    for dy, dx, dname in [(-_step,0,'UP'),(_step,0,'DOWN'),(0,-_step,'LEFT'),(0,_step,'RIGHT')]:
                        ny, nx = cy+dy, cx+dx
                        if (ny, nx) in _bfs_visited or ny < 0 or ny >= _h or nx < 0 or nx >= _w:
                            continue
                        # Check for walls: known wall positions + color-based check along path
                        _blocked = False
                        # Check every pixel along the movement path (catches thin walls)
                        _path_steps = max(abs(dy), abs(dx))
                        for _pi in range(1, _path_steps + 1):
                            _mid_y = cy + (dy * _pi) // _path_steps
                            _mid_x = cx + (dx * _pi) // _path_steps
                            if 0 <= _mid_y < _h and 0 <= _mid_x < _w:
                                if (_mid_y, _mid_x) in _known_walls:
                                    _blocked = True; break
                                if int(frame[_mid_y, _mid_x]) not in _walkable:
                                    _blocked = True; break
                        # Also check destination pixel itself
                        if not _blocked and 0 <= ny < _h and 0 <= nx < _w:
                            if int(frame[ny, nx]) not in _walkable:
                                _blocked = True
                        if not _blocked:
                            _bfs_visited.add((ny, nx))
                            _bfs_q.append((ny, nx, path + [dname]))
                # Skip nearby targets (likely modifiers, not goals)
                _dist_to_target = abs(_ty - _py) + abs(_tx - _px)
                # Score: prefer targets far from player AND close to far corner (bottom-right)
                _corner_dist = abs(_ty - (_h-1)) + abs(_tx - (_w-1))
                _target_score = _dist_to_target - _corner_dist  # high = far from player + close to corner
                if _seg_path and _dist_to_target < 3 * _step:
                    if verbose:
                        print(f"  [nav-autopilot] Target #{_ti}: ({_ty},{_tx}) — {len(_seg_path)} moves (too close, skip)")
                elif _seg_path and _target_score > _best_score:
                    _best_path = _seg_path
                    _best_target = (_ty, _tx, _ti)
                    _best_score = _target_score
                    if verbose:
                        print(f"  [nav-autopilot] Target #{_ti}: ({_ty},{_tx}) — {len(_seg_path)} moves, score={_target_score} (new best)")
                elif _seg_path and verbose:
                    print(f"  [nav-autopilot] Target #{_ti}: ({_ty},{_tx}) — {len(_seg_path)} moves, score={_target_score}")
                elif verbose:
                    print(f"  [nav-autopilot] Target #{_ti}: ({_ty},{_tx}) — no path")
            _full_nav = _best_path if _best_path else []
            if _full_nav and _name_to_act:
                _researcher_clicks = []
                for dname in _full_nav:
                    if dname in _name_to_act:
                        _researcher_clicks.append((_name_to_act[dname], None))
                if verbose:
                    _compressed = []
                    i = 0
                    while i < len(_full_nav):
                        d = _full_nav[i]
                        c = 1
                        while i+c < len(_full_nav) and _full_nav[i+c] == d:
                            c += 1
                        _compressed.append(f"{d} {c}" if c > 1 else d)
                        i += c
                    print(f"  [nav-autopilot] Full nav: {' → '.join(_compressed)} ({len(_full_nav)} moves, {len(_researcher_clicks)} actions)")
            elif verbose:
                print(f"  [nav-autopilot] No paths found to any target")
    else:
        # Try researcher-mode: systematic probe + one LLM call for rule + programmatic execution
        _researcher_clicks = _researcher_solve(env, mind, _extract_frame, frame, action_info, obs, verbose)
    # === TIGHT RESEARCHER LOOP: solve ALL levels without LLM ===
    _researcher_game_done = False
    while _researcher_clicks is not None and len(_researcher_clicks) > 0:
        if verbose:
            print(f"  [researcher] Executing {len(_researcher_clicks)} computed actions (level {level})")
        _researcher_solved_this_level = False
        for act_id, data in _researcher_clicks:
            act_id = max(0, min(act_id, n_actions - 1))
            actual_action = obs.available_actions[act_id]
            frame_before = frame.copy()
            try:
                obs = env.step(actual_action, data) if data else env.step(actual_action)
            except Exception as e:
                if verbose:
                    print(f"  [researcher] Step failed: {e}")
                continue
            frame = _extract_frame(obs)
            n_actions = len(obs.available_actions)

            act_desc = action_info.get(act_id, f"action {act_id}")
            mind.observe_effect(lvl, act_id, act_desc, frame_before, frame,
                               obs.levels_completed > prev_levels_completed)

            won = obs.levels_completed > prev_levels_completed
            done = ('WIN' in str(obs.state).upper() or 'LOSS' in str(obs.state).upper() or 'GAME_OVER' in str(obs.state).upper() or (str(obs.state).upper().endswith('FINISHED') and 'NOT_FINISHED' not in str(obs.state).upper()))

            if won:
                mind.memory.levels_solved = obs.levels_completed
                # Reinforce CNN with researcher's winning actions
                if mind.action_predictor and lvl.experiments:
                    _win_seq = [e.action_id for e in lvl.experiments]
                    mind.action_predictor.reinforce(_win_seq)
                if verbose:
                    print(f"\n  *** LEVEL {level} SOLVED by researcher! ***\n")
                prev_levels_completed = obs.levels_completed
                _researcher_solved_this_level = True
                if done:
                    _researcher_game_done = True
                break  # exit inner for-loop to re-run researcher on new level

            if done:
                _researcher_game_done = True
                break

        if _researcher_game_done:
            if verbose:
                print(f"\n  === GAME COMPLETE: {mind.memory.levels_solved}/{total_levels} ===")
            break  # exit while loop

        if not _researcher_solved_this_level:
            break  # researcher couldn't solve this level -- fall through to LLM

        # Level solved -- prepare for next level
        level = obs.levels_completed
        frame = _extract_frame(obs)  # CRITICAL: get the NEW level's frame
        n_actions = len(obs.available_actions)
        lvl = mind.observe_initial(frame, n_actions, action_info, level)
        mind.total_llm_calls = 0

        # Re-run researcher on new level
        _researcher_clicks = _researcher_solve(env, mind, _extract_frame, frame, action_info, obs, verbose)
        # For navigation games, _researcher_solve returns None — re-run nav-autopilot BFS
        if _researcher_clicks is None and _skip_researcher:
            _dir_map = getattr(mind.spatial, 'action_map', {}) or getattr(lvl, 'direction_map', {})
            _player_pos = mind.spatial.player_pos if hasattr(mind, 'spatial') else (-1, -1)
            _target_pos = mind.spatial.target_pos if hasattr(mind, 'spatial') else (-1, -1)
            _targets_to_try = getattr(mind.spatial, '_nav_targets', [])
            if not _targets_to_try:
                # Re-detect targets from the new frame
                _wall_colors_r = getattr(mind.spatial, 'wall_colors', set())
                _pc = mind.spatial.player_color
                if _pc >= 0 and frame.ndim == 2:
                    from collections import Counter as _Ctr
                    _cnts_r = _Ctr(frame.flatten())
                    _cands_r = []
                    for c, cnt in _cnts_r.items():
                        if int(c) != _pc and int(c) not in _wall_colors_r and 5 < cnt < 200:
                            _ys, _xs = np.where(frame == c)
                            if len(_ys) > 0:
                                _cands_r.append((int(np.mean(_ys)), int(np.mean(_xs))))
                    _targets_to_try = _cands_r
            if _target_pos != (-1, -1) and _target_pos not in _targets_to_try:
                _targets_to_try = [_target_pos] + _targets_to_try
            if _dir_map and _player_pos != (-1, -1) and _targets_to_try and frame.ndim == 2:
                from collections import deque as _dq, Counter as _Ctr2
                _name_to_act_r = {}
                for act_idx, (dy, dx, dname) in _dir_map.items():
                    _name_to_act_r[dname.upper()] = act_idx
                _observed_steps_r = [max(abs(dy), abs(dx)) for dy, dx, _ in _dir_map.values() if max(abs(dy), abs(dx)) > 0]
                _step_r = int(round(sum(_observed_steps_r) / len(_observed_steps_r))) if _observed_steps_r else 5
                _h_r, _w_r = frame.shape[:2]
                _py_r, _px_r = _player_pos
                _wall_colors_r2 = getattr(mind.spatial, 'wall_colors', set())
                _counts_r = _Ctr2(frame.flatten())
                _walkable_r = set()
                if _wall_colors_r2:
                    for c in _counts_r:
                        if int(c) not in _wall_colors_r2:
                            _walkable_r.add(int(c))
                else:
                    _cell_r = 5
                    _gh_r, _gw_r = _h_r // _cell_r, _w_r // _cell_r
                    if _gh_r >= 3 and _gw_r >= 3:
                        _mini_r = _Ctr2()
                        for _gy in range(_gh_r):
                            for _gx in range(_gw_r):
                                _reg = frame[_gy*_cell_r:(_gy+1)*_cell_r, _gx*_cell_r:(_gx+1)*_cell_r]
                                _v, _cn = np.unique(_reg.flatten(), return_counts=True)
                                _mini_r[int(_v[np.argmax(_cn)])] += 1
                        _sc = _mini_r.most_common()
                        _dw = _sc[1][0] if len(_sc) > 1 else -1
                        for c in _counts_r:
                            if int(c) != _dw:
                                _walkable_r.add(int(c))
                if mind.spatial.player_color >= 0:
                    _walkable_r.add(mind.spatial.player_color)
                for _ty2, _tx2 in _targets_to_try[:7]:
                    _walkable_r.add(int(frame[min(_ty2, frame.shape[0]-1), min(_tx2, frame.shape[1]-1)]))
                _known_walls_r = getattr(mind.spatial, 'walls', set())
                _best_path_r = None
                for _ti, (_ty, _tx) in enumerate(_targets_to_try[:7]):
                    _bfs_v = set()
                    _bfs_q = _dq([(_py_r, _px_r, [])])
                    _bfs_v.add((_py_r, _px_r))
                    _seg = None
                    while _bfs_q and len(_bfs_v) < 10000:
                        cy, cx, path = _bfs_q.popleft()
                        if abs(cy - _ty) <= _step_r and abs(cx - _tx) <= _step_r:
                            _seg = path; break
                        for dy, dx, dn in [(-_step_r,0,'UP'),(_step_r,0,'DOWN'),(0,-_step_r,'LEFT'),(0,_step_r,'RIGHT')]:
                            ny, nx = cy+dy, cx+dx
                            if (ny, nx) in _bfs_v or ny < 0 or ny >= _h_r or nx < 0 or nx >= _w_r:
                                continue
                            _blk = False
                            _ps = max(abs(dy), abs(dx))
                            for _pi in range(1, _ps + 1):
                                _my = cy + (dy * _pi) // _ps
                                _mx = cx + (dx * _pi) // _ps
                                if 0 <= _my < _h_r and 0 <= _mx < _w_r:
                                    if (_my, _mx) in _known_walls_r or int(frame[_my, _mx]) not in _walkable_r:
                                        _blk = True; break
                            if not _blk and 0 <= ny < _h_r and 0 <= nx < _w_r:
                                if int(frame[ny, nx]) not in _walkable_r:
                                    _blk = True
                            if not _blk:
                                _bfs_v.add((ny, nx))
                                _bfs_q.append((ny, nx, path + [dn]))
                    _dt = abs(_ty - _py_r) + abs(_tx - _px_r)
                    if _seg and _dt >= 3 * _step_r and (not _best_path_r or len(_seg) < len(_best_path_r)):
                        _best_path_r = _seg
                if _best_path_r and _name_to_act_r:
                    _researcher_clicks = [(_name_to_act_r[d], None) for d in _best_path_r if d in _name_to_act_r]
                    if verbose:
                        print(f"  [nav-autopilot] Re-planned for level {level}: {len(_researcher_clicks)} moves")
        # while-loop continues with new clicks (or exits if None)

    # Check if researcher solved everything
    done = ('WIN' in str(obs.state).upper() or 'LOSS' in str(obs.state).upper() or 'GAME_OVER' in str(obs.state).upper() or (str(obs.state).upper().endswith('FINISHED') and 'NOT_FINISHED' not in str(obs.state).upper()))
    if done:
        if verbose:
            total_solved = mind.memory.levels_solved
            print(f"\n{'='*60}")
            print(f"  RESEARCHER MODE RESULTS: {total_solved}/{total_levels} levels solved")
            print(f"  LLM calls: {mind.total_llm_calls}")
            print(f"{'='*60}")
        mind.remember_game()
        return mind

    # === SWITCH PUZZLE SOLVER (timing/routing games like VC33) ===
    # For games with auto-advancing state (timers, trains): find responsive click
    # positions and try systematic clicking patterns via deepcopy validation.
    # This runs BEFORE the LLM loop since it's zero-cost.
    if not done and n_actions == 1 and len(obs.available_actions) == 1:
        _click_act_val = obs.available_actions[0]
        _switch_genre = getattr(getattr(mind, '_env_profile', None), 'genre', '') or ''
        # Only run for click-only games that aren't grid puzzles (researcher handles those)
        _has_grids = bool(getattr(mind, '_detected_grids', None))
        if not _has_grids and not done:
            import copy as _sw_cp
            if verbose:
                print(f"\n  [switch-solver] Click-only game, no grids — scanning for switches...")
            # Find responsive click positions (coarse scan)
            _sw_responsive = []
            _sw_base = _extract_frame(obs)
            for _sy in range(0, 64, 4):
                for _sx in range(0, 64, 4):
                    _sw_env2 = _sw_cp.deepcopy(env)
                    _sw_obs2 = _sw_env2.step(_click_act_val, {"x": _sx, "y": _sy})
                    _sw_f2 = _extract_frame(_sw_obs2)
                    _sw_diff = int(np.sum(_sw_base != _sw_f2))
                    if _sw_diff > 5:
                        _sw_responsive.append((_sx, _sy, _sw_diff))
            if verbose:
                print(f"  [switch-solver] Found {len(_sw_responsive)} responsive positions")
                for _sx, _sy, _sd in _sw_responsive:
                    print(f"    ({_sx},{_sy}): {_sd}px")
            # GF(2) Gaussian solver for medium-size switch puzzles (10-30 positions)
            # Same math the researcher uses, applied to switch-solver's responsive positions
            if _sw_responsive and 10 < len(_sw_responsive) <= 30:
                import copy as _gf_cp
                if verbose:
                    print(f"  [switch-gf2] {len(_sw_responsive)} responsive positions — building toggle matrix")
                _gf_total_solved = obs.levels_completed
                _gf_done = False
                for _gf_level in range(_gf_total_solved, total_levels):
                    _gf_solved = False
                    n_pos = len(_sw_responsive)
                    # Build toggle matrix: A[i,j] = does clicking position j affect position i?
                    # Sample color at each responsive position before any clicks
                    _gf_base = _extract_frame(obs)
                    _gf_colors_before = []
                    for _gx, _gy, _ in _sw_responsive:
                        _gf_colors_before.append(int(_gf_base[_gy, _gx]) if _gf_base.ndim == 2 else int(_gf_base[0, _gy, _gx]))
                    A_sw = np.zeros((n_pos, n_pos), dtype=int)
                    for j in range(n_pos):
                        _gx, _gy, _ = _sw_responsive[j]
                        _gf_env = _gf_cp.deepcopy(env)
                        _gf_obs_j = _gf_env.step(_click_act_val, {"x": _gx, "y": _gy})
                        _gf_frame_j = _extract_frame(_gf_obs_j)
                        for i in range(n_pos):
                            _ix, _iy, _ = _sw_responsive[i]
                            c_after = int(_gf_frame_j[_iy, _ix]) if _gf_frame_j.ndim == 2 else int(_gf_frame_j[0, _iy, _ix])
                            if c_after != _gf_colors_before[i]:
                                A_sw[i, j] = 1
                    # Try multiple target hypotheses
                    from collections import Counter
                    _cc = Counter(_gf_colors_before)
                    _majority_color = _cc.most_common(1)[0][0]
                    # Build target hypotheses: majority, minority, checkerboard, each-color-is-target
                    _target_hypotheses = [
                        ("majority", np.array([0 if c == _majority_color else 1 for c in _gf_colors_before], dtype=int)),
                        ("minority", np.array([1 if c == _majority_color else 0 for c in _gf_colors_before], dtype=int)),
                    ]
                    # Checkerboard: alternate 0/1 based on position index
                    _target_hypotheses.append(
                        ("checker", np.array([i % 2 for i in range(n_pos)], dtype=int)))
                    _target_hypotheses.append(
                        ("checker_inv", np.array([(i + 1) % 2 for i in range(n_pos)], dtype=int)))
                    # Per-color targets: for each unique color, "make everything THIS color"
                    for _tc in set(_gf_colors_before):
                        if _tc == _majority_color:
                            continue  # already covered by majority
                        _target_hypotheses.append(
                            (f"all_{_tc}", np.array([0 if c == _tc else 1 for c in _gf_colors_before], dtype=int)))
                    for _hyp_label, _b_vec in _target_hypotheses:
                        if not np.any(_b_vec):
                            continue
                        _sol = _gf2_solve(A_sw, _b_vec)
                        if _sol is not None and np.any(_sol):
                            # Verify via deepcopy
                            _gf_env_v = _gf_cp.deepcopy(env)
                            _gf_obs_v = None
                            _gf_clicks = []
                            for j in range(n_pos):
                                if _sol[j]:
                                    _gx, _gy, _ = _sw_responsive[j]
                                    _gf_obs_v = _gf_env_v.step(_click_act_val, {"x": _gx, "y": _gy})
                                    _gf_clicks.append((_gx, _gy))
                            if _gf_obs_v and getattr(_gf_obs_v, 'levels_completed', 0) > _gf_total_solved:
                                if verbose:
                                    print(f"  [switch-gf2] L{_gf_level} SOLVED ({_hyp_label}): {len(_gf_clicks)} clicks")
                                # Replay on real env
                                for _gx, _gy in _gf_clicks:
                                    obs = env.step(_click_act_val, {"x": _gx, "y": _gy})
                                frame = _extract_frame(obs)
                                _gf_total_solved = obs.levels_completed
                                mind.memory.levels_solved = _gf_total_solved
                                prev_levels_completed = _gf_total_solved
                                _gf_solved = True
                                break
                    # Brute-force small subsets (1-5 clicks) — cheap with deepcopy
                    if not _gf_solved and n_pos <= 30:
                        from itertools import combinations
                        _max_clicks = 5 if n_pos <= 20 else (4 if n_pos <= 26 else 3)
                        if verbose:
                            print(f"  [switch-gf2] Trying brute-force subsets (1-{_max_clicks} clicks from {n_pos} positions)")
                        for _bf_size in range(1, min(_max_clicks + 1, n_pos + 1)):
                            if _gf_solved:
                                break
                            for _bf_combo in combinations(range(n_pos), _bf_size):
                                _bf_env = _gf_cp.deepcopy(env)
                                _bf_obs = None
                                for j in _bf_combo:
                                    _gx, _gy, _ = _sw_responsive[j]
                                    _bf_obs = _bf_env.step(_click_act_val, {"x": _gx, "y": _gy})
                                if _bf_obs and getattr(_bf_obs, 'levels_completed', 0) > _gf_total_solved:
                                    if verbose:
                                        print(f"  [switch-gf2] L{_gf_level} SOLVED (brute {_bf_size}-click): positions {_bf_combo}")
                                    for j in _bf_combo:
                                        _gx, _gy, _ = _sw_responsive[j]
                                        obs = env.step(_click_act_val, {"x": _gx, "y": _gy})
                                    frame = _extract_frame(obs)
                                    _gf_total_solved = obs.levels_completed
                                    mind.memory.levels_solved = _gf_total_solved
                                    prev_levels_completed = _gf_total_solved
                                    _gf_solved = True
                                    break
                    if not _gf_solved:
                        if verbose:
                            print(f"  [switch-gf2] Could not solve L{_gf_level}")
                        # Save toggle matrix for LLM to reason about
                        mind._switch_toggle_info = {
                            'positions': [(x, y) for x, y, _ in _sw_responsive],
                            'colors_before': _gf_colors_before,
                            'toggle_matrix_density': float(np.mean(A_sw)),
                            'n_positions': n_pos,
                        }
                        break
                    # Re-scan responsive positions for next level
                    _gf_base_new = _extract_frame(obs)
                    _sw_responsive_new = []
                    for _sy2 in range(0, 64, 4):
                        for _sx2 in range(0, 64, 4):
                            _gf_env2 = _gf_cp.deepcopy(env)
                            _gf_obs2 = _gf_env2.step(_click_act_val, {"x": _sx2, "y": _sy2})
                            _gf_f2 = _extract_frame(_gf_obs2)
                            _gf_diff = int(np.sum(_gf_base_new != _gf_f2))
                            if _gf_diff > 5:
                                _sw_responsive_new.append((_sx2, _sy2, _gf_diff))
                    if _sw_responsive_new:
                        _sw_responsive = _sw_responsive_new
                    if verbose:
                        print(f"  [switch-gf2] L{_gf_level+1}: {len(_sw_responsive)} responsive positions")
                _gf_done_check = ('WIN' in str(obs.state).upper() or 'LOSS' in str(obs.state).upper() or
                    'GAME_OVER' in str(obs.state).upper() or
                    (str(obs.state).upper().endswith('FINISHED') and 'NOT_FINISHED' not in str(obs.state).upper()))
                if _gf_done_check or _gf_total_solved >= total_levels:
                    if verbose:
                        print(f"\n{'='*60}")
                        print(f"  SWITCH-GF2 RESULTS: {_gf_total_solved}/{total_levels} levels solved")
                        print(f"{'='*60}")
                    mind.remember_game()
                    return mind
                done = _gf_done_check

            # Try solving levels by clicking responsive positions
            if _sw_responsive and len(_sw_responsive) <= 10:
                _sw_total_solved = obs.levels_completed
                _sw_game_done = False
                for _sw_level in range(obs.levels_completed, total_levels):
                    _sw_solved_this = False
                    # Single-position: try N clicks (1-7) then let game auto-advance
                    for _sx, _sy, _ in _sw_responsive:
                        if _sw_solved_this:
                            break
                        for _sw_n in range(1, 8):
                            _sw_env2 = _sw_cp.deepcopy(env)
                            _sw_obs2 = None
                            for _sw_c in range(_sw_n):
                                _sw_obs2 = _sw_env2.step(_click_act_val, {"x": _sx, "y": _sy})
                            # Let game auto-advance (80 steps max)
                            for _sw_wait in range(80):
                                _sw_obs2 = _sw_env2.step(_click_act_val, {})
                                if _sw_obs2.levels_completed > _sw_total_solved:
                                    if verbose:
                                        print(f"  [switch-solver] L{_sw_level} SOLVED: {_sw_n}x click at ({_sx},{_sy})")
                                    _sw_solved_this = True
                                    # Replay solution on REAL env (for scorecard tracking)
                                    for _sw_r in range(_sw_n):
                                        obs = env.step(_click_act_val, {"x": _sx, "y": _sy})
                                    for _sw_rw in range(_sw_wait + 1):
                                        obs = env.step(_click_act_val, {})
                                    frame = _extract_frame(obs)
                                    _sw_total_solved = obs.levels_completed
                                    mind.memory.levels_solved = _sw_total_solved
                                    prev_levels_completed = _sw_total_solved
                                    break
                                if obs.state.name != 'NOT_FINISHED' if hasattr(obs, 'state') else False:
                                    break
                                _sw_st = str(_sw_obs2.state).upper()
                                if 'GAME_OVER' in _sw_st or ('FINISHED' in _sw_st and 'NOT_FINISHED' not in _sw_st):
                                    break
                            if _sw_solved_this:
                                break
                    if not _sw_solved_this and len(_sw_responsive) >= 2:
                        # 2-position combos (order matters — timing game)
                        for _sw_i, (_sx1, _sy1, _) in enumerate(_sw_responsive):
                            if _sw_solved_this:
                                break
                            for _sw_j, (_sx2, _sy2, _) in enumerate(_sw_responsive):
                                if _sw_i == _sw_j or _sw_solved_this:
                                    continue
                                for _sw_n1 in range(1, 6):
                                    if _sw_solved_this:
                                        break
                                    for _sw_n2 in range(1, 6):
                                        _sw_env2 = _sw_cp.deepcopy(env)
                                        for _sw_c in range(_sw_n1):
                                            _sw_obs2 = _sw_env2.step(_click_act_val, {"x": _sx1, "y": _sy1})
                                        for _sw_c in range(_sw_n2):
                                            _sw_obs2 = _sw_env2.step(_click_act_val, {"x": _sx2, "y": _sy2})
                                        for _sw_wait in range(80):
                                            _sw_obs2 = _sw_env2.step(_click_act_val, {})
                                            if _sw_obs2.levels_completed > _sw_total_solved:
                                                if verbose:
                                                    print(f"  [switch-solver] L{_sw_level} SOLVED: {_sw_n1}x({_sx1},{_sy1}) + {_sw_n2}x({_sx2},{_sy2})")
                                                _sw_solved_this = True
                                                # Replay on real env for scorecard
                                                for _sw_r in range(_sw_n1):
                                                    obs = env.step(_click_act_val, {"x": _sx1, "y": _sy1})
                                                for _sw_r in range(_sw_n2):
                                                    obs = env.step(_click_act_val, {"x": _sx2, "y": _sy2})
                                                for _sw_rw in range(_sw_wait + 1):
                                                    obs = env.step(_click_act_val, {})
                                                frame = _extract_frame(obs)
                                                _sw_total_solved = obs.levels_completed
                                                mind.memory.levels_solved = _sw_total_solved
                                                prev_levels_completed = _sw_total_solved
                                                break
                                            _sw_st = str(_sw_obs2.state).upper()
                                            if 'GAME_OVER' in _sw_st or ('FINISHED' in _sw_st and 'NOT_FINISHED' not in _sw_st):
                                                break
                                        if _sw_solved_this:
                                            break
                    if not _sw_solved_this and len(_sw_responsive) >= 3:
                        # 3-position combos (for complex switch puzzles)
                        from itertools import permutations
                        _sw_positions = list(set((x,y) for x,y,_ in _sw_responsive))
                        if len(_sw_positions) <= 6:  # tractable
                            for _sw_perm in permutations(_sw_positions, 3):
                                if _sw_solved_this:
                                    break
                                for _sw_n1 in range(1, 4):
                                    if _sw_solved_this:
                                        break
                                    for _sw_n2 in range(1, 4):
                                        if _sw_solved_this:
                                            break
                                        for _sw_n3 in range(1, 4):
                                            _sw_env2 = _sw_cp.deepcopy(env)
                                            for _sw_c in range(_sw_n1):
                                                _sw_obs2 = _sw_env2.step(_click_act_val, {"x": _sw_perm[0][0], "y": _sw_perm[0][1]})
                                            for _sw_c in range(_sw_n2):
                                                _sw_obs2 = _sw_env2.step(_click_act_val, {"x": _sw_perm[1][0], "y": _sw_perm[1][1]})
                                            for _sw_c in range(_sw_n3):
                                                _sw_obs2 = _sw_env2.step(_click_act_val, {"x": _sw_perm[2][0], "y": _sw_perm[2][1]})
                                            for _sw_wait in range(80):
                                                _sw_obs2 = _sw_env2.step(_click_act_val, {})
                                                if _sw_obs2.levels_completed > _sw_total_solved:
                                                    if verbose:
                                                        print(f"  [switch-solver] L{_sw_level} SOLVED: {_sw_n1}x({_sw_perm[0][0]},{_sw_perm[0][1]}) + {_sw_n2}x({_sw_perm[1][0]},{_sw_perm[1][1]}) + {_sw_n3}x({_sw_perm[2][0]},{_sw_perm[2][1]})")
                                                    _sw_solved_this = True
                                                    # Replay on real env for scorecard
                                                    for _sw_r in range(_sw_n1):
                                                        obs = env.step(_click_act_val, {"x": _sw_perm[0][0], "y": _sw_perm[0][1]})
                                                    for _sw_r in range(_sw_n2):
                                                        obs = env.step(_click_act_val, {"x": _sw_perm[1][0], "y": _sw_perm[1][1]})
                                                    for _sw_r in range(_sw_n3):
                                                        obs = env.step(_click_act_val, {"x": _sw_perm[2][0], "y": _sw_perm[2][1]})
                                                    for _sw_rw in range(_sw_wait + 1):
                                                        obs = env.step(_click_act_val, {})
                                                    frame = _extract_frame(obs)
                                                    _sw_total_solved = obs.levels_completed
                                                    mind.memory.levels_solved = _sw_total_solved
                                                    prev_levels_completed = _sw_total_solved
                                                    break
                                                _sw_st = str(_sw_obs2.state).upper()
                                                if 'GAME_OVER' in _sw_st or ('FINISHED' in _sw_st and 'NOT_FINISHED' not in _sw_st):
                                                    break
                                            if _sw_solved_this:
                                                break
                    if not _sw_solved_this:
                        if verbose:
                            print(f"  [switch-solver] Could not solve L{_sw_level}")
                        break
                    # Re-scan responsive positions for new level
                    _sw_base = _extract_frame(obs)
                    _sw_responsive_new = []
                    for _sy in range(0, 64, 4):
                        for _sx in range(0, 64, 4):
                            _sw_env2 = _sw_cp.deepcopy(env)
                            _sw_obs2 = _sw_env2.step(_click_act_val, {"x": _sx, "y": _sy})
                            _sw_f2 = _extract_frame(_sw_obs2)
                            _sw_diff = int(np.sum(_sw_base != _sw_f2))
                            if _sw_diff > 5:
                                _sw_responsive_new.append((_sx, _sy, _sw_diff))
                    _sw_responsive = _sw_responsive_new if _sw_responsive_new else _sw_responsive
                    if verbose:
                        print(f"  [switch-solver] L{_sw_level+1}: {len(_sw_responsive)} responsive positions")

                done = ('WIN' in str(obs.state).upper() or 'LOSS' in str(obs.state).upper() or 'GAME_OVER' in str(obs.state).upper() or (str(obs.state).upper().endswith('FINISHED') and 'NOT_FINISHED' not in str(obs.state).upper()))
                if done or _sw_total_solved >= total_levels:
                    if verbose:
                        print(f"\n{'='*60}")
                        print(f"  SWITCH SOLVER RESULTS: {_sw_total_solved}/{total_levels} levels solved")
                        print(f"{'='*60}")
                    mind.remember_game()
                    return mind

    # === FALLBACK: PROGRAMMATIC SOLVER (lights-out specific) ===
    # If researcher mode didn't work, try lights-out solver before burning LLM budget
    # PETER DIRECTIVE (2026-03-20): Specialized solvers disabled by default.
    # Architecture: Qwen-VL (eyes) + Claude (brain) + corpus callosum.
    # Set ARC_USE_SOLVER=1 to re-enable for offline benchmarking only.
    _solver_disabled = not os.environ.get('ARC_USE_SOLVER')
    _lo_solution = None  # lights-out solution queue
    mind._lo_solution_found_this_level = False  # tracks if oracle solved CURRENT level
    _lo_click_idx = getattr(mind, '_click_action_idx', None)

    def _try_lights_out_solve():
        """Attempt programmatic solve on current frame. Returns click list or None."""
        if _solver_disabled:
            return None
        if _lo_click_idx is None:
            return None
        cycle = getattr(mind, '_color_cycle', None)
        period = cycle['period'] if cycle else 2
        click_act = obs.available_actions[_lo_click_idx]

        # PRIMARY: probe-based oracle — discovers clickable positions from frame,
        # then tries all combinations using game win-check as oracle.
        # Bypasses grid detection entirely — more robust.
        # Pass researcher's prediction if available (for Hamming-guided search)
        researcher_target = getattr(mind, '_researcher_target_map', None)
        researcher_current = getattr(mind, '_researcher_current_map', None)
        oracle_result = _oracle_solve(
            env, click_act, _extract_frame,
            grid_cells=None,  # discover from probing
            cycle_period=period,
            levels_completed=prev_levels_completed,
            base_frame=frame,
            predicted_target_map=researcher_target,
            predicted_current_map=researcher_current,
            verbose=verbose
        )
        # Save discovered positions for researcher AND LLM context
        if hasattr(_oracle_solve, '_last_cell_positions'):
            mind._oracle_cell_positions = _oracle_solve._last_cell_positions
            # Also set as interactive positions so the LLM knows WHERE to click
            if not getattr(mind, '_interactive_positions', []):
                mind._interactive_positions = list(_oracle_solve._last_cell_positions)

        if oracle_result is not None:
            return oracle_result

        # FALLBACK: grid-detection-based math solver
        grids = getattr(mind, '_detected_grids', None)
        if grids:
            target_grid = next((g for g in grids if g.get('is_target')), grids[-1])
            if target_grid and target_grid.get('cells'):
                result = _solve_lights_out(
                    env, click_act, _extract_frame, frame,
                    target_grid['cells'], cycle_period=period, verbose=verbose
                )
                if result is not None:
                    return result

        return None

    _lo_solution = _try_lights_out_solve()
    if _lo_solution is not None and len(_lo_solution) > 0:
        if verbose:
            print(f"  [gundam] LIGHTS-OUT SOLVER: {len(_lo_solution)} clicks computed")

    _muscle_path = []
    _muscle_mode = ""

    _is_oracle = False  # flag: skip walk-before-run for oracle solutions
    _level_just_changed = False  # flag: researcher should re-analyze
    level_done = False
    for turn in range(max_turns_per_level * 10):  # global turn limit
        # If we have a programmatic solution, use it instead of LLM
        _is_oracle = False

        # ── RE-RUN RESEARCHER AFTER LEVEL TRANSITION ──
        # The researcher runs once during preflight. After a level transition,
        # the new level needs fresh analysis. Re-run researcher and populate
        # _lo_solution so the batch execution code below picks it up.
        if verbose and turn < 20:
            print(f"  [TRACE turn={turn}] _level_just_changed={_level_just_changed}, _skip_researcher={_skip_researcher}, _lo_click_idx={_lo_click_idx}, _lo_solution={_lo_solution is not None and len(_lo_solution) if _lo_solution else 0}, level_done={level_done}")
        if _level_just_changed and not _skip_researcher and _lo_click_idx is not None:
            _level_just_changed = False
            try:
                _new_frame = _extract_frame(env)
                _new_clicks = _researcher_solve(env, mind, _extract_frame, _new_frame, action_info, obs, verbose)
                if _new_clicks:
                    _lo_solution = []
                    for _act_id, _data in _new_clicks:
                        if _data and isinstance(_data, dict):
                            _lo_solution.append((_data.get('x', 0), _data.get('y', 0)))
                    if verbose:
                        print(f"  [researcher-rerun] Level {level}: {len(_lo_solution)} clicks from GF solver")
                    mind._last_progress_at_call = mind.total_llm_calls
            except Exception as e:
                if verbose:
                    print(f"  [researcher-rerun] Failed: {e}")

        if _lo_solution is not None and len(_lo_solution) > 0:
            # Batch ALL remaining oracle clicks into one sequence — no LLM calls needed
            _is_oracle = True
            all_clicks = list(_lo_solution)  # copy
            _lo_solution.clear()
            decision = {
                'action': _lo_click_idx,
                'data': {'x': all_clicks[0][0], 'y': all_clicks[0][1]},
                'hypothesis': 'Lights-out solver (GF2 Gaussian elimination)',
                'sequence': [(_lo_click_idx, {'x': cp[0], 'y': cp[1]}) for cp in all_clicks]
            }
            if verbose:
                print(f"  [lights-out] Executing {len(all_clicks)} clicks in batch")
        else:
            # ── MUSCLE MEMORY: "I think I know how this works, lemme try it" ──
            # If the spatial model has a path to the target, execute it without
            # burning an LLM call. This is learned knowledge from past observation,
            # not brute-force search. Like a musician replaying a practiced passage.
            # If it fails (blocks/stalls), fall back to LLM thinking.
            # Approved by Peter, 2026-03-16.
            _muscle_path = []
            _muscle_mode = ""

            # ── DEAD-END ESCAPE ──
            # If all directions are blocked, clear stale blocks and retry.
            # Modifiers change the world — walls may have moved.
            if (mind.spatial.action_map
                    and mind.spatial.player_pos != (0, 0)
                    and mind.spatial.is_dead_end()):
                cleared = mind.spatial.clear_dead_end()
                if verbose and cleared:
                    print(f"  [dead-end-escape] Cleared {cleared} stale blocks from {mind.spatial.player_pos} — retrying")

            if (mind.spatial.action_map
                    and mind.spatial.player_pos != (0, 0)
                    and mind.total_llm_calls > 0):  # at least one LLM call first — observe before assuming
                if mind.spatial.target_pos != (-1, -1):
                    _muscle_path = mind.spatial.plan_path()
                    _muscle_mode = "navigate"
                elif mind.spatial.known_modifiers:
                    # We know modifiers exist — navigate to closest one
                    # Try unvisited first, but if stuck, revisit ANY modifier
                    py, px = mind.spatial.player_pos
                    unvisited_mods = [(my, mx) for my, mx in mind.spatial.known_modifiers
                                      if (my, mx) not in mind.spatial.visited]
                    # Fallback: if no unvisited, try ALL modifiers (revisit with new state)
                    candidate_mods = unvisited_mods if unvisited_mods else [
                        (my, mx) for my, mx in mind.spatial.known_modifiers
                        if (my, mx) != mind.spatial.player_pos]
                    if candidate_mods:
                        closest_mod = min(candidate_mods, key=lambda p: abs(p[0]-py) + abs(p[1]-px))
                        _muscle_path = mind.spatial.find_path(target=closest_mod)
                        _muscle_mode = "modifier-seek"
                    else:
                        _muscle_path = mind.spatial.explore_path(max_steps=15)
                        _muscle_mode = "explore"
                else:
                    # No target, no modifiers — explore unvisited frontier
                    _muscle_path = mind.spatial.explore_path(max_steps=15)
                    _muscle_mode = "explore"

            if _muscle_path:
                # Cap sequence length — explore longer, navigate shorter
                # Navigate: longer cap if path goes through known-visited positions (safe corridor)
                _base_caps = {"explore": 15, "modifier-seek": 10, "navigate": 5}
                _mm_cap = _base_caps.get(_muscle_mode, 8)
                if _muscle_mode == "navigate" and mind.spatial.visited:
                    # Count how many path steps go through already-visited positions
                    _known_steps = 0
                    _pos = mind.spatial.player_pos
                    for _aid in _muscle_path:
                        if _aid in mind.spatial.action_map:
                            _dy, _dx, _ = mind.spatial.action_map[_aid]
                            _next = (_pos[0] + _dy, _pos[1] + _dx)
                            if _next in mind.spatial.visited:
                                _known_steps += 1
                                _pos = _next
                            else:
                                break
                        else:
                            break
                    # Through known territory: extend cap (max 12)
                    if _known_steps >= 3:
                        _mm_cap = min(12, max(_mm_cap, _known_steps))
                _mm_len = min(len(_muscle_path), _mm_cap)
                _muscle_seq = [(_muscle_path[i], {}) for i in range(_mm_len)]
                # Modifier probe: after reaching a modifier, cycle on/off to test effect
                if _muscle_mode == "modifier-seek" and _mm_len == len(_muscle_path):
                    # We'll reach the modifier — append probe cycle (step away + step back)
                    last_action = _muscle_path[-1] if _muscle_path else 0
                    # Find a reverse action (opposite direction)
                    reverse = None
                    if last_action in mind.spatial.action_map:
                        dy, dx, _ = mind.spatial.action_map[last_action]
                        for aid, (ady, adx, _) in mind.spatial.action_map.items():
                            if ady == -dy and adx == -dx:
                                reverse = aid
                                break
                    if reverse is not None:
                        # Step away, step back = probe the modifier twice
                        _muscle_seq.append((reverse, {}))    # step off
                        _muscle_seq.append((last_action, {}))  # step back on
                        if verbose:
                            print(f"  [modifier-probe] Will cycle on/off at modifier to test effect")
                decision = {
                    'action': _muscle_path[0],
                    'data': {},
                    'hypothesis': f'Muscle memory ({_muscle_mode}) — {len(_muscle_path)} steps planned, executing {_mm_len}',
                    'sequence': _muscle_seq,
                }
                if verbose:
                    _dirs = []
                    for _aid in _muscle_path[:_mm_len]:
                        if _aid in mind.spatial.action_map:
                            _dirs.append(mind.spatial.action_map[_aid][2])
                        else:
                            _dirs.append(f'a{_aid}')
                    print(f"  [muscle-memory:{_muscle_mode}] Path: {' → '.join(_dirs)} ({_mm_len}/{len(_muscle_path)} steps)")
            else:
                # THINK: ask the mind what to do — returns a SEQUENCE of actions
                decision = mind.think(lvl, phase="act")

        # EXECUTE: run sequence — cap length for navigation (observe often!)
        sequence = decision.get('sequence', [(decision['action'], decision.get('data', {}))])
        if not sequence:
            sequence = [(decision['action'], decision.get('data', {}))]

        # Walk before you run — short sequences early, longer once you know the rules
        # Only apply to LLM-generated plans, not muscle memory or oracle solutions
        # For non-navigation games (deduction, sequence), allow longer sequences from start
        _is_muscle = _muscle_path and len(_muscle_path) > 0
        _env_genre = getattr(getattr(mind, '_env_profile', None), 'genre', '')
        _is_nav = _env_genre in ('navigation_maze', 'navigation', 'exploration', 'push_block')
        if not _is_muscle and not _is_oracle:
            llm_calls = mind.total_llm_calls
            if _is_nav:
                # Navigation: short sequences early, observe often
                if llm_calls <= 1:
                    max_seq = 4
                elif llm_calls <= 3:
                    max_seq = 8
                else:
                    max_seq = len(sequence)
            else:
                # Non-navigation (deduction, sequence, toggle): allow full sequences
                # The pilot needs to submit complete answers, not piecemeal
                if llm_calls <= 1:
                    max_seq = 20  # generous first attempt for deduction puzzles
                else:
                    max_seq = len(sequence)  # trust the pilot
            if len(sequence) > max_seq:
                if verbose:
                    print(f"  [gundam] Walk-before-run: capped {len(sequence)} → {max_seq} steps (call #{llm_calls})")
                sequence = sequence[:max_seq]

        # ── Auto-exploration ──
        # If LLM returned very short plan and we have unexplored frontier,
        # append exploration actions. Saves LLM calls by moving between them.
        if len(sequence) <= 2 and mind.spatial.action_map:
            explore_actions = mind.spatial.explore_path(max_steps=15)
            if explore_actions:
                for ea in explore_actions[:12]:
                    sequence.append((ea, None))
            elif mind.action_predictor:
                # No spatial frontier — use CNN predictions to extend the sequence
                _cnn_frame = frame[0] if frame.ndim == 3 else frame
                _cnn_preds = mind.predict_best_actions(_cnn_frame, top_k=3)
                if _cnn_preds:
                    # Add predicted best actions as exploration
                    for _cnn_aid, _cnn_conf in _cnn_preds[:6]:
                        sequence.append((_cnn_aid, {}))
                    if verbose:
                        print(f"  [gundam] CNN-explore: appended {min(len(_cnn_preds), 6)} predicted actions")
                if verbose:
                    print(f"  [gundam] Auto-explore: appended {min(len(explore_actions), 15)} steps toward frontier")

        if verbose and len(sequence) > 1:
            print(f"  [gundam] Executing sequence of {len(sequence)} actions")

        level_done = False
        for seq_idx, (action_id, data) in enumerate(sequence):
            action_id = max(0, min(action_id, n_actions - 1))  # clamp
            frame_before = frame.copy()

            try:
                actual_action = obs.available_actions[action_id] if action_id < len(obs.available_actions) else obs.available_actions[0]
                if data:
                    obs = env.step(actual_action, data)
                else:
                    obs = env.step(actual_action)
            except Exception as e:
                if verbose:
                    print(f"  [gundam] Step failed: {e}")
                continue

            frame = _extract_frame(obs)
            n_actions = len(obs.available_actions)

            won = obs.levels_completed > prev_levels_completed
            done = ('WIN' in str(obs.state).upper() or 'LOSS' in str(obs.state).upper() or 'GAME_OVER' in str(obs.state).upper() or (str(obs.state).upper().endswith('FINISHED') and 'NOT_FINISHED' not in str(obs.state).upper()))

            action_desc = action_info.get(action_id, f"action_{action_id}")
            mind._last_action_data = data or {}
            exp = mind.observe_effect(lvl, action_id, action_desc, frame_before, frame, won)
            exp.hypothesis_at_time = decision.get('hypothesis', '')

            # ABORT sequence on consecutive blocks — don't ram into walls
            # Fuel ticks where player didn't move also count as blocks
            # For click/toggle puzzles: 0px on a click is just a miss, not a wall
            _env_prof = getattr(mind, '_env_profile', None)
            _is_click_action = _env_prof and action_id in getattr(_env_prof, 'param_actions', [])
            _is_toggle = _env_prof and getattr(_env_prof, 'genre', '') == 'toggle_puzzle'
            # Click/parameterized actions: 0px = miss, not wall (applies to all click games, not just toggles)
            is_blocked = exp.pixels_changed == 0 and not _is_click_action
            is_fuel_blocked = getattr(mind, '_last_was_fuel_blocked', False)
            mind._last_was_fuel_blocked = False  # reset after checking
            if (is_blocked or is_fuel_blocked) and len(sequence) > 1:
                consecutive_blocks = getattr(mind, '_seq_blocks', 0) + 1
                mind._seq_blocks = consecutive_blocks
                if consecutive_blocks >= 2:
                    if verbose:
                        reason = "wall (0px)" if is_blocked else f"fuel-tick ({exp.pixels_changed}px)"
                        print(f"  [gundam] Aborting sequence — {consecutive_blocks} consecutive {reason}")
                    mind._seq_blocks = 0
                    break
            else:
                mind._seq_blocks = 0

            # ── Causal target demotion (deferred) ──
            # If player reached a candidate but level didn't complete, demote it
            if not won and hasattr(mind, '_pending_demotions') and mind._pending_demotions:
                for tc, pos in mind._pending_demotions:
                    mind.spatial.demote_target(tc, f'player reached {pos} but level did not complete')
                    if verbose:
                        print(f"  [causal-target] Demoted color {tc} — reached {pos}, no level completion")
                mind._pending_demotions.clear()
            elif won and hasattr(mind, '_pending_demotions'):
                mind._pending_demotions.clear()  # don't demote on win

            # ── Corpus callosum: incremental visual changelog ──
            # Triggers on: (1) unexpected changes, (2) first action, (3) every 5th action as fallback
            _total_acts = getattr(mind, '_total_actions_this_level', 0) + 1
            mind._total_actions_this_level = _total_acts
            # Track action history for cross-level carryover
            if not hasattr(mind, '_level_action_history'):
                mind._level_action_history = []
            mind._level_action_history.append(action_id)
            _should_perceive = False
            _is_nav_game = _env_prof and getattr(_env_prof, 'genre', '').startswith('navigation')
            if HAS_VLM and not won and not getattr(mind, '_lo_solution_found_this_level', False):
                if _total_acts == 1:  # first action — always perceive
                    _should_perceive = True
                elif _is_nav_game:
                    # Navigation: perceive only on large unexpected changes or every 8th action
                    if exp.pixels_changed > 200:  # much larger change than normal movement
                        _should_perceive = True
                    elif _total_acts % 8 == 0 and exp.pixels_changed > 5:
                        _should_perceive = True
                elif exp.pixels_changed > 20:
                    # Click/toggle games: perceive on any significant change
                    _should_perceive = True
                elif _total_acts % 3 == 0 and exp.pixels_changed > 5:
                    # Periodic fallback — every 3rd action with visible change
                    _should_perceive = True
            if _should_perceive:
                try:
                    # Multi-frame diff: compare against FIRST keyframe (baseline) and LAST keyframe
                    if not hasattr(lvl, '_keyframes'):
                        lvl._keyframes = [frame_before.copy()]  # baseline = frame before first action

                    # Compute ASCII grids for both frames
                    _grid_before = frame_to_grid_text(frame_before)
                    _grid_after = frame_to_grid_text(frame)
                    _action_str = action_desc[:60] if isinstance(action_desc, str) else str(action_id)

                    # Try Qwen-VL first (ASCII + pixel art), fall back to Florence
                    _diff_desc = None
                    if os.environ.get('ARC_USE_QWEN_VL'):
                        _diff_desc = corpus_callosum_qwen(
                            frame_before, frame,
                            action_desc=_action_str,
                            grid_before=_grid_before, grid_after=_grid_after)
                    if not _diff_desc:
                        _diff_desc = arc_vlm.perceive_diff(
                            frame_before, frame, action_desc=_action_str)

                    if _diff_desc and '(VLM error' not in _diff_desc:
                        if not hasattr(lvl, '_visual_changelog'):
                            lvl._visual_changelog = []
                        lvl._visual_changelog.append(f"After action {_total_acts} ({action_desc[:30]}): {_diff_desc}")
                        # Store for turn prompt — triggers fresh Qwen perception on next reasoning call
                        lvl._corpus_callosum_report = _diff_desc
                        lvl._keyframes.append(frame.copy())
                        # Every 3rd keyframe: also diff baseline vs current for trend detection
                        if len(lvl._keyframes) >= 3 and len(lvl._keyframes) % 3 == 0:
                            _trend = None
                            if os.environ.get('ARC_USE_QWEN_VL'):
                                _grid_base = frame_to_grid_text(lvl._keyframes[0])
                                _trend = corpus_callosum_qwen(
                                    lvl._keyframes[0], frame,
                                    action_desc=f"TREND: start vs now ({_total_acts} actions later)",
                                    grid_before=_grid_base, grid_after=_grid_after)
                            if not _trend:
                                _trend = arc_vlm.perceive_diff(
                                    lvl._keyframes[0], frame,
                                    action_desc=f"TREND: comparing start of level vs now ({_total_acts} actions later)")
                            if _trend and '(VLM error' not in _trend:
                                lvl._visual_changelog.append(f"TREND (actions 1→{_total_acts}): {_trend}")
                                if verbose:
                                    print(f"  [corpus-callosum-trend] {_trend[:80]}")
                        if verbose:
                            print(f"  [corpus-callosum] {_diff_desc[:80]}")
                except Exception:
                    pass  # VLM failure shouldn't break the game

            if won:
                lvl.solved = True
                mind.memory.levels_solved = obs.levels_completed
                # Record winning action sequence for cross-level carryover
                lvl.winning_sequence = getattr(mind, '_level_action_history', [])[:]
                # Reinforce CNN with winning action sequence
                if mind.action_predictor and lvl.winning_sequence:
                    mind.action_predictor.reinforce(lvl.winning_sequence)
                # ── Causal target confirmation ──
                # Level completed = whatever we just reached IS the target
                if mind.spatial.target_color >= 0:
                    mind.spatial.confirm_target(mind.spatial.target_color)
                    if verbose:
                        print(f"  [causal-target] CONFIRMED: color {mind.spatial.target_color} = target (level completed on contact)")
                if verbose:
                    print(f"\n  *** LEVEL {level} SOLVED in {lvl.total_actions} actions ({seq_idx+1}/{len(sequence)} in sequence)! ***\n")
                # ── Dopamine: level solved ──
                try:
                    from scripts.agent_dopamine import AgentMoodTracker
                    AgentMoodTracker().signal("apollo", "craft_win", f"Level {level} solved in {lvl.total_actions} actions — {mind.memory.game_id}")
                except Exception:
                    pass  # dopamine is optional — don't break the flight

                prev_levels_completed = obs.levels_completed
                # Track if oracle solved this level (for model selection)
                if _lo_solution is not None:
                    mind._lo_solution_found_this_level = True

                if done:
                    if verbose:
                        print(f"\n  === GAME COMPLETE: {mind.memory.levels_solved}/{total_levels} levels solved ===")
                    level_done = True
                    break

                # New level — fresh eyes, stop old sequence
                level = obs.levels_completed
                mind._current_level = level
                mind._total_actions_this_level = 0  # reset visual changelog counter
                mind._level_action_history = []  # reset action history for new level
                mind._lo_solution_found_this_level = False  # reset oracle flag
                # Scenery diff: what's constant between levels is infrastructure
                # _prev_level_frame = first frame of OLD level, frame = first frame of NEW level
                mind.spatial.learn_scenery_from_diff(_prev_level_frame, frame)
                if verbose and mind.spatial.scenery_colors:
                    print(f"  [causal-target] Scenery colors (constant between levels): {mind.spatial.scenery_colors}")
                _prev_level_frame = frame.copy()  # store first frame of new level for next transition
                lvl = mind.observe_initial(frame, n_actions, action_info, level)
                mind.total_llm_calls = 0
                # Preflight on new level
                try:
                    mind.preflight(env, _extract_frame, level)
                    frame = _extract_frame(env)
                    lvl.current_frame = frame[0].copy() if frame.ndim == 3 else frame.copy()
                    lvl.current_frame_b64 = frame_to_b64(frame)
                except Exception:
                    pass
                # Clear stale state from previous level
                mind._researcher_target_map = {}
                mind._researcher_current_map = {}
                mind._oracle_cell_positions = None  # clear stale positions
                mind._lo_solution_found_this_level = False  # reset for new level
                if verbose:
                    print(f"  [gundam] Level transition: _lo_click_idx={_lo_click_idx}, llm_calls={mind.total_llm_calls}/{mind.budget_per_level}")
                # Run oracle first (discovers cell positions for researcher)
                _lo_solution = _try_lights_out_solve()
                # Re-run standalone researcher on new level (handles GF(2) without LLM)
                if _lo_click_idx is not None:
                    try:
                        new_frame = _extract_frame(obs)
                        new_clicks = _researcher_solve(env, mind, _extract_frame, new_frame, action_info, obs, verbose)
                        if new_clicks:
                            # Convert researcher clicks to _lo_solution format so
                            # the main loop picks them up at line 8648
                            _lo_solution = []
                            for act_id, data in new_clicks:
                                if data and isinstance(data, dict):
                                    _lo_solution.append((data.get('x', 0), data.get('y', 0)))
                                elif data and isinstance(data, (list, tuple)):
                                    _lo_solution.append(tuple(data))
                            if not _lo_solution:
                                # Non-click actions — fall through to researcher_clicks
                                _researcher_clicks = new_clicks
                            if verbose:
                                print(f"  [gundam] Researcher re-solved level {level}: {len(new_clicks)} clicks → _lo_solution")
                    except Exception as e:
                        if verbose:
                            print(f"  [gundam] Researcher re-run failed: {e}")
                # Always ask researcher for target prediction on new levels
                # (oracle may return clicks that don't actually solve — wrong target guess)
                if _lo_click_idx is not None and mind.total_llm_calls < mind.budget_per_level:
                    try:
                        if verbose:
                            n_cells = len(mind._oracle_cell_positions) if getattr(mind, '_oracle_cell_positions', None) else 0
                            print(f"  [gundam] Running researcher for new level ({n_cells} cells)...")
                        new_frame = _extract_frame(obs)
                        _ask_researcher_for_level(mind, env, new_frame, _lo_click_idx, obs, verbose)
                        if verbose:
                            has_target = bool(getattr(mind, '_researcher_target_map', {}))
                            print(f"  [gundam] Researcher result: has_target={has_target}")
                        # Re-solve with researcher's target prediction
                        if getattr(mind, '_researcher_target_map', None):
                            _lo_solution = _try_lights_out_solve()
                    except Exception as e:
                        if verbose:
                            print(f"  [gundam] Researcher re-run failed: {e}")
                if _lo_solution is not None and len(_lo_solution) > 0 and verbose:
                    print(f"  [gundam] LIGHTS-OUT SOLVER: {len(_lo_solution)} clicks for level {level}")
                _level_just_changed = True  # trigger researcher re-run on next turn
                level_done = True
                mind._last_progress_at_call = mind.total_llm_calls
                break  # new level needs fresh thinking

            if done:
                level_done = True
                mind._last_progress_at_call = mind.total_llm_calls
                break

        if level_done and done:
            if verbose and not won:
                print(f"\n  Game ended. Solved {mind.memory.levels_solved}/{total_levels} levels.")
            break

        if mind.total_llm_calls >= max(mind.budget_per_level, 1) and not level_done:
            if verbose:
                print(f"\n  Budget exhausted for level {level}. Moving on.")
            break

        # Global cost ceiling check — abort game if approaching limit
        if _api_cost_usd >= _COST_CEILING * 0.95:
            if verbose:
                print(f"\n  ⚠️ Global cost ceiling ({_COST_CEILING:.2f}) approaching. Ending game.")
            break

        # Early abort: if N consecutive LLM calls produced zero level progress, give up
        # Toggle puzzles: 3 calls (oracle verifies quickly). Navigation: 15 (needs many moves). Others: 6
        _has_oracle_solution = getattr(mind, '_lo_solution_found_this_level', False)
        _genre = getattr(getattr(mind, '_env_profile', None), 'genre', '') or ''
        _is_nav = 'nav' in _genre
        _no_prog_limit = 3 if _has_oracle_solution else (15 if _is_nav else 10)
        _stall_count = (mind.total_llm_calls - getattr(mind, '_last_progress_at_call', 0)
                        if hasattr(mind, '_last_progress_at_call') else mind.total_llm_calls)
        # Strategy pivot at half-limit: force micro-dose to loosen assumptions
        if (_stall_count == _no_prog_limit // 2 and not level_done
                and not getattr(mind, '_pivot_triggered', False)):
            mind._pivot_triggered = True
            mind._force_microdose = True  # next think() call uses micro-dose prompt
            if verbose:
                print(f"\n  [PIVOT] Stalled for {_stall_count} calls — forcing micro-dose to challenge assumptions")
        if (mind.total_llm_calls >= _no_prog_limit and not level_done
                and hasattr(mind, '_last_progress_at_call')
                and _stall_count >= _no_prog_limit):
            if verbose:
                print(f"\n  No progress in {_no_prog_limit} consecutive LLM calls. Giving up on level {level}.")
            break

    # Summary
    if verbose:
        print(f"\n{'='*60}")
        print(f"  RESULTS: {mind.memory.levels_solved}/{mind.memory.levels_total or '?'} levels solved")
        print(f"  LLM calls: {mind.total_llm_calls}")
        print(f"  Rules discovered: {len(mind.memory.rules_discovered)}")
        for r in mind.memory.rules_discovered:
            print(f"    - {r}")
        print(f"  Game hypothesis: {mind.memory.game_hypothesis}")
        if mind.spatial.action_map:
            print(f"  Action map: {mind.spatial.action_map}")
        print(f"{'='*60}")

    # ── Dopamine: game complete ──
    try:
        from scripts.agent_dopamine import AgentMoodTracker
        _dopamine = AgentMoodTracker()
        solved = mind.memory.levels_solved
        total = mind.memory.levels_total or 1
        if solved > 0:
            _dopamine.signal("apollo", "craft_win", f"Game {mind.memory.game_id}: {solved}/{total} levels solved")
        else:
            _dopamine.signal("apollo", "craft_miss", f"Game {mind.memory.game_id}: 0/{total} — no levels solved")
    except Exception:
        pass

    # REMEMBER: store what we learned for future games
    mind.remember_game()

    return mind


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(message)s')

    if len(sys.argv) > 1 and sys.argv[1] == "--bench":
        # Benchmark mode — run all 3 preview games, output summary
        budget = int(sys.argv[2]) if len(sys.argv) > 2 else 10
        turns = int(sys.argv[3]) if len(sys.argv) > 3 else 50
        games = ["ft09", "vc33", "ls20"]
        results = {}
        for gid in games:
            print(f"\n{'='*60}")
            print(f"BENCHMARK: {gid}")
            print(f"{'='*60}")
            try:
                mind = run_game(gid, max_turns_per_level=turns, budget_per_level=budget)
                solved = mind.memory.levels_solved
                total = mind.memory.levels_total or '?'
                results[gid] = f"{solved}/{total}"
            except Exception as e:
                results[gid] = f"ERROR: {e}"
        print(f"\n{'='*60}")
        print("BENCHMARK RESULTS")
        print(f"{'='*60}")
        for gid, result in results.items():
            print(f"  {gid}: {result}")
        total_solved = sum(int(r.split('/')[0]) for r in results.values() if '/' in r and 'ERROR' not in r)
        total_levels = sum(int(r.split('/')[1]) for r in results.values() if '/' in r and 'ERROR' not in r)
        print(f"  TOTAL: {total_solved}/{total_levels}")
    else:
        game_id = sys.argv[1] if len(sys.argv) > 1 else "vc33"
        budget = int(sys.argv[2]) if len(sys.argv) > 2 else 15
        turns = int(sys.argv[3]) if len(sys.argv) > 3 else 50

        print(f"Launching GUNDAM against {game_id}")
        print(f"Budget: {budget} LLM calls/level | Max {turns} turns/level")

        mind = run_game(game_id, max_turns_per_level=turns, budget_per_level=budget)
