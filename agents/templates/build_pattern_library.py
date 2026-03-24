#!/usr/bin/env python3
"""Build trilingual pattern library for ARC solver + Qwen VLM.

Three encodings of the same knowledge:
1. Machine (JSON) — solver consumes: effect matrices, routes, parameters
2. Human (text) — strategy descriptions, analogies, reasoning steps
3. Visual (PNG frames) — reference frames for few-shot VLM prompting

Captures key frames during game solving and distills them into
retrievable patterns for unknown games.
"""

import copy
import json
import hashlib
import logging
import os
import sys
import time
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import Optional

import numpy as np

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s',
                    datefmt='%H:%M:%S')

LIBRARY_DIR = Path(__file__).parent.parent / "knowledge" / "pattern_library"
FRAMES_DIR = LIBRARY_DIR / "frames"


@dataclass
class GamePattern:
    """Trilingual pattern entry."""
    pattern_id: str
    genre: str  # e.g., "toggle_grid", "pump_overflow", "navigation_maze"

    # Machine language (solver consumes)
    machine: dict = field(default_factory=dict)
    # {solver_route, effect_matrix, modular_period, action_count, ...}

    # Human language (readable strategy)
    human: dict = field(default_factory=dict)
    # {description, analogy, strategy_steps, difficulty, ...}

    # Visual language (VLM consumes)
    visual: dict = field(default_factory=dict)
    # {initial_frame, solved_frame, action_effects: [{before, after, description}], visual_cues}

    # Metadata
    source_games: list = field(default_factory=list)
    confidence: float = 0.0
    times_matched: int = 0


def save_frame(frame: np.ndarray, name: str) -> str:
    """Save a frame as PNG, return relative path."""
    FRAMES_DIR.mkdir(parents=True, exist_ok=True)
    # Use PIL if available, else save as raw numpy
    try:
        from PIL import Image
        if frame.ndim == 3 and frame.shape[0] in (1, 3, 4):
            # CHW -> HWC
            img_data = np.transpose(frame, (1, 2, 0))
        else:
            img_data = frame
        if img_data.ndim == 2:
            img = Image.fromarray(img_data.astype(np.uint8), mode='L')
        elif img_data.ndim == 3 and img_data.shape[2] == 1:
            img = Image.fromarray(img_data[:, :, 0].astype(np.uint8), mode='L')
        elif img_data.ndim == 3 and img_data.shape[2] == 3:
            img = Image.fromarray(img_data.astype(np.uint8), mode='RGB')
        elif img_data.ndim == 3 and img_data.shape[2] == 4:
            img = Image.fromarray(img_data.astype(np.uint8), mode='RGBA')
        else:
            # Unusual shape (e.g., 64 channels) — just take first channel as grayscale
            img = Image.fromarray(img_data[:, :, 0].astype(np.uint8), mode='L')
        path = FRAMES_DIR / f"{name}.png"
        img.save(path)
        return f"frames/{name}.png"
    except ImportError:
        # Fallback: save as .npy
        path = FRAMES_DIR / f"{name}.npy"
        np.save(path, frame)
        return f"frames/{name}.npy"


def capture_game_patterns(arcade, game_id: str) -> Optional[GamePattern]:
    """Play a game and capture its pattern in all three languages."""
    sys.path.insert(0, str(Path(__file__).parent))
    from arc_agent_v05 import (discover_actions, analyze_game, Action,
                                constraint_solve_v2, is_level_won)

    env = arcade.make(game_id)
    if not env:
        return None

    obs = env.reset()
    frame0 = np.array(obs.frame)

    # Discover actions
    env_snap = copy.deepcopy(env)
    actions = discover_actions(env_snap, frame0, obs.available_actions)
    env_snap = copy.deepcopy(env)
    profile = analyze_game(env_snap, actions, frame0)

    click_actions = [a for a in actions if a.data and 'x' in a.data]
    dir_actions = [a for a in actions if not a.data or 'x' not in a.data]

    pid = hashlib.md5(f"{game_id}_{profile.game_type}".encode()).hexdigest()[:8]

    pattern = GamePattern(
        pattern_id=pid,
        genre=profile.game_type.lower(),
        source_games=[game_id],
        confidence=0.8,
    )

    # ── Visual language: capture key frames ──
    # Initial frame
    init_path = save_frame(frame0, f"{game_id}_initial")
    pattern.visual['initial_frame'] = init_path
    pattern.visual['frame_shape'] = list(frame0.shape)
    pattern.visual['visual_cues'] = []

    # Probe each click action and capture before/after
    action_effects = []
    for i, act in enumerate(click_actions[:8]):  # Cap at 8 for storage
        ec = copy.deepcopy(env)
        before = np.array(ec.step(actions[0].game_action,
                                   actions[0].data if actions[0].data else None).frame)
        ec2 = copy.deepcopy(env)
        obs_after = ec2.step(act.game_action, act.data if act.data else None)
        if obs_after is None:
            continue
        after = np.array(obs_after.frame)

        # Save frames
        before_path = save_frame(before, f"{game_id}_btn{i}_before")
        after_path = save_frame(after, f"{game_id}_btn{i}_after")

        # Compute diff stats
        if before.shape == after.shape:
            diff = (before != after)
            n_changed = int(diff.any(axis=0).sum()) if diff.ndim == 3 else int(diff.sum())
            total_pixels = diff.shape[-2] * diff.shape[-1] if diff.ndim == 3 else diff.size

            action_effects.append({
                'action_id': act.game_action,
                'action_data': act.data,
                'before_frame': before_path,
                'after_frame': after_path,
                'pixels_changed': n_changed,
                'change_fraction': round(n_changed / max(total_pixels, 1), 4),
                'description': f"Button {i} at ({act.data.get('x','?')},{act.data.get('y','?')}): {n_changed} pixels changed ({n_changed/max(total_pixels,1)*100:.1f}%)"
            })

    pattern.visual['action_effects'] = action_effects

    # Visual cues from profile
    if click_actions:
        pattern.visual['visual_cues'].append("click-based interaction")
    if dir_actions:
        pattern.visual['visual_cues'].append("directional movement")
    if profile.game_type == "CLICK_TOGGLE":
        pattern.visual['visual_cues'].append("cells toggle on click")
        pattern.visual['visual_cues'].append("neighbor coupling possible")
    elif profile.game_type == "NAVIGATION":
        pattern.visual['visual_cues'].append("player sprite moves")
        pattern.visual['visual_cues'].append("maze or obstacle layout")

    # ── Machine language: solver parameters ──
    pattern.machine = {
        'game_type': profile.game_type,
        'n_actions': len(actions),
        'n_click_actions': len(click_actions),
        'n_dir_actions': len(dir_actions),
        'solver_route': _best_route_for_type(profile.game_type),
        'frame_shape': list(frame0.shape),
    }

    # For click games: probe for modular period
    if click_actions and len(click_actions) <= 30:
        periods = _detect_periods(env, click_actions)
        if periods:
            pattern.machine['modular_periods'] = periods

    # ── Human language: strategy description ──
    pattern.human = {
        'description': _human_description(profile.game_type, len(click_actions), len(dir_actions)),
        'analogy': _human_analogy(profile.game_type),
        'strategy_steps': _strategy_steps(profile.game_type),
        'difficulty': 'unknown',
    }

    return pattern


def _best_route_for_type(game_type: str) -> str:
    routes = {
        'CLICK_TOGGLE': 'constraint_solve_v2',
        'CLICK_SEQUENCE': 'mechanic_learning_solve',
        'NAVIGATION': 'navigation_solve',
        'SORTING': 'beam_search_solve',
        'UNKNOWN': 'full_cascade',
    }
    return routes.get(game_type, 'full_cascade')


def _detect_periods(env, click_actions, max_presses=20):
    """Detect modular period of each button by pressing until state repeats."""
    periods = {}
    for act in click_actions[:8]:
        ec = copy.deepcopy(env)
        frames = []
        obs = ec.step(act.game_action, act.data if act.data else None)
        if obs is None:
            continue
        frames.append(np.array(obs.frame).tobytes())
        for press in range(1, max_presses):
            obs = ec.step(act.game_action, act.data if act.data else None)
            if obs is None:
                break
            fb = np.array(obs.frame).tobytes()
            if fb in frames:
                periods[f"btn_{act.game_action}_{act.data}"] = press
                break
            frames.append(fb)
    return periods


def _human_description(game_type, n_click, n_dir):
    descs = {
        'CLICK_TOGGLE': f"A toggle puzzle with {n_click} clickable cells. Clicking changes the state of cells — possibly neighbors too. Goal: reach a target configuration. Think of it like Lights Out.",
        'CLICK_SEQUENCE': f"A sequence puzzle with {n_click} buttons. Each button has a specific effect. The ORDER matters. Think of it like a combination lock — you need the right sequence.",
        'NAVIGATION': f"A maze/pathfinding game with {n_dir} movement directions. Navigate to collect items or reach a goal. Walls block movement. Resources may be limited.",
        'SORTING': f"An ordering puzzle. Arrange elements into the correct sequence. Think of it like sorting cards — each swap has a cost.",
        'UNKNOWN': f"Unknown game type with {n_click} click and {n_dir} directional actions. Observe carefully before committing.",
    }
    return descs.get(game_type, f"Game with {n_click} click + {n_dir} directional actions.")


def _human_analogy(game_type):
    analogies = {
        'CLICK_TOGGLE': "Like a Rubik's cube face — each move affects multiple pieces. Find the algebra, don't brute force.",
        'CLICK_SEQUENCE': "Like a combination lock with visible dials. Watch what each button does. The solution is a formula, not a search.",
        'NAVIGATION': "Like a maze with keys and doors. Map the world first. Plan the route second. Execute third.",
        'SORTING': "Like sorting a hand of cards. Minimize swaps by identifying the longest already-sorted subsequence.",
        'UNKNOWN': "Look before you leap. Press each button once. Watch. Think. Then act.",
    }
    return analogies.get(game_type, "Observe, hypothesize, test, execute.")


def _strategy_steps(game_type):
    steps = {
        'CLICK_TOGGLE': [
            "1. Press each button once and observe what changes",
            "2. Build effect matrix: button → affected pixels",
            "3. Check for modular period (does pressing K times return to start?)",
            "4. Solve as linear algebra over Z/kZ",
            "5. If hidden state detected (same frame, different outcome): track internal state"
        ],
        'CLICK_SEQUENCE': [
            "1. Press each button once from fresh state — catalog effects",
            "2. Look for cascading/chain reactions (one button triggers another)",
            "3. Identify dependency order (which buttons must come first?)",
            "4. Try the dependency-ordered sequence",
            "5. If order-dependent: probe pairs (A then B vs B then A)"
        ],
        'NAVIGATION': [
            "1. Map available movement directions",
            "2. Identify player sprite and goal/target sprites",
            "3. Check for resource constraints (fuel bar, step counter)",
            "4. A* search with heuristic toward nearest target",
            "5. If keys/doors: determine collection order (dependency graph)"
        ],
        'UNKNOWN': [
            "1. Press each button once — observe effects (LOOK)",
            "2. Classify: toggle, sequence, navigation, or hybrid?",
            "3. Check for self-inverse actions (press twice = undo)",
            "4. Check for commutativity (order independence)",
            "5. Match to closest known pattern and apply that strategy"
        ],
    }
    return steps.get(game_type, ["1. Observe", "2. Hypothesize", "3. Test", "4. Execute"])


def save_library(patterns: list[GamePattern], path: Path = None):
    """Save pattern library as JSON."""
    if path is None:
        path = LIBRARY_DIR / "patterns.json"
    path.parent.mkdir(parents=True, exist_ok=True)

    data = {
        'version': '1.0',
        'created': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
        'n_patterns': len(patterns),
        'patterns': [asdict(p) for p in patterns],
    }
    with open(path, 'w') as f:
        json.dump(data, f, indent=2, default=str)
    log.info(f"Saved {len(patterns)} patterns to {path}")


def load_library(path: Path = None) -> list[dict]:
    """Load pattern library."""
    if path is None:
        path = LIBRARY_DIR / "patterns.json"
    if not path.exists():
        return []
    with open(path) as f:
        data = json.load(f)
    return data.get('patterns', [])


def format_for_vlm(patterns: list[dict], unknown_frame: np.ndarray = None) -> str:
    """Format patterns as VLM prompt context.

    Returns a text description suitable for injecting into a VLM system prompt,
    with references to visual frames that should be included as images.
    """
    lines = ["# Game Pattern Library — Reference Strategies\n"]
    lines.append("Below are patterns learned from previous games. Use these to identify")
    lines.append("the type of game you're facing and choose the right strategy.\n")

    for p in patterns:
        lines.append(f"## Pattern: {p['genre'].upper()}")
        lines.append(f"**Analogy**: {p['human'].get('analogy', 'N/A')}")
        lines.append(f"**Description**: {p['human'].get('description', 'N/A')}")
        lines.append(f"\n**Strategy**:")
        for step in p['human'].get('strategy_steps', []):
            lines.append(f"  {step}")
        lines.append(f"\n**Visual cues to look for**: {', '.join(p['visual'].get('visual_cues', []))}")
        if p['visual'].get('action_effects'):
            lines.append(f"**Action effects** ({len(p['visual']['action_effects'])} probed):")
            for ae in p['visual']['action_effects'][:3]:
                lines.append(f"  - {ae['description']}")
        lines.append("")

    return "\n".join(lines)


if __name__ == '__main__':
    from arc_agi import Arcade

    arcade = Arcade()
    patterns = []

    for env_info in arcade.get_environments():
        gid = env_info.game_id
        log.info(f"Capturing pattern for {gid}...")
        pattern = capture_game_patterns(arcade, gid)
        if pattern:
            patterns.append(pattern)
            log.info(f"  Genre: {pattern.genre}, {len(pattern.visual.get('action_effects',[]))} action effects captured")

    save_library(patterns)

    # Print human-readable summary
    for p in patterns:
        print(f"\n{'='*60}")
        print(f"  {p.genre.upper()} (from {p.source_games})")
        print(f"  {p.human['description']}")
        print(f"  Analogy: {p.human['analogy']}")
        print(f"  Solver: {p.machine['solver_route']}")
        if p.machine.get('modular_periods'):
            print(f"  Periods: {p.machine['modular_periods']}")

    # Print VLM prompt format
    print("\n" + "="*60)
    print("VLM PROMPT FORMAT:")
    print("="*60)
    print(format_for_vlm([asdict(p) for p in patterns]))
