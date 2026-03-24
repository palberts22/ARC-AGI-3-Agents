"""ARC Eyes — Perception module for human-like game solving.

See the game. Poke things. Watch what happens. Form theories. Win.

No source code reading. No env._game. Just pixels and actions.
"""

import numpy as np
import copy
import time
import logging
from collections import defaultdict, Counter
from dataclasses import dataclass, field
from typing import Optional

log = logging.getLogger("arc_eyes")


# ---------------------------------------------------------------------------
# 0. EPISODIC MEMORY — Remember what happened
# ---------------------------------------------------------------------------

@dataclass
class FrameTransition:
    """A single observed transition: I did action A in state S, got state S'."""
    frame_hash_before: int
    frame_hash_after: int
    action_id: int
    action_data: dict
    pixel_delta_count: int  # how many pixels changed
    sprite_movements: list  # [(color, dy, dx)] — what moved
    level: int
    won: bool  # did this action win the level?


@dataclass
class ActionMemory:
    """Aggregated memory of what a specific action does."""
    action_id: int
    action_data_repr: str  # hashable representation
    total_observations: int = 0
    avg_pixel_change: float = 0.0
    movement_histogram: dict = field(default_factory=dict)  # (dy,dx) -> count
    win_count: int = 0
    noop_count: int = 0  # times it changed nothing


class EpisodicMemory:
    """Episodic memory for game-playing — remembers frame transitions.

    Like a human remembering: "last time I clicked that button, the wall
    moved right." Accumulates observations across levels so later levels
    solve faster.
    """

    def __init__(self, max_transitions: int = 10000):
        self.transitions: list[FrameTransition] = []
        self.action_memories: dict[str, ActionMemory] = {}  # key -> ActionMemory
        self.frame_hashes_seen: set[int] = set()
        self.max_transitions = max_transitions
        self.levels_played = 0
        self.current_game: str = ""
        self.game_patterns: list[dict] = []  # cross-game meta-patterns
        self.game_guides: list[dict] = []  # gamer's guide — genre knowledge from solvers

    def _action_key(self, action_id: int, action_data: dict) -> str:
        """Create hashable key for an action."""
        if action_data and 'x' in action_data:
            return f"act{action_id}_x{action_data['x']}_y{action_data['y']}"
        return f"act{action_id}_nodata"

    def record(self, frame_before: np.ndarray, action_id: int,
               action_data: dict, frame_after: np.ndarray,
               noise_mask: Optional[np.ndarray] = None,
               level: int = 0, won: bool = False):
        """Record a single transition. Called every time the solver steps."""
        fb = frame_before if frame_before.ndim == 2 else frame_before[0]
        fa = frame_after if frame_after.ndim == 2 else frame_after[0]

        # Compute hashes (noise-masked if available)
        if noise_mask is not None:
            h_before = _masked_hash(fb, noise_mask)
            h_after = _masked_hash(fa, noise_mask)
        else:
            h_before = hash(fb.tobytes())
            h_after = hash(fa.tobytes())

        # Guard: empty or mismatched frames
        if fb.shape != fa.shape or fb.size == 0 or fa.size == 0:
            return  # skip recording — bad frame

        # Compute pixel delta
        diff_mask = fb != fa
        pixel_delta = int(diff_mask.sum())

        # Detect sprite movements (lightweight — just color + displacement)
        sprite_moves = []
        if pixel_delta > 0 and pixel_delta < 2000:
            sprites_b = detect_sprites(fb, min_size=3)
            sprites_a = detect_sprites(fa, min_size=3)
            movements = track_sprite_movement(sprites_b, sprites_a)
            sprite_moves = [(m[0].color, int(m[2][0]), int(m[2][1]))
                           for m in movements]

        # Store transition
        t = FrameTransition(
            frame_hash_before=h_before,
            frame_hash_after=h_after,
            action_id=action_id,
            action_data=dict(action_data) if action_data else {},
            pixel_delta_count=pixel_delta,
            sprite_movements=sprite_moves,
            level=level,
            won=won,
        )

        if len(self.transitions) < self.max_transitions:
            self.transitions.append(t)

        self.frame_hashes_seen.add(h_before)
        self.frame_hashes_seen.add(h_after)

        # Update aggregated action memory
        key = self._action_key(action_id, action_data)
        if key not in self.action_memories:
            self.action_memories[key] = ActionMemory(
                action_id=action_id,
                action_data_repr=key,
            )
        am = self.action_memories[key]
        am.total_observations += 1
        # Running average of pixel change
        am.avg_pixel_change += (pixel_delta - am.avg_pixel_change) / am.total_observations
        if won:
            am.win_count += 1
        if pixel_delta == 0:
            am.noop_count += 1
        for color, dy, dx in sprite_moves:
            mv = (dy, dx)
            am.movement_histogram[mv] = am.movement_histogram.get(mv, 0) + 1

    def get_action_effect(self, action_id: int, action_data: dict) -> Optional[ActionMemory]:
        """Query: what does this action typically do?"""
        key = self._action_key(action_id, action_data)
        return self.action_memories.get(key)

    def get_effective_actions(self, min_observations: int = 1) -> list[ActionMemory]:
        """Get all actions that have been observed to do something."""
        return [am for am in self.action_memories.values()
                if am.total_observations >= min_observations and am.noop_count < am.total_observations]

    def get_winning_actions(self) -> list[ActionMemory]:
        """Which actions have won levels before?"""
        return [am for am in self.action_memories.values() if am.win_count > 0]

    def dominant_movement(self, action_id: int, action_data: dict) -> Optional[tuple]:
        """What's the most common movement for this action? Returns (dy, dx) or None."""
        am = self.get_action_effect(action_id, action_data)
        if am is None or not am.movement_histogram:
            return None
        return max(am.movement_histogram, key=am.movement_histogram.get)

    def start_game(self, game_id: str):
        """Mark the start of a new game. Resets per-game state but keeps
        cross-game meta-patterns."""
        self.current_game = game_id
        self.levels_played = 0
        # Per-game action memories reset — different games have different actions
        self.action_memories.clear()
        # But frame hashes and meta-patterns persist

    def end_game(self, game_id: str, game_type: str, levels_solved: int,
                 total_actions: int, n_dir_actions: int, n_click_actions: int):
        """Record cross-game meta-pattern after finishing a game."""
        self.game_patterns.append({
            'game_id': game_id,
            'game_type': game_type,
            'levels_solved': levels_solved,
            'total_actions': total_actions,
            'n_dir_actions': n_dir_actions,
            'n_click_actions': n_click_actions,
        })

    # ─── Gamer's Guide (genre knowledge from solvers) ─────────────────

    def add_game_guide(self, genre: str, mechanic: str,
                       visual_cues: list[str], strategies: list[str],
                       has_hidden_state: bool = False):
        """Register genre knowledge from a specialized solver.

        Like a page in a gamer's guide — not the solution to a specific game,
        but the genre knowledge that helps with similar games.
        E.g., "toggle puzzles have neighbor effects, try systematic ordering."
        """
        if not hasattr(self, 'game_guides'):
            self.game_guides = []
        # Don't duplicate
        for g in self.game_guides:
            if g['genre'] == genre:
                g.update({'mechanic': mechanic, 'visual_cues': visual_cues,
                          'strategies': strategies, 'has_hidden_state': has_hidden_state})
                return
        self.game_guides.append({
            'genre': genre,
            'mechanic': mechanic,
            'visual_cues': visual_cues,
            'strategies': strategies,
            'has_hidden_state': has_hidden_state,
        })

    def match_guide(self, theory) -> Optional[dict]:
        """Find the best matching game guide for the current theory.

        Compares visual observations (sprites, actions) against known guides.
        Returns the guide entry if a match is found, None otherwise.
        """
        if not hasattr(self, 'game_guides') or not self.game_guides:
            return None
        gt = theory.game_type.lower() if theory else ''
        for g in self.game_guides:
            if g['genre'].lower() in gt or gt in g['genre'].lower():
                return g
        return None

    # ─── Strategy Memory (recursive self-improvement) ──────────────────

    def record_strategy_result(self, game_type: str, strategy: str,
                               solved: bool, actions: int, level: int,
                               game_id: str = ""):
        """Record which strategy worked (or failed) for which game type.

        This is the feedback loop: after each level, we learn which
        approach actually works. Next time we see a similar game, we
        try the winning strategy first.
        """
        if not hasattr(self, 'strategy_results'):
            self.strategy_results = []
        self.strategy_results.append({
            'game_type': game_type,
            'strategy': strategy,
            'solved': solved,
            'actions': actions,
            'level': level,
            'game_id': game_id,
        })

    def best_strategy_for(self, game_type: str) -> list[str]:
        """Return strategies ranked by success rate for this game type.

        Used at the start of each level to prioritize which solver to try
        first. This is the "getting recursively better" part — accumulated
        experience changes solver order.
        """
        if not hasattr(self, 'strategy_results') or not self.strategy_results:
            return []  # no experience yet — use defaults

        # Aggregate: for each strategy, compute win rate + avg actions
        from collections import defaultdict
        stats = defaultdict(lambda: {'wins': 0, 'total': 0, 'total_actions': 0})
        for r in self.strategy_results:
            if r['game_type'] == game_type:
                key = r['strategy']
                stats[key]['total'] += 1
                if r['solved']:
                    stats[key]['wins'] += 1
                    stats[key]['total_actions'] += r['actions']

        if not stats:
            # No exact match — try cross-game-type transfer
            for r in self.strategy_results:
                key = r['strategy']
                stats[key]['total'] += 1
                if r['solved']:
                    stats[key]['wins'] += 1
                    stats[key]['total_actions'] += r['actions']

        # Rank by: win rate first, then efficiency (fewer actions better)
        ranked = []
        for strat, s in stats.items():
            wr = s['wins'] / max(1, s['total'])
            avg_act = s['total_actions'] / max(1, s['wins']) if s['wins'] else 9999
            ranked.append((strat, wr, avg_act))
        ranked.sort(key=lambda x: (-x[1], x[2]))  # best win rate, then fewest actions

        return [r[0] for r in ranked]

    def strategy_summary(self) -> str:
        """Human-readable summary of strategy performance."""
        if not hasattr(self, 'strategy_results') or not self.strategy_results:
            return "No strategy experience yet."
        from collections import defaultdict
        by_type = defaultdict(lambda: defaultdict(lambda: [0, 0]))
        for r in self.strategy_results:
            s = by_type[r['game_type']][r['strategy']]
            s[1] += 1
            if r['solved']:
                s[0] += 1
        lines = ["Strategy Memory:"]
        for gt, strats in by_type.items():
            parts = []
            for strat, (w, t) in sorted(strats.items(), key=lambda x: -x[1][0]):
                parts.append(f"{strat}={w}/{t}")
            lines.append(f"  {gt}: {', '.join(parts)}")
        return "\n".join(lines)

    def predict_game_type(self, n_dir: int, n_click: int) -> Optional[str]:
        """From cross-game experience: given action counts, what type of game?"""
        if not self.game_patterns:
            return None
        # Find games with similar action profiles
        best_type = None
        best_score = -1
        type_votes = defaultdict(float)
        for gp in self.game_patterns:
            # Similarity: how close is this game's action profile?
            dir_match = 1.0 if (n_dir > 0) == (gp['n_dir_actions'] > 0) else 0.0
            click_match = 1.0 if (n_click > 0) == (gp['n_click_actions'] > 0) else 0.0
            score = dir_match + click_match
            type_votes[gp['game_type']] += score

        if type_votes:
            best_type = max(type_votes, key=type_votes.get)
            if type_votes[best_type] > 1.0:  # at least some confidence
                return best_type
        return None

    def save(self, path: str):
        """Persist memory to disk as JSON. Survives between runs."""
        import json
        data = {
            'version': 2,
            'games_played': len(self.game_patterns),
            'total_transitions': len(self.transitions),
            'game_patterns': self.game_patterns,
            'strategy_results': getattr(self, 'strategy_results', []),
            'game_guides': getattr(self, 'game_guides', []),
            # Don't save raw transitions (too large) — save aggregated knowledge
            'meta': {
                'total_frames_seen': len(self.frame_hashes_seen),
                'levels_played_total': sum(gp.get('levels_solved', 0)
                                          for gp in self.game_patterns),
            }
        }
        with open(path, 'w') as f:
            json.dump(data, f, indent=2)
        n_strats = len(getattr(self, 'strategy_results', []))
        log.info(f"  [memory] Saved to {path}: {len(self.game_patterns)} games, "
                f"{len(self.transitions)} transitions, {n_strats} strategy results")

    @classmethod
    def load(cls, path: str) -> 'EpisodicMemory':
        """Load persisted memory from disk."""
        import json, os
        mem = cls()
        if not os.path.exists(path):
            return mem
        try:
            with open(path) as f:
                data = json.load(f)
            mem.game_patterns = data.get('game_patterns', [])
            mem.strategy_results = data.get('strategy_results', [])
            mem.game_guides = data.get('game_guides', [])
            n_strats = len(mem.strategy_results)
            log.info(f"  [memory] Loaded from {path}: "
                    f"{len(mem.game_patterns)} games, {n_strats} strategy results")
        except Exception as e:
            log.warning(f"  [memory] Failed to load {path}: {e}")
        return mem

    def summary(self) -> str:
        """Human-readable summary of what we've learned."""
        lines = [f"EpisodicMemory: {len(self.transitions)} transitions, "
                 f"{len(self.action_memories)} actions, "
                 f"{len(self.frame_hashes_seen)} unique frames, "
                 f"{len(self.game_patterns)} games played"]
        # Cross-game patterns
        if self.game_patterns:
            type_counts = defaultdict(int)
            for gp in self.game_patterns:
                type_counts[gp['game_type']] += 1
            lines.append(f"  Game types seen: {dict(type_counts)}")
        # Current game action memories
        for key, am in sorted(self.action_memories.items()):
            eff = "noop" if am.noop_count == am.total_observations else "active"
            wins = f", {am.win_count} wins" if am.win_count else ""
            dom = ""
            if am.movement_histogram:
                top = max(am.movement_histogram, key=am.movement_histogram.get)
                dom = f", moves ({top[0]},{top[1]})"
            lines.append(f"  {key}: {am.total_observations} obs, "
                        f"avg Δ{am.avg_pixel_change:.0f}px, {eff}{dom}{wins}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# 1. SEE — Sprite & Region Detection
# ---------------------------------------------------------------------------

@dataclass
class Sprite:
    """A connected region of same-colored pixels."""
    sid: int
    color: int
    pixels: set  # set of (y, x) tuples
    bbox: tuple  # (y_min, x_min, y_max, x_max)
    center: tuple  # (cy, cx)
    size: int

    @property
    def width(self):
        return self.bbox[3] - self.bbox[1] + 1

    @property
    def height(self):
        return self.bbox[2] - self.bbox[0] + 1

    def overlaps(self, other: 'Sprite') -> bool:
        return bool(self.pixels & other.pixels)

    def distance_to(self, other: 'Sprite') -> float:
        return ((self.center[0] - other.center[0])**2 +
                (self.center[1] - other.center[1])**2) ** 0.5


def flood_fill(grid: np.ndarray, y: int, x: int, visited: set) -> set:
    """4-connected flood fill. Returns set of (y, x) pixels."""
    h, w = grid.shape
    color = grid[y, x]
    stack = [(y, x)]
    region = set()
    while stack:
        cy, cx = stack.pop()
        if (cy, cx) in visited or cy < 0 or cy >= h or cx < 0 or cx >= w:
            continue
        if grid[cy, cx] != color:
            continue
        visited.add((cy, cx))
        region.add((cy, cx))
        stack.extend([(cy-1, cx), (cy+1, cx), (cy, cx-1), (cy, cx+1)])
    return region


def detect_sprites(frame: np.ndarray, min_size: int = 2,
                   bg_colors: Optional[set] = None) -> list[Sprite]:
    """Find all connected regions (sprites) in a frame.

    Args:
        frame: 2D numpy array (64x64 palette-indexed)
        min_size: minimum pixels to count as a sprite
        bg_colors: colors to treat as background (skip). Auto-detected if None.
    """
    if frame.ndim == 3:
        frame = frame[0]  # squeeze channel dim

    h, w = frame.shape

    # Auto-detect background: the most common color
    if bg_colors is None:
        counts = Counter(frame.flatten().tolist())
        if not counts:
            return []  # empty frame — nothing to detect
        most_common = counts.most_common(1)[0]
        # Background is the dominant color (>25% of pixels)
        bg_colors = set()
        for color, count in counts.items():
            if count > h * w * 0.25:
                bg_colors.add(color)

    visited = set()
    sprites = []
    sid = 0

    for y in range(h):
        for x in range(w):
            if (y, x) in visited:
                continue
            color = int(frame[y, x])
            if color in bg_colors:
                visited.add((y, x))
                continue
            region = flood_fill(frame, y, x, visited)
            if len(region) < min_size:
                continue
            ys = [p[0] for p in region]
            xs = [p[1] for p in region]
            bbox = (min(ys), min(xs), max(ys), max(xs))
            center = (sum(ys) / len(ys), sum(xs) / len(xs))
            sprites.append(Sprite(
                sid=sid, color=color, pixels=region,
                bbox=bbox, center=center, size=len(region)
            ))
            sid += 1

    return sprites


def frame_to_ascii(frame: np.ndarray, step: int = 1,
                   color: bool = True) -> str:
    """Render frame as ASCII art for human-readable display.

    With color=True, uses ANSI colors for a richer display.
    """
    if frame.ndim == 3:
        frame = frame[0]

    # ANSI color codes for palette indices
    COLORS = [
        '\033[90m',   # 0: dark gray (background)
        '\033[97m',   # 1: bright white
        '\033[91m',   # 2: red
        '\033[92m',   # 3: green
        '\033[94m',   # 4: blue
        '\033[93m',   # 5: yellow
        '\033[95m',   # 6: magenta
        '\033[96m',   # 7: cyan
        '\033[33m',   # 8: orange/dark yellow
        '\033[31m',   # 9: dark red
        '\033[32m',   # 10: dark green
        '\033[34m',   # 11: dark blue
        '\033[35m',   # 12: dark magenta
        '\033[36m',   # 13: dark cyan
        '\033[37m',   # 14: light gray
        '\033[90m',   # 15: dim gray
    ]
    RESET = '\033[0m'
    # Block characters for denser display
    charset = '·█▓░▒▚▞╬◆●○□■◇▲▼'
    lines = []
    for y in range(0, frame.shape[0], step):
        row = ''
        for x in range(0, frame.shape[1], max(1, step // 2)):
            v = int(frame[y, x])
            ch = charset[v] if v < len(charset) else '?'
            if color and v < len(COLORS):
                row += COLORS[v] + ch + RESET
            else:
                row += ch
        lines.append(row)
    return '\n'.join(lines)


# ---------------------------------------------------------------------------
# 2. TRACK — Change Detection Between Frames
# ---------------------------------------------------------------------------

@dataclass
class FrameDelta:
    """What changed between two frames."""
    changed_pixels: int
    appeared: dict  # color -> set of (y,x) that are new
    disappeared: dict  # color -> set of (y,x) that are gone
    moved_sprites: list  # (before_sprite, after_sprite, displacement)
    region_shifts: list  # (color, direction, amount) for boundary moves
    is_animation: bool  # minor change, likely animation frame


def diff_frames(f0: np.ndarray, f1: np.ndarray) -> FrameDelta:
    """Compute semantic diff between two frames."""
    if f0.ndim == 3:
        f0 = f0[0]
    if f1.ndim == 3:
        f1 = f1[0]

    mask = f0 != f1
    changed = int(mask.sum())

    appeared = defaultdict(set)
    disappeared = defaultdict(set)

    if changed > 0:
        ys, xs = np.where(mask)
        for y, x in zip(ys, xs):
            old_c = int(f0[y, x])
            new_c = int(f1[y, x])
            disappeared[old_c].add((int(y), int(x)))
            appeared[new_c].add((int(y), int(x)))

    return FrameDelta(
        changed_pixels=changed,
        appeared=dict(appeared),
        disappeared=dict(disappeared),
        moved_sprites=[],
        region_shifts=[],
        is_animation=changed < 20 and changed > 0
    )


def track_sprite_movement(sprites_before: list[Sprite],
                          sprites_after: list[Sprite]) -> list[tuple]:
    """Match sprites between frames, detect movement.

    Returns list of (before, after, (dy, dx)) for moved sprites.
    """
    movements = []
    used = set()

    for sb in sprites_before:
        best_match = None
        best_dist = float('inf')
        for i, sa in enumerate(sprites_after):
            if i in used:
                continue
            if sa.color != sb.color:
                continue
            # Same color, similar size — likely same sprite
            if abs(sa.size - sb.size) > max(sb.size, sa.size) * 0.3:
                continue
            dist = sb.distance_to(sa)
            if dist < best_dist:
                best_dist = dist
                best_match = (i, sa)

        if best_match is not None:
            idx, sa = best_match
            dy = sa.center[0] - sb.center[0]
            dx = sa.center[1] - sb.center[1]
            if abs(dy) > 0.5 or abs(dx) > 0.5:
                movements.append((sb, sa, (dy, dx)))
            used.add(idx)

    return movements


# ---------------------------------------------------------------------------
# 3. INFER — Rule Learning from Observation
# ---------------------------------------------------------------------------

@dataclass
class ActionEffect:
    """What an action does, learned from observation."""
    action_id: int
    action_data: dict
    description: str
    avg_pixel_change: float
    sprite_movements: list  # [(dy, dx), ...] across observations
    consistent: bool  # same effect each time?
    observations: int


@dataclass
class GameTheory:
    """Current theory about how the game works."""
    game_type: str  # 'navigation', 'click_puzzle', 'sorting', 'unknown'
    player_sprite: Optional[Sprite]  # which sprite is "me"?
    targets: list[Sprite]  # things I need to reach/activate
    obstacles: list[Sprite]  # things in the way
    action_effects: dict  # action_key -> ActionEffect
    rules: list[str]  # human-readable rule descriptions
    confidence: float  # 0-1


def observe_action(env, action_id: int, data: dict,
                   n_trials: int = 1) -> list[FrameDelta]:
    """Try an action and observe what happens. Non-destructive (uses deepcopy)."""
    results = []
    for _ in range(n_trials):
        env_copy = copy.deepcopy(env)
        f_before = np.array(env_copy._last_obs.frame if hasattr(env_copy, '_last_obs')
                           else env_copy.reset().frame)
        obs = env_copy.step(action_id, data)
        if obs is None:
            continue
        f_after = np.array(obs.frame)
        delta = diff_frames(f_before, f_after)
        results.append(delta)
    return results


def build_theory(env, available_actions: list[int],
                 frame: np.ndarray, deadline: float,
                 memory: Optional[EpisodicMemory] = None,
                 skip_click_scan: bool = False) -> GameTheory:
    """Build a theory about the game by observing action effects.

    This is the core "human-like" reasoning:
    1. Look at the screen — identify sprites
    2. Try each action — watch what changes
    3. Classify: navigation? click puzzle? sorting?
    4. Identify: player, targets, obstacles

    If memory is provided, uses accumulated observations to skip
    re-probing actions we've already learned about.
    """
    if frame.ndim == 3:
        frame = frame[0]

    sprites = detect_sprites(frame, min_size=2)
    action_effects = {}

    # Phase 1: Probe each action type
    for act_id in available_actions:
        if time.time() >= deadline:
            break

        # Check memory first — do we already know what this action does?
        if memory is not None:
            mem_nodata = memory.get_action_effect(act_id, {})
            if mem_nodata and mem_nodata.total_observations >= 3:
                # We've seen this enough times — trust memory
                dom = memory.dominant_movement(act_id, {})
                sprite_mvs = [dom] if dom else []
                key = f"act{act_id}_nodata"
                action_effects[key] = ActionEffect(
                    action_id=act_id, action_data={},
                    description=f"memory: avg {mem_nodata.avg_pixel_change:.0f}px",
                    avg_pixel_change=mem_nodata.avg_pixel_change,
                    sprite_movements=sprite_mvs,
                    consistent=True,
                    observations=mem_nodata.total_observations
                )
                continue  # Skip probing — we remember

        # Try without data (directional actions)
        env_copy = copy.deepcopy(env)
        obs_before_frame = frame.copy()
        obs = env_copy.step(act_id, {})
        if obs is not None:
            f_after = np.array(obs.frame)
            if f_after.ndim == 3:
                f_after = f_after[0]
            delta = diff_frames(obs_before_frame, f_after)

            # Record to memory
            if memory is not None:
                memory.record(obs_before_frame, act_id, {}, f_after,
                             level=memory.levels_played)

            key = f"act{act_id}_nodata"
            if delta.changed_pixels > 0:
                # Detect sprite movement
                sprites_after = detect_sprites(f_after, min_size=2)
                movements = track_sprite_movement(sprites, sprites_after)

                action_effects[key] = ActionEffect(
                    action_id=act_id, action_data={},
                    description=_describe_effect(delta, movements),
                    avg_pixel_change=delta.changed_pixels,
                    sprite_movements=[(m[2]) for m in movements],
                    consistent=True, observations=1
                )
            else:
                # Zero-change action — still store it (needed for
                # navigation games where pickup/toggle/forward are
                # context-dependent and show 0px from initial state)
                action_effects[key] = ActionEffect(
                    action_id=act_id, action_data={},
                    description=f"act{act_id}: no visible change",
                    avg_pixel_change=0,
                    sprite_movements=[],
                    consistent=True, observations=1
                )

        # Try with click positions (for click-based games)
        # Sample a few sprite locations
        if skip_click_scan:
            continue  # MiniGrid-style games: no click actions, save budget
        for sprite in sprites[:5]:
            if time.time() >= deadline:
                break
            cx, cy = int(sprite.center[1]), int(sprite.center[0])
            env_copy = copy.deepcopy(env)
            obs = env_copy.step(act_id, {'x': cx, 'y': cy})
            if obs is not None:
                f_after = np.array(obs.frame)
                if f_after.ndim == 3:
                    f_after = f_after[0]
                delta = diff_frames(obs_before_frame, f_after)

                # Record click observation to memory
                if memory is not None:
                    memory.record(obs_before_frame, act_id, {'x': cx, 'y': cy},
                                 f_after, level=memory.levels_played)

                if delta.changed_pixels > 0:
                    sprites_after = detect_sprites(f_after, min_size=2)
                    movements = track_sprite_movement(sprites, sprites_after)

                    key = f"act{act_id}_click_{sprite.sid}"
                    action_effects[key] = ActionEffect(
                        action_id=act_id, action_data={'x': cx, 'y': cy},
                        description=_describe_effect(delta, movements),
                        avg_pixel_change=delta.changed_pixels,
                        sprite_movements=[(m[2]) for m in movements],
                        consistent=True, observations=1
                    )

    # Phase 2: Classify game type
    game_type = _classify_game(action_effects, sprites)

    # Phase 3: Identify player, targets, obstacles
    player = _find_player(sprites, action_effects)
    targets = _find_targets(sprites, player)

    rules = _generate_rules(action_effects, game_type, player, sprites)

    return GameTheory(
        game_type=game_type,
        player_sprite=player,
        targets=targets,
        obstacles=[],
        action_effects=action_effects,
        rules=rules,
        confidence=0.5 + 0.1 * len(action_effects)
    )


def _describe_effect(delta: FrameDelta, movements: list) -> str:
    """Generate human-readable description of what happened."""
    parts = []
    if movements:
        for sb, sa, (dy, dx) in movements:
            direction = ""
            if abs(dy) > abs(dx):
                direction = "down" if dy > 0 else "up"
            else:
                direction = "right" if dx > 0 else "left"
            parts.append(f"color-{sb.color} sprite moved {direction} by ({dy:.0f},{dx:.0f})")
    if delta.appeared:
        for color, pixels in delta.appeared.items():
            parts.append(f"color-{color} appeared ({len(pixels)}px)")
    if delta.disappeared:
        for color, pixels in delta.disappeared.items():
            parts.append(f"color-{color} disappeared ({len(pixels)}px)")
    if not parts:
        parts.append(f"{delta.changed_pixels}px changed")
    return "; ".join(parts)


def _classify_game(effects: dict, sprites: list) -> str:
    """Classify game type from observed effects."""
    has_directional = any('nodata' in k for k in effects)
    has_click = any('click' in k for k in effects)

    # Check for consistent directional movement (navigation)
    dir_movements = []
    for k, e in effects.items():
        if 'nodata' in k and e.sprite_movements:
            dir_movements.extend(e.sprite_movements)

    if dir_movements and len(dir_movements) >= 2:
        # Check if movements are axis-aligned and consistent
        axis_aligned = all(
            (abs(dy) < 2 or abs(dx) < 2) for dy, dx in dir_movements
        )
        if axis_aligned:
            return 'navigation'

    if has_click and not has_directional:
        return 'click_puzzle'
    if has_click and has_directional:
        return 'hybrid'
    if has_directional:
        return 'navigation'
    return 'unknown'


def _find_player(sprites: list, effects: dict) -> Optional[Sprite]:
    """Find the player sprite — the one that moves with directional actions."""
    for k, e in effects.items():
        if 'nodata' not in k:
            continue
        if not e.sprite_movements:
            continue
        # The sprite that moved is likely the player
        # Find which sprite in the original list matches
        for dy, dx in e.sprite_movements:
            # Look for small, distinct sprites that moved
            for s in sprites:
                if s.size < 100:  # player sprites are usually small
                    return s
    return None


def _find_targets(sprites: list, player: Optional[Sprite]) -> list[Sprite]:
    """Find target sprites — things that look like goals."""
    targets = []
    for s in sprites:
        if player and s.sid == player.sid:
            continue
        # Small, distinct sprites that aren't the player
        if s.size < 50:
            targets.append(s)
    return targets


def _generate_rules(effects: dict, game_type: str,
                    player: Optional[Sprite], sprites: list) -> list[str]:
    """Generate human-readable rules from observations."""
    rules = [f"Game type: {game_type}"]
    if player:
        rules.append(f"Player: color-{player.color} sprite at {player.center}")
    rules.append(f"Found {len(sprites)} sprites, {len(effects)} action effects")
    for k, e in effects.items():
        rules.append(f"  {k}: {e.description}")
    return rules


# ---------------------------------------------------------------------------
# 3b. REASON — Causal Models from Observation
# ---------------------------------------------------------------------------

@dataclass
class ClickTransform:
    """What a single click does to the frame, learned by watching."""
    position: tuple  # (x, y)
    action_id: int
    pixel_delta: dict  # {(y,x): (old_val, new_val)} — the transformation
    is_toggle: bool  # clicking twice returns to original?
    cycle_length: int  # how many clicks to return to start (1=noop, 2=toggle)
    changes_count: int  # number of pixels changed


def learn_click_transforms(env, frame: np.ndarray, click_positions: list[dict],
                           act_id: int, deadline: float) -> list[ClickTransform]:
    """For each click position, learn what transformation it applies.

    Like a human clicking each button once and watching: "oh, this one
    moves the left wall right. This one toggles the color."
    """
    if frame.ndim == 3:
        frame = frame[0]

    transforms = []

    for pos in click_positions:
        if time.time() >= deadline:
            break

        x, y = pos['x'], pos['y']

        # Click once — what changes?
        env_c1 = copy.deepcopy(env)
        obs1 = env_c1.step(act_id, {'x': x, 'y': y})
        if obs1 is None:
            continue

        # Pump animation
        game = getattr(env_c1, '_game', None)
        if game and hasattr(game, 'vai'):
            for _ in range(200):
                if getattr(game, 'vai', None) is None:
                    break
                obs1 = env_c1.step(act_id, {'x': x, 'y': y})

        f1 = np.array(obs1.frame)
        if f1.ndim == 3:
            f1 = f1[0]

        # Record the pixel delta
        diff_mask = frame != f1
        if not diff_mask.any():
            continue

        pixel_delta = {}
        ys, xs = np.where(diff_mask)
        for dy, dx in zip(ys, xs):
            pixel_delta[(int(dy), int(dx))] = (int(frame[dy, dx]), int(f1[dy, dx]))

        # Click again — does it toggle back?
        env_c2 = copy.deepcopy(env_c1)
        obs2 = env_c2.step(act_id, {'x': x, 'y': y})
        if obs2 is not None:
            game2 = getattr(env_c2, '_game', None)
            if game2 and hasattr(game2, 'vai'):
                for _ in range(200):
                    if getattr(game2, 'vai', None) is None:
                        break
                    obs2 = env_c2.step(act_id, {'x': x, 'y': y})

            f2 = np.array(obs2.frame)
            if f2.ndim == 3:
                f2 = f2[0]
            is_toggle = np.array_equal(frame, f2)

            # If not toggle, check cycle length up to 10
            cycle_length = 2 if is_toggle else 0
            if not is_toggle:
                env_cn = copy.deepcopy(env_c2)
                fn = f2.copy()
                for click_n in range(3, 11):
                    if time.time() >= deadline:
                        break
                    obs_n = env_cn.step(act_id, {'x': x, 'y': y})
                    if obs_n is None:
                        break
                    g = getattr(env_cn, '_game', None)
                    if g and hasattr(g, 'vai'):
                        for _ in range(200):
                            if getattr(g, 'vai', None) is None:
                                break
                            obs_n = env_cn.step(act_id, {'x': x, 'y': y})
                    fn = np.array(obs_n.frame)
                    if fn.ndim == 3:
                        fn = fn[0]
                    if np.array_equal(frame, fn):
                        cycle_length = click_n
                        break
        else:
            is_toggle = False
            cycle_length = 0

        transforms.append(ClickTransform(
            position=(x, y), action_id=act_id,
            pixel_delta=pixel_delta,
            is_toggle=is_toggle,
            cycle_length=cycle_length,
            changes_count=len(pixel_delta)
        ))

    return transforms


@dataclass
class SpatialEffect:
    """Spatial understanding of what a click does."""
    position: tuple  # (x, y) click position
    moving_sprites: list  # [(color, displacement_per_click)]
    affected_region: tuple  # bounding box of changes


def learn_spatial_effects(env, frame: np.ndarray, click_positions: list[dict],
                          act_id: int, deadline: float) -> list[SpatialEffect]:
    """Learn HOW each click moves things spatially.

    Like a human thinking: "clicking this button moves the green bar
    2 pixels to the right each time."
    """
    if frame.ndim == 3:
        frame = frame[0]

    effects = []
    for pos in click_positions:
        if time.time() >= deadline:
            break
        x, y = pos['x'], pos['y']

        # Click once, get sprites before and after
        sprites_before = detect_sprites(frame, min_size=2)
        env_copy = copy.deepcopy(env)
        obs = env_copy.step(act_id, {'x': x, 'y': y})
        if obs is None:
            continue
        game = getattr(env_copy, '_game', None)
        if game and hasattr(game, 'vai'):
            for _ in range(200):
                if getattr(game, 'vai', None) is None:
                    break
                obs = env_copy.step(act_id, {'x': x, 'y': y})

        f_after = np.array(obs.frame)
        if f_after.ndim == 3:
            f_after = f_after[0]

        sprites_after = detect_sprites(f_after, min_size=2)
        movements = track_sprite_movement(sprites_before, sprites_after)

        moving = []
        for sb, sa, (dy, dx) in movements:
            moving.append((sb.color, (dy, dx)))

        # Bounding box of all changes
        diff_mask = frame != f_after
        if diff_mask.any():
            ys, xs = np.where(diff_mask)
            bbox = (int(ys.min()), int(xs.min()), int(ys.max()), int(xs.max()))
        else:
            bbox = (0, 0, 0, 0)

        effects.append(SpatialEffect(
            position=(x, y), moving_sprites=moving,
            affected_region=bbox
        ))

    return effects


def visual_constraint_solve(env, frame: np.ndarray, transforms: list[ClickTransform],
                            act_id: int, deadline: float,
                            start_lc: int = 0,
                            max_depth: int = 100) -> Optional[list]:
    """Solve a click puzzle by reasoning about transformations.

    For toggle games (cycle_length=2):
      Each click either applies or un-applies a transformation.
      This is equivalent to: which SUBSET of clicks produces the win state?
      Try all 2^N subsets (feasible for N<20).

    For cyclic games (cycle_length=K):
      Each click can be applied 0..K-1 times.
      Try modular combinations.

    This is the REASONING step — like a human thinking
    "if I click A and C but not B, the colors should line up."
    """
    if frame.ndim == 3:
        frame = frame[0]

    if not transforms:
        return None

    # Check if all transforms are toggles
    all_toggles = all(t.is_toggle for t in transforms)
    all_cyclic = all(t.cycle_length > 0 for t in transforms)

    if all_toggles and len(transforms) <= 20:
        return _solve_toggle_subset(env, frame, transforms, act_id, deadline,
                                    start_lc)
    elif all_cyclic and len(transforms) <= 15:
        return _solve_cyclic_modular(env, frame, transforms, act_id, deadline,
                                     start_lc)
    else:
        # Fall back to BFS with transforms as moves
        return _solve_transform_bfs(env, frame, transforms, act_id, deadline,
                                    max_depth, start_lc)


def _solve_toggle_subset(env, frame: np.ndarray,
                         transforms: list[ClickTransform],
                         act_id: int, deadline: float,
                         start_lc: int = 0) -> Optional[list]:
    """Try all 2^N subsets of toggle clicks."""
    n = len(transforms)
    log.info(f"  [eyes] Toggle subset search: 2^{n} = {2**n} combinations")

    for mask in range(1, 2**n):
        if time.time() >= deadline:
            break

        # Apply this subset of transforms
        env_copy = copy.deepcopy(env)
        actions = []
        for i in range(n):
            if mask & (1 << i):
                x, y = transforms[i].position
                obs = env_copy.step(act_id, {'x': x, 'y': y})
                actions.append((act_id, {'x': x, 'y': y}))
                # Pump animation
                game = getattr(env_copy, '_game', None)
                if game and hasattr(game, 'vai'):
                    for _ in range(200):
                        if getattr(game, 'vai', None) is None:
                            break
                        obs = env_copy.step(act_id, {'x': x, 'y': y})

                if obs and hasattr(obs, 'levels_completed'):
                    if obs.levels_completed > start_lc:
                        return actions

    return None


def _solve_cyclic_modular(env, frame: np.ndarray,
                          transforms: list[ClickTransform],
                          act_id: int, deadline: float,
                          start_lc: int = 0) -> Optional[list]:
    """Try modular combinations for cyclic transforms.

    Each transform has cycle_length K. Try counts 0..K-1 for each.
    Total combinations: product of cycle lengths.
    """
    from itertools import product as iproduct

    cycles = [t.cycle_length for t in transforms]
    total = 1
    for c in cycles:
        total *= c
    log.info(f"  [eyes] Cyclic modular search: {total} combinations")

    if total > 100000:
        return None  # too many, fall back to BFS

    ranges = [range(c) for c in cycles]
    for counts in iproduct(*ranges):
        if time.time() >= deadline:
            break

        if all(c == 0 for c in counts):
            continue

        env_copy = copy.deepcopy(env)
        actions = []
        for i, click_count in enumerate(counts):
            x, y = transforms[i].position
            for _ in range(click_count):
                obs = env_copy.step(act_id, {'x': x, 'y': y})
                actions.append((act_id, {'x': x, 'y': y}))
                game = getattr(env_copy, '_game', None)
                if game and hasattr(game, 'vai'):
                    for _ in range(200):
                        if getattr(game, 'vai', None) is None:
                            break
                        obs = env_copy.step(act_id, {'x': x, 'y': y})

                if obs and hasattr(obs, 'levels_completed'):
                    if obs.levels_completed > start_lc:
                        return actions

    return None


def _solve_greedy_chain(env, frame: np.ndarray,
                        click_positions: list[dict],
                        act_id: int, deadline: float,
                        start_lc: int = 0) -> Optional[list]:
    """Solve by greedy chaining: click → observe → re-evaluate → click.

    Like a human playing: after each click, look at the new state,
    figure out what changed, and pick the next best click.

    Strategy: at each step, try all click positions, pick the one
    that produces the most "progress" (most pixel novelty relative
    to what we've seen before).

    Also tries: clicking the same position multiple times when it
    keeps producing change (human: "this is still doing something,
    keep clicking").
    """
    if frame.ndim == 3:
        frame = frame[0]

    env_state = copy.deepcopy(env)
    current_frame = frame.copy()
    actions = []
    seen_frames = {hash(current_frame.tobytes())}
    stale_count = 0  # how many clicks with no novel frame

    log.info(f"  [eyes] Greedy chain: {len(click_positions)} positions")

    for step in range(200):
        if time.time() >= deadline or stale_count > len(click_positions) * 3:
            break

        # Try each click position, measure novelty
        best_click = None
        best_novelty = -1
        best_env = None
        best_obs = None
        best_frame = None

        for pos in click_positions:
            if time.time() >= deadline:
                break

            env_try = copy.deepcopy(env_state)
            obs = env_try.step(act_id, {'x': pos['x'], 'y': pos['y']})
            if obs is None:
                continue

            # Pump animation
            game = getattr(env_try, '_game', None)
            if game and hasattr(game, 'vai'):
                for _ in range(200):
                    if getattr(game, 'vai', None) is None:
                        break
                    obs = env_try.step(act_id, {'x': pos['x'], 'y': pos['y']})

            # Check for win
            if obs and hasattr(obs, 'levels_completed'):
                if obs.levels_completed > start_lc:
                    actions.append((act_id, {'x': pos['x'], 'y': pos['y']}))
                    log.info(f"  [eyes] Greedy chain: solved in "
                            f"{len(actions)} clicks!")
                    return actions

            f_new = np.array(obs.frame)
            if f_new.ndim == 3:
                f_new = f_new[0]
            if f_new.shape != current_frame.shape:
                continue

            h = hash(f_new.tobytes())
            # Novelty = new frame? + pixel change magnitude
            pixel_change = int(np.sum(current_frame != f_new))
            is_novel = h not in seen_frames
            novelty = pixel_change * (2 if is_novel else 0.1)

            if novelty > best_novelty:
                best_novelty = novelty
                best_click = pos
                best_env = env_try
                best_obs = obs
                best_frame = f_new

        if best_click is None or best_novelty <= 0:
            break

        # Commit the best click
        actions.append((act_id, {'x': best_click['x'], 'y': best_click['y']}))
        env_state = best_env
        h = hash(best_frame.tobytes())
        if h in seen_frames:
            stale_count += 1
        else:
            stale_count = 0
            seen_frames.add(h)
        current_frame = best_frame

        # Re-detect click positions periodically (new sprites may appear)
        if step > 0 and step % 10 == 0 and time.time() < deadline - 5:
            new_positions = _find_click_positions(
                env_state, current_frame, act_id,
                time.time() + 2)
            existing = {(p['x'], p['y']) for p in click_positions}
            for np_ in new_positions:
                if (np_['x'], np_['y']) not in existing:
                    click_positions.append(np_)

    log.info(f"  [eyes] Greedy chain: {len(actions)} clicks, "
            f"{len(seen_frames)} unique frames")
    return None


def _pump_animation(env_copy, act_id, x, y):
    """Pump animation frames after a click. VC33-style games animate."""
    g = getattr(env_copy, '_game', None)
    if g and hasattr(g, 'vai'):
        for _ in range(200):
            if getattr(g, 'vai', None) is None:
                break
            env_copy.step(act_id, {'x': x, 'y': y})


def _get_frame(env_copy) -> np.ndarray:
    """Extract frame from env state (handles deepcopy'd envs)."""
    g = getattr(env_copy, '_game', None)
    if g and hasattr(g, 'frame'):
        f = np.array(g.frame)
    else:
        # Fallback: do a noop step and read obs
        return np.zeros((64, 64), dtype=np.uint8)
    if f.ndim == 3:
        f = f[0]
    return f


def _test_commutativity(env, click_positions: list[dict],
                       act_id: int, frame: np.ndarray) -> bool:
    """Test if click order doesn't matter (AB == BA for sample pairs).

    Commutative clicks mean we only need to search COUNT vectors, not
    permutations. Reduces search space from N^D to C(D+N-1, N-1).
    """
    if frame.ndim == 3:
        frame = frame[0]
    if len(click_positions) < 2:
        return True

    import random
    pairs = []
    for i in range(len(click_positions)):
        for j in range(i + 1, len(click_positions)):
            pairs.append((i, j))
    random.shuffle(pairs)
    pairs = pairs[:3]

    for i, j in pairs:
        pi, pj = click_positions[i], click_positions[j]

        # A then B
        e1 = copy.deepcopy(env)
        obs_a = e1.step(act_id, {'x': pi['x'], 'y': pi['y']})
        if obs_a is None:
            continue
        _pump_animation(e1, act_id, pi['x'], pi['y'])
        obs_ab = e1.step(act_id, {'x': pj['x'], 'y': pj['y']})
        if obs_ab is None:
            continue
        _pump_animation(e1, act_id, pj['x'], pj['y'])
        f_ab = np.array(obs_ab.frame)
        if f_ab.ndim == 3:
            f_ab = f_ab[0]

        # B then A
        e2 = copy.deepcopy(env)
        obs_b = e2.step(act_id, {'x': pj['x'], 'y': pj['y']})
        if obs_b is None:
            continue
        _pump_animation(e2, act_id, pj['x'], pj['y'])
        obs_ba = e2.step(act_id, {'x': pi['x'], 'y': pi['y']})
        if obs_ba is None:
            continue
        _pump_animation(e2, act_id, pi['x'], pi['y'])
        f_ba = np.array(obs_ba.frame)
        if f_ba.ndim == 3:
            f_ba = f_ba[0]

        if not np.array_equal(f_ab, f_ba):
            return False

    return True


def _find_rail_lengths(env, click_positions: list[dict],
                       act_id: int, frame: np.ndarray,
                       max_clicks: int = 30) -> list[int]:
    """For each click position, find how many distinct states it produces.

    A "rail" is the sequence of frames from clicking the same position
    repeatedly. Rail length = number of distinct frames (including start).
    """
    if frame.ndim == 3:
        frame = frame[0]

    rail_lengths = []
    for pos in click_positions:
        env_c = copy.deepcopy(env)
        seen = {hash(frame.tobytes())}
        length = 0
        for _ in range(max_clicks):
            obs = env_c.step(act_id, {'x': pos['x'], 'y': pos['y']})
            if obs is None:
                break
            g = getattr(env_c, '_game', None)
            if g and hasattr(g, 'vai'):
                for _a in range(200):
                    if getattr(g, 'vai', None) is None:
                        break
                    env_c.step(act_id, {'x': pos['x'], 'y': pos['y']})
            f = np.array(obs.frame)
            if f.ndim == 3:
                f = f[0]
            h = hash(f.tobytes())
            if h in seen:
                break  # cycled back or stuck
            seen.add(h)
            length += 1
        rail_lengths.append(length)
    return rail_lengths


def _solve_count_vector(env, frame: np.ndarray,
                        click_positions: list[dict],
                        act_id: int, deadline: float,
                        start_lc: int = 0) -> Optional[list]:
    """Solve commutative click games by searching count vectors.

    For commutative games where each position has a finite "rail length",
    search over how many times to click each position. Uses iterative
    deepening on total clicks to find shortest solution first.

    Since order doesn't matter, we replay clicks in position order.
    Single deepcopy per candidate (no tree of deepcopies).
    """
    if frame.ndim == 3:
        frame = frame[0]

    n = len(click_positions)
    if n == 0:
        return None

    # Learn rail lengths (max distinct clicks per position)
    rail_lengths = _find_rail_lengths(env, click_positions, act_id, frame)
    total_combos = 1
    for rl in rail_lengths:
        total_combos *= (rl + 1)

    log.info(f"  [eyes] Count vector: {n} positions, rails={rail_lengths}, "
             f"combos={total_combos}")

    if total_combos > 5_000_000 or total_combos == 0:
        return None

    def _replay_and_check(counts: list[int]) -> Optional[list]:
        """Apply count vector to fresh env copy, return actions if win."""
        env_c = copy.deepcopy(env)
        actions = []
        for i, k in enumerate(counts):
            pos = click_positions[i]
            for _ in range(k):
                obs = env_c.step(act_id, {'x': pos['x'], 'y': pos['y']})
                if obs is None:
                    return None
                g = getattr(env_c, '_game', None)
                if g and hasattr(g, 'vai'):
                    for _a in range(200):
                        if getattr(g, 'vai', None) is None:
                            break
                        env_c.step(act_id, {'x': pos['x'], 'y': pos['y']})
                actions.append((act_id, {'x': pos['x'], 'y': pos['y']}))
                if obs and hasattr(obs, 'levels_completed'):
                    if obs.levels_completed > start_lc:
                        return actions
        return None

    # Iterative deepening: try total clicks d=1,2,3,...
    max_total = sum(rail_lengths)
    tested = 0

    for d in range(1, max_total + 1):
        if time.time() >= deadline:
            break

        # Generate all count vectors summing to d, respecting rail limits
        # Use recursive generation to avoid materializing all combos
        def _gen(pos_idx, remaining, partial):
            nonlocal tested
            if time.time() >= deadline:
                return None
            if pos_idx == n:
                if remaining == 0:
                    tested += 1
                    return _replay_and_check(partial)
                return None
            # How many clicks at this position? 0..min(remaining, rail_i)
            max_here = min(remaining, rail_lengths[pos_idx])
            # Remaining positions need at least 0 clicks each
            for k in range(0, max_here + 1):
                partial.append(k)
                result = _gen(pos_idx + 1, remaining - k, partial)
                partial.pop()
                if result is not None:
                    return result
            return None

        result = _gen(0, d, [])
        if result is not None:
            log.info(f"  [eyes] Count vector: WIN at d={d}, "
                     f"tested={tested}")
            return result

        # Log progress periodically
        if d % 5 == 0:
            log.info(f"  [eyes] Count vector: depth {d}, "
                     f"tested={tested}")

    log.info(f"  [eyes] Count vector: exhausted (tested={tested})")
    return None


def _solve_click_mcts(env, frame: np.ndarray,
                      click_positions: list[dict],
                      act_id: int, deadline: float,
                      start_lc: int = 0,
                      max_depth: int = 100) -> Optional[list]:
    """MCTS for click puzzles — handles deep solution spaces.

    Like a human trying things semi-randomly but remembering
    what worked and doing more of that. Key advantage over BFS:
    can explore depth 50+ by following promising rollout paths.
    """
    import random
    import math

    if not click_positions:
        return None

    n_actions = len(click_positions)

    # MCTS tree node
    class Node:
        __slots__ = ['parent', 'action_idx', 'children', 'visits',
                     'value', 'env_snap', 'frame_hash', 'untried']

        def __init__(self, parent, action_idx, env_snap, frame_hash):
            self.parent = parent
            self.action_idx = action_idx
            self.children = []
            self.visits = 0
            self.value = 0.0
            self.env_snap = env_snap
            self.frame_hash = frame_hash
            self.untried = list(range(n_actions))
            random.shuffle(self.untried)

    def uct_select(node):
        best = None
        best_score = -1
        for child in node.children:
            if child.visits == 0:
                return child
            exploit = child.value / child.visits
            explore = math.sqrt(2 * math.log(node.visits) / child.visits)
            score = exploit + 1.4 * explore
            if score > best_score:
                best_score = score
                best = child
        return best

    def expand(node):
        if not node.untried:
            return None
        action_idx = node.untried.pop()
        pos = click_positions[action_idx]
        env_copy = copy.deepcopy(node.env_snap)
        obs = env_copy.step(act_id, {'x': pos['x'], 'y': pos['y']})
        if obs is None:
            return None

        # Pump animation
        game = getattr(env_copy, '_game', None)
        if game and hasattr(game, 'vai'):
            for _ in range(200):
                if getattr(game, 'vai', None) is None:
                    break
                obs = env_copy.step(act_id, {'x': pos['x'], 'y': pos['y']})

        # Check win
        if obs and hasattr(obs, 'levels_completed'):
            if obs.levels_completed > start_lc:
                return ('WIN', action_idx, env_copy)

        f = np.array(obs.frame)
        if f.ndim == 3:
            f = f[0]
        fh = hash(f.tobytes())

        child = Node(node, action_idx, env_copy, fh)
        node.children.append(child)
        return child

    def rollout(env_snap, depth, seen):
        """Random rollout from a state."""
        env_r = copy.deepcopy(env_snap)
        novelty = 0
        for _ in range(min(50, max_depth - depth)):
            idx = random.randint(0, n_actions - 1)
            pos = click_positions[idx]
            obs = env_r.step(act_id, {'x': pos['x'], 'y': pos['y']})
            if obs is None:
                break

            game = getattr(env_r, '_game', None)
            if game and hasattr(game, 'vai'):
                for _ in range(200):
                    if getattr(game, 'vai', None) is None:
                        break
                    obs = env_r.step(act_id, {'x': pos['x'], 'y': pos['y']})

            if obs and hasattr(obs, 'levels_completed'):
                if obs.levels_completed > start_lc:
                    return 1.0  # win!

            f = np.array(obs.frame)
            if f.ndim == 3:
                f = f[0]
            if f.shape == (64, 64):
                h = hash(f.tobytes())
                if h not in seen:
                    seen.add(h)
                    novelty += 1

        return novelty / 50  # normalized novelty as reward

    def backprop(node, value):
        while node is not None:
            node.visits += 1
            node.value += value
            node = node.parent

    def get_path(node):
        path = []
        while node.parent is not None:
            pos = click_positions[node.action_idx]
            path.append((act_id, {'x': pos['x'], 'y': pos['y']}))
            node = node.parent
        path.reverse()
        return path

    # Run MCTS
    if frame.ndim == 3:
        frame = frame[0]
    root = Node(None, -1, copy.deepcopy(env), hash(frame.tobytes()))
    seen_global = {root.frame_hash}
    iterations = 0

    log.info(f"  [eyes] MCTS: {n_actions} actions, "
            f"budget {deadline - time.time():.0f}s")

    while time.time() < deadline:
        # SELECT
        node = root
        depth = 0
        while node.untried == [] and node.children:
            node = uct_select(node)
            depth += 1

        # EXPAND
        if node.untried:
            result = expand(node)
            if result is not None:
                if isinstance(result, tuple) and result[0] == 'WIN':
                    # Found a win! Reconstruct path
                    _, win_idx, _ = result
                    path = get_path(node)
                    pos = click_positions[win_idx]
                    path.append((act_id, {'x': pos['x'], 'y': pos['y']}))
                    log.info(f"  [eyes] MCTS: WIN at depth {depth+1}, "
                            f"{iterations} iters, {len(path)} actions")
                    return path
                node = result

        # ROLLOUT — use local seen set (not global) for novelty
        value = rollout(node.env_snap, depth, {node.frame_hash})

        # BACKPROPAGATE
        backprop(node, value)
        iterations += 1

    log.info(f"  [eyes] MCTS: {iterations} iterations, "
            f"{len(seen_global)} unique frames, no solution")
    return None


def _solve_transform_bfs(env, frame: np.ndarray,
                         transforms: list[ClickTransform],
                         act_id: int, deadline: float,
                         max_depth: int,
                         start_lc: int = 0) -> Optional[list]:
    """BFS using known transforms as moves."""
    from collections import deque

    if frame.ndim == 3:
        frame = frame[0]

    initial_hash = hash(frame.tobytes())
    queue = deque([(copy.deepcopy(env), [], initial_hash)])
    visited = {initial_hash}

    while queue and time.time() < deadline:
        env_state, path, _ = queue.popleft()
        if len(path) >= max_depth:
            continue

        for t in transforms:
            if time.time() >= deadline:
                break
            x, y = t.position
            env_copy = copy.deepcopy(env_state)
            obs = env_copy.step(act_id, {'x': x, 'y': y})
            if obs is None:
                continue

            game = getattr(env_copy, '_game', None)
            if game and hasattr(game, 'vai'):
                for _ in range(200):
                    if getattr(game, 'vai', None) is None:
                        break
                    obs = env_copy.step(act_id, {'x': x, 'y': y})

            if obs and hasattr(obs, 'levels_completed'):
                if obs.levels_completed > start_lc:
                    full_path = path + [(act_id, {'x': x, 'y': y})]
                    shorten_dl = time.time() + min(30, max(5, deadline - time.time()))
                    return _shorten_explore_path(env, full_path, start_lc,
                                                 shorten_dl)

            f_new = np.array(obs.frame)
            if f_new.ndim == 3:
                f_new = f_new[0]
            h = hash(f_new.tobytes())

            if h not in visited and len(visited) < 200000:
                visited.add(h)
                queue.append((env_copy, path + [(act_id, {'x': x, 'y': y})], h))

    log.info(f"  [eyes] Transform BFS: {len(visited)} states")
    return None


# ---------------------------------------------------------------------------
# 4. PLAN — Use Theory to Solve the Game
# ---------------------------------------------------------------------------

def solve_with_eyes(env, theory: GameTheory, frame: np.ndarray,
                    deadline: float, max_steps: int = 500,
                    memory: Optional[EpisodicMemory] = None) -> Optional[list]:
    """Solve a game level using the perception-based theory.

    Strategy selection is ADAPTIVE: if we have accumulated experience
    (strategy_results in memory), we try strategies in order of past
    success rate. Otherwise, use default heuristic order.

    This is the recursive self-improvement loop:
    play -> record which strategy worked -> next time, try winners first.
    """
    has_dir = any('nodata' in k for k in theory.action_effects)
    has_click = any('click' in k for k in theory.action_effects)

    # Build the strategy pipeline based on accumulated experience
    strategies = _build_strategy_pipeline(theory, has_dir, has_click, memory)

    if memory and hasattr(memory, 'strategy_results') and memory.strategy_results:
        log.info(f"  [eyes] Strategy order (from experience): "
                f"{[s[0] for s in strategies]}")

    for strat_name, strat_fn, budget_frac in strategies:
        remaining = deadline - time.time()
        if remaining < 2:
            break
        strat_deadline = time.time() + remaining * budget_frac
        # Navigation modifies env (commit-first). Use deepcopy so
        # subsequent strategies get a clean env if nav fails.
        env_for_strat = copy.deepcopy(env) if strat_name == 'navigation' else env
        result = strat_fn(env_for_strat, theory, frame, strat_deadline, max_steps,
                          memory=memory) if 'memory' in strat_fn.__code__.co_varnames else \
                 strat_fn(env_for_strat, theory, frame, strat_deadline, max_steps)
        if result:
            # Record success
            if memory:
                memory.record_strategy_result(
                    theory.game_type, strat_name, True, len(result),
                    memory.levels_played, memory.current_game)
            return result
        else:
            # Record failure
            if memory:
                memory.record_strategy_result(
                    theory.game_type, strat_name, False, 0,
                    memory.levels_played, memory.current_game)

    return None


def _build_strategy_pipeline(theory: GameTheory, has_dir: bool, has_click: bool,
                              memory: Optional[EpisodicMemory] = None
                              ) -> list[tuple]:
    """Build ordered list of (name, function, budget_fraction) strategies.

    If memory has strategy experience, reorder to try winners first.
    Otherwise, use default heuristic order based on game type.
    """
    # Default pipeline (heuristic order)
    pipeline = []
    if has_dir:
        nav_budget = 0.6 if theory.game_type == 'navigation' else 0.3
        pipeline.append(('navigation', _solve_navigation, nav_budget))
    if has_click or theory.game_type in ('click_puzzle', 'hybrid'):
        pipeline.append(('click', _solve_click, 0.5))
    pipeline.append(('explore', _solve_explore, 0.4))

    # If we have strategy experience, reorder based on past success
    if memory and hasattr(memory, 'strategy_results') and memory.strategy_results:
        ranked = memory.best_strategy_for(theory.game_type)
        if ranked:
            # Build reordered pipeline: experienced strategies first, then defaults
            reordered = []
            seen = set()
            strat_map = {name: (fn, budget) for name, fn, budget in pipeline}
            for strat_name in ranked:
                if strat_name in strat_map and strat_name not in seen:
                    fn, budget = strat_map[strat_name]
                    reordered.append((strat_name, fn, budget))
                    seen.add(strat_name)
            # Append any strategies not yet seen (new strategies get a chance)
            for name, fn, budget in pipeline:
                if name not in seen:
                    reordered.append((name, fn, budget))
            pipeline = reordered

    return pipeline


def _detect_noise_mask(env, frame: np.ndarray, dir_actions: list,
                       n_probes: int = 3) -> np.ndarray:
    """Detect pixels that change on every action (counters, fuel bars).

    A "noise pixel" changes regardless of which action you take. These
    are UI elements (fuel bars, counters, timers) that change every step.
    Mask them for frame hashing so BFS can deduplicate effectively.

    Returns boolean mask: True = noise pixel (should be masked).
    """
    if not dir_actions:
        return np.zeros_like(frame, dtype=bool)

    # Take the first action from current state
    always_change = None

    for probe in range(min(n_probes, len(dir_actions))):
        effect = dir_actions[probe]
        ec = copy.deepcopy(env)
        obs = ec.step(effect.action_id, effect.action_data)
        if obs is None:
            continue
        f_after = np.array(obs.frame)
        if f_after.ndim == 3:
            f_after = f_after[0]
        if f_after.shape != frame.shape:
            continue

        changed = frame != f_after
        if always_change is None:
            always_change = changed
        else:
            always_change = always_change & changed  # intersection

    if always_change is None:
        return np.zeros_like(frame, dtype=bool)

    # Extend mask to full rows that have noise (resource bars span rows)
    noise_rows = np.any(always_change, axis=1)
    noise_mask = np.zeros_like(frame, dtype=bool)
    for r in range(frame.shape[0]):
        if noise_rows[r]:
            # Mask all pixels in this row that have the same value
            # as any noise pixel (catches full bar, not just shrinking part)
            noise_vals = set(frame[r, always_change[r, :]])
            for c in range(frame.shape[1]):
                if frame[r, c] in noise_vals:
                    noise_mask[r, c] = True

    n_noise = int(noise_mask.sum())
    if n_noise > 0:
        log.info(f"  [eyes] Detected {n_noise} noise pixels "
                f"(rows: {list(np.where(noise_rows)[0][:5])})")

    return noise_mask


def _masked_hash(frame: np.ndarray, noise_mask: np.ndarray) -> int:
    """Hash frame with noise pixels zeroed out."""
    if frame.ndim == 1:
        side = int(np.sqrt(frame.size))
        if side * side == frame.size:
            frame = frame.reshape(side, side)
        else:
            return hash(frame.tobytes())
    f = frame.copy()
    if f.shape == noise_mask.shape:
        f[noise_mask] = 0
    return hash(f.tobytes())


def _find_frontier_path(current_hash, transitions, dir_indices, opposites):
    """BFS on the learned transition graph to find the shortest path to a
    frontier state (one with at least one untried action).

    Returns list of action indices, or None if no reachable frontier.
    This is pure graph search — no env interaction needed.
    """
    from collections import deque

    queue = deque([(current_hash, [], -1)])  # (hash, path, last_action_idx)
    seen = {current_hash}

    while queue:
        h, p, last_idx = queue.popleft()

        # Check if this state is a frontier (has untried actions)
        untried = False
        for idx in dir_indices:
            if (h, idx) not in transitions:
                untried = True
                break
        if untried and p:  # not the start state
            return p

        # Expand via known transitions
        for idx in dir_indices:
            # Reversal pruning
            if idx in opposites and opposites[idx] == last_idx:
                continue
            key = (h, idx)
            if key in transitions:
                next_h = transitions[key]
                if next_h not in seen:
                    seen.add(next_h)
                    queue.append((next_h, p + [idx], idx))

    return None


def _nav_pilot(env, all_nodata, dir_actions, interact_actions, frame,
               noise_mask, deadline, start_levels, _check_win, _check_game_over,
               opposites=None, nav_max_steps: int = 300) -> Optional[list]:
    """LLM-guided navigation. The pilot observes, reasons, and drives.

    Uses Gundam's perception (minimap + spatial model) with a lightweight
    LLM loop. The pilot sees the game, decides where to move, and adapts
    when it hits walls or discovers new information.

    Falls back to None if LLM unavailable or budget exhausted — caller
    should try adaptive explorer next.
    """
    try:
        from gundam import call_llm, frame_to_minimap, SpatialModel, frame_to_b64
    except ImportError:
        log.info("  [pilot-nav] gundam not importable — skipping pilot nav")
        return None

    import os
    if not os.environ.get('ANTHROPIC', ''):
        log.info("  [pilot-nav] No ANTHROPIC key — skipping pilot nav")
        return None

    # Budget: max LLM calls for this nav attempt
    time_budget = deadline - time.time()
    if time_budget < 5:
        return None

    # Use Haiku for speed — nav decisions are simple directional choices
    pilot_model = os.environ.get('ARC_NAV_MODEL', 'claude-haiku-4-5-20251001')
    max_llm_calls = min(20, int(time_budget / 1.5))  # ~1.5s per call

    spatial = SpatialModel()
    env_copy = copy.deepcopy(env)
    path = []
    llm_calls = 0
    current_frame = frame.copy() if frame.ndim == 2 else frame[0].copy()
    action_history = []  # [(action_idx, pixels_changed, direction)]
    consecutive_blocked = 0

    # Build action index mapping
    action_list = [(i, e) for i, e in enumerate(all_nodata)]

    # First: take each directional action once to map directions (no LLM needed)
    # This gives the pilot a complete action map before it starts reasoning
    probe_env = copy.deepcopy(env)
    probe_frame = current_frame.copy()
    for idx, e in enumerate(all_nodata):
        if e.avg_pixel_change <= 0:
            continue  # skip interact actions for probing
        probe_snap = copy.deepcopy(probe_env)
        obs_probe = probe_snap.step(e.action_id, e.action_data)
        if obs_probe is None:
            continue
        f_after = np.array(obs_probe.frame)
        if f_after.ndim == 3:
            f_after = f_after[0]

        # Track movement via sprite detection
        try:
            sprites_b = detect_sprites(probe_frame, min_size=2)
            sprites_a = detect_sprites(f_after, min_size=2)
            movements = track_sprite_movement(sprites_b, sprites_a)
            if movements:
                for s_before, s_after, (dy, dx) in movements:
                    if spatial.player_color < 0:
                        spatial.player_color = s_before.color
                    if s_before.color == spatial.player_color:
                        direction = ""
                        if abs(dy) > abs(dx):
                            direction = "DOWN" if dy > 0 else "UP"
                        elif abs(dx) > 0:
                            direction = "RIGHT" if dx > 0 else "LEFT"
                        if direction:
                            spatial.action_map[idx] = (int(dy), int(dx), direction)
        except Exception:
            pass

    if spatial.action_map:
        log.info(f"  [pilot-nav] Action map from probes: "
                 f"{{{', '.join(f'{k}={v[2]}' for k, v in spatial.action_map.items())}}}")

    # Find player and objects in initial frame
    try:
        sprites = detect_sprites(current_frame, min_size=2)
        sprites.sort(key=lambda s: s.size)
        for s in sprites:
            role = "small" if s.size < 20 else ("structure" if s.size > 500 else "medium")
            spatial.objects.append((s.color, int(s.center[0]), int(s.center[1]), role))
            if s.size < 20 and spatial.player_color >= 0 and s.color == spatial.player_color:
                spatial.player_pos = (int(s.center[0]), int(s.center[1]))
    except Exception:
        pass

    # Phase 1: Discover unmapped actions by stepping each one
    # (probe phase may miss low-pixel-change movements like fuel bars)
    unmapped_dir = [i for i, e in enumerate(all_nodata)
                    if e.avg_pixel_change > 0 and i not in spatial.action_map]
    if unmapped_dir:
        log.info(f"  [pilot-nav] Discovering {len(unmapped_dir)} unmapped dir actions")
        for idx in unmapped_dir:
            act = all_nodata[idx]
            frame_before = current_frame.copy()
            obs_d = env_copy.step(act.action_id, act.action_data)
            if obs_d is None:
                continue
            path.append((act.action_id, act.action_data))
            if _check_win(obs_d):
                log.info(f"  [pilot-nav] Won during discovery!")
                return _shorten_explore_path(env, path, start_levels,
                                            time.time() + 10)
            f_d = np.array(obs_d.frame)
            if f_d.ndim == 3:
                f_d = f_d[0]
            try:
                sprites_b = detect_sprites(frame_before, min_size=2)
                sprites_a = detect_sprites(f_d, min_size=2)
                movements = track_sprite_movement(sprites_b, sprites_a)
                for s_before, s_after, (dy, dx) in movements:
                    if spatial.player_color < 0:
                        spatial.player_color = s_before.color
                    if s_before.color == spatial.player_color:
                        spatial.player_pos = (int(s_after.center[0]),
                                            int(s_after.center[1]))
                        direction = ""
                        if abs(dy) > abs(dx):
                            direction = "DOWN" if dy > 0 else "UP"
                        elif abs(dx) > 0:
                            direction = "RIGHT" if dx > 0 else "LEFT"
                        if direction:
                            spatial.action_map[idx] = (int(dy), int(dx), direction)
                            log.info(f"  [pilot-nav] Discovered action {idx} = {direction}")
            except Exception:
                pass
            current_frame = f_d

        if spatial.action_map:
            log.info(f"  [pilot-nav] Full action map: "
                     f"{{{', '.join(f'{k}={v[2]}' for k, v in spatial.action_map.items())}}}")

    # Main pilot loop
    total_steps = 0
    while total_steps < nav_max_steps:
        if time.time() >= deadline - 1.0:  # leave 1s margin
            break
        if llm_calls >= max_llm_calls:
            log.info(f"  [pilot-nav] LLM budget exhausted ({llm_calls} calls)")
            break

        # Build minimap
        minimap = frame_to_minimap(
            current_frame,
            player_pos=spatial.player_pos if spatial.player_color >= 0 else None,
            player_color=spatial.player_color,
            target_pos=spatial.target_pos if spatial.target_pos != (-1, -1) else None,
            target_color=spatial.target_color,
        )

        # Build spatial context
        spatial_desc = spatial.describe()

        # Recent history (last 10 actions)
        history_lines = []
        for ah in action_history[-10:]:
            idx_h, px_h, dir_h = ah
            if px_h == 0:
                history_lines.append(f"  Action {idx_h} → BLOCKED (wall)")
            else:
                history_lines.append(f"  Action {idx_h} → moved {dir_h}")
        history_str = "\n".join(history_lines) if history_lines else "  (no actions yet)"

        # Interact actions available
        interact_str = ""
        if interact_actions:
            interact_str = f"\nYou also have {len(interact_actions)} interact actions (pickup, toggle, etc.) at indices: {[i for i, e in enumerate(all_nodata) if e.avg_pixel_change == 0]}"

        system_prompt = f"""You are a navigation pilot for a grid-based game. You see a minimap and must navigate to win.

RULES:
- You control a sprite on a grid. Move it to the goal to win the level.
- P = your position, T = target (if known), # = wall, . = open, ~ = other object
- When you hit a wall (0 pixels changed), that direction is blocked. Go around.
- Actions are indexed 0-{len(all_nodata)-1}. Your action map shows which index = which direction.

ACTION MAP:
{chr(10).join(f'  {k} = {v[2]}' for k, v in spatial.action_map.items()) if spatial.action_map else '  (probe first action each direction to discover)'}
{interact_str}

STRATEGY:
- Navigate toward T if you can see it. Plan a path around walls.
- If blocked, try perpendicular directions to route around walls.
- If you can't see T, explore systematically — prefer directions you haven't tried.
- Keep track of which directions are blocked at each position.
- After {consecutive_blocked}+ consecutive blocks, you MUST try a DIFFERENT action.

Respond with EXACTLY this format:
PLAN: <sequence of action indices, space-separated, up to 15 steps>
REASON: <brief why>

Example: if action 1=UP and 3=RIGHT, and you want to go up 3 then right 2:
PLAN: 1 1 1 3 3
REASON: moving up to clear the wall, then right toward target"""

        user_prompt = f"""Step {total_steps} | {spatial_desc}

MINIMAP:
```
{minimap if minimap else '(too small to render)'}
```

RECENT HISTORY:
{history_str}

What action do you take?"""

        # Call LLM
        old_model = os.environ.get('ARC_PILOT_MODEL', '')
        os.environ['ARC_PILOT_MODEL'] = pilot_model
        try:
            response = call_llm(
                [{'role': 'system', 'content': system_prompt},
                 {'role': 'user', 'content': user_prompt}],
                images=[frame_to_b64(current_frame)] if total_steps < 3 or total_steps % 5 == 0 else [],
                max_tokens=200,
                temperature=0.2,
            )
        finally:
            if old_model:
                os.environ['ARC_PILOT_MODEL'] = old_model
            elif 'ARC_PILOT_MODEL' in os.environ:
                del os.environ['ARC_PILOT_MODEL']

        llm_calls += 1

        if response is None:
            log.info("  [pilot-nav] LLM call failed — aborting pilot nav")
            break

        # Parse multi-step plan from response
        plan = []
        for line in response.strip().split('\n'):
            line = line.strip()
            if line.upper().startswith('PLAN:'):
                tokens = line.split(':')[1].strip().split()
                for tok in tokens:
                    try:
                        idx = int(tok)
                        if 0 <= idx < len(all_nodata):
                            plan.append(idx)
                    except ValueError:
                        pass

        if not plan:
            # Fallback: try parsing ACTION: format
            for line in response.strip().split('\n'):
                line = line.strip()
                if line.upper().startswith('ACTION:'):
                    try:
                        idx = int(line.split(':')[1].strip().split()[0])
                        if 0 <= idx < len(all_nodata):
                            plan = [idx]
                    except (ValueError, IndexError):
                        pass

        if not plan:
            # Bad parse — explore sequentially
            if dir_actions:
                di = [i for i, e in enumerate(all_nodata) if e.avg_pixel_change > 0]
                plan = [di[total_steps % len(di)]]
            else:
                plan = [0]

        log.info(f"  [pilot-nav] Plan ({len(plan)} steps): {plan[:15]}")

        # Execute plan step by step
        for action_idx in plan:
            if time.time() >= deadline - 1.0:
                break

            act = all_nodata[action_idx]
            frame_before = current_frame.copy()
            obs = env_copy.step(act.action_id, act.action_data)
            if obs is None:
                break
            path.append((act.action_id, act.action_data))
            total_steps += 1

            if _check_win(obs):
                log.info(f"  [pilot-nav] WON at step {total_steps} ({llm_calls} LLM calls)")
                shorten_dl = time.time() + min(30, max(5, deadline - time.time()))
                return _shorten_explore_path(env, path, start_levels, shorten_dl)

            if _check_game_over(obs):
                log.info(f"  [pilot-nav] Game over at step {total_steps}")
                break

            # Observe effect
            f_after = np.array(obs.frame)
            if f_after.ndim == 3:
                f_after = f_after[0]

            pixels_changed = int(np.sum(current_frame != f_after))
            direction = ""

            if pixels_changed > 0:
                consecutive_blocked = 0
                try:
                    sprites_b = detect_sprites(frame_before, min_size=2)
                    sprites_a = detect_sprites(f_after, min_size=2)
                    movements = track_sprite_movement(sprites_b, sprites_a)
                    for s_before, s_after, (dy, dx) in movements:
                        if spatial.player_color < 0:
                            spatial.player_color = s_before.color
                        if s_before.color == spatial.player_color:
                            spatial.player_pos = (int(s_after.center[0]),
                                                int(s_after.center[1]))
                            if abs(dy) > abs(dx):
                                direction = "DOWN" if dy > 0 else "UP"
                            elif abs(dx) > 0:
                                direction = "RIGHT" if dx > 0 else "LEFT"
                            spatial.action_map[action_idx] = (int(dy), int(dx), direction)
                except Exception:
                    pass
            else:
                consecutive_blocked += 1
                direction = "BLOCKED"
                if spatial.player_color >= 0:
                    spatial.blocked.add(
                        (spatial.player_pos[0], spatial.player_pos[1], action_idx))
                # If blocked mid-plan, abort remaining plan and re-consult LLM
                action_history.append((action_idx, pixels_changed, direction))
                current_frame = f_after
                break

            action_history.append((action_idx, pixels_changed, direction))
            current_frame = f_after

        # After plan execution (or block), try interact actions at novel positions
        if interact_actions and consecutive_blocked >= 2:
            for ia_idx in [i for i, e in enumerate(all_nodata) if e.avg_pixel_change == 0]:
                ia = all_nodata[ia_idx]
                obs_ia = env_copy.step(ia.action_id, ia.action_data)
                if obs_ia is None:
                    continue
                path.append((ia.action_id, ia.action_data))
                if _check_win(obs_ia):
                    log.info(f"  [pilot-nav] WON via interact at step {total_steps}")
                    shorten_dl = time.time() + min(30, max(5, deadline - time.time()))
                    return _shorten_explore_path(env, path, start_levels, shorten_dl)
                f_ia = np.array(obs_ia.frame)
                if f_ia.ndim == 3:
                    f_ia = f_ia[0]
                if int(np.sum(current_frame != f_ia)) > 0:
                    current_frame = f_ia
                    consecutive_blocked = 0
                    action_history.append((ia_idx, 1, "INTERACT"))
                    break

    log.info(f"  [pilot-nav] Finished: {len(path)} steps, {llm_calls} LLM calls")
    return None


def _solve_navigation(env, theory: GameTheory, frame: np.ndarray,
                      deadline: float, max_steps: int,
                      memory: Optional[EpisodicMemory] = None) -> Optional[list]:
    """Navigation solver: LLM pilot first, adaptive explorer fallback.

    1. Pilot nav (LLM-guided): observes minimap, reasons about direction,
       drives step by step. The AGI approach.
    2. Adaptive explorer (heuristic): novelty-scoring with transition table.
       Fallback when LLM unavailable or exhausted.
    """
    if frame.ndim == 3:
        frame = frame[0]

    # Classify actions
    dir_actions = []
    interact_actions = []
    all_nodata = []
    for k, e in theory.action_effects.items():
        if 'nodata' in k:
            all_nodata.append(e)
            if e.avg_pixel_change > 0:
                dir_actions.append(e)
            else:
                interact_actions.append(e)

    if not all_nodata:
        return None

    log.info(f"  [eyes] Navigation: {len(dir_actions)} dir, "
             f"{len(interact_actions)} interact")

    # Detect noise pixels and reversal pairs
    noise_mask = _detect_noise_mask(env, frame, dir_actions)

    start_levels = getattr(env, '_start_levels', 0)

    def _check_win(obs):
        """Check if observation indicates a win."""
        if hasattr(obs, 'levels_completed'):
            if obs.levels_completed > start_levels:
                return True
        if hasattr(obs, 'state') and obs.state is not None:
            sv = obs.state.value if hasattr(obs.state, 'value') else str(obs.state)
            if sv == 'WIN':
                return True
        return False

    def _check_game_over(obs):
        """Check if game ended (loss/timeout)."""
        if hasattr(obs, 'state') and obs.state is not None:
            sv = obs.state.value if hasattr(obs.state, 'value') else str(obs.state)
            if sv in ('GAME_OVER', 'LOSS', 'TIMEOUT'):
                return True
        return False

    # Try LLM pilot first — AGI approach
    # Give pilot 60% of budget, keep 40% for adaptive fallback
    pilot_deadline = time.time() + (deadline - time.time()) * 0.6
    pilot_result = _nav_pilot(
        env, all_nodata, dir_actions, interact_actions, frame,
        noise_mask, pilot_deadline, start_levels, _check_win, _check_game_over)
    if pilot_result is not None:
        return pilot_result
    log.info("  [eyes] Pilot nav didn't solve — falling back to adaptive explorer")

    # Adaptive explorer fallback
    opposites = _detect_opposites(env, dir_actions, all_nodata, frame, noise_mask)
    if opposites:
        log.info(f"  [eyes] Reversal pairs: {len(opposites)//2}")

    # --- Adaptive explorer with multi-attempt restart ---
    # Each attempt gets fresh env steps but CARRIES the transition table.
    # Attempt N navigates directly to the frontier of explored territory.
    import random

    initial_hash = _masked_hash(frame, noise_mask)
    transitions = {}  # (hash, action_idx) → result_hash — persists across attempts
    all_visited = {initial_hash}  # all states ever seen — persists across attempts

    # Build dir_action indices for fast lookup
    dir_indices = [i for i, e in enumerate(all_nodata) if e.avg_pixel_change > 0]
    interact_indices = [i for i, e in enumerate(all_nodata) if e.avg_pixel_change == 0]

    max_attempts = max(1, int((deadline - time.time()) / 3))  # ~3s per attempt
    for attempt in range(max_attempts):
        if time.time() >= deadline:
            break

        # Fresh env copy for each attempt (fresh step counter)
        env_attempt = copy.deepcopy(env)
        visit_time = {initial_hash: 0}
        visit_time.update({h: -100 for h in all_visited})  # bias against old states
        visit_time[initial_hash] = 0
        path = []
        current_hash = initial_hash
        current_frame = frame.copy()
        last_action_idx = -1
        unique_states = len(all_visited)
        stall = 0
        attempt_steps = 0

        if attempt > 0:
            # Navigate to frontier using learned transitions
            frontier_path = _find_frontier_path(
                initial_hash, transitions, dir_indices, opposites)
            if frontier_path:
                for fp_idx in frontier_path:
                    fp_act = all_nodata[fp_idx]
                    obs_fp = env_attempt.step(fp_act.action_id, fp_act.action_data)
                    if obs_fp is None:
                        break
                    path.append((fp_act.action_id, fp_act.action_data))
                    attempt_steps += 1
                    if _check_win(obs_fp):
                        shorten_dl = time.time() + min(30, max(5, deadline - time.time()))
                        return _shorten_explore_path(env, path, start_levels,
                                                     shorten_dl)
                    if _check_game_over(obs_fp):
                        break
                    f_fp = np.array(obs_fp.frame)
                    if f_fp.ndim == 3:
                        f_fp = f_fp[0]
                    h_fp = _masked_hash(f_fp, noise_mask)
                    transitions[(current_hash, fp_idx)] = h_fp
                    current_hash = h_fp
                    current_frame = f_fp
                    last_action_idx = fp_idx
                log.info(f"  [eyes] Attempt {attempt+1}: navigated to frontier "
                         f"in {len(frontier_path)} steps")
            else:
                log.info(f"  [eyes] Attempt {attempt+1}: no reachable frontier")
                break  # fully explored

        for step in range(max_steps):
            if time.time() >= deadline:
                break

            # Score each directional action using learned transitions
            candidates = []
            for idx in dir_indices:
                if idx in opposites and opposites[idx] == last_action_idx:
                    continue
                key = (current_hash, idx)
                if key in transitions:
                    predicted_hash = transitions[key]
                    if predicted_hash not in all_visited:
                        score = 100  # known to lead somewhere novel!
                    elif predicted_hash not in visit_time:
                        score = 90  # visited in prior attempt, not this one
                    else:
                        age = step - visit_time.get(predicted_hash, -100)
                        score = min(age, 50)
                else:
                    score = 80  # unknown transition — explore it!
                candidates.append((score, idx))

            if not candidates:
                break

            candidates.sort(key=lambda x: (-x[0], random.random()))
            best_score, best_idx = candidates[0]

            act = all_nodata[best_idx]
            obs = env_attempt.step(act.action_id, act.action_data)
            if obs is None:
                break
            path.append((act.action_id, act.action_data))
            attempt_steps += 1

            if _check_win(obs):
                shorten_dl = time.time() + min(30, max(5, deadline - time.time()))
                return _shorten_explore_path(env, path, start_levels,
                                             shorten_dl)
            if _check_game_over(obs):
                log.info(f"  [eyes] Game over at step {step} "
                         f"(attempt {attempt+1}) — "
                         f"{len(all_visited)} total unique states")
                break

            f_new = np.array(obs.frame)
            if f_new.ndim == 3:
                f_new = f_new[0]
            new_hash = _masked_hash(f_new, noise_mask)
            transitions[(current_hash, best_idx)] = new_hash

            is_novel = new_hash not in all_visited
            all_visited.add(new_hash)
            visit_time[new_hash] = step
            last_action_idx = best_idx

            if is_novel:
                unique_states = len(all_visited)
                stall = 0

                # At novel positions, try interact actions
                for ia_idx in interact_indices:
                    ia = all_nodata[ia_idx]
                    ec = copy.deepcopy(env_attempt)
                    obs_ia = ec.step(ia.action_id, ia.action_data)
                    if obs_ia is None:
                        continue
                    if _check_win(obs_ia):
                        full_path = path + [(ia.action_id, ia.action_data)]
                        shorten_dl = time.time() + min(30, max(5, deadline - time.time()))
                        return _shorten_explore_path(env, full_path, start_levels,
                                                     shorten_dl)
                    f_ia = np.array(obs_ia.frame)
                    if f_ia.ndim == 3:
                        f_ia = f_ia[0]
                    h_ia = _masked_hash(f_ia, noise_mask)
                    if h_ia != new_hash:
                        env_attempt = ec
                        path.append((ia.action_id, ia.action_data))
                        new_hash = h_ia
                        f_new = f_ia
                        all_visited.add(h_ia)
                        visit_time[h_ia] = step
                        log.info(f"  [eyes] Interact at step {step}! "
                                 f"({len(all_visited)} unique)")
                        break

                if memory is not None and len(all_visited) % 5 == 0:
                    memory.record(current_frame, act.action_id,
                                 act.action_data, f_new,
                                 noise_mask=noise_mask,
                                 level=memory.levels_played)
            else:
                stall += 1

            current_hash = new_hash
            current_frame = f_new

            # Anti-stall: plan path to nearest frontier
            if stall > len(dir_indices) * 3:
                fp = _find_frontier_path(
                    current_hash, transitions, dir_indices, opposites)
                if fp:
                    escaped = False
                    for fp_idx in fp:
                        if time.time() >= deadline:
                            break
                        fp_act = all_nodata[fp_idx]
                        obs_fp = env_attempt.step(fp_act.action_id,
                                                   fp_act.action_data)
                        if obs_fp is None:
                            break
                        path.append((fp_act.action_id, fp_act.action_data))
                        attempt_steps += 1
                        if _check_win(obs_fp):
                            shorten_dl = time.time() + min(30, max(5, deadline - time.time()))
                            return _shorten_explore_path(env, path,
                                                         start_levels, shorten_dl)
                        if _check_game_over(obs_fp):
                            break
                        f_fp = np.array(obs_fp.frame)
                        if f_fp.ndim == 3:
                            f_fp = f_fp[0]
                        h_fp = _masked_hash(f_fp, noise_mask)
                        transitions[(current_hash, fp_idx)] = h_fp
                        current_hash = h_fp
                        if h_fp not in all_visited:
                            all_visited.add(h_fp)
                            escaped = True
                        visit_time[h_fp] = step
                        current_frame = f_fp
                        last_action_idx = fp_idx
                    stall = 0 if escaped else stall + 1
                else:
                    break  # no frontier reachable — end this attempt

    log.info(f"  [eyes] Adaptive nav: {max_attempts} attempts, "
             f"{len(all_visited)} total unique states, "
             f"{len(transitions)} learned transitions")
    return None


def _nav_bfs(env, dir_actions, frame, noise_mask, deadline,
             max_steps, start_levels, memory, check_win):
    """BFS navigation for pure directional games (no interact actions).

    Explores all reachable states breadth-first with reversal pruning.
    Optimal for finding shortest path to win state.
    """
    from collections import deque

    # Detect reversal pairs
    opposites = {}
    if len(dir_actions) >= 2:
        ref = copy.deepcopy(env)
        for i, a1 in enumerate(dir_actions):
            for j, a2 in enumerate(dir_actions):
                if i >= j:
                    continue
                ec = copy.deepcopy(ref)
                ec.step(a1.action_id, a1.action_data)
                obs = ec.step(a2.action_id, a2.action_data)
                if obs is not None:
                    f_check = np.array(obs.frame)
                    if f_check.ndim == 3:
                        f_check = f_check[0]
                    if _masked_hash(f_check, noise_mask) == _masked_hash(frame, noise_mask):
                        opposites[i] = j
                        opposites[j] = i
        if opposites:
            log.info(f"  [eyes] Reversal pairs: {len(opposites)//2}")

    initial_hash = _masked_hash(frame, noise_mask)
    queue = deque([(copy.deepcopy(env), [], initial_hash, -1)])
    visited = {initial_hash}

    while queue and time.time() < deadline:
        env_state, path, _, last_action_idx = queue.popleft()

        if len(path) >= max_steps:
            continue

        for idx, effect in enumerate(dir_actions):
            if time.time() >= deadline:
                break

            if idx in opposites and opposites[idx] == last_action_idx:
                continue

            env_copy = copy.deepcopy(env_state)
            obs = env_copy.step(effect.action_id, effect.action_data)
            if obs is None:
                continue

            if check_win(obs):
                full_path = path + [(effect.action_id, effect.action_data)]
                shorten_dl = time.time() + min(30, max(5, deadline - time.time()))
                return _shorten_explore_path(env, full_path, start_levels,
                                             shorten_dl)

            f_new = np.array(obs.frame)
            if f_new.ndim == 3:
                f_new = f_new[0]
            h = _masked_hash(f_new, noise_mask)

            if h not in visited and len(visited) < 50000:
                visited.add(h)
                queue.append((env_copy,
                             path + [(effect.action_id, effect.action_data)],
                             h, idx))
                if memory is not None and len(visited) % 10 == 0:
                    memory.record(frame, effect.action_id,
                                 effect.action_data, f_new,
                                 noise_mask=noise_mask,
                                 level=memory.levels_played)

    if len(visited) > 1:
        log.info(f"  [eyes] Navigation BFS: {len(visited)} states, "
                f"depth {len(path) if queue else 'exhausted'}")
    return None


def _detect_opposites(env, dir_actions, all_nodata, frame, noise_mask):
    """Find pairs of actions that reverse each other."""
    opposites = {}
    if len(dir_actions) < 2:
        return opposites
    ref = copy.deepcopy(env)
    for i, a1 in enumerate(all_nodata):
        if a1.avg_pixel_change == 0:
            continue
        for j, a2 in enumerate(all_nodata):
            if a2.avg_pixel_change == 0 or i >= j:
                continue
            ec = copy.deepcopy(ref)
            ec.step(a1.action_id, a1.action_data)
            obs = ec.step(a2.action_id, a2.action_data)
            if obs is not None:
                f_check = np.array(obs.frame)
                if f_check.ndim == 3:
                    f_check = f_check[0]
                if _masked_hash(f_check, noise_mask) == _masked_hash(frame, noise_mask):
                    opposites[i] = j
                    opposites[j] = i
    return opposites


def _find_click_positions(env, frame: np.ndarray, act_id: int,
                          deadline: float) -> list[dict]:
    """Scan the frame grid to find positions where clicking does something.

    Like a human moving the cursor: "where can I click?"
    Two-pass scan:
    1. Coarse pass (step=4): find approximate click regions
    2. Fine pass (step=2): refine within active regions

    Returns list of {'x': x, 'y': y, 'effect_size': n_changed_pixels}.
    """
    if frame.ndim == 3:
        frame = frame[0]

    positions = []
    seen_effects = set()
    active_regions = set()  # (x//8, y//8) cells with click activity

    # Pass 1: Coarse scan at step=4
    for y in range(0, 64, 4):
        for x in range(0, 64, 4):
            if time.time() >= deadline:
                break
            ec = copy.deepcopy(env)
            obs = ec.step(act_id, {'x': x, 'y': y})
            if obs is None:
                continue
            f_after = np.array(obs.frame)
            if f_after.ndim == 3:
                f_after = f_after[0]
            if f_after.shape != frame.shape:
                continue
            n_changed = int(np.sum(frame != f_after))
            if n_changed > 2:  # more than just a counter tick
                effect_hash = hash(f_after.tobytes())
                if effect_hash not in seen_effects:
                    seen_effects.add(effect_hash)
                    positions.append({'x': x, 'y': y, 'effect_size': n_changed})
                # Mark this region as active for fine scan
                active_regions.add((x // 8, y // 8))
                # Also mark neighbors
                for dx in [-1, 0, 1]:
                    for dy in [-1, 0, 1]:
                        active_regions.add((x // 8 + dx, y // 8 + dy))

    # Pass 2: Fine scan (step=2) in active regions
    for y in range(0, 64, 2):
        for x in range(0, 64, 2):
            if time.time() >= deadline:
                break
            if (x // 8, y // 8) not in active_regions:
                continue
            if x % 4 == 0 and y % 4 == 0:
                continue  # already scanned in pass 1
            ec = copy.deepcopy(env)
            obs = ec.step(act_id, {'x': x, 'y': y})
            if obs is None:
                continue
            f_after = np.array(obs.frame)
            if f_after.ndim == 3:
                f_after = f_after[0]
            if f_after.shape != frame.shape:
                continue
            n_changed = int(np.sum(frame != f_after))
            if n_changed > 2:
                effect_hash = hash(f_after.tobytes())
                if effect_hash not in seen_effects:
                    seen_effects.add(effect_hash)
                    positions.append({'x': x, 'y': y, 'effect_size': n_changed})

    # Pass 3: Full step=2 scan if coarse missed positions (narrow sprites
    # between grid points). Only runs if we have time and found few positions.
    if len(positions) < 4 and time.time() < deadline - 2.0:
        for y in range(0, 64, 2):
            for x in range(0, 64, 2):
                if time.time() >= deadline:
                    break
                if (x // 8, y // 8) in active_regions:
                    continue  # already covered by pass 1+2
                ec = copy.deepcopy(env)
                obs = ec.step(act_id, {'x': x, 'y': y})
                if obs is None:
                    continue
                f_after = np.array(obs.frame)
                if f_after.ndim == 3:
                    f_after = f_after[0]
                if f_after.shape != frame.shape:
                    continue
                n_changed = int(np.sum(frame != f_after))
                if n_changed > 2:
                    effect_hash = hash(f_after.tobytes())
                    if effect_hash not in seen_effects:
                        seen_effects.add(effect_hash)
                        positions.append({'x': x, 'y': y, 'effect_size': n_changed})

    return positions


def _solve_click(env, theory: GameTheory, frame: np.ndarray,
                 deadline: float, max_steps: int,
                 memory: Optional[EpisodicMemory] = None) -> Optional[list]:
    """Solve click puzzle by BFS over frame states.

    Human approach: scan for clickable spots, try each, track what
    state each click produces, BFS for the win state.
    """
    from collections import deque

    if frame.ndim == 3:
        frame = frame[0]

    # Find which action ID is the click action
    click_act_id = 6  # default
    for k, e in theory.action_effects.items():
        if 'click' in k and e.avg_pixel_change > 10:
            click_act_id = e.action_id
            break

    # Phase 1: Find all clickable positions (scan the screen)
    scan_deadline = time.time() + min(8, (deadline - time.time()) * 0.15)
    click_positions = _find_click_positions(env, frame, click_act_id, scan_deadline)
    log.info(f"  [eyes] Found {len(click_positions)} click positions")

    if not click_positions:
        return None

    # Detect current level count for win detection
    start_levels = 0
    try:
        env_probe = copy.deepcopy(env)
        probe_obs = env_probe.step(click_act_id, {'x': 0, 'y': 0})
        if probe_obs and hasattr(probe_obs, 'levels_completed'):
            start_levels = probe_obs.levels_completed
    except Exception:
        pass

    # Phase 2: REASON — learn what each click does
    reason_deadline = time.time() + min(15, (deadline - time.time()) * 0.2)
    transforms = learn_click_transforms(env, frame, click_positions,
                                        click_act_id, reason_deadline)

    # Learn spatial relationships — which sprites move when I click where?
    spatial_deadline = time.time() + min(10, (deadline - time.time()) * 0.1)
    spatial = learn_spatial_effects(env, frame, click_positions,
                                    click_act_id, spatial_deadline)
    if spatial:
        for s in spatial:
            if s.moving_sprites:
                log.info(f"  [eyes] Click ({s.position[0]},{s.position[1]}) → " +
                        ", ".join(f"color-{c} moves ({dy:.0f},{dx:.0f})"
                                 for c, (dy, dx) in s.moving_sprites))

    if transforms:
        log.info(f"  [eyes] Learned {len(transforms)} transforms: " +
                ", ".join(f"({t.position[0]},{t.position[1]}) "
                         f"{'toggle' if t.is_toggle else f'cycle={t.cycle_length}'} "
                         f"{t.changes_count}px"
                         for t in transforms))

        # Try constraint solving first (fast when it works)
        constraint_budget = min(20, (deadline - time.time()) * 0.3)
        constraint_deadline = time.time() + constraint_budget
        solution = visual_constraint_solve(env, frame, transforms,
                                           click_act_id, constraint_deadline,
                                           start_lc=start_levels)
        if solution:
            return solution

    # Phase 2b: Count vector search (for commutative click games)
    # Test if clicks commute — if so, only click COUNTS matter
    if len(click_positions) >= 3 and (deadline - time.time()) > 20:
        is_commutative = _test_commutativity(env, click_positions,
                                              click_act_id, frame)
        if is_commutative:
            log.info(f"  [eyes] Clicks are commutative — count vector search")
            cv_budget = min(90, (deadline - time.time()) * 0.6)
            cv_deadline = time.time() + cv_budget
            solution = _solve_count_vector(env, frame, click_positions,
                                           click_act_id, cv_deadline,
                                           start_lc=start_levels)
            if solution:
                return solution

    # Phase 2c: Greedy chain — click, observe, re-evaluate, repeat
    greedy_budget = min(60, (deadline - time.time()) * 0.5)
    greedy_deadline = time.time() + greedy_budget
    solution = _solve_greedy_chain(env, frame, click_positions,
                                   click_act_id, greedy_deadline,
                                   start_lc=start_levels)
    if solution:
        return solution

    # Phase 2d: MCTS — deeper search with rollouts
    mcts_budget = min(60, (deadline - time.time()) * 0.6)
    if mcts_budget > 10:
        mcts_deadline = time.time() + mcts_budget
        solution = _solve_click_mcts(env, frame, click_positions,
                                      click_act_id, mcts_deadline,
                                      start_lc=start_levels)
        if solution:
            return solution

    # Phase 3: Fall back to BFS over frame states
    initial_hash = hash(frame.tobytes())
    queue = deque([(copy.deepcopy(env), [], initial_hash)])
    visited = {initial_hash}
    states_explored = 0

    while queue and time.time() < deadline:
        env_state, path, _ = queue.popleft()

        if len(path) >= max_steps:
            continue

        for pos in click_positions:
            if time.time() >= deadline:
                break

            env_copy = copy.deepcopy(env_state)
            obs = env_copy.step(click_act_id, {'x': pos['x'], 'y': pos['y']})
            if obs is None:
                continue

            # Pump animation frames (some games animate after clicks)
            game = getattr(env_copy, '_game', None)
            if game and hasattr(game, 'vai'):
                for _ in range(200):
                    if getattr(game, 'vai', None) is None:
                        break
                    obs = env_copy.step(click_act_id, {'x': pos['x'], 'y': pos['y']})

            # Check for level completion
            if hasattr(obs, 'levels_completed'):
                if obs.levels_completed > start_levels:
                    full_path = path + [(click_act_id, {'x': pos['x'], 'y': pos['y']})]
                    shorten_dl = time.time() + min(30, max(5, deadline - time.time()))
                    return _shorten_explore_path(env, full_path, start_levels,
                                                 shorten_dl)

            f_new = np.array(obs.frame)
            if f_new.ndim == 3:
                f_new = f_new[0]
            h = hash(f_new.tobytes())

            if h not in visited and len(visited) < 100000:
                visited.add(h)
                queue.append((env_copy,
                             path + [(click_act_id, {'x': pos['x'], 'y': pos['y']})],
                             h))
                states_explored += 1

                # Re-scan for new click positions periodically
                # (the game state changed — new sprites may have appeared)
                if states_explored % 500 == 0 and time.time() < deadline - 5:
                    new_positions = _find_click_positions(
                        env_copy, f_new, click_act_id,
                        time.time() + 3)
                    # Merge new positions
                    existing = {(p['x'], p['y']) for p in click_positions}
                    for np_ in new_positions:
                        if (np_['x'], np_['y']) not in existing:
                            click_positions.append(np_)
                    if new_positions:
                        log.info(f"  [eyes] Re-scan: {len(click_positions)} positions now")

    log.info(f"  [eyes] Click BFS: {len(visited)} states explored")
    return None


def _solve_explore(env, theory: GameTheory, frame: np.ndarray,
                   deadline: float, max_steps: int) -> Optional[list]:
    """Explore unknown game: novelty-seeking random walk."""
    import random

    if frame.ndim == 3:
        frame = frame[0]

    best_path = None
    best_novelty = 0
    seen_frames = {hash(frame.tobytes())}

    all_actions = []
    for k, e in theory.action_effects.items():
        all_actions.append((e.action_id, e.action_data))
    if not all_actions:
        return None

    # Random walks biased toward novelty
    for walk in range(200):
        if time.time() >= deadline:
            break

        env_copy = copy.deepcopy(env)
        path = []
        novelty = 0

        for step in range(max_steps):
            if time.time() >= deadline:
                break

            act_id, data = random.choice(all_actions)
            obs = env_copy.step(act_id, data)
            if obs is None:
                break

            path.append((act_id, data))

            if hasattr(obs, 'levels_completed'):
                start_levels = getattr(env, '_start_levels', 0)
                if obs.levels_completed > start_levels:
                    # Shorten path — allocate up to 30s for shortening
                    shorten_deadline = time.time() + min(30, max(5, deadline - time.time()))
                    shortened = _shorten_explore_path(
                        env, path, start_levels, shorten_deadline)
                    return shortened

            f_new = np.array(obs.frame)
            if f_new.ndim == 3:
                f_new = f_new[0]
            h = hash(f_new.tobytes())
            if h not in seen_frames:
                seen_frames.add(h)
                novelty += 1

        if novelty > best_novelty:
            best_novelty = novelty
            best_path = path

    # Only return if a walk actually solved the level (early return above).
    # best_path is just the most novel exploration — not a solution.
    return None


def _shorten_explore_path(env, path: list[tuple], start_levels: int,
                           deadline: float) -> list[tuple]:
    """Remove unnecessary actions from an explore path.

    Two phases:
    1. Binary search for shortest SUFFIX that solves (fast, O(log n) verifies)
    2. Single-action removal on the short path (O(n^2) but n is now small)
    """
    def _verify(candidate):
        ec = copy.deepcopy(env)
        for aid, data in candidate:
            obs = ec.step(aid, data)
            if obs is None:
                return False
            if hasattr(obs, 'levels_completed'):
                if obs.levels_completed > start_levels:
                    return True
        return False

    current = list(path)

    # Phase 1: Binary search — find shortest prefix that solves
    # (Many actions at the start are wasted exploration)
    if len(current) > 20 and time.time() < deadline:
        lo, hi = 1, len(current)
        while lo < hi and time.time() < deadline:
            mid = (lo + hi) // 2
            # Try last `mid` actions (suffix)
            if _verify(current[-mid:]):
                hi = mid
            else:
                lo = mid + 1
        if lo < len(current) and _verify(current[-lo:]):
            current = current[-lo:]

    # Phase 2: Single-action removal (fine-tuning)
    improved = True
    while improved and time.time() < deadline:
        improved = False
        i = 0
        while i < len(current) and time.time() < deadline:
            candidate = current[:i] + current[i+1:]
            if _verify(candidate):
                current = candidate
                improved = True
            else:
                i += 1
        # Window removal (2-3 consecutive)
        for w in [2, 3]:
            i = 0
            while i < len(current) - w + 1 and time.time() < deadline:
                candidate = current[:i] + current[i+w:]
                if _verify(candidate):
                    current = candidate
                    improved = True
                else:
                    i += 1

    log.info(f"  [eyes] Path shortened: {len(path)} → {len(current)} actions")
    return current


# ---------------------------------------------------------------------------
# 5. MAIN — The Full "Human" Loop
# ---------------------------------------------------------------------------

def play_level(env, obs, timeout: float = 60.0,
               theory: Optional[GameTheory] = None,
               verbose: bool = True,
               memory: Optional[EpisodicMemory] = None,
               skip_click_scan: bool = False
               ) -> tuple[Optional[list], Optional[GameTheory], Optional[EpisodicMemory]]:
    """Play a single game level like a human.

    1. SCAN: Look at the screen — what's here?
    2. ANALYZE: Try actions, build/refine theory
    3. MOVE: Execute solution using theory
    4. COLLECT: Update theory and episodic memory

    Args:
        env: game environment (already at the right level)
        obs: current observation
        timeout: seconds for this level
        theory: reuse theory from previous level (humans remember!)
        memory: episodic memory — accumulated observations across levels

    Returns: (action_list_or_None, updated_theory, updated_memory)
    """
    deadline = time.time() + timeout
    frame = np.array(obs.frame)
    if frame.ndim == 3 and frame.shape[0] == 1:
        frame = frame[0]
    elif frame.ndim == 1:
        # Flatten edge case — try to reshape to square
        side = int(np.sqrt(frame.size))
        if side * side == frame.size:
            frame = frame.reshape(side, side)
    available = list(obs.available_actions)

    # Initialize memory if not provided
    if memory is None:
        memory = EpisodicMemory()

    # 1. SCAN — what do I see?
    if verbose:
        sprites = detect_sprites(frame, min_size=2)
        log.info(f"  [eyes] I see {len(sprites)} sprites, "
                f"{len(available)} actions")

    # 2. ANALYZE — build or refine theory
    # If memory has enough observations, skip expensive re-probing
    theory_budget = min(8, timeout * 0.12)
    if memory.levels_played > 0 and len(memory.action_memories) >= len(available):
        # We've seen these actions before — use shorter probe
        theory_budget = min(3, timeout * 0.05)
        if verbose:
            log.info(f"  [eyes] Memory shortcut: {len(memory.action_memories)} "
                    f"actions already known")

    theory_deadline = time.time() + theory_budget
    new_theory = build_theory(env, available, frame, theory_deadline,
                              memory=memory,
                              skip_click_scan=skip_click_scan)

    # Merge: keep the old theory's classification if the new one is unsure
    if theory and new_theory.confidence < theory.confidence:
        new_theory.game_type = theory.game_type
        new_theory.confidence = max(new_theory.confidence, theory.confidence * 0.8)

    if verbose:
        log.info(f"  [eyes] Theory: {new_theory.game_type} "
                f"({new_theory.confidence:.1f})")

    # 3. MOVE — solve using theory (strategy memory guides solver order)
    solution = solve_with_eyes(env, new_theory, frame, deadline,
                               memory=memory)

    if solution and verbose:
        log.info(f"  [eyes] Solved: {len(solution)} actions")

    # 4. COLLECT — theory + memory carry to next level
    memory.levels_played += 1

    # Log accumulated strategy knowledge
    if verbose and hasattr(memory, 'strategy_results') and memory.strategy_results:
        log.info(f"  [eyes] {memory.strategy_summary()}")

    return solution, new_theory, memory


def play_game(env, timeout_per_level: float = 60.0,
              verbose: bool = True, game_id: str = "",
              memory: Optional[EpisodicMemory] = None
              ) -> tuple[int, int, int, EpisodicMemory]:
    """Play an entire multi-level game like a human.

    Returns (levels_solved, total_actions, total_baseline, memory).
    Memory persists between games for cross-game learning.
    """
    obs = env.reset()
    game = getattr(env, '_game', None)
    n_levels = getattr(game, 'level_index', 0)
    # Count levels by checking if the game reports them
    if hasattr(obs, 'win_levels'):
        n_levels = obs.win_levels
    else:
        n_levels = 7  # default guess

    # Initialize or reuse cross-game memory
    if memory is None:
        memory = EpisodicMemory()
    memory.start_game(game_id)

    theory = None
    total_actions = 0
    levels_solved = 0
    total_baseline = 0
    time_bank = 0.0  # fast levels donate time to hard levels

    for level in range(n_levels):
        budget = timeout_per_level + time_bank * 0.8
        level_start = time.time()

        if verbose:
            log.info(f"\n--- Level {level} (budget: {budget:.0f}s) ---")

        solution, theory, memory = play_level(env, obs, budget, theory,
                                               verbose, memory)

        if solution:
            # Execute the solution
            action_count = 0
            for act_id, data in solution:
                obs = env.step(act_id, data)
                action_count += 1
                # Pump animations
                game = getattr(env, '_game', None)
                if game and hasattr(game, 'vai'):
                    while getattr(game, 'vai', None) is not None:
                        obs = env.step(act_id, data)

            levels_solved += 1
            total_actions += action_count

            # Time banking — fast levels donate to slow ones
            elapsed = time.time() - level_start
            saved = budget - elapsed
            if saved > 0:
                time_bank += saved

            if verbose:
                log.info(f"  SOLVED: {action_count} actions "
                        f"(banked {saved:.0f}s)")

            # Check if game is complete
            if hasattr(obs, 'state') and obs.state is not None:
                sv = obs.state.value if hasattr(obs.state, 'value') else str(obs.state)
                if sv == 'WIN':
                    break
        else:
            if verbose:
                log.info(f"  FAILED level {level}")
            break

    # Record cross-game meta-pattern
    n_dir = sum(1 for k in (theory.action_effects if theory else {})
                if 'nodata' in k)
    n_click = sum(1 for k in (theory.action_effects if theory else {})
                  if 'click' in k)
    game_type = theory.game_type if theory else 'unknown'
    memory.end_game(game_id, game_type, levels_solved, total_actions,
                    n_dir, n_click)

    if verbose and memory.transitions:
        log.info(f"\n  [memory] {memory.summary()}")

    return levels_solved, total_actions, total_baseline, memory


# ---------------------------------------------------------------------------
# CLI — Test the eyes on a game
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse, os
    logging.basicConfig(level=logging.INFO, format='%(message)s')

    parser = argparse.ArgumentParser(description="ARC Eyes — see games like a human")
    parser.add_argument("--games", nargs='+', default=['vc33', 'ft09', 'ls20'],
                        help="Game IDs")
    parser.add_argument("--timeout", type=float, default=60, help="Seconds per level")
    parser.add_argument("--watch", action="store_true",
                        help="Watch mode — render each frame as ASCII art during play")
    parser.add_argument("--ascii", action="store_true", help="Print initial ASCII art")
    parser.add_argument("--no-color", action="store_true",
                        help="Disable ANSI colors in ASCII output")
    parser.add_argument("--memory", type=str,
                        default="results/arc_eyes_memory.json",
                        help="Path to persistent memory file")
    parser.add_argument("--no-memory", action="store_true",
                        help="Don't load/save persistent memory")
    args = parser.parse_args()

    import sys, glob as globmod
    from arc_agi import Arcade
    arcade = Arcade()

    # Load persistent cross-game memory
    if not args.no_memory:
        memory = EpisodicMemory.load(args.memory)
        if memory.game_patterns:
            log.info(f"[memory] Loaded experience from "
                    f"{len(memory.game_patterns)} previous games")
    else:
        memory = EpisodicMemory()

    grand_total_levels = 0
    grand_total_actions = 0
    grand_total_solved = 0

    # Monkey-patch env.step for --watch mode
    _original_steps = {}

    for game_id in args.games:
        # Add game path
        subdirs = globmod.glob(f'environment_files/{game_id}/*/')
        for sd in subdirs:
            if sd not in sys.path:
                sys.path.insert(0, sd)

        env = arcade.make(game_id)
        log.info(f"\n{'='*60}")
        log.info(f"Game: {game_id.upper()}")

        # Cross-game prediction
        predicted = memory.predict_game_type(0, 0)
        if predicted:
            log.info(f"  [memory] Based on {len(memory.game_patterns)} games, "
                    f"predicting: {predicted}")
        log.info(f"{'='*60}")

        if args.ascii or args.watch:
            obs = env.reset()
            use_color = not args.no_color
            print(frame_to_ascii(np.array(obs.frame), step=2, color=use_color))

        if args.watch:
            # Wrap play_game to render frames after each action
            _real_step = env.step.__func__ if hasattr(env.step, '__func__') else None
            _action_count = [0]

            class WatchWrapper:
                """Wraps env to render frames during play."""
                def __init__(self, env, use_color):
                    self._env = env
                    self._use_color = use_color
                    self._action_count = 0
                    # Proxy all attributes to the real env
                    for attr in dir(env):
                        if not attr.startswith('_') and attr != 'step':
                            try:
                                setattr(self, attr, getattr(env, attr))
                            except (AttributeError, TypeError):
                                pass

                def step(self, action_id, data=None):
                    obs = self._env.step(action_id, data)
                    self._action_count += 1
                    if obs is not None and hasattr(obs, 'frame'):
                        # Clear screen and redraw
                        print(f"\033[2J\033[H", end='')  # clear + home
                        frame = np.array(obs.frame)
                        print(f"Game: {game_id.upper()} | "
                              f"Action #{self._action_count}: "
                              f"act={action_id} data={data}")
                        print(frame_to_ascii(frame, step=2,
                                            color=self._use_color))
                        time.sleep(0.15)  # 150ms pause for human viewing
                    return obs

                def reset(self):
                    obs = self._env.reset()
                    self._action_count = 0
                    if obs is not None and hasattr(obs, 'frame'):
                        print(f"\033[2J\033[H", end='')
                        print(f"Game: {game_id.upper()} | RESET")
                        print(frame_to_ascii(np.array(obs.frame), step=2,
                                            color=self._use_color))
                    return obs

                def __getattr__(self, name):
                    return getattr(self._env, name)

                def __deepcopy__(self, memo):
                    # Deepcopy returns unwrapped env (solvers shouldn't render)
                    return copy.deepcopy(self._env, memo)

            watch_env = WatchWrapper(env, not args.no_color)
            solved, actions, baseline, memory = play_game(
                watch_env, args.timeout, game_id=game_id, memory=memory)
        else:
            solved, actions, baseline, memory = play_game(
                env, args.timeout, game_id=game_id, memory=memory)

        log.info(f"\n{game_id.upper()}: {solved} levels, {actions} actions")

        grand_total_solved += solved
        grand_total_actions += actions

    # Save persistent memory
    if not args.no_memory:
        os.makedirs(os.path.dirname(args.memory) or '.', exist_ok=True)
        memory.save(args.memory)

    log.info(f"\n{'='*60}")
    log.info(f"TOTAL: {grand_total_solved} levels solved, "
            f"{grand_total_actions} actions")
    if memory.game_patterns:
        log.info(f"MEMORY: {len(memory.game_patterns)} games played, "
                f"experience persisted to {args.memory}")
    if hasattr(memory, 'strategy_results') and memory.strategy_results:
        log.info(f"\n{memory.strategy_summary()}")
    log.info(f"{'='*60}")
