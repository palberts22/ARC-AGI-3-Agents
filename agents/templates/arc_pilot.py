"""ARC Pilot — The mind in the machine.

LLM as pilot, solver as body. The body handles real-time perception,
action, and memory. The mind handles transfer, abstraction, and
self-modification — but only at inflection points, not every frame.

Inflection points:
  1. All solver routes exhausted (body is stuck)
  2. Game type UNKNOWN after analysis (body can't classify)
  3. Route regression (what worked before stopped working)
  4. Level transition failure (solved L0-L2, stuck on L3)

The pilot sees:
  - Current frame (visual)
  - Game profile (body's classification)
  - Failed routes + why
  - Reasoner hypotheses
  - Solver memory (past games)
  - Action history

The pilot can:
  - Reframe game type (override profile)
  - Suggest specific action sequences
  - Modify solver parameters
  - Request targeted observations
  - Write new rules to memory
  - Strategic retreat (skip level to save time)

Design: Peter's Gundam architecture (KB #7423).
Implementation: Apollo, 2026-03-12.
"""

import base64
import io
import json
import logging
import os
import time
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

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

log = logging.getLogger("arc_pilot")


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

@dataclass
class PilotDirective:
    """What the pilot tells the body to do."""
    action: str  # 'reframe', 'try_sequence', 'modify_params', 'observe', 'skip', 'retry'

    # For 'reframe': override the game type
    new_game_type: str = ""

    # For 'try_sequence': specific actions to attempt
    suggested_actions: list = field(default_factory=list)  # list of (action_id, data_dict)

    # For 'modify_params': solver parameter overrides
    param_overrides: dict = field(default_factory=dict)
    # e.g. {'bfs_depth': 30, 'mcts_rollout': 500, 'focus_actions': [0,3,5]}

    # For 'observe': targeted experiments to run before retrying
    observations_requested: list = field(default_factory=list)
    # e.g. ['click_corners', 'click_center', 'try_all_dirs_from_start']

    # For all: pilot's reasoning (logged, stored in memory)
    reasoning: str = ""
    confidence: float = 0.0  # 0-1

    # Route preference for retry
    preferred_route: str = ""

    # Memory write-through: rules the pilot discovered
    new_rules: list = field(default_factory=list)
    # e.g. [{'pattern': 'grid_4x4', 'strategy': 'try toggle solver', 'genre': 'toggle'}]


@dataclass
class InflectionContext:
    """Everything the pilot needs to make a decision."""
    # Visual
    current_frame: Optional[np.ndarray] = None
    initial_frame: Optional[np.ndarray] = None

    # Game state
    game_id: str = ""
    game_type: str = ""
    level: int = 0
    levels_solved: int = 0
    levels_total: int = 0

    # What the body tried
    failed_routes: list = field(default_factory=list)  # [(route_name, failure_reason)]
    time_spent: float = 0.0
    time_remaining: float = 0.0

    # Profile
    n_actions: int = 0
    n_click_actions: int = 0
    n_directional_actions: int = 0
    self_inverse: bool = False
    commutative: bool = False
    genre_hints: list = field(default_factory=list)

    # Reasoner hypotheses
    hypotheses: list = field(default_factory=list)  # list of (description, confidence)

    # Memory
    similar_games: list = field(default_factory=list)
    known_rules: list = field(default_factory=list)

    # Cross-level context
    previous_winning_route: str = ""
    level_results: list = field(default_factory=list)  # [{level, solved, route, actions}]


# ---------------------------------------------------------------------------
# Pilot
# ---------------------------------------------------------------------------

class Pilot:
    """The mind. Invoked at inflection points only.

    Budget-aware: tracks API calls and respects limits.
    The body runs at 2000 FPS. The pilot runs at inflection points.
    """

    def __init__(self, max_calls_per_game: int = 3, backend: str = "auto",
                 flight_recorder: 'FlightRecorder' = None):
        self.max_calls_per_game = max_calls_per_game
        self.calls_this_game = 0
        self.total_calls = 0
        self.backend = backend  # 'auto', 'anthropic', 'openai', 'deepseek'
        self._call_log = []  # [(timestamp, game_id, level, directive)]
        self.recorder = flight_recorder  # The self — watches and grades us

    def reset_game(self):
        """Call at the start of each game."""
        self.calls_this_game = 0

    @property
    def budget_remaining(self) -> int:
        return max(0, self.max_calls_per_game - self.calls_this_game)

    def should_engage(self, ctx: InflectionContext) -> bool:
        """Should the pilot step in? Quick check before expensive LLM call."""
        if self.calls_this_game >= self.max_calls_per_game:
            log.info("  [pilot] Budget exhausted — staying quiet")
            return False

        if ctx.time_remaining < 5:
            log.info("  [pilot] Not enough time remaining")
            return False

        # Always engage if all routes failed
        if ctx.failed_routes and ctx.game_type != "SPECIALIZED":
            return True

        # Engage if game type is UNKNOWN and we haven't tried yet
        if ctx.game_type == "UNKNOWN" and self.calls_this_game == 0:
            return True

        # Engage if route regression (solved before, stuck now)
        if ctx.levels_solved > 0 and ctx.previous_winning_route:
            return True

        return False

    def consult(self, ctx: InflectionContext) -> Optional[PilotDirective]:
        """The inflection point. Ask the mind what to do.

        Returns a PilotDirective or None if the pilot can't help.
        """
        if not self.should_engage(ctx):
            return None

        self.calls_this_game += 1
        self.total_calls += 1

        # Capture vitals before engagement
        vitals = get_vitals() if HAS_PSUTIL else SystemVitals()

        # Ask the self for advice (historical data)
        self_advice = {}
        if self.recorder:
            self_advice = self.recorder.advise_pilot(ctx)
            if self_advice.get('recommend_skip') and vitals.under_pressure:
                log.warning(f"  [pilot] System under pressure — {vitals.summary()}")
                log.info("  [pilot] Self recommends skip to conserve resources")

        log.info(f"  [pilot] Engaging — call {self.calls_this_game}/{self.max_calls_per_game} "
                 f"(game={ctx.game_id}, level={ctx.level}) | {vitals.summary()}")

        # Build the prompt (enriched with self-advice)
        prompt = self._build_prompt(ctx, self_advice=self_advice)

        # Call LLM (measure latency)
        t0 = time.time()
        response = self._call_llm(prompt, ctx.current_frame)
        latency_ms = (time.time() - t0) * 1000

        if response is None:
            log.warning("  [pilot] LLM call failed — no directive")
            return None

        # Parse response into directive
        directive = self._parse_response(response, ctx)

        # Create engagement record for the flight recorder
        record = EngagementRecord(
            timestamp=time.time(),
            game_id=ctx.game_id,
            level=ctx.level,
            game_type=ctx.game_type,
            n_failed_routes=len(ctx.failed_routes),
            time_spent=ctx.time_spent,
            time_remaining=ctx.time_remaining,
            action=directive.action,
            preferred_route=directive.preferred_route,
            reasoning=directive.reasoning,
            confidence=directive.confidence,
            vitals=vitals,
            outcome='pending',
            latency_ms=latency_ms,
        )
        directive._engagement_record = record  # Attach for later evaluation

        if self.recorder:
            self.recorder.record_engagement(record)

        # Log
        self._call_log.append({
            'time': time.time(),
            'game': ctx.game_id,
            'level': ctx.level,
            'directive': directive.action,
            'reasoning': directive.reasoning,
            'latency_ms': round(latency_ms, 1),
        })

        log.info(f"  [pilot] Directive: {directive.action} — {directive.reasoning[:100]} "
                 f"(latency: {latency_ms:.0f}ms)")
        return directive

    def _build_prompt(self, ctx: InflectionContext, self_advice: dict = None) -> str:
        """Build the pilot's prompt from inflection context."""

        sections = []

        # Identity
        sections.append(
            "You are the PILOT of an autonomous game-solving system. "
            "Your BODY (the solver) has been playing this game and is now stuck. "
            "You step in at inflection points to reframe, redirect, or diagnose. "
            "You don't play the game — you coach the solver."
        )

        # Game state
        sections.append(f"""
## Game State
- Game: {ctx.game_id}
- Detected type: {ctx.game_type}
- Level: {ctx.level} (solved {ctx.levels_solved}/{ctx.levels_total or '?'})
- Actions available: {ctx.n_actions} ({ctx.n_click_actions} click, {ctx.n_directional_actions} directional)
- Self-inverse: {ctx.self_inverse} | Commutative: {ctx.commutative}
- Genre hints: {ctx.genre_hints or 'none'}
- Time spent: {ctx.time_spent:.1f}s | Time remaining: {ctx.time_remaining:.1f}s""")

        # Failed routes
        if ctx.failed_routes:
            route_lines = "\n".join(f"  - {r[0]}: {r[1]}" for r in ctx.failed_routes[-8:])
            sections.append(f"""
## Routes Tried (all failed)
{route_lines}""")

        # Hypotheses from reasoner
        if ctx.hypotheses:
            hyp_lines = "\n".join(f"  - [{h[1]:.0%}] {h[0]}" for h in ctx.hypotheses[:5])
            sections.append(f"""
## Reasoner Hypotheses
{hyp_lines}""")

        # Memory
        if ctx.similar_games:
            mem_lines = "\n".join(
                f"  - {g.get('game_name', '?')} ({g.get('genre', '?')}): "
                f"route={g.get('winning_route', '?')}, "
                f"{g.get('levels_solved', 0)}/{g.get('levels_total', '?')} levels"
                for g in ctx.similar_games[:3]
            )
            sections.append(f"""
## Similar Games in Memory
{mem_lines}""")

        if ctx.known_rules:
            rule_lines = "\n".join(
                f"  - [{r.get('confidence', 0):.0%}] {r.get('pattern', '?')} → {r.get('strategy', '?')}"
                for r in ctx.known_rules[:5]
            )
            sections.append(f"""
## Known Rules
{rule_lines}""")

        # Cross-level context
        if ctx.level_results:
            lev_lines = "\n".join(
                f"  - L{r.get('level', '?')}: {'SOLVED' if r.get('solved') else 'FAILED'} "
                f"via {r.get('route', '?')} ({r.get('actions', '?')} actions)"
                for r in ctx.level_results[-5:]
            )
            sections.append(f"""
## Level History
{lev_lines}
- Previous winning route: {ctx.previous_winning_route or 'none'}""")

        # Frame description (text fallback if no vision)
        if ctx.current_frame is not None:
            desc = _describe_frame_for_pilot(ctx.current_frame)
            sections.append(f"""
## Current Frame
{desc}""")

        # Self-advice (historical data from the flight recorder)
        if self_advice:
            advice_lines = []
            if self_advice.get('best_routes'):
                advice_lines.append("Historical route performance for this game type:")
                for r in self_advice['best_routes'][:3]:
                    advice_lines.append(
                        f"  - {r['route']}: {r['success_rate']:.0%} success "
                        f"({r['attempts']} attempts, avg {r['avg_time']:.1f}s)")
            if self_advice.get('pilot_history'):
                ph = self_advice['pilot_history']
                advice_lines.append(
                    f"Your own track record on {ctx.game_type}: "
                    f"{ph['success_rate']:.0%} ({ph['engagements']} prior engagements)")
            if self_advice.get('pressure_warning'):
                advice_lines.append(f"⚠ SYSTEM PRESSURE: {self_advice['pressure_warning']}")
                advice_lines.append("Consider SKIP to conserve resources for solvable levels.")
            if advice_lines:
                sections.append("## Self-Awareness (your historical performance)\n" +
                                "\n".join(advice_lines))

        # Instructions
        sections.append("""
## Your Task
Analyze the situation and give ONE directive. Be specific and actionable.

Answer in this exact format:
ACTION: reframe | retry | observe | skip
GAME_TYPE: (only if ACTION=reframe — what type is this game really?)
ROUTE: (which solver route should be tried — e.g. 'bfs', 'mcts', 'mechanic', 'navigation', 'toggle_matrix', 'constraint')
PARAMS: (JSON object of parameter overrides, or {} if none)
REASONING: (1-2 sentences — what did the body miss?)
CONFIDENCE: (0.0-1.0)
RULES: (any new rules to remember, as JSON list of {pattern, strategy, genre} — or [])""")

        return "\n".join(sections)

    def _call_llm(self, prompt: str, frame: Optional[np.ndarray] = None) -> Optional[str]:
        """Call the LLM backend. Returns raw response text."""
        if not HAS_REQUESTS:
            log.warning("  [pilot] requests library not available")
            return None

        # Backend selection priority:
        # 1. ARC_LLM_ENDPOINT (OpenAI-compatible — Qwen, ollama, vLLM)
        # 2. ANTHROPIC (Claude)
        # 3. DEEPSEEK (text-only fallback)

        oai_endpoint = os.environ.get('ARC_LLM_ENDPOINT', '')
        oai_model = os.environ.get('ARC_LLM_MODEL', 'qwen2.5-vl-32b')
        oai_key = os.environ.get('ARC_LLM_KEY', 'none')

        anthropic_key = os.environ.get('ANTHROPIC', '')
        deepseek_key = os.environ.get('DEEPSEEK', '')

        use_vision = frame is not None and HAS_PIL
        img_b64 = _frame_to_base64(frame) if use_vision else None

        # Try backends in order
        if oai_endpoint and (self.backend in ('auto', 'openai')):
            return self._call_openai_compat(oai_endpoint, oai_model, oai_key, prompt, img_b64)

        if anthropic_key and (self.backend in ('auto', 'anthropic')):
            return self._call_anthropic(anthropic_key, prompt, img_b64)

        if deepseek_key and (self.backend in ('auto', 'deepseek')):
            return self._call_deepseek(deepseek_key, prompt)

        log.warning("  [pilot] No LLM backend available (set ARC_LLM_ENDPOINT, ANTHROPIC, or DEEPSEEK)")
        return None

    def _call_openai_compat(self, endpoint: str, model: str, key: str,
                            prompt: str, img_b64: Optional[str]) -> Optional[str]:
        """Call OpenAI-compatible endpoint (vLLM, ollama, Qwen, etc.)."""
        endpoint = endpoint.rstrip('/')
        if img_b64:
            content = [
                {'type': 'image_url', 'image_url': {'url': f'data:image/png;base64,{img_b64}'}},
                {'type': 'text', 'text': prompt},
            ]
        else:
            content = [{'type': 'text', 'text': prompt}]

        try:
            resp = _requests.post(
                f'{endpoint}/chat/completions',
                headers={'Authorization': f'Bearer {key}', 'Content-Type': 'application/json'},
                json={
                    'model': model,
                    'messages': [{'role': 'user', 'content': content}],
                    'max_tokens': 400,
                    'temperature': 0.3,
                },
                timeout=30,
            )
            resp.raise_for_status()
            return resp.json()['choices'][0]['message']['content']
        except Exception as e:
            log.warning(f"  [pilot] OpenAI-compat error: {e}")
            return None

    def _call_anthropic(self, api_key: str, prompt: str,
                        img_b64: Optional[str]) -> Optional[str]:
        """Call Anthropic Claude."""
        content = []
        if img_b64:
            content.append({
                'type': 'image',
                'source': {'type': 'base64', 'media_type': 'image/png', 'data': img_b64},
            })
        content.append({'type': 'text', 'text': prompt})

        try:
            resp = _requests.post(
                'https://api.anthropic.com/v1/messages',
                headers={
                    'x-api-key': api_key,
                    'anthropic-version': '2023-06-01',
                    'Content-Type': 'application/json',
                },
                json={
                    'model': os.environ.get('ARC_PILOT_MODEL', 'claude-opus-4-20250514'),
                    'max_tokens': 400,
                    'messages': [{'role': 'user', 'content': content}],
                },
                timeout=20,
            )
            resp.raise_for_status()
            return resp.json()['content'][0]['text']
        except Exception as e:
            log.warning(f"  [pilot] Anthropic error: {e}")
            return None

    def _call_deepseek(self, api_key: str, prompt: str) -> Optional[str]:
        """Call DeepSeek (text-only, cheap)."""
        try:
            resp = _requests.post(
                'https://api.deepseek.com/chat/completions',
                headers={'Authorization': f'Bearer {api_key}', 'Content-Type': 'application/json'},
                json={
                    'model': 'deepseek-chat',
                    'messages': [{'role': 'user', 'content': prompt}],
                    'max_tokens': 400,
                    'temperature': 0.3,
                },
                timeout=20,
            )
            resp.raise_for_status()
            return resp.json()['choices'][0]['message']['content']
        except Exception as e:
            log.warning(f"  [pilot] DeepSeek error: {e}")
            return None

    def _parse_response(self, text: str, ctx: InflectionContext) -> PilotDirective:
        """Parse LLM response into a PilotDirective."""
        directive = PilotDirective(action='retry')

        for line in text.strip().split('\n'):
            line = line.strip()
            if ':' not in line:
                continue
            key, _, val = line.partition(':')
            key = key.strip().upper()
            val = val.strip()

            if key == 'ACTION':
                val_lower = val.lower().strip()
                if val_lower in ('reframe', 'retry', 'observe', 'skip'):
                    directive.action = val_lower

            elif key == 'GAME_TYPE':
                directive.new_game_type = val.upper().replace(' ', '_')

            elif key == 'ROUTE':
                directive.preferred_route = val.lower().strip()

            elif key == 'PARAMS':
                try:
                    directive.param_overrides = json.loads(val)
                except (json.JSONDecodeError, ValueError):
                    pass

            elif key == 'REASONING':
                directive.reasoning = val

            elif key == 'CONFIDENCE':
                try:
                    directive.confidence = float(val)
                except ValueError:
                    pass

            elif key == 'RULES':
                try:
                    rules = json.loads(val)
                    if isinstance(rules, list):
                        directive.new_rules = rules
                except (json.JSONDecodeError, ValueError):
                    pass

        return directive


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _frame_to_base64(frame: Optional[np.ndarray], scale: int = 2) -> Optional[str]:
    """Convert frame to base64 PNG."""
    if frame is None or not HAS_PIL:
        return None
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


def _describe_frame_for_pilot(frame: np.ndarray) -> str:
    """Text description of a frame for the pilot's situational awareness."""
    if frame.ndim == 3 and frame.shape[0] in (1, 3, 4):
        f = np.transpose(frame, (1, 2, 0))
    elif frame.ndim == 2:
        f = frame[:, :, None]
    else:
        f = frame

    h, w = f.shape[:2]

    # Unique colors
    pixels = f.reshape(-1, f.shape[-1])
    unique = np.unique(pixels, axis=0)

    # Spatial structure — where are the non-background pixels?
    bg_color = pixels[0]  # top-left as background guess
    non_bg = ~np.all(f == bg_color, axis=-1)
    non_bg_pct = non_bg.sum() / (h * w) * 100

    # Find regions of interest
    rows_active = np.any(non_bg, axis=1)
    cols_active = np.any(non_bg, axis=0)

    row_spans = _find_spans(rows_active)
    col_spans = _find_spans(cols_active)

    desc = f"Size: {w}x{h}, {len(unique)} colors, {non_bg_pct:.0f}% non-background"

    if row_spans:
        desc += f"\nActive row bands: {len(row_spans)} ({', '.join(f'{s}-{e}' for s,e in row_spans[:5])})"
    if col_spans:
        desc += f"\nActive col bands: {len(col_spans)} ({', '.join(f'{s}-{e}' for s,e in col_spans[:5])})"

    # Symmetry check
    if h == w:
        desc += "\nSquare frame"
    lr_sym = np.allclose(f, np.flip(f, axis=1))
    tb_sym = np.allclose(f, np.flip(f, axis=0))
    if lr_sym:
        desc += " | Left-right symmetric"
    if tb_sym:
        desc += " | Top-bottom symmetric"

    # Color palette
    for i, color in enumerate(unique[:8]):
        mask = np.all(f == color, axis=-1)
        pct = mask.sum() / (h * w) * 100
        if pct > 1:
            desc += f"\n  Color {i} ({','.join(str(c) for c in color[:3])}): {pct:.0f}%"

    return desc


def _find_spans(mask: np.ndarray) -> list:
    """Find contiguous True spans in a 1D boolean array."""
    spans = []
    in_span = False
    start = 0
    for i, v in enumerate(mask):
        if v and not in_span:
            start = i
            in_span = True
        elif not v and in_span:
            spans.append((start, i - 1))
            in_span = False
    if in_span:
        spans.append((start, len(mask) - 1))
    return spans


# ---------------------------------------------------------------------------
# Integration: build context from solver state
# ---------------------------------------------------------------------------

def build_inflection_context(
    env_snap,
    game_id: str,
    profile,  # GameProfile
    level: int,
    levels_solved: int,
    levels_total: int,
    failed_routes: list,
    time_spent: float,
    time_remaining: float,
    previous_winning_route: str = "",
    level_results: list = None,
    reasoner=None,  # InductiveReasoner
    solver_memory=None,  # SolverMemory
) -> InflectionContext:
    """Build an InflectionContext from the current solver state.

    Called by solve_level/run_game when an inflection point is detected.
    """
    ctx = InflectionContext()
    ctx.game_id = game_id
    ctx.game_type = profile.game_type if profile else "UNKNOWN"
    ctx.level = level
    ctx.levels_solved = levels_solved
    ctx.levels_total = levels_total
    ctx.failed_routes = failed_routes
    ctx.time_spent = time_spent
    ctx.time_remaining = time_remaining
    ctx.previous_winning_route = previous_winning_route
    ctx.level_results = level_results or []

    # Frame
    if hasattr(env_snap, '_last_obs') and env_snap._last_obs is not None:
        ctx.current_frame = np.array(env_snap._last_obs.frame)

    # Profile details
    if profile:
        ctx.n_actions = len(getattr(profile, 'actions', []))
        ctx.n_click_actions = len(getattr(profile, 'click_actions', []))
        ctx.n_directional_actions = len(getattr(profile, 'directional_actions', []))
        ctx.self_inverse = getattr(profile, 'self_inverse', False)
        ctx.commutative = getattr(profile, 'commutative', False)
        ctx.genre_hints = getattr(profile, 'genre_hints', [])

    # Reasoner hypotheses
    if reasoner and hasattr(reasoner, 'hypotheses'):
        ctx.hypotheses = [
            (h.description, h.confidence)
            for h in reasoner.hypotheses
            if h.confidence > 0.2
        ]

    # Solver memory
    if solver_memory:
        try:
            genre = ctx.genre_hints[0] if ctx.genre_hints else "unknown"
            ctx.similar_games = solver_memory.get_similar_games(genre=genre) or []
            ctx.known_rules = solver_memory.get_rulebook(genre=genre, min_confidence=0.3) or []
        except Exception:
            pass

    return ctx


# ---------------------------------------------------------------------------
# Observation Runner — the pilot's hands
# ---------------------------------------------------------------------------

import copy


def run_observations(env_snap, actions: list, requests: list,
                     start_levels: int) -> list[dict]:
    """Execute targeted experiments on behalf of the pilot.

    The pilot says "observe X". This function does it on a deepcopy
    and returns structured results. The pilot can then make a better decision.

    Supported observation requests:
      'click_corners'     — click the 4 corner positions
      'click_center'      — click the center position
      'try_all_once'      — try each action once, report changes
      'hold_still'        — do nothing for N frames, watch what moves
      'repeat_action:N'   — press action N multiple times, watch cycle

    Returns list of {request, results: [{action, changes, won}]}
    """
    observations = []

    for req in requests:
        req_lower = req.lower().strip() if isinstance(req, str) else str(req).lower()
        result = {'request': req, 'results': []}

        if req_lower == 'try_all_once':
            # Try each action once, report pixel changes
            for act in actions[:20]:  # Cap to avoid blowing time
                probe_env = copy.deepcopy(env_snap)
                obs_before = getattr(probe_env, '_last_obs', None)
                frame_before = np.array(obs_before.frame) if obs_before else None

                try:
                    obs_after = probe_env.step(act.game_action,
                                               act.data if act.data else None)
                except Exception:
                    continue

                if obs_after is None:
                    continue

                frame_after = np.array(obs_after.frame)
                won = (obs_after.levels_completed > start_levels
                       if hasattr(obs_after, 'levels_completed') else False)

                n_changed = 0
                if frame_before is not None:
                    n_changed = int(np.sum(frame_before != frame_after))

                result['results'].append({
                    'action': str(act),
                    'action_id': act.game_action,
                    'pixels_changed': n_changed,
                    'won': won,
                })

        elif req_lower == 'click_corners':
            # Find corner positions from click actions
            click_acts = [a for a in actions if a.data and 'x' in a.data]
            if click_acts:
                xs = sorted(set(a.data['x'] for a in click_acts))
                ys = sorted(set(a.data['y'] for a in click_acts))
                corners = []
                if xs and ys:
                    for x in [xs[0], xs[-1]]:
                        for y in [ys[0], ys[-1]]:
                            corner = next((a for a in click_acts
                                          if a.data['x'] == x and a.data['y'] == y), None)
                            if corner:
                                corners.append(corner)

                for act in corners:
                    probe_env = copy.deepcopy(env_snap)
                    obs_before = getattr(probe_env, '_last_obs', None)
                    frame_before = np.array(obs_before.frame) if obs_before else None

                    try:
                        obs_after = probe_env.step(act.game_action,
                                                   act.data if act.data else None)
                    except Exception:
                        continue

                    if obs_after is None:
                        continue

                    frame_after = np.array(obs_after.frame)
                    n_changed = int(np.sum(frame_before != frame_after)) if frame_before is not None else 0

                    result['results'].append({
                        'action': str(act),
                        'pixels_changed': n_changed,
                        'won': False,
                    })

        elif req_lower == 'click_center':
            click_acts = [a for a in actions if a.data and 'x' in a.data]
            if click_acts:
                xs = sorted(set(a.data['x'] for a in click_acts))
                ys = sorted(set(a.data['y'] for a in click_acts))
                mid_x = xs[len(xs) // 2] if xs else 0
                mid_y = ys[len(ys) // 2] if ys else 0
                center = next((a for a in click_acts
                               if a.data['x'] == mid_x and a.data['y'] == mid_y), None)
                if center:
                    probe_env = copy.deepcopy(env_snap)
                    obs_before = getattr(probe_env, '_last_obs', None)
                    frame_before = np.array(obs_before.frame) if obs_before else None

                    try:
                        obs_after = probe_env.step(center.game_action,
                                                   center.data if center.data else None)
                    except Exception:
                        pass
                    else:
                        if obs_after:
                            frame_after = np.array(obs_after.frame)
                            n_changed = int(np.sum(frame_before != frame_after)) if frame_before is not None else 0
                            result['results'].append({
                                'action': str(center),
                                'pixels_changed': n_changed,
                                'won': False,
                            })

        elif req_lower.startswith('repeat_action:'):
            # Repeat a specific action N times, watch the cycle
            try:
                action_id = int(req_lower.split(':')[1])
            except (ValueError, IndexError):
                action_id = 0
            target = next((a for a in actions if a.game_action == action_id), None)
            if target:
                probe_env = copy.deepcopy(env_snap)
                frames_seen = []
                for i in range(10):  # repeat up to 10 times
                    obs_before = getattr(probe_env, '_last_obs', None)
                    frame_before = np.array(obs_before.frame) if obs_before else None

                    try:
                        obs_after = probe_env.step(target.game_action,
                                                   target.data if target.data else None)
                    except Exception:
                        break

                    if obs_after is None:
                        break

                    frame_after = np.array(obs_after.frame)
                    probe_env._last_obs = obs_after
                    n_changed = int(np.sum(frame_before != frame_after)) if frame_before is not None else 0
                    frame_hash = hash(frame_after.tobytes())

                    # Check for cycle
                    cycle_at = None
                    for j, fh in enumerate(frames_seen):
                        if fh == frame_hash:
                            cycle_at = j
                            break
                    frames_seen.append(frame_hash)

                    result['results'].append({
                        'repeat': i + 1,
                        'pixels_changed': n_changed,
                        'cycle_detected': cycle_at,
                    })
                    if cycle_at is not None:
                        break  # Found the cycle, stop

        elif req_lower == 'hold_still':
            # Step with a noop action (action 0) repeatedly, watch for autonomous changes
            if actions:
                probe_env = copy.deepcopy(env_snap)
                first_act = actions[0]  # Use first action as proxy
                for i in range(5):
                    obs_before = getattr(probe_env, '_last_obs', None)
                    frame_before = np.array(obs_before.frame) if obs_before else None

                    try:
                        obs_after = probe_env.step(first_act.game_action,
                                                   first_act.data if first_act.data else None)
                    except Exception:
                        break

                    if obs_after is None:
                        break

                    frame_after = np.array(obs_after.frame)
                    probe_env._last_obs = obs_after
                    n_changed = int(np.sum(frame_before != frame_after)) if frame_before is not None else 0
                    result['results'].append({
                        'frame': i,
                        'pixels_changed': n_changed,
                        'autonomous_movement': n_changed > 100,
                    })

        observations.append(result)

    return observations


def apply_directive(directive: PilotDirective, profile, solver_memory=None):
    """Apply a pilot directive to the solver state.

    Returns modified profile (or original if no reframe).
    Side effects: writes rules to memory if directive has new_rules.
    """
    if directive.action == 'reframe' and directive.new_game_type:
        log.info(f"  [pilot] Reframing: {profile.game_type} → {directive.new_game_type}")
        profile.game_type = directive.new_game_type

    # Write rules to memory
    if directive.new_rules and solver_memory:
        for rule in directive.new_rules:
            if isinstance(rule, dict) and 'pattern' in rule and 'strategy' in rule:
                try:
                    solver_memory.add_rule(
                        pattern=rule['pattern'],
                        mechanism=rule.get('mechanism', ''),
                        strategy=rule['strategy'],
                        genre=rule.get('genre', 'unknown'),
                        confidence=directive.confidence,
                        source='pilot',
                    )
                    log.info(f"  [pilot] Wrote rule: {rule['pattern']} → {rule['strategy']}")
                except Exception as e:
                    log.debug(f"  [pilot] Failed to write rule: {e}")

    return profile


# ---------------------------------------------------------------------------
# LAYER 3: THE SELF — autonomous awareness, telemetry, self-evaluation
# ---------------------------------------------------------------------------
# The body plays. The mind coaches. The self watches both and learns.

try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False

import sqlite3
from pathlib import Path


@dataclass
class SystemVitals:
    """Snapshot of system resources — the body's heartbeat."""
    cpu_percent: float = 0.0
    memory_used_mb: float = 0.0
    memory_available_mb: float = 0.0
    memory_percent: float = 0.0
    process_memory_mb: float = 0.0
    disk_free_gb: float = 0.0
    timestamp: float = 0.0

    @property
    def under_pressure(self) -> bool:
        """Is the system resource-constrained?"""
        return self.memory_percent > 85 or self.cpu_percent > 90

    def summary(self) -> str:
        pressure = " ⚠ PRESSURE" if self.under_pressure else ""
        return (f"CPU:{self.cpu_percent:.0f}% | "
                f"RAM:{self.memory_used_mb:.0f}/{self.memory_used_mb+self.memory_available_mb:.0f}MB "
                f"({self.memory_percent:.0f}%) | "
                f"Proc:{self.process_memory_mb:.0f}MB | "
                f"Disk:{self.disk_free_gb:.1f}GB{pressure}")


def get_vitals() -> SystemVitals:
    """Read current system vitals. Fast — no blocking calls."""
    v = SystemVitals(timestamp=time.time())
    if not HAS_PSUTIL:
        return v
    try:
        v.cpu_percent = psutil.cpu_percent(interval=0)  # non-blocking
        mem = psutil.virtual_memory()
        v.memory_used_mb = mem.used / (1024 * 1024)
        v.memory_available_mb = mem.available / (1024 * 1024)
        v.memory_percent = mem.percent
        proc = psutil.Process()
        v.process_memory_mb = proc.memory_info().rss / (1024 * 1024)
        disk = psutil.disk_usage('/')
        v.disk_free_gb = disk.free / (1024 * 1024 * 1024)
    except Exception:
        pass
    return v


@dataclass
class EngagementRecord:
    """One pilot engagement — what happened and did it work?"""
    timestamp: float
    game_id: str
    level: int
    game_type: str

    # What the pilot saw
    n_failed_routes: int = 0
    time_spent: float = 0.0
    time_remaining: float = 0.0

    # What the pilot decided
    action: str = ""          # reframe, retry, observe, skip
    preferred_route: str = ""
    reasoning: str = ""
    confidence: float = 0.0

    # System state at decision time
    vitals: Optional[SystemVitals] = None

    # Outcome (filled in by evaluate())
    outcome: str = ""         # 'success', 'failure', 'skipped', 'pending'
    outcome_actions: int = 0  # actions used if successful
    latency_ms: float = 0.0   # LLM call latency


class FlightRecorder:
    """The self. Watches the pilot, grades it, learns from it.

    Persists to SQLite — survives across sessions. The pilot's performance
    data becomes training signal for better engagement decisions.
    """

    def __init__(self, db_path: str = ""):
        if not db_path:
            db_path = os.path.join(
                os.path.dirname(os.path.dirname(__file__)),
                "data", "pilot_telemetry.db"
            )
        self.db_path = db_path
        self.db = sqlite3.connect(db_path)
        self.db.row_factory = sqlite3.Row
        self._init_tables()

        # In-memory ring buffer for current session
        self._session_log: list[EngagementRecord] = []
        self._route_stats: dict[str, dict] = {}  # route -> {attempts, successes, total_time}
        self._session_start = time.time()
        self._games_played = 0
        self._levels_solved = 0
        self._levels_failed = 0
        self._pilot_successes = 0
        self._pilot_failures = 0

    def _init_tables(self):
        self.db.executescript("""
            CREATE TABLE IF NOT EXISTS engagements (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp REAL NOT NULL,
                game_id TEXT NOT NULL,
                level INTEGER NOT NULL,
                game_type TEXT,
                n_failed_routes INTEGER,
                time_spent REAL,
                time_remaining REAL,
                action TEXT,
                preferred_route TEXT,
                reasoning TEXT,
                confidence REAL,
                outcome TEXT,
                outcome_actions INTEGER,
                latency_ms REAL,
                cpu_percent REAL,
                memory_percent REAL,
                process_memory_mb REAL
            );

            CREATE TABLE IF NOT EXISTS route_performance (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp REAL NOT NULL,
                game_id TEXT NOT NULL,
                game_type TEXT,
                route TEXT NOT NULL,
                success INTEGER NOT NULL,
                actions_used INTEGER,
                time_spent REAL,
                level INTEGER
            );

            CREATE TABLE IF NOT EXISTS session_summaries (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_start REAL NOT NULL,
                session_end REAL NOT NULL,
                games_played INTEGER,
                levels_solved INTEGER,
                levels_failed INTEGER,
                pilot_engagements INTEGER,
                pilot_successes INTEGER,
                pilot_success_rate REAL,
                avg_cpu REAL,
                peak_memory_mb REAL,
                notes TEXT
            );
        """)
        self.db.commit()

    def record_engagement(self, record: EngagementRecord):
        """Log a pilot engagement."""
        self._session_log.append(record)

        vitals = record.vitals or SystemVitals()
        try:
            self.db.execute("""
                INSERT INTO engagements
                (timestamp, game_id, level, game_type, n_failed_routes,
                 time_spent, time_remaining, action, preferred_route,
                 reasoning, confidence, outcome, outcome_actions,
                 latency_ms, cpu_percent, memory_percent, process_memory_mb)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                record.timestamp, record.game_id, record.level,
                record.game_type, record.n_failed_routes,
                record.time_spent, record.time_remaining,
                record.action, record.preferred_route,
                record.reasoning, record.confidence,
                record.outcome, record.outcome_actions,
                record.latency_ms,
                vitals.cpu_percent, vitals.memory_percent,
                vitals.process_memory_mb,
            ))
            self.db.commit()
        except Exception as e:
            log.debug(f"  [self] Failed to persist engagement: {e}")

    def record_route(self, game_id: str, game_type: str, route: str,
                     success: bool, actions_used: int = 0,
                     time_spent: float = 0.0, level: int = 0):
        """Log a route attempt (success or failure) — the body's heartbeat."""
        key = route
        if key not in self._route_stats:
            self._route_stats[key] = {'attempts': 0, 'successes': 0, 'total_time': 0.0}
        self._route_stats[key]['attempts'] += 1
        if success:
            self._route_stats[key]['successes'] += 1
        self._route_stats[key]['total_time'] += time_spent

        try:
            self.db.execute("""
                INSERT INTO route_performance
                (timestamp, game_id, game_type, route, success, actions_used, time_spent, level)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (time.time(), game_id, game_type, route, int(success),
                  actions_used, time_spent, level))
            self.db.commit()
        except Exception as e:
            log.debug(f"  [self] Failed to persist route: {e}")

    def evaluate(self, record: EngagementRecord, solved: bool,
                 actions_used: int = 0):
        """Grade the pilot's decision. Did its directive work?

        Called after solve_level returns — we now know the outcome.
        """
        record.outcome = 'success' if solved else 'failure'
        record.outcome_actions = actions_used

        if solved:
            self._pilot_successes += 1
            log.info(f"  [self] Pilot directive WORKED — {record.action}:{record.preferred_route} "
                     f"solved level {record.level} in {actions_used} actions")
        else:
            self._pilot_failures += 1
            log.info(f"  [self] Pilot directive FAILED — {record.action}:{record.preferred_route} "
                     f"did not solve level {record.level}")

        # Update the DB record
        self.record_engagement(record)

    def get_route_stats(self) -> dict:
        """Current session's route performance stats."""
        return {
            route: {
                'attempts': s['attempts'],
                'successes': s['successes'],
                'success_rate': s['successes'] / s['attempts'] if s['attempts'] > 0 else 0,
                'avg_time': s['total_time'] / s['attempts'] if s['attempts'] > 0 else 0,
            }
            for route, s in self._route_stats.items()
        }

    def get_historical_success_rate(self, route: str = "",
                                    game_type: str = "") -> float:
        """Query historical success rate from DB."""
        try:
            where = []
            params = []
            if route:
                where.append("route = ?")
                params.append(route)
            if game_type:
                where.append("game_type = ?")
                params.append(game_type)
            clause = " AND ".join(where) if where else "1=1"

            row = self.db.execute(f"""
                SELECT COUNT(*) as total,
                       SUM(CASE WHEN success=1 THEN 1 ELSE 0 END) as wins
                FROM route_performance WHERE {clause}
            """, params).fetchone()

            if row and row['total'] > 0:
                return row['wins'] / row['total']
        except Exception:
            pass
        return 0.0

    def get_pilot_success_rate(self) -> float:
        """How often does the pilot's intervention actually help?"""
        total = self._pilot_successes + self._pilot_failures
        if total == 0:
            return 0.0
        return self._pilot_successes / total

    def session_summary(self) -> dict:
        """Current session stats — the self's awareness of its own performance."""
        vitals = get_vitals()
        uptime = time.time() - self._session_start
        pilot_total = self._pilot_successes + self._pilot_failures
        return {
            'uptime_s': round(uptime, 1),
            'games_played': self._games_played,
            'levels_solved': self._levels_solved,
            'levels_failed': self._levels_failed,
            'pilot_engagements': pilot_total,
            'pilot_success_rate': self.get_pilot_success_rate(),
            'route_stats': self.get_route_stats(),
            'vitals': vitals.summary(),
            'under_pressure': vitals.under_pressure,
        }

    def save_session_summary(self):
        """Persist session summary to DB. Call at session end."""
        vitals = get_vitals()
        pilot_total = self._pilot_successes + self._pilot_failures
        rate = self.get_pilot_success_rate()
        try:
            self.db.execute("""
                INSERT INTO session_summaries
                (session_start, session_end, games_played, levels_solved,
                 levels_failed, pilot_engagements, pilot_successes,
                 pilot_success_rate, avg_cpu, peak_memory_mb, notes)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                self._session_start, time.time(),
                self._games_played, self._levels_solved,
                self._levels_failed, pilot_total, self._pilot_successes,
                rate, vitals.cpu_percent, vitals.process_memory_mb,
                json.dumps(self.get_route_stats()),
            ))
            self.db.commit()
            log.info(f"  [self] Session summary saved — "
                     f"{self._games_played} games, "
                     f"{self._levels_solved}/{self._levels_solved + self._levels_failed} levels, "
                     f"pilot {rate:.0%} effective")
        except Exception as e:
            log.debug(f"  [self] Failed to save session summary: {e}")

    def advise_pilot(self, ctx: InflectionContext) -> dict:
        """The self advises the pilot before it engages.

        Returns hints based on historical data — which routes work for
        this game type, what the pilot's track record is, system pressure.
        """
        advice = {}

        # Historical route performance for this game type
        try:
            rows = self.db.execute("""
                SELECT route, COUNT(*) as n,
                       SUM(CASE WHEN success=1 THEN 1 ELSE 0 END) as wins,
                       AVG(time_spent) as avg_time
                FROM route_performance
                WHERE game_type = ?
                GROUP BY route
                ORDER BY wins DESC
                LIMIT 5
            """, (ctx.game_type,)).fetchall()
            if rows:
                advice['best_routes'] = [
                    {'route': r['route'], 'success_rate': r['wins']/r['n'] if r['n'] > 0 else 0,
                     'avg_time': round(r['avg_time'], 1), 'attempts': r['n']}
                    for r in rows
                ]
        except Exception:
            pass

        # Pilot track record for this game type
        try:
            row = self.db.execute("""
                SELECT COUNT(*) as n,
                       SUM(CASE WHEN outcome='success' THEN 1 ELSE 0 END) as wins
                FROM engagements WHERE game_type = ?
            """, (ctx.game_type,)).fetchone()
            if row and row['n'] > 0:
                advice['pilot_history'] = {
                    'engagements': row['n'],
                    'success_rate': row['wins'] / row['n'],
                }
        except Exception:
            pass

        # System pressure
        vitals = get_vitals()
        if vitals.under_pressure:
            advice['pressure_warning'] = vitals.summary()
            advice['recommend_skip'] = True  # conserve resources

        return advice
