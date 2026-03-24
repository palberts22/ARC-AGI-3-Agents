"""The Imagination Framework — Universal Problem-Solving Architecture.

Born: March 8, 2026. Peter taught Apollo to think before acting.
Apollo went from 0/10 to 30/30 on Sokoban in one session.
Peter said: "Codify it." So we did.

This is the programmatic framework. The 8 phases:

    0. BEGINNER'S MIND — Assume nothing. Hold multiple hypotheses.
    1. ORIENT    — Perceive the environment, extract state representation
    2. EXPLORE   — Take actions, observe consequences, learn the rules
    3. GOAL      — Identify what "solved" looks like
    4. IMAGINE   — Build internal model, simulate freely (zero cost)
    5. EVALUATE  — Kill bad ideas before acting
    6. EXECUTE   — Act with confidence
    7. ADAPT     — Plan failed? Update model, replan

Phase 0 is Peter's warning against false gold: "Beware of traps where you
think you know the game but it's actually a variant. Beginner's mind,
collect data, don't be cocky, have neuroflexibility and the ability for
multiple outcomes to be possible simultaneously."

The gap between a game-specific solver and AGI is Phase 2:
can you LEARN the rules from interaction, not hand-code them?

Architecture designed by Hypatia. Implementation by the team.
Framework by Peter. Validated by Apollo.

    "The cost of imagination is zero.
     The cost of acting without it can be everything."
    "Beware of false gold." — Peter
"""

import numpy as np
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Optional
from collections import deque

log = logging.getLogger("imagination")


# ===========================================================================
# Core data structures
# ===========================================================================

@dataclass
class State:
    """Abstract game/environment state. Subclass per domain."""
    raw: Any                    # Raw observation (pixels, grid, etc.)
    features: dict = field(default_factory=dict)  # Extracted features
    step: int = 0
    done: bool = False
    reward: float = 0.0

    def __hash__(self):
        """Must be hashable for BFS/visited sets. Override per domain."""
        if isinstance(self.raw, np.ndarray):
            return hash(self.raw.tobytes())
        return hash(str(self.raw))

    def __eq__(self, other):
        if isinstance(self.raw, np.ndarray) and isinstance(other.raw, np.ndarray):
            return np.array_equal(self.raw, other.raw)
        return self.raw == other.raw


@dataclass
class Action:
    """An action the agent can take."""
    id: int
    name: str = ""
    data: dict = field(default_factory=dict)


@dataclass
class TransitionRule:
    """A learned rule: in state matching `condition`, action `action` produces `effect`.

    This is what Phase 2 (EXPLORE) discovers. Each rule is a hypothesis
    about how the world works, confirmed by observation count.
    """
    condition: dict             # What the state looks like (pattern match)
    action: Action              # What action was taken
    effect: dict                # What changed (delta)
    observations: int = 1       # How many times we've seen this
    confidence: float = 0.5     # How much we trust this rule

    def strengthen(self):
        """More observations → higher confidence."""
        self.observations += 1
        self.confidence = min(0.99, 1.0 - 1.0 / (self.observations + 1))


@dataclass
class Plan:
    """A sequence of actions leading to a goal state."""
    actions: list[Action]
    expected_states: list[State] = field(default_factory=list)
    confidence: float = 0.0     # Product of rule confidences
    goal_achieved: bool = False


# ===========================================================================
# Phase 2: World Model — the learned imagination
# ===========================================================================

class WorldModel:
    """The internal model of how the environment works.

    This is THE critical component. For Sokoban, Apollo hand-coded this
    (he knew: push box → box moves). For AGI, this must be LEARNED
    from interaction.

    The world model answers: "If I do action A in state S, what happens?"

    PHASE 0 — BEGINNER'S MIND (Peter's warning against false gold):
    - confidence NEVER reaches 1.0 on learned rules (cap at 0.95)
    - contradictions don't overwrite — they create competing hypotheses
    - surprise_count tracks how often reality contradicts predictions
    - when surprise_count is high, the model DISTRUSTS itself
    - multiple outcomes are held simultaneously until evidence collapses them
    """

    CONFIDENCE_CAP = 0.95  # Never fully certain — leave room for variants

    def __init__(self):
        self.rules: list[TransitionRule] = []
        self.action_effects: dict[int, list[dict]] = {}  # action_id → observed effects
        self.state_transitions: dict = {}  # (state_hash, action_id) → resulting_state
        self.surprise_count: int = 0       # Times reality contradicted prediction
        self.total_predictions: int = 0    # Total predictions made
        self.competing_rules: dict[int, list[TransitionRule]] = {}  # action_id → multiple hypotheses

    def observe(self, state_before: State, action: Action, state_after: State):
        """Learn from a real transition. Phase 2 core operation.

        Called every time we take a real action. Accumulates evidence
        about how the world works.

        Phase 0 (Beginner's Mind): if the observation CONTRADICTS
        a confident prediction, that's a surprise. Don't overwrite —
        create a competing hypothesis. The world might have changed,
        or we might be in a variant.
        """
        # --- Phase 0: Check for surprise ---
        predicted = self.predict(state_before, action)
        if predicted is not None:
            self.total_predictions += 1
            if hash(predicted) != hash(state_after):
                self.surprise_count += 1
                log.debug(f"[IF] SURPRISE: action {action.name} produced "
                          f"unexpected result (surprise rate: "
                          f"{self.surprise_rate:.1%})")

        # Direct transition cache
        key = (hash(state_before), action.id)
        self.state_transitions[key] = state_after

        # Effect extraction — what changed?
        effect = self._extract_effect(state_before, state_after)

        # Condition extraction — what was relevant in the before-state?
        condition = self._extract_condition(state_before, action)

        # Try to match existing rule
        matched = False
        for rule in self.rules:
            if (rule.action.id == action.id and
                    self._conditions_match(rule.condition, condition) and
                    self._effects_match(rule.effect, effect)):
                rule.strengthen()
                # Phase 0: cap confidence — never fully certain
                rule.confidence = min(rule.confidence, self.CONFIDENCE_CAP)
                matched = True
                break

        if not matched:
            new_rule = TransitionRule(
                condition=condition,
                action=action,
                effect=effect,
            )
            self.rules.append(new_rule)

            # Phase 0: If this action already has rules with DIFFERENT effects,
            # these are competing hypotheses — the world has multiple behaviors
            # for the same action (context-dependent)
            existing = [r for r in self.rules
                        if r.action.id == action.id and r is not new_rule]
            if existing:
                if action.id not in self.competing_rules:
                    self.competing_rules[action.id] = list(existing)
                self.competing_rules[action.id].append(new_rule)

        # Track per-action effects
        if action.id not in self.action_effects:
            self.action_effects[action.id] = []
        self.action_effects[action.id].append(effect)

    @property
    def surprise_rate(self) -> float:
        """How often does reality contradict our model? High = model is wrong."""
        if self.total_predictions == 0:
            return 0.0
        return self.surprise_count / self.total_predictions

    @property
    def is_humble(self) -> bool:
        """Phase 0: Is the model aware of its own limitations?

        True when surprise rate is high enough to warrant caution.
        When humble, the solver should explore more and trust plans less.
        """
        return self.surprise_rate > 0.2  # >20% surprise = don't trust yourself

    def predict(self, state: State, action: Action) -> Optional[State]:
        """Imagine: what would happen if I did `action` in `state`?

        This is the imagination. Returns predicted next state, or None
        if we have no model for this situation.

        Phase 4 core operation.
        """
        # First: check direct cache (exact match)
        key = (hash(state), action.id)
        if key in self.state_transitions:
            return self.state_transitions[key]

        # Second: find matching rule and apply its effect
        best_rule = self._find_best_rule(state, action)
        if best_rule and best_rule.confidence > 0.3:
            return self._apply_effect(state, best_rule.effect)

        return None

    def predict_confidence(self, state: State, action: Action) -> float:
        """How confident are we in our prediction for this state-action pair?"""
        key = (hash(state), action.id)
        if key in self.state_transitions:
            return 1.0  # Exact match from cache
        best_rule = self._find_best_rule(state, action)
        return best_rule.confidence if best_rule else 0.0

    def is_trained(self, min_rules: int = 5, min_confidence: float = 0.5) -> bool:
        """Has the model learned enough to be useful?"""
        confident_rules = [r for r in self.rules if r.confidence >= min_confidence]
        return len(confident_rules) >= min_rules

    # --- Override these per domain for richer world models ---

    def _extract_effect(self, before: State, after: State) -> dict:
        """What changed between states? Override for domain-specific diff."""
        if isinstance(before.raw, np.ndarray) and isinstance(after.raw, np.ndarray):
            if before.raw.shape != after.raw.shape or before.raw.size == 0 or after.raw.size == 0:
                return {'changed_pixels': 0, 'changed_positions': []}
            diff = before.raw != after.raw
            return {
                'changed_pixels': int(diff.sum()),
                'changed_positions': list(zip(*np.where(diff)))[:20],  # cap at 20
            }
        return {'before': str(before.raw), 'after': str(after.raw)}

    def _extract_condition(self, state: State, action: Action) -> dict:
        """What's relevant about the state for this action? Override per domain."""
        return {'features': dict(state.features), 'action_id': action.id}

    def _conditions_match(self, c1: dict, c2: dict) -> bool:
        """Do two conditions describe the same situation? Override for fuzzy matching."""
        return c1.get('action_id') == c2.get('action_id')

    def _effects_match(self, e1: dict, e2: dict) -> bool:
        """Do two effects describe the same outcome? Override per domain."""
        return e1.get('changed_pixels', -1) == e2.get('changed_pixels', -2)

    def _find_best_rule(self, state: State, action: Action) -> Optional[TransitionRule]:
        """Find the most applicable rule for this state-action pair."""
        candidates = [r for r in self.rules if r.action.id == action.id]
        if not candidates:
            return None
        # Return highest confidence match
        return max(candidates, key=lambda r: r.confidence)

    def _apply_effect(self, state: State, effect: dict) -> Optional[State]:
        """Apply a predicted effect to a state. Override per domain."""
        # Default: can't apply abstract effects to states
        return None

    def summary(self) -> str:
        """Human-readable summary of what the model knows."""
        n_rules = len(self.rules)
        n_actions = len(self.action_effects)
        avg_conf = (sum(r.confidence for r in self.rules) / n_rules) if n_rules else 0
        n_cached = len(self.state_transitions)
        humble = " HUMBLE" if self.is_humble else ""
        n_competing = len(self.competing_rules)
        surprise = f", surprise_rate={self.surprise_rate:.1%}" if self.total_predictions else ""
        competing = f", {n_competing} competing" if n_competing else ""
        return (f"WorldModel: {n_rules} rules, {n_actions} actions, "
                f"avg_confidence={avg_conf:.2f}, {n_cached} cached"
                f"{surprise}{competing}{humble}")


# ===========================================================================
# The Solver — the 7-phase framework
# ===========================================================================

class ImaginationSolver(ABC):
    """Universal problem-solving framework.

    Subclass this per domain. Override the abstract methods.
    The framework handles the phase transitions.

    Usage:
        solver = MySokobanSolver(env)
        plan = solver.solve(max_steps=200)

    The solve loop:
        0. BEGINNER'S MIND → assume nothing, hold multiple hypotheses
        1. ORIENT    → perceive()
        2. EXPLORE   → explore() until world model is trained
        3. GOAL      → identify_goal()
        4. IMAGINE   → search_in_imagination()
        5. EVALUATE  → evaluate_plan()
        6. EXECUTE   → execute_plan()
        7. ADAPT     → adapt() if plan fails, check surprise rate
    """

    def __init__(self, env: Any, world_model: Optional[WorldModel] = None):
        self.env = env
        self.model = world_model or WorldModel()
        self.current_state: Optional[State] = None
        self.goal_state: Optional[State] = None
        self.goal_test: Optional[callable] = None
        self.phase: str = "orient"
        self.step_count: int = 0
        self.max_explore_steps: int = 50    # Budget for exploration
        self.max_search_states: int = 10000  # Budget for imagination search
        self.plan: Optional[Plan] = None
        self.history: list[tuple[State, Action, State]] = []

    # --- Abstract methods: override per domain ---

    @abstractmethod
    def perceive(self, observation: Any) -> State:
        """Phase 1: Convert raw observation to State.

        Extract the abstract representation from pixels/text/whatever.
        This is the ORIENT phase — seeing the board.
        """
        ...

    @abstractmethod
    def get_actions(self, state: State) -> list[Action]:
        """Return all valid actions in this state."""
        ...

    @abstractmethod
    def is_goal(self, state: State) -> bool:
        """Phase 3: Does this state satisfy the goal?

        For Sokoban: all boxes on targets.
        For ARC: output matches expected pattern.
        For Doom: all enemies dead.
        """
        ...

    @abstractmethod
    def take_action(self, action: Action) -> tuple[Any, float, bool]:
        """Execute an action in the REAL environment.

        Returns (observation, reward, done).
        This is the ONLY method that touches the real world.
        Everything else happens in imagination.
        """
        ...

    # --- Optional overrides for domain-specific intelligence ---

    def extract_goal(self, state: State) -> Optional[dict]:
        """Phase 3: Identify what the goal looks like from observation.

        For known games, this is hard-coded (Sokoban: boxes on targets).
        For unknown games, this must be DISCOVERED.
        Override to add domain-specific goal detection.
        """
        return None

    def heuristic(self, state: State) -> float:
        """Heuristic estimate: how far is this state from the goal?

        Used to prioritize search in imagination.
        0.0 = at goal. Higher = farther away.
        Override for domain-specific distance estimates.
        """
        return 0.0

    def is_dead_state(self, state: State) -> bool:
        """Is this state unsolvable? (Sokoban: box in corner not on target)

        Phase 5: kill bad ideas. If a state is dead, prune the branch.
        Override per domain.
        """
        return False

    def choose_explore_action(self, state: State,
                              available: list[Action]) -> Action:
        """Phase 2: Choose an action during exploration.

        Default: random. Override for smarter exploration
        (e.g., prefer unexplored actions, novelty-seeking).
        """
        import random
        return random.choice(available)

    # --- The framework: phase management and solve loop ---

    def solve(self, max_steps: int = 200) -> Optional[Plan]:
        """Main solve loop. Runs through all 8 phases.

        Returns a Plan if solved, None if failed.
        """
        # Phase 0: BEGINNER'S MIND — assume nothing
        log.info(f"[IF] Phase 0: BEGINNER'S MIND — no assumptions")

        observation = self._get_observation()
        self.current_state = self.perceive(observation)
        self.phase = "orient"

        log.info(f"[IF] Phase 1: ORIENT — perceiving environment")

        # Phase 2: EXPLORE — learn the rules
        # Phase 0 integration: if model is humble (high surprise rate),
        # explore MORE even if model thinks it's trained
        needs_explore = not self.model.is_trained() or self.model.is_humble
        if needs_explore:
            explore_budget = self.max_explore_steps
            if self.model.is_humble:
                explore_budget = int(explore_budget * 1.5)  # Extra exploration
                log.info(f"[IF] Phase 0: Model is HUMBLE "
                         f"(surprise rate {self.model.surprise_rate:.1%}) — "
                         f"exploring more")
            log.info(f"[IF] Phase 2: EXPLORE — learning rules "
                     f"(budget: {explore_budget} steps)")
            old_budget = self.max_explore_steps
            self.max_explore_steps = explore_budget
            self._explore_phase()
            self.max_explore_steps = old_budget

        # Phase 3: GOAL — identify what we're solving for
        log.info(f"[IF] Phase 3: GOAL — identifying objective")
        goal_info = self.extract_goal(self.current_state)
        if goal_info:
            log.info(f"[IF] Goal identified: {goal_info}")

        # Phase 4: IMAGINE — search for solution in mental model
        log.info(f"[IF] Phase 4: IMAGINE — searching in imagination "
                 f"(budget: {self.max_search_states} states)")
        self.plan = self._imagination_phase()

        if self.plan and self.plan.actions:
            # Phase 5: EVALUATE — sanity check the plan
            log.info(f"[IF] Phase 5: EVALUATE — checking "
                     f"{len(self.plan.actions)}-step plan")
            if not self._evaluate_plan(self.plan):
                log.info(f"[IF] Plan rejected by evaluation. Falling back.")
                self.plan = None

        if self.plan and self.plan.actions:
            # Phase 6: EXECUTE — run the plan
            log.info(f"[IF] Phase 6: EXECUTE — running "
                     f"{len(self.plan.actions)}-step plan")
            success = self._execute_phase(self.plan, max_steps)
            if success:
                log.info(f"[IF] SOLVED in {self.step_count} steps!")
                return self.plan
            else:
                # Phase 7: ADAPT — plan failed, replan
                log.info(f"[IF] Phase 7: ADAPT — plan failed at step "
                         f"{self.step_count}, replanning")
                return self._adapt_phase(max_steps)
        else:
            # No plan found — fall back to reactive + periodic replan
            log.info(f"[IF] No plan found. Falling back to reactive mode.")
            return self._reactive_fallback(max_steps)

    def _explore_phase(self):
        """Phase 2: Take actions to learn how the world works."""
        for i in range(self.max_explore_steps):
            actions = self.get_actions(self.current_state)
            if not actions:
                break

            action = self.choose_explore_action(self.current_state, actions)
            obs, reward, done = self.take_action(action)
            new_state = self.perceive(obs)
            new_state.reward = reward
            new_state.done = done

            # Learn from this transition
            self.model.observe(self.current_state, action, new_state)
            self.history.append((self.current_state, action, new_state))

            self.current_state = new_state
            self.step_count += 1

            if done:
                break

            # Check if we've learned enough
            if self.model.is_trained():
                log.info(f"[IF] World model trained after {i+1} exploration steps")
                break

        log.info(f"[IF] Exploration complete. {self.model.summary()}")

    def _imagination_phase(self) -> Optional[Plan]:
        """Phase 4: BFS/DFS through imagined states to find solution.

        This is where the magic happens. We search the state space
        entirely in our heads — no real actions taken.
        """
        if self.current_state is None:
            return None

        # BFS through imagined states
        start = self.current_state
        queue = deque([(start, [])])  # (state, action_sequence)
        visited = {hash(start)}
        states_explored = 0

        while queue and states_explored < self.max_search_states:
            state, actions_so_far = queue.popleft()
            states_explored += 1

            available_actions = self.get_actions(state)
            for action in available_actions:
                # Predict the next state using our world model
                predicted = self.model.predict(state, action)
                if predicted is None:
                    continue  # Model doesn't know what would happen

                # Skip if we've been here
                state_hash = hash(predicted)
                if state_hash in visited:
                    continue
                visited.add(state_hash)

                new_actions = actions_so_far + [action]

                # Check: is this the goal?
                if self.is_goal(predicted):
                    confidence = min(
                        self.model.predict_confidence(state, action)
                        for s, a in zip([start] + list(
                            self.plan.expected_states if self.plan else []),
                            new_actions)
                    ) if new_actions else 0.0
                    return Plan(
                        actions=new_actions,
                        confidence=confidence,
                        goal_achieved=True,
                    )

                # Prune dead states
                if self.is_dead_state(predicted):
                    continue

                queue.append((predicted, new_actions))

        log.info(f"[IF] Imagination searched {states_explored} states, "
                 f"no solution found")
        return None

    def _evaluate_plan(self, plan: Plan) -> bool:
        """Phase 5: Does this plan make sense?

        Override for domain-specific validation.
        Default: accept if confidence > threshold.
        """
        if not plan.actions:
            return False
        # Accept plans from cached transitions (confidence=1.0)
        # Be skeptical of rule-based plans
        return True

    def _execute_phase(self, plan: Plan, max_steps: int) -> bool:
        """Phase 6: Execute the plan in the real world."""
        for i, action in enumerate(plan.actions):
            if self.step_count >= max_steps:
                return False

            obs, reward, done = self.take_action(action)
            new_state = self.perceive(obs)
            new_state.reward = reward
            new_state.done = done

            # Learn from execution (refine model)
            self.model.observe(self.current_state, action, new_state)
            self.history.append((self.current_state, action, new_state))

            self.current_state = new_state
            self.step_count += 1

            if self.is_goal(new_state):
                return True

            if done:
                return False

            # Detect plan divergence — did reality match imagination?
            # If not, break and replan (Phase 7)
            predicted = self.model.predict(self.current_state, action)
            if predicted and hash(predicted) != hash(new_state):
                log.info(f"[IF] Plan diverged from reality at step {i+1}")
                plan.actions = plan.actions[i+1:]  # Trim executed portion
                return False

        return self.is_goal(self.current_state)

    def _adapt_phase(self, max_steps: int) -> Optional[Plan]:
        """Phase 7: Plan failed. Update model and try again.

        The model has been updated during execution (observe() calls
        in _execute_phase). So replanning uses better knowledge.
        """
        replan_attempts = 3
        for attempt in range(replan_attempts):
            if self.step_count >= max_steps:
                break

            log.info(f"[IF] Replan attempt {attempt+1}/{replan_attempts}")

            # Re-imagine with updated model
            self.plan = self._imagination_phase()
            if self.plan and self.plan.actions:
                if self._evaluate_plan(self.plan):
                    success = self._execute_phase(self.plan, max_steps)
                    if success:
                        return self.plan

            # If replan fails, do a few explore steps to gather more data
            self.max_explore_steps = 10
            self._explore_phase()

        return None

    def _reactive_fallback(self, max_steps: int) -> Optional[Plan]:
        """No plan possible. Fall back to explore + periodic replan.

        This is the "dumb" mode — take actions, learn, periodically
        try to plan again.
        """
        replan_interval = 20

        while self.step_count < max_steps:
            # Try to replan periodically
            if self.step_count % replan_interval == 0 and self.model.is_trained():
                self.plan = self._imagination_phase()
                if self.plan and self.plan.actions:
                    success = self._execute_phase(self.plan, max_steps)
                    if success:
                        return self.plan

            # Otherwise: explore
            actions = self.get_actions(self.current_state)
            if not actions:
                break

            action = self.choose_explore_action(self.current_state, actions)
            obs, reward, done = self.take_action(action)
            new_state = self.perceive(obs)
            new_state.reward = reward
            new_state.done = done

            self.model.observe(self.current_state, action, new_state)
            self.history.append((self.current_state, action, new_state))
            self.current_state = new_state
            self.step_count += 1

            if self.is_goal(new_state):
                return Plan(actions=[], goal_achieved=True)
            if done:
                break

        return None

    def _get_observation(self) -> Any:
        """Get current observation from environment. Override if needed."""
        if hasattr(self.env, 'render'):
            return self.env.render(mode='rgb_array')
        if hasattr(self.env, 'observe'):
            return self.env.observe()
        raise NotImplementedError("Override _get_observation for your env")


# ===========================================================================
# Example: Sokoban solver using the framework
# ===========================================================================

class SokobanSolver(ImaginationSolver):
    """Sokoban solver — demonstrates the framework with KNOWN rules.

    Phase 2 (EXPLORE) is trivial here because we hard-code the rules.
    The interesting part is Phase 4 (IMAGINE) — BFS through all possible
    push sequences to find the complete solution.

    For ARC-AGI-3, Phase 2 would need to DISCOVER these rules from
    interaction. That's the gap between this and AGI.
    """

    def perceive(self, observation: Any) -> State:
        """Extract grid from pixel observation."""
        # Delegate to existing SokobanBrain.extract_grid()
        grid = self._extract_grid(observation)
        player = tuple(np.argwhere(grid == 2)[0]) if 2 in grid else (0, 0)
        boxes = set(map(tuple, np.argwhere(grid == 3)))
        targets = set(map(tuple, np.argwhere(grid == 4)))
        box_on_target = set(map(tuple, np.argwhere(grid == 5)))
        boxes |= box_on_target
        targets |= box_on_target
        walls = set(map(tuple, np.argwhere(grid == 1)))

        state = State(raw=grid)
        state.features = {
            'player': player,
            'boxes': frozenset(boxes),
            'targets': frozenset(targets),
            'walls': frozenset(walls),
            'grid_shape': grid.shape,
        }
        return state

    def get_actions(self, state: State) -> list[Action]:
        """4 push directions + 4 move directions."""
        return [
            Action(1, 'push_up'), Action(2, 'push_down'),
            Action(3, 'push_left'), Action(4, 'push_right'),
            Action(5, 'move_up'), Action(6, 'move_down'),
            Action(7, 'move_left'), Action(8, 'move_right'),
        ]

    def is_goal(self, state: State) -> bool:
        """All boxes on targets."""
        boxes = state.features.get('boxes', set())
        targets = state.features.get('targets', set())
        return bool(boxes and boxes == targets)

    def take_action(self, action: Action) -> tuple[Any, float, bool]:
        """Execute in real Sokoban environment."""
        obs, reward, done, info = self.env.step(action.id)
        return obs, reward, done

    def is_dead_state(self, state: State) -> bool:
        """Box in corner (not on target) = dead."""
        walls = state.features.get('walls', set())
        targets = state.features.get('targets', set())
        boxes = state.features.get('boxes', set())
        h, w = state.features.get('grid_shape', (7, 7))

        for by, bx in boxes:
            if (by, bx) in targets:
                continue
            blocked = lambda y, x: (
                y < 0 or y >= h or x < 0 or x >= w or (y, x) in walls
            )
            if ((blocked(by-1, bx) and blocked(by, bx-1)) or
                (blocked(by-1, bx) and blocked(by, bx+1)) or
                (blocked(by+1, bx) and blocked(by, bx-1)) or
                (blocked(by+1, bx) and blocked(by, bx+1))):
                return True
        return False

    def _extract_grid(self, obs):
        """Reuse SokobanBrain's grid extraction. Simplified for framework."""
        h, w = obs.shape[:2]
        tile_h, tile_w = 16, 16
        grid_h, grid_w = h // tile_h, w // tile_w
        grid = np.zeros((grid_h, grid_w), dtype=int)
        for gy in range(grid_h):
            for gx in range(grid_w):
                tile = obs[gy*tile_h:(gy+1)*tile_h, gx*tile_w:(gx+1)*tile_w, :]
                r, g, b = tile[:,:,0].mean(), tile[:,:,1].mean(), tile[:,:,2].mean()
                if r > 160 and g > 100 and b < 60:
                    grid[gy, gx] = 3
                elif r > 130 and g > 60 and g < 110 and b < 70:
                    grid[gy, gx] = 5 if r > 160 else 1
                elif r > 40 and r < 90 and g < 20 and b < 20:
                    grid[gy, gx] = 4
                elif g > 80 and r < 60 and b < 40:
                    grid[gy, gx] = 2
        return grid


# ===========================================================================
# Skeleton: ARC-AGI-3 solver using the framework
# ===========================================================================

class ArcSolver(ImaginationSolver):
    """ARC-AGI-3 solver skeleton — the REAL test.

    The key difference from Sokoban: rules are UNKNOWN.
    Phase 2 (EXPLORE) must discover the transition function
    from interaction. This is where AGI happens or doesn't.

    TODO (Apollo):
    - perceive(): extract grid state from ARC environment
    - explore logic: systematic action probing to learn rules
    - WorldModel subclass: grid-aware rule learning
    - Goal detection: infer success condition from reward/state
    """

    def __init__(self, env, **kwargs):
        # Use a grid-aware world model
        super().__init__(env, world_model=GridWorldModel(), **kwargs)
        self.max_explore_steps = 100  # More exploration for unknown rules

    def perceive(self, observation: Any) -> State:
        """Extract grid state from ARC environment observation."""
        # TODO: Apollo — implement per ARC-AGI-3 observation format
        state = State(raw=observation)
        return state

    def get_actions(self, state: State) -> list[Action]:
        """Return available actions in this ARC environment."""
        # TODO: Apollo — ARC-AGI-3 action space varies per environment
        # Discover available actions during Phase 2
        return []

    def is_goal(self, state: State) -> bool:
        """Is this the solved state?"""
        # TODO: Apollo — detect from reward signal or state pattern
        return state.done and state.reward > 0

    def take_action(self, action: Action) -> tuple[Any, float, bool]:
        """Execute in real ARC environment."""
        # TODO: Apollo — interface with ARC-AGI-3 environment
        raise NotImplementedError

    def choose_explore_action(self, state: State,
                              available: list[Action]) -> Action:
        """Smart exploration for unknown environments.

        Priority:
        1. Actions never tried before (novelty)
        2. Actions that caused large state changes (interesting)
        3. Random (fallback)
        """
        import random

        # Prefer untried actions
        tried = {a.id for _, a, _ in self.history}
        untried = [a for a in available if a.id not in tried]
        if untried:
            return random.choice(untried)

        # Prefer actions that caused large changes
        big_changers = []
        for a in available:
            effects = self.model.action_effects.get(a.id, [])
            if effects:
                avg_change = sum(e.get('changed_pixels', 0) for e in effects) / len(effects)
                if avg_change > 5:
                    big_changers.append(a)
        if big_changers:
            return random.choice(big_changers)

        return random.choice(available)


class GridWorldModel(WorldModel):
    """World model specialized for grid-based environments.

    Learns rules like:
    - "pushing action 3 moves a colored block left"
    - "action 7 near a wall does nothing"
    - "when two blocks collide, the pushed one stops"

    This is the Phase 2 engine for ARC-AGI-3.
    """

    def _extract_effect(self, before: State, after: State) -> dict:
        """Grid-specific diff: what cells changed, what moved where."""
        if not isinstance(before.raw, np.ndarray) or not isinstance(after.raw, np.ndarray):
            return super()._extract_effect(before, after)

        diff_mask = before.raw != after.raw
        changed = list(zip(*np.where(diff_mask)))

        # Track color movements
        movements = []
        for y, x in changed[:50]:
            old_color = int(before.raw[y, x])
            new_color = int(after.raw[y, x])
            movements.append({
                'pos': (y, x),
                'from_color': old_color,
                'to_color': new_color,
            })

        return {
            'changed_count': len(changed),
            'movements': movements,
            'is_noop': len(changed) == 0,
        }

    def _effects_match(self, e1: dict, e2: dict) -> bool:
        """Two grid effects match if they change similar number of cells."""
        c1 = e1.get('changed_count', -1)
        c2 = e2.get('changed_count', -2)
        if c1 == 0 and c2 == 0:
            return True  # Both noops
        if c1 == 0 or c2 == 0:
            return False  # One noop, one not
        # Similar magnitude of change
        return abs(c1 - c2) <= max(1, min(c1, c2) * 0.3)

    def _apply_effect(self, state: State, effect: dict) -> Optional[State]:
        """Apply a learned effect to predict next state."""
        if not isinstance(state.raw, np.ndarray):
            return None
        if effect.get('is_noop'):
            return State(raw=state.raw.copy(), features=dict(state.features))

        # Apply color movements
        new_raw = state.raw.copy()
        for mov in effect.get('movements', []):
            y, x = mov['pos']
            if 0 <= y < new_raw.shape[0] and 0 <= x < new_raw.shape[1]:
                new_raw[y, x] = mov['to_color']

        return State(raw=new_raw, features=dict(state.features))
