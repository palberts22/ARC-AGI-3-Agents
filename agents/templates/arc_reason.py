"""ARC Reason — Inductive reasoning from visual observation.

Watch. Hypothesize. Test. Codify. Solve.

This module sits between Eyes (perception) and the BFS solver (search).
Instead of brute-force exploring all states, it:
1. Observes a few transitions (Eyes records them)
2. Induces rules ("clicking cell X toggles its neighbors")
3. Tests rules with targeted experiments
4. Once confident, solves analytically — no BFS needed

The goal: solve games by *understanding* them, not by exhaustive search.
This is critical for ARC-AGI-3 if they restrict API access to visual-only.
"""

import numpy as np
import copy
import logging
from collections import defaultdict, Counter
from dataclasses import dataclass, field
from typing import Optional, Callable

log = logging.getLogger("arc_reason")


# ---------------------------------------------------------------------------
# 1. HYPOTHESIS — A testable claim about game mechanics
# ---------------------------------------------------------------------------

@dataclass
class Hypothesis:
    """A hypothesis about how the game works.

    Examples:
    - "Clicking cell (r,c) toggles it and its 4 neighbors" (Lights Out)
    - "Action 0 moves the player up by 1 cell" (Navigation)
    - "Green cells are goals — filling all of them wins" (Painting)
    - "Each click cycles cell color: 0→1→2→0" (Color cycling)
    """
    rule_type: str          # 'toggle', 'move', 'cycle', 'fill', 'swap', 'gravity', 'custom'
    description: str        # Human-readable
    parameters: dict        # Rule-specific params (e.g., neighborhood pattern, cycle sequence)
    confidence: float = 0.0 # 0-1, increases with confirming evidence
    tests_passed: int = 0
    tests_failed: int = 0

    @property
    def tested(self) -> bool:
        return self.tests_passed + self.tests_failed > 0

    def confirm(self):
        self.tests_passed += 1
        self.confidence = self.tests_passed / (self.tests_passed + self.tests_failed)

    def reject(self):
        self.tests_failed += 1
        self.confidence = self.tests_passed / (self.tests_passed + self.tests_failed)


# ---------------------------------------------------------------------------
# 2. OBSERVATION — Structured representation of what changed
# ---------------------------------------------------------------------------

@dataclass
class CellChange:
    """A single cell that changed between two frames."""
    row: int
    col: int
    old_color: int
    new_color: int


@dataclass
class Observation:
    """What happened when we took an action."""
    action_id: int
    action_data: dict
    changes: list  # list of CellChange
    click_pos: tuple = None  # (row, col) if click-based
    won: bool = False
    frame_before: np.ndarray = None
    frame_after: np.ndarray = None


# ---------------------------------------------------------------------------
# 3. INDUCTIVE REASONER — The core reasoning engine
# ---------------------------------------------------------------------------

class InductiveReasoner:
    """Induces game rules from visual observations.

    Usage:
        reasoner = InductiveReasoner()

        # Feed it observations from Eyes
        reasoner.observe(action, frame_before, frame_after)

        # Ask it to hypothesize
        hypotheses = reasoner.hypothesize()

        # Test hypotheses with targeted experiments
        experiments = reasoner.design_experiments(hypotheses)

        # Once confident, solve analytically
        solution = reasoner.solve_analytically(current_frame, goal_frame)
    """

    def __init__(self):
        self.observations: list[Observation] = []
        self.hypotheses: list[Hypothesis] = []
        self.confirmed_rules: list[Hypothesis] = []  # confidence > 0.8
        self.grid_size: tuple = None
        self.color_palette: set = set()
        self.win_observations: list[Observation] = []

    # ── Observation ──

    def detect_grid(self, frame: np.ndarray):
        """Detect grid structure from frame — map pixels to game cells.

        Returns (cell_size, grid_origin, grid_dims) or None.
        """
        f = frame if frame.ndim == 2 else frame[0]
        h, w = f.shape

        # Look for regular grid lines (rows/cols where all pixels are same color)
        # Grid lines are typically the darkest color (0 = black)
        row_is_line = np.array([len(np.unique(f[r, :])) <= 2 and np.min(f[r, :]) == 0
                                for r in range(h)])
        col_is_line = np.array([len(np.unique(f[:, c])) <= 2 and np.min(f[:, c]) == 0
                                for c in range(w)])

        # Find grid line positions
        line_rows = np.where(row_is_line)[0]
        line_cols = np.where(col_is_line)[0]

        if len(line_rows) < 2 or len(line_cols) < 2:
            return None

        # Compute cell size from gaps between grid lines
        row_gaps = np.diff(line_rows)
        col_gaps = np.diff(line_cols)

        # Filter out gaps of 1 (consecutive line pixels = thick lines)
        row_gaps = row_gaps[row_gaps > 1]
        col_gaps = col_gaps[col_gaps > 1]

        if len(row_gaps) == 0 or len(col_gaps) == 0:
            return None

        cell_h = int(np.median(row_gaps))
        cell_w = int(np.median(col_gaps))

        if cell_h < 3 or cell_w < 3:
            return None

        # Grid origin: first non-line row/col after first line
        origin_r = int(line_rows[0]) + 1
        origin_c = int(line_cols[0]) + 1

        # Grid dimensions
        n_rows = len(row_gaps)
        n_cols = len(col_gaps)

        self._grid = {
            'cell_h': cell_h, 'cell_w': cell_w,
            'origin_r': origin_r, 'origin_c': origin_c,
            'n_rows': n_rows, 'n_cols': n_cols
        }
        self.grid_size = (n_rows, n_cols)
        log.info(f"  Grid detected: {n_rows}x{n_cols}, cell_size={cell_h}x{cell_w}")
        return self._grid

    def _pixel_to_grid(self, pr, pc) -> tuple:
        """Convert pixel coordinates to grid cell coordinates."""
        if not hasattr(self, '_grid') or self._grid is None:
            return (pr, pc)
        g = self._grid
        gr = (pr - g['origin_r']) // g['cell_h']
        gc = (pc - g['origin_c']) // g['cell_w']
        # Clamp to valid range
        gr = max(0, min(int(gr), g['n_rows'] - 1))
        gc = max(0, min(int(gc), g['n_cols'] - 1))
        return (gr, gc)

    def _cell_color(self, frame, gr, gc) -> int:
        """Get the dominant color of a grid cell."""
        if not hasattr(self, '_grid') or self._grid is None:
            return int(frame[gr, gc])
        g = self._grid
        r0 = g['origin_r'] + gr * g['cell_h']
        c0 = g['origin_c'] + gc * g['cell_w']
        cell = frame[r0:r0+g['cell_h'], c0:c0+g['cell_w']]
        if cell.size == 0:
            return 0
        # Most common non-zero value (skip grid lines)
        vals = cell[cell > 0]
        if len(vals) == 0:
            return 0
        counts = np.bincount(vals)
        return int(np.argmax(counts))

    def observe(self, action_id: int, action_data: dict,
                frame_before: np.ndarray, frame_after: np.ndarray,
                won: bool = False):
        """Record an observation — what changed when we acted."""
        fb = frame_before if frame_before.ndim == 2 else frame_before[0]
        fa = frame_after if frame_after.ndim == 2 else frame_after[0]

        # Detect grid on first observation
        if self.grid_size is None:
            detected = self.detect_grid(fb)
            if not detected:
                self.grid_size = fb.shape  # Fallback to pixel-level

        # Find what changed at the GRID level (not pixel level)
        changes = []
        if hasattr(self, '_grid') and self._grid is not None:
            g = self._grid
            for gr in range(g['n_rows']):
                for gc in range(g['n_cols']):
                    old_c = self._cell_color(fb, gr, gc)
                    new_c = self._cell_color(fa, gr, gc)
                    if old_c != new_c:
                        changes.append(CellChange(gr, gc, old_c, new_c))
                        self.color_palette.add(old_c)
                        self.color_palette.add(new_c)
        else:
            # Fallback: pixel-level diff
            diff = fb != fa
            for r, c in zip(*np.where(diff)):
                changes.append(CellChange(int(r), int(c), int(fb[r, c]), int(fa[r, c])))
                self.color_palette.add(int(fb[r, c]))
                self.color_palette.add(int(fa[r, c]))

        click_pos = None
        if action_data and 'x' in action_data:
            px_r, px_c = action_data['y'], action_data['x']
            click_pos = self._pixel_to_grid(px_r, px_c)  # Convert to grid coords

        obs = Observation(
            action_id=action_id, action_data=action_data,
            changes=changes, click_pos=click_pos, won=won,
            frame_before=fb.copy(), frame_after=fa.copy()
        )
        self.observations.append(obs)
        if won:
            self.win_observations.append(obs)

        return obs

    # ── Hypothesis generation ──

    def hypothesize(self) -> list[Hypothesis]:
        """Generate hypotheses from accumulated observations."""
        hypotheses = []

        if not self.observations:
            return hypotheses

        # Separate click-based vs directional observations
        click_obs = [o for o in self.observations if o.click_pos is not None]
        dir_obs = [o for o in self.observations if o.click_pos is None]

        if click_obs:
            hypotheses.extend(self._hypothesize_click_mechanics(click_obs))
        if dir_obs:
            hypotheses.extend(self._hypothesize_movement_mechanics(dir_obs))

        # Cross-cutting: counter/selector detection (small, consistent changes)
        small_change_obs = [o for o in self.observations if 0 < len(o.changes) <= 5]
        if small_change_obs:
            hypotheses.extend(self._hypothesize_counter_mechanics(small_change_obs))

        # Commit button detection (large state changes)
        hypotheses.extend(self._hypothesize_commit_button())

        # Win condition hypotheses
        if self.win_observations:
            hypotheses.extend(self._hypothesize_win_condition())

        self.hypotheses = hypotheses

        # Auto-promote high-confidence hypotheses to confirmed rules
        for h in hypotheses:
            if h.confidence >= 0.7 and h not in self.confirmed_rules:
                self.confirmed_rules.append(h)
                log.info(f"  Rule auto-confirmed: {h.description} (conf={h.confidence:.2f})")

        return hypotheses

    def _hypothesize_click_mechanics(self, observations: list[Observation]) -> list[Hypothesis]:
        """Induce click-based game mechanics."""
        hypotheses = []

        # Pattern 1: Toggle neighborhood
        # For each click, check if changes form a consistent pattern around click point
        # Strategy: use the LARGEST observed pattern as the full neighborhood
        # (interior cells show the full pattern; edge/corner cells show truncated versions)
        patterns_by_click = defaultdict(list)
        all_relative = []
        for obs in observations:
            if not obs.changes or obs.click_pos is None:
                continue
            cr, cc = obs.click_pos
            # Compute relative positions of changes
            relative = tuple(sorted((ch.row - cr, ch.col - cc) for ch in obs.changes))
            patterns_by_click[relative].append(obs)
            all_relative.append(set(relative))

        # Use the LARGEST pattern (interior cells) as the canonical neighborhood
        # Edge/corner observations are truncated subsets of this
        if patterns_by_click:
            # Pick the largest pattern (most cells changed = interior click)
            largest = max(patterns_by_click.items(), key=lambda x: len(x[0]))
            pattern, supporting_exact = largest

            # Count all observations whose pattern is a SUBSET of the largest
            supporting = []
            pattern_set = set(pattern)
            for obs in observations:
                if not obs.changes or obs.click_pos is None:
                    continue
                cr, cc = obs.click_pos
                rel = set((ch.row - cr, ch.col - cc) for ch in obs.changes)
                if rel.issubset(pattern_set):
                    supporting.append(obs)

            if len(supporting) >= 2:  # Need at least 2 observations
                # Check if it's a standard neighborhood
                if set(pattern) == {(0, 0), (-1, 0), (1, 0), (0, -1), (0, 1)}:
                    desc = "Plus-shaped toggle (Lights Out pattern)"
                elif set(pattern) == {(0, 0)}:
                    desc = "Single-cell toggle"
                else:
                    # Truncate large patterns to avoid context pollution in pilot prompts
                    if len(pattern) <= 10:
                        desc = f"Custom neighborhood toggle: {pattern}"
                    else:
                        desc = f"Custom neighborhood toggle ({len(pattern)} cells, radius ~{max(abs(r) for r, c in pattern)})"

                # Check if it's a binary toggle (2 colors) or color cycle
                color_transitions = defaultdict(set)
                for obs in supporting:
                    for ch in obs.changes:
                        color_transitions[ch.old_color].add(ch.new_color)

                is_binary = all(len(v) == 1 for v in color_transitions.values())
                n_colors = len(color_transitions)

                if is_binary and n_colors == 2:
                    rule_type = 'toggle'
                    params = {'neighborhood': list(pattern), 'colors': list(color_transitions.keys())}
                else:
                    rule_type = 'cycle'
                    # Infer cycle order
                    cycle = []
                    if color_transitions:
                        start = min(color_transitions.keys())
                        current = start
                        visited = set()
                        while current not in visited:
                            visited.add(current)
                            cycle.append(current)
                            nexts = color_transitions.get(current, set())
                            if nexts:
                                current = min(nexts)  # deterministic choice
                            else:
                                break
                    params = {'neighborhood': list(pattern), 'cycle': cycle}

                hypotheses.append(Hypothesis(
                    rule_type=rule_type,
                    description=desc,
                    parameters=params,
                    confidence=len(supporting) / len(observations),
                    tests_passed=len(supporting)
                ))

        # Pattern 2: Fill/paint — click changes only the clicked cell to a target color
        single_cell_obs = [o for o in observations
                          if len(o.changes) == 1 and o.click_pos
                          and o.changes[0].row == o.click_pos[0]
                          and o.changes[0].col == o.click_pos[1]]
        if len(single_cell_obs) >= 2:
            target_colors = set(o.changes[0].new_color for o in single_cell_obs)
            if len(target_colors) == 1:
                hypotheses.append(Hypothesis(
                    rule_type='fill',
                    description=f"Click fills cell with color {target_colors.pop()}",
                    parameters={'target_color': list(target_colors)[0] if target_colors else None},
                    confidence=len(single_cell_obs) / len(observations),
                    tests_passed=len(single_cell_obs)
                ))

        return hypotheses

    def _hypothesize_movement_mechanics(self, observations: list[Observation]) -> list[Hypothesis]:
        """Induce movement-based game mechanics."""
        hypotheses = []

        # Group observations by action_id
        by_action = defaultdict(list)
        for obs in observations:
            by_action[obs.action_id].append(obs)

        for action_id, obs_list in by_action.items():
            # Check if this action consistently moves a specific color
            movements = []
            for obs in obs_list:
                if len(obs.changes) >= 2:
                    # Look for a color that disappeared from one spot and appeared in another
                    disappeared = {(ch.row, ch.col, ch.old_color) for ch in obs.changes}
                    appeared = {(ch.row, ch.col, ch.new_color) for ch in obs.changes}

                    for dr, dc, old_c in disappeared:
                        for ar, ac, new_c in appeared:
                            if old_c == new_c and old_c != 0:  # Same non-background color
                                movements.append((ar - dr, ac - dc, old_c))

            if movements:
                # Most common movement vector
                vec_counts = Counter(movements)
                (dy, dx, color), count = vec_counts.most_common(1)[0]
                if count >= 2:
                    hypotheses.append(Hypothesis(
                        rule_type='move',
                        description=f"Action {action_id} moves color {color} by ({dy},{dx})",
                        parameters={'action_id': action_id, 'dy': dy, 'dx': dx, 'color': color},
                        confidence=count / len(obs_list),
                        tests_passed=count
                    ))

        return hypotheses

    def _hypothesize_counter_mechanics(self, observations: list[Observation]) -> list[Hypothesis]:
        """Detect counter/selector mechanics — actions that cycle small regions.

        Common in ARC games: buttons that increment a counter, selectors that
        cycle through colors/states. Key signal: same action always changes
        the same set of cells, and the color transitions form a cycle.
        """
        hypotheses = []

        # Group by action_id
        by_action = defaultdict(list)
        for obs in observations:
            by_action[obs.action_id].append(obs)

        for action_id, obs_list in by_action.items():
            if len(obs_list) < 2:
                continue

            # Check: do all observations change the same cell positions?
            cell_sets = []
            for obs in obs_list:
                cells = frozenset((ch.row, ch.col) for ch in obs.changes)
                cell_sets.append(cells)

            if not cell_sets:
                continue

            # If >70% share the same cell set, it's a counter/selector
            most_common_cells = max(set(cell_sets), key=cell_sets.count)
            match_count = cell_sets.count(most_common_cells)

            if match_count / len(obs_list) >= 0.7:
                # Track color transitions to detect cycling
                transitions = defaultdict(set)
                for obs in obs_list:
                    for ch in obs.changes:
                        transitions[ch.old_color].add(ch.new_color)

                # Is it a cycle? (each color maps to exactly one next color)
                is_cycle = all(len(v) == 1 for v in transitions.values())
                n_states = len(transitions)

                hypotheses.append(Hypothesis(
                    rule_type='counter',
                    description=f"Action {action_id} cycles {len(most_common_cells)} cells through {n_states} states",
                    parameters={
                        'action_id': action_id,
                        'cells': list(most_common_cells),
                        'transitions': {k: list(v) for k, v in transitions.items()},
                        'n_states': n_states,
                        'is_cycle': is_cycle
                    },
                    confidence=match_count / len(obs_list),
                    tests_passed=match_count
                ))

        return hypotheses

    def _hypothesize_commit_button(self) -> list[Hypothesis]:
        """Detect actions that trigger large state changes — 'submit/commit' buttons.

        Pattern: One or few actions cause >50 cell changes while others cause <10.
        These are validation/commit buttons (VC33 pattern: click to submit answer).
        """
        hypotheses = []
        by_action = defaultdict(list)
        for obs in self.observations:
            by_action[obs.action_id].append(obs)

        # Compute average change size per action
        action_sizes = {}
        for action_id, obs_list in by_action.items():
            sizes = [len(obs.changes) for obs in obs_list]
            action_sizes[action_id] = np.mean(sizes) if sizes else 0

        if not action_sizes:
            return hypotheses

        max_size = max(action_sizes.values())
        if max_size < 50:
            return hypotheses  # No large-change actions

        # Actions with >10x the median change size are likely commit buttons
        median_size = np.median(list(action_sizes.values()))
        if median_size == 0:
            median_size = 1

        for action_id, avg_size in action_sizes.items():
            if avg_size > 50 and avg_size > median_size * 5:
                hypotheses.append(Hypothesis(
                    rule_type='commit_button',
                    description=f"Action {action_id} is a commit/submit button ({avg_size:.0f} cells change)",
                    parameters={
                        'action_id': action_id,
                        'avg_change_size': float(avg_size),
                        'ratio_to_median': float(avg_size / median_size)
                    },
                    confidence=min(0.9, avg_size / 200),  # Higher confidence for larger changes
                    tests_passed=len(by_action[action_id])
                ))

        return hypotheses

    def _hypothesize_win_condition(self) -> list[Hypothesis]:
        """Induce what the winning state looks like."""
        hypotheses = []

        if not self.win_observations:
            return hypotheses

        # Analyze winning frames — what's common?
        win_frames = [obs.frame_after for obs in self.win_observations if obs.frame_after is not None]

        if not win_frames:
            return hypotheses

        # Check: are all cells the same color in winning frame? (fill-all games)
        for wf in win_frames:
            unique_colors = np.unique(wf)
            if len(unique_colors) <= 2:  # background + 1 target
                hypotheses.append(Hypothesis(
                    rule_type='win_condition',
                    description=f"Win when all cells are same color ({unique_colors[-1]})",
                    parameters={'type': 'uniform', 'target_color': int(unique_colors[-1])},
                    confidence=0.5,
                    tests_passed=1
                ))

        # Check: does the winning frame have spatial symmetry?
        for wf in win_frames:
            # Horizontal symmetry
            if np.array_equal(wf, np.fliplr(wf)):
                hypotheses.append(Hypothesis(
                    rule_type='win_condition',
                    description="Win when frame is horizontally symmetric",
                    parameters={'type': 'symmetry', 'axis': 'horizontal'},
                    confidence=0.5, tests_passed=1
                ))
            # Vertical symmetry
            if np.array_equal(wf, np.flipud(wf)):
                hypotheses.append(Hypothesis(
                    rule_type='win_condition',
                    description="Win when frame is vertically symmetric",
                    parameters={'type': 'symmetry', 'axis': 'vertical'},
                    confidence=0.5, tests_passed=1
                ))

        return hypotheses

    # ── Experiment design ──

    def design_experiments(self, env_snap, actions) -> list[dict]:
        """Design targeted experiments to test hypotheses.

        Instead of BFS exploring everything, we test specific predictions:
        - "If I click (3,3), cells (2,3), (3,2), (3,3), (3,4), (4,3) should toggle"
        - Run the action, check if prediction matches reality
        """
        experiments = []

        for hyp in self.hypotheses:
            if hyp.confidence >= 0.8:
                continue  # Already confident enough

            if hyp.rule_type in ('toggle', 'cycle'):
                # Pick an untested click position
                tested_positions = set()
                for obs in self.observations:
                    if obs.click_pos:
                        tested_positions.add(obs.click_pos)

                # Find click actions
                click_actions = [a for a in actions if a.data and 'x' in a.data]
                untested = [a for a in click_actions
                           if (a.data['y'], a.data['x']) not in tested_positions]

                if untested:
                    target = untested[0]
                    experiments.append({
                        'hypothesis': hyp,
                        'action': target,
                        'prediction': self._predict(hyp, target, env_snap),
                        'type': 'confirm_mechanic'
                    })

            elif hyp.rule_type == 'move':
                # Test movement prediction
                experiments.append({
                    'hypothesis': hyp,
                    'action_id': hyp.parameters['action_id'],
                    'type': 'confirm_movement'
                })

        return experiments

    def _predict(self, hyp: Hypothesis, action, env_snap) -> set:
        """Predict which cells will change if hypothesis is correct."""
        if hyp.rule_type in ('toggle', 'cycle') and action.data and 'x' in action.data:
            cr, cc = action.data['y'], action.data['x']
            neighborhood = hyp.parameters.get('neighborhood', [(0, 0)])
            predicted_cells = set()
            for dr, dc in neighborhood:
                nr, nc = cr + dr, cc + dc
                if 0 <= nr < self.grid_size[0] and 0 <= nc < self.grid_size[1]:
                    predicted_cells.add((nr, nc))
            return predicted_cells
        return set()

    def verify_experiment(self, experiment: dict, observation: Observation) -> bool:
        """Check if the experiment's prediction matched reality."""
        hyp = experiment['hypothesis']

        if experiment['type'] == 'confirm_mechanic':
            predicted = experiment.get('prediction', set())
            actual = set((ch.row, ch.col) for ch in observation.changes)

            if predicted and predicted == actual:
                hyp.confirm()
                if hyp.confidence >= 0.8 and hyp not in self.confirmed_rules:
                    self.confirmed_rules.append(hyp)
                    log.info(f"  RULE CONFIRMED: {hyp.description} (conf={hyp.confidence:.2f})")
                return True
            else:
                hyp.reject()
                return False

        return False

    # ── Analytical solving ──

    def can_solve_analytically(self) -> bool:
        """Can we solve the game without BFS, using confirmed rules?"""
        has_toggle = any(r.rule_type == 'toggle' for r in self.confirmed_rules)
        has_mechanic = any(r.rule_type in ('toggle', 'cycle', 'fill', 'move')
                         for r in self.confirmed_rules)
        has_win = any(r.rule_type == 'win_condition' for r in self.confirmed_rules)

        # Toggle games: infer win condition (all cells same color) if not explicitly observed
        if has_toggle and not has_win:
            toggle_rule = next(r for r in self.confirmed_rules if r.rule_type == 'toggle')
            colors = toggle_rule.parameters.get('colors', [])
            if len(colors) == 2:
                # Binary toggle: win = all cells are one color (try the "off" state = min color)
                target = min(colors)
                inferred_win = Hypothesis(
                    rule_type='win_condition',
                    description=f"Inferred: all cells should be color {target}",
                    parameters={'type': 'uniform', 'target_color': target},
                    confidence=0.6, tests_passed=0
                )
                self.confirmed_rules.append(inferred_win)
                has_win = True
                log.info(f"  Inferred win condition: all cells → color {target}")

        return has_mechanic and has_win

    def solve_analytically(self, current_frame: np.ndarray,
                           actions: list, env_snap=None) -> list:
        """Attempt to solve the game using confirmed rules alone.

        For toggle games (Lights Out): solve the linear system over GF(2).
        For fill games: click every non-target cell.
        For navigation: path-plan using movement rules.
        """
        mechanic = next((r for r in self.confirmed_rules
                        if r.rule_type in ('toggle', 'cycle', 'fill', 'move')), None)
        win_cond = next((r for r in self.confirmed_rules
                        if r.rule_type == 'win_condition'), None)

        if not mechanic:
            return []

        frame = current_frame if current_frame.ndim == 2 else current_frame[0]

        if mechanic.rule_type == 'toggle' and win_cond:
            return self._solve_toggle(frame, mechanic, win_cond, actions)
        elif mechanic.rule_type == 'fill':
            return self._solve_fill(frame, mechanic, actions)

        return []  # Can't solve this type analytically yet

    def _solve_toggle(self, frame, mechanic, win_cond, actions) -> list:
        """Solve toggle game as linear system over GF(2).

        Classic Lights Out solution: each cell is a variable (0 or 1 = click or not).
        Each click toggles a neighborhood. We want all cells to reach target color.
        This is a system of linear equations mod 2.
        """
        target_color = win_cond.parameters.get('target_color', 0)
        colors = mechanic.parameters.get('colors', [0, 1])
        neighborhood = mechanic.parameters.get('neighborhood', [(0, 0)])

        # Work at grid level if we have grid info, else pixel level
        if hasattr(self, '_grid') and self._grid is not None:
            g = self._grid
            rows, cols = g['n_rows'], g['n_cols']
        else:
            rows, cols = frame.shape

        n = rows * cols

        # Build the toggle matrix A (n x n) over GF(2)
        # A[i][j] = 1 if clicking cell j affects cell i
        A = np.zeros((n, n), dtype=np.uint8)

        # Map click actions to grid cell positions
        click_actions = {}
        for act in actions:
            if act.data and 'x' in act.data:
                gr, gc = self._pixel_to_grid(act.data['y'], act.data['x'])
                if 0 <= gr < rows and 0 <= gc < cols:
                    j = gr * cols + gc
                    if j not in click_actions:  # First action per cell wins
                        click_actions[j] = act
                        for dr, dc in neighborhood:
                            nr, nc = gr + dr, gc + dc
                            if 0 <= nr < rows and 0 <= nc < cols:
                                i = nr * cols + nc
                                A[i][j] = 1

        # Build target vector b: which cells need to toggle?
        b = np.zeros(n, dtype=np.uint8)
        for gr in range(rows):
            for gc in range(cols):
                cell_color = self._cell_color(frame, gr, gc) if hasattr(self, '_grid') and self._grid else int(frame[gr, gc])
                if cell_color != target_color:
                    b[gr * cols + gc] = 1

        # Solve Ax = b over GF(2) using Gaussian elimination
        solution = self._gaussian_elimination_gf2(A, b)

        if solution is None:
            log.info("  Toggle system has no solution")
            return []

        # Convert solution to action sequence
        action_seq = []
        for j in range(n):
            if solution[j] == 1 and j in click_actions:
                action_seq.append(click_actions[j])

        log.info(f"  Analytical solution: {len(action_seq)} clicks (toggle system solved over GF(2))")
        return action_seq

    def _gaussian_elimination_gf2(self, A, b):
        """Solve Ax = b over GF(2) using Gaussian elimination."""
        n = len(b)
        # Augmented matrix [A|b]
        M = np.zeros((n, n + 1), dtype=np.uint8)
        M[:, :n] = A.copy()
        M[:, n] = b.copy()

        pivot_cols = []
        row = 0

        for col in range(n):
            # Find pivot
            pivot = None
            for r in range(row, n):
                if M[r, col] == 1:
                    pivot = r
                    break
            if pivot is None:
                continue

            # Swap rows
            M[[row, pivot]] = M[[pivot, row]]
            pivot_cols.append(col)

            # Eliminate
            for r in range(n):
                if r != row and M[r, col] == 1:
                    M[r] = (M[r] + M[row]) % 2

            row += 1

        # Check consistency
        for r in range(row, n):
            if M[r, n] == 1:
                return None  # No solution

        # Extract solution (free variables set to 0)
        x = np.zeros(n, dtype=np.uint8)
        for i, col in enumerate(pivot_cols):
            x[col] = M[i, n]

        return x

    def _solve_fill(self, frame, mechanic, actions) -> list:
        """Solve fill game: click every cell that isn't the target color."""
        target = mechanic.parameters.get('target_color')
        if target is None:
            return []

        action_seq = []
        for act in actions:
            if act.data and 'x' in act.data:
                r, c = act.data['y'], act.data['x']
                if 0 <= r < frame.shape[0] and 0 <= c < frame.shape[1]:
                    if frame[r, c] != target:
                        action_seq.append(act)

        return action_seq

    # ── Summary for persistence ──

    def summary(self) -> dict:
        """Summarize what was learned — for the Rosetta Stone."""
        return {
            'observations': len(self.observations),
            'hypotheses': len(self.hypotheses),
            'confirmed_rules': [
                {'type': r.rule_type, 'desc': r.description,
                 'conf': r.confidence, 'params': r.parameters}
                for r in self.confirmed_rules
            ],
            'grid_size': self.grid_size,
            'colors': sorted(self.color_palette),
            'can_solve_analytically': self.can_solve_analytically(),
        }
