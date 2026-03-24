#!/usr/bin/env python3
"""Lightweight action predictor — numpy only, no PyTorch.

Drop-in replacement for arc_action_predictor.ActionPredictor.
Uses pixel-change statistics + color-aware spatial tracking instead of CNN.
Much lighter (~200 lines vs ~300 lines + 2GB PyTorch dependency).

Key insights (informed by StochasticGoose 1st-place ARC-AGI-3 solution):
- What matters: "does this action change the frame, and by how much?"
- Color separation: track which COLORS moved, not just pixel count.
  StochasticGoose uses 16-channel one-hot CNN; we use per-color change stats.
- Spatial awareness: track which REGIONS changed (quadrant-level).
"""

import logging
import numpy as np

log = logging.getLogger(__name__)


class ActionPredictor:
    """Lightweight action predictor using pixel-change statistics.

    API-compatible with arc_action_predictor.ActionPredictor.
    """

    def __init__(self, n_actions: int = 6, frame_h: int = 64,
                 frame_w: int = 64, max_colors: int = 16,
                 buffer_size: int = 50000):
        self.n_actions = n_actions
        self.max_colors = max_colors
        self.frame_h = frame_h
        self.frame_w = frame_w
        self.enabled = True
        self.buffer_size = buffer_size
        self.total_obs = 0
        self.confidence = 0.0

        # Per-action statistics
        self._change_counts = np.zeros(n_actions)  # how many times action caused change
        self._total_counts = np.zeros(n_actions)    # how many times action was tried
        self._change_magnitudes = np.zeros(n_actions)  # total pixels changed
        self._success_counts = np.zeros(n_actions)  # actions that were part of winning sequences

        # Color-aware: per-action, per-color change counts (inspired by StochasticGoose one-hot)
        self._color_change_counts = np.zeros((n_actions, max_colors))  # action × color
        self._color_total_counts = np.zeros((n_actions, max_colors))

        # Spatial-aware: per-action quadrant change counts (4 quadrants)
        self._quad_change_counts = np.zeros((n_actions, 4))
        self._quad_total_counts = np.zeros((n_actions, 4))

        # Frame-conditioned features (simple hash → action outcome)
        self._frame_action_cache = {}  # (frame_hash, action) → change_ratio

    def observe(self, frame_before: np.ndarray, action_idx: int,
                frame_after: np.ndarray, changed=None):
        """Record an observation: action_idx was taken, frame changed from before→after."""
        if action_idx < 0 or action_idx >= self.n_actions:
            return

        self.total_obs += 1
        self._total_counts[action_idx] += 1

        # Compute change
        try:
            fb = frame_before.squeeze()
            fa = frame_after.squeeze()
            if fb.shape != fa.shape:
                return
            diff = (fb != fa)
            n_changed = np.sum(diff)
            total_pixels = max(diff.size, 1)
            change_ratio = n_changed / total_pixels
        except Exception:
            return

        if change_ratio > 0.001:  # threshold: >0.1% pixels changed
            self._change_counts[action_idx] += 1
            self._change_magnitudes[action_idx] += change_ratio

            # Color-aware tracking: which colors were affected?
            try:
                changed_pixels_before = fb[diff]
                changed_pixels_after = fa[diff]
                for color in np.unique(np.concatenate([changed_pixels_before, changed_pixels_after])):
                    c = int(color) % self.max_colors
                    self._color_change_counts[action_idx, c] += 1
                    self._color_total_counts[action_idx, c] += 1
            except Exception:
                pass

            # Spatial tracking: which quadrants changed?
            try:
                h2, w2 = fb.shape[0] // 2, fb.shape[1] // 2
                quads = [diff[:h2, :w2], diff[:h2, w2:],
                         diff[h2:, :w2], diff[h2:, w2:]]
                for qi, q in enumerate(quads):
                    if np.any(q):
                        self._quad_change_counts[action_idx, qi] += 1
                    self._quad_total_counts[action_idx, qi] += 1
            except Exception:
                pass
        else:
            # No change — still update color/quad totals for base rate
            try:
                for color in np.unique(fb):
                    c = int(color) % self.max_colors
                    self._color_total_counts[action_idx, c] += 1
                for qi in range(4):
                    self._quad_total_counts[action_idx, qi] += 1
            except Exception:
                pass

        # Cache frame-specific outcome
        fhash = hash(frame_before.tobytes()[:1024])  # fast partial hash
        self._frame_action_cache[(fhash, action_idx)] = change_ratio

        # Trim cache if too large
        if len(self._frame_action_cache) > 10000:
            # Keep most recent half
            items = list(self._frame_action_cache.items())
            self._frame_action_cache = dict(items[len(items)//2:])

        # Update confidence
        min_obs = min(self._total_counts[self._total_counts > 0]) if np.any(self._total_counts > 0) else 0
        self.confidence = min(min_obs / 5.0, 1.0)  # confident after 5 obs per active action

    def train(self, epochs: int = 1, batch_size: int = 32):
        """No-op for API compatibility. Statistics are updated in observe()."""
        pass

    def predict(self, frame: np.ndarray) -> np.ndarray:
        """Predict action-change probabilities for the current frame.

        Returns array of shape (n_actions,) with probabilities [0, 1].
        Uses color-aware and spatial-aware features when available.
        """
        probs = np.zeros(self.n_actions)

        # Base rate: what fraction of times did each action cause a change?
        for i in range(self.n_actions):
            if self._total_counts[i] > 0:
                probs[i] = self._change_counts[i] / self._total_counts[i]
            else:
                probs[i] = 0.5  # unknown = 50%

        # Color-aware boost: which actions affect colors present in THIS frame?
        try:
            fr = frame.squeeze()
            present_colors = np.unique(fr)
            color_boost = np.zeros(self.n_actions)
            for i in range(self.n_actions):
                relevant_changes = 0
                relevant_total = 0
                for color in present_colors:
                    c = int(color) % self.max_colors
                    relevant_changes += self._color_change_counts[i, c]
                    relevant_total += self._color_total_counts[i, c]
                if relevant_total > 0:
                    color_boost[i] = relevant_changes / relevant_total

            # Blend color info if we have enough data
            if np.any(self._color_total_counts > 0):
                probs = 0.6 * probs + 0.4 * color_boost
        except Exception:
            pass

        # Frame-specific boost: if we've seen this frame before, use cached data
        fhash = hash(frame.tobytes()[:1024])
        for i in range(self.n_actions):
            cached = self._frame_action_cache.get((fhash, i))
            if cached is not None:
                # Blend: 70% frame-specific, 30% base rate
                probs[i] = 0.7 * (1.0 if cached > 0.001 else 0.0) + 0.3 * probs[i]

        # Success boost: actions from winning sequences get priority
        total_success = self._success_counts.sum()
        if total_success > 0:
            success_probs = self._success_counts / total_success
            probs = 0.6 * probs + 0.4 * success_probs  # 40% weight on success history

        # Normalize to sum to 1 (for compatibility)
        total = probs.sum()
        if total > 0:
            probs = probs / total
        else:
            probs = np.ones(self.n_actions) / self.n_actions

        return probs

    def color_profile(self) -> dict:
        """Return per-action color change profile for debugging/logging.

        Returns dict of {action_idx: [(color, change_rate), ...]} sorted by change rate.
        """
        profile = {}
        for i in range(self.n_actions):
            rates = []
            for c in range(self.max_colors):
                if self._color_total_counts[i, c] > 0:
                    rate = self._color_change_counts[i, c] / self._color_total_counts[i, c]
                    if rate > 0:
                        rates.append((c, round(rate, 3)))
            if rates:
                profile[i] = sorted(rates, key=lambda x: -x[1])
        return profile

    def reinforce(self, action_sequence: list):
        """Boost actions from a winning sequence. Called when a level is solved.

        action_sequence: list of action indices that led to the win.
        This is the "CNN remembers" mechanism — successful actions get weighted higher.
        """
        for action_idx in action_sequence:
            if 0 <= action_idx < self.n_actions:
                self._success_counts[action_idx] += 1
        log.info(f"CNN reinforced {len(action_sequence)} actions from winning sequence")

    def reset(self):
        """Reset observation buffer and success bias for new level.

        Keeps base change statistics (which actions cause frame changes)
        but resets success reinforcement — each level may need different actions.
        """
        self._frame_action_cache.clear()
        self._success_counts = np.zeros(self.n_actions)  # reset per-level — different levels need different actions
        # Keep _change_counts, _total_counts, _color_*, _quad_* — these transfer across levels
        # The knowledge of "action 3 moves red objects" is game-level, not level-level
