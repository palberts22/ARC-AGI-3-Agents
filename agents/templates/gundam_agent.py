#!/usr/bin/env python3
"""ARC-AGI-3 Agent adapter for the Gundam solver.

Bridges the ARC-AGI-3-Agents framework (choose_action/is_done interface)
with Gundam's observe-think-act loop.

Strategy: Run Gundam's full pipeline in a background thread. Gundam steps a
PROXY env whose step() queues actions. The framework's main loop pops from
that queue via choose_action().

Usage:
    Copy this file to ARC-AGI-3-Agents/agents/templates/gundam_agent.py
    Register in agents/__init__.py:
        from .templates.gundam_agent import GundamAgent
    Run:
        uv run main.py --agent=gundamagent --game=ft09
"""

import copy
import logging
import os
import queue
import sys
import threading
import time
from typing import Any, Optional

import numpy as np
from arcengine import FrameData, GameAction, GameState

# Add parent dir so we can import gundam
_submission_dir = os.path.dirname(os.path.abspath(__file__))
if _submission_dir not in sys.path:
    sys.path.insert(0, _submission_dir)

try:
    from agents.agent import Agent
except ImportError:
    class Agent:
        MAX_ACTIONS = 500
        def __init__(self, *args, **kwargs):
            self.game_id = kwargs.get('game_id', 'unknown')
            self.arc_env = kwargs.get('arc_env')
            self.action_counter = 0
            self.frames = [FrameData(levels_completed=0)]

logger = logging.getLogger(__name__)


class ProxyEnv:
    """Wraps the real env so Gundam's step() calls get intercepted.

    Each step() from Gundam:
    1. Puts the action into the action_queue
    2. Blocks until the framework executes it and puts the result into result_queue
    3. Returns the obs to Gundam

    This synchronizes Gundam's internal loop with the framework's
    choose_action() → take_action() cycle.
    """

    def __init__(self, real_env, game_id: str):
        self.real_env = real_env
        self.game_id = game_id
        self.action_queue = queue.Queue()   # Gundam → framework
        self.result_queue = queue.Queue()   # framework → Gundam
        self.done = threading.Event()
        self._last_obs = None
        self._levels_completed = 0

    def __deepcopy__(self, memo):
        """Deepcopy returns a plain env copy (not proxied).

        Gundam deepcopies env for OPRAH probing. Copies should step
        directly on the game state without going through the queue.
        """
        return copy.deepcopy(self.real_env, memo)

    def reset(self):
        """Reset returns the initial observation without queuing."""
        obs = self.real_env.reset()
        self._last_obs = obs
        # Expose observation_space for preflight compatibility
        self.frame = obs.frame if hasattr(obs, 'frame') else None
        return obs

    def step(self, action_idx, data=None):
        """Called by Gundam. Steps own env AND queues action for framework.

        The proxy steps its own real_env (for correct state) and also
        queues the action so the framework records it in the scorecard.
        Both envs receive the same actions → same results (deterministic).
        """
        if self.done.is_set():
            return self._last_obs

        # Step our own env copy (keeps deepcopy state correct for probing)
        obs = self.real_env.step(action_idx, data)
        self._last_obs = obs
        if hasattr(obs, 'levels_completed') and obs.levels_completed:
            self._levels_completed = obs.levels_completed

        # Queue the action for the framework to replay on self.arc_env
        self.action_queue.put((action_idx, data))

        # Wait for framework to confirm it processed the action
        try:
            self.result_queue.get(timeout=60)
        except queue.Empty:
            logger.warning("[proxy] Framework didn't ack action in time")

        return obs

    def close(self):
        self.done.set()


class GundamAgent(Agent):
    """Gundam-powered ARC-AGI-3 agent.

    Runs Gundam's full pipeline (OPRAH → genre → researcher → oracle → LLM)
    in a background thread, synchronized with the framework via ProxyEnv.
    """

    MAX_ACTIONS = 500

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._proxy = None
        self._gundam_thread = None
        self._started = False
        self._game_done = False
        logger.info(f"[gundam-agent] Created for game {self.game_id}")

    def _start_gundam(self) -> None:
        """Launch Gundam in a background thread with proxy env."""
        self._proxy = ProxyEnv(copy.deepcopy(self.arc_env), self.game_id)

        def _run():
            try:
                from gundam import run_game
                run_game(
                    game_id=self.game_id,
                    budget_per_level=50,
                    max_turns_per_level=50,
                    verbose=True,
                    env_override=self._proxy,
                )
            except Exception as e:
                logger.error(f"[gundam-agent] Gundam thread error: {e}", exc_info=True)
            finally:
                self._proxy.done.set()
                # Signal framework that Gundam is done
                self._proxy.action_queue.put(None)

        self._gundam_thread = threading.Thread(target=_run, daemon=True)
        self._gundam_thread.start()
        self._started = True
        logger.info(f"[gundam-agent] Gundam thread started for {self.game_id}")

    def _action_idx_to_game_action(self, action_idx, data: dict = None) -> GameAction:
        """Convert Gundam's action to GameAction."""
        # action_idx may already be a GameAction enum (Gundam passes obs.available_actions[i])
        if isinstance(action_idx, GameAction):
            action = action_idx
        elif isinstance(action_idx, int):
            action = GameAction.from_id(action_idx)
        else:
            action = GameAction.from_id(int(action_idx))

        if action.is_complex() and data:
            x = data.get('x', 32)
            y = data.get('y', 32)
            action.set_data({"x": x, "y": y, "game_id": self.game_id})
        elif data:
            action.set_data({"game_id": self.game_id, **data})

        action.reasoning = {"source": "gundam", "action_idx": action_idx}
        return action

    def is_done(self, frames: list[FrameData], latest_frame: FrameData) -> bool:
        if latest_frame.state is GameState.WIN:
            if self._proxy:
                self._proxy.done.set()
            return True
        if self._proxy and self._proxy.done.is_set() and self._proxy.action_queue.empty():
            return True
        return False

    def choose_action(
        self, frames: list[FrameData], latest_frame: FrameData
    ) -> GameAction:
        """Get next action from Gundam's background thread."""

        # Handle game start
        if latest_frame.state in (GameState.NOT_PLAYED, GameState.GAME_OVER):
            return GameAction.RESET

        # Start Gundam on first real frame
        if not self._started:
            self._start_gundam()
            # Gundam's run_game() calls env.reset() which returns immediately.
            # The first actual step() will be queued. Wait for it.

        # Ack previous action so Gundam thread can continue
        if self._proxy and self._started and self.action_counter > 0:
            try:
                self._proxy.result_queue.put_nowait(True)
            except queue.Full:
                pass

        # Get next action from Gundam
        try:
            item = self._proxy.action_queue.get(timeout=120)
        except queue.Empty:
            logger.warning("[gundam-agent] Timeout waiting for Gundam action")
            return GameAction.ACTION1

        if item is None:
            # Gundam finished
            self._game_done = True
            return GameAction.ACTION1  # will trigger is_done on next check

        action_idx, data = item
        return self._action_idx_to_game_action(action_idx, data)

    def cleanup(self, scorecard=None) -> None:
        """Clean up Gundam thread."""
        if self._proxy:
            self._proxy.done.set()
            self._proxy.result_queue.put(None)  # unblock Gundam if waiting
        if self._gundam_thread and self._gundam_thread.is_alive():
            self._gundam_thread.join(timeout=5)
        super().cleanup(scorecard)
