#!/usr/bin/env python3
"""ARC-AGI-3 submission entry point.

Supports TWO modes:
  1. Server mode (competition): listen_and_serve on port 8001, competition runner sends games
  2. Pull mode (testing): iterate through local environments

Environment variables:
    OPERATION_MODE  — 'competition' for server mode, 'normal'/'offline' for pull mode
    ARC_TIMEOUT     — Total time budget in seconds (default: 28800 = 8hrs)
    ARC_PER_LEVEL   — Per-level timeout in seconds (default: 60)
    ARC_GAMES       — Comma-separated game IDs (optional, default: all)
    ARC_GENERIC     — Set to '1' to force generic solver only
    ARC_PORT        — Server port for competition mode (default: 8001)
"""

import json
import logging
import os
import sys
import time

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)
log = logging.getLogger("arc-submission")


def on_scorecard_complete(scorecard):
    """Called when competition runner closes a scorecard."""
    log.info(f"Scorecard closed: {scorecard.total_environments_completed}/"
             f"{scorecard.total_environments} envs, "
             f"{scorecard.total_levels_completed}/{scorecard.total_levels} levels, "
             f"{scorecard.total_actions} actions")


def run_server():
    """Competition mode: start HTTP server, let runner send games via scorecard API."""
    import arc_agi

    port = int(os.environ.get("ARC_PORT", "8001"))
    log.info(f"ARC-AGI-3 Competition Server — port={port}")

    arcade = arc_agi.Arcade(operation_mode=arc_agi.OperationMode.COMPETITION)
    arcade.listen_and_serve(
        host="0.0.0.0",
        port=port,
        competition_mode=True,
        save_all_recordings=True,
        on_scorecard_close=on_scorecard_complete,
    )


def run_pull():
    """Pull mode: iterate through environments locally (for testing)."""
    total_timeout = float(os.environ.get("ARC_TIMEOUT", "28800"))
    per_level = float(os.environ.get("ARC_PER_LEVEL", "60"))
    game_filter = os.environ.get("ARC_GAMES", "")
    force_generic = os.environ.get("ARC_GENERIC", "") == "1"

    # Pure AGI mode by default
    if "ARC_PURE_AGI" not in os.environ:
        os.environ["ARC_PURE_AGI"] = "1"
        log.info("Pure AGI mode (Gundam pilot only)")

    llm_mode = os.environ.get("ARC_LLM_MODE", "auto")
    if llm_mode == "none":
        os.environ["ARC_NO_LLM"] = "1"
        log.info("LLM disabled — algorithmic-only mode")

    log.info(f"ARC-AGI-3 Pull Mode — timeout={total_timeout}s, per_level={per_level}s, llm={llm_mode}")

    import arc_agi
    from arc_eyes import EpisodicMemory

    memory_path = "/app/arc_eyes_memory.json"
    eyes_memory = EpisodicMemory.load(memory_path) if os.path.exists(memory_path) else EpisodicMemory()
    if eyes_memory.game_patterns:
        log.info(f"Loaded Eyes memory: {len(eyes_memory.game_patterns)} game patterns")

    solver_mem = None
    try:
        from solver_memory import SolverMemory
        solver_mem = SolverMemory(db_path="/app/solver_memory.db")
        stats = solver_mem.stats()
        log.info(f"Solver memory: {stats['rules']} rules, {stats['games']} games")
    except Exception as e:
        log.info(f"No solver memory: {e}")

    arcade = arc_agi.Arcade()

    # Open scorecard for competition scoring
    card_id = None
    try:
        card_id = arcade.open_scorecard(tags=["gundam", "autobots"])
        log.info(f"Scorecard opened: {card_id}")
    except Exception as e:
        log.warning(f"Could not open scorecard: {e} — results won't be recorded on ARC platform")

    envs = arcade.get_environments()
    game_ids = list(dict.fromkeys(e.game_id.split('-')[0] for e in envs))  # deduplicate, preserve order
    env_info_map = {e.game_id.split('-')[0]: e for e in envs}  # for baseline_actions lookup

    if game_filter:
        allowed = {g.strip() for g in game_filter.split(",") if g.strip()}
        game_ids = [g for g in game_ids if g in allowed]

    log.info(f"Solving {len(game_ids)} games")

    from gundam import run_game as gundam_run_game
    log.info("Gundam AGI mode — pilot + oracle + 16 algo transforms")

    # Competition budget from env (default: 10 calls/level)
    _budget_per_level = int(os.environ.get('ARC_BUDGET_PER_LEVEL', '10'))

    deadline = time.time() + total_timeout
    all_results = []
    total_solved = 0
    total_actions = 0
    n_games = len(game_ids)

    for i, gid in enumerate(game_ids):
        if time.time() >= deadline:
            log.warning(f"Timeout after {i}/{len(game_ids)} games")
            break

        remaining_time = deadline - time.time()
        remaining_games = max(n_games - i, 1)
        avg_levels = 7
        dynamic_per_level = min(
            per_level,
            max(remaining_time / (remaining_games * avg_levels), 10)
        )
        log.info(f"[{i+1}/{n_games}] {gid} (budget: {dynamic_per_level:.0f}s/level, "
                 f"{remaining_time:.0f}s remaining)")

        try:
            # Create env from arcade, pass to Gundam
            env = arcade.make(gid, scorecard_id=card_id)
            if env is None:
                log.error(f"  → Failed to create env for {gid}")
                all_results.append({"game": gid, "error": "failed to create env"})
                continue

            mind = gundam_run_game(
                game_id=gid,
                budget_per_level=_budget_per_level,
                verbose=True,
                env_override=env,
            )

            # Convert Gundam result to standard dict format
            solved = mind.memory.levels_solved
            total_lvl = mind.memory.levels_total or 0
            actions = sum(lvl.total_actions for lvl in mind.memory.levels.values())
            # Action efficiency: compare to human baseline (per solved levels only)
            baseline = getattr(env_info_map.get(gid), 'baseline_actions', None)
            baseline_total = sum(baseline) if baseline else None
            baseline_solved = sum(baseline[:solved]) if baseline and solved > 0 else None
            efficiency = None
            if baseline_solved and actions > 0 and solved > 0:
                # Efficiency = human_actions_for_solved_levels / our_actions (1.0 = human-level)
                efficiency = round(baseline_solved / actions, 3)
            result = {
                "game": gid,
                "levels_solved": solved,
                "total_levels": total_lvl,
                "total_actions": actions,
                "baseline_actions_total": baseline_total,
                "baseline_actions_solved": baseline_solved,
                "action_efficiency": efficiency,
                "llm_calls": mind.total_llm_calls,
            }
            all_results.append(result)
            if solved == total_lvl and total_lvl > 0:
                total_solved += 1
            total_actions += actions
            eff_str = f", efficiency={efficiency:.1%}" if efficiency else ""
            log.info(f"  → {solved}/{total_lvl} levels, {actions} actions (baseline={baseline_total}){eff_str}, {mind.total_llm_calls} LLM calls")
        except Exception as e:
            log.error(f"  → FAILED: {e}")
            all_results.append({"game": gid, "error": str(e)})

    summary = {
        "total_games": len(game_ids),
        "games_attempted": len(all_results),
        "fully_solved": total_solved,
        "total_actions": total_actions,
        "time_used": total_timeout - max(deadline - time.time(), 0),
        "results": all_results,
    }
    results_path = os.environ.get("ARC_RESULTS", "/app/results.json")
    try:
        os.makedirs(os.path.dirname(results_path) or ".", exist_ok=True)
        with open(results_path, "w") as f:
            json.dump(summary, f, indent=2)
    except OSError:
        # Fallback to current dir if /app/ doesn't exist
        with open("results.json", "w") as f:
            json.dump(summary, f, indent=2)

    # Close scorecard to finalize results
    if card_id:
        try:
            scorecard = arcade.close_scorecard(card_id)
            if scorecard:
                log.info(f"Scorecard closed: {scorecard.total_levels_completed}/{scorecard.total_levels} levels, "
                         f"{scorecard.total_actions} actions")
        except Exception as e:
            log.warning(f"Could not close scorecard: {e}")

    log.info(f"Done: {total_solved}/{len(game_ids)} fully solved, {total_actions} total actions")


def main():
    op_mode = os.environ.get("OPERATION_MODE", "normal").lower()

    if op_mode == "competition":
        run_server()
    else:
        run_pull()


if __name__ == "__main__":
    main()
