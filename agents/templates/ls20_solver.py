#!/usr/bin/env python3
"""LS20 solver v7 — fuel management + multi-target support."""

import numpy as np
from collections import deque

GAME_DIRS = {1: (0, -5), 2: (0, 5), 3: (-5, 0), 4: (5, 0)}
REVERSE = {1: 2, 2: 1, 3: 4, 4: 3}
NUM_SHAPES = 6
NUM_COLORS = 4
NUM_ROTS = 4
FUEL_MAX = 42
# Gauge starts at 42, pca() decrements AFTER move. So 43 steps before death
# (step 0: gauge=42, step 1: gauge=41, ..., step 42: gauge=0, step 43: gauge=-1 → lose)
# But fuel item step doesn't decrement (xpb=True skips pca).
# So effectively 43 moves allowed between refuels.
FUEL_STEPS = 43


def get_walls(g):
    """Get blocked cells by checking rbt at every position on the player grid."""
    px = g.mgu.x % 5
    py = g.mgu.y % 5
    blocked = set()
    for gy in range(py, 60, 5):
        for gx in range(px, 60, 5):
            sprites_at = g.rbt(gx, gy, 5, 5)
            for s in sprites_at:
                tags = s.tags if s.tags else []
                if "jdd" in tags:
                    blocked.add((gx, gy))
                    break
    return blocked


def bfs_path(start, goal, walls, avoid=None):
    """BFS pathfinding. avoid = set of cells to treat as impassable (except goal)."""
    if start == goal: return []
    avoid = avoid or set()
    queue = deque([(start, [])])
    seen = {start}
    while queue:
        p, path = queue.popleft()
        for gid, (dx, dy) in GAME_DIRS.items():
            npos = (p[0]+dx, p[1]+dy)
            if npos in seen or npos in walls or npos[0] < 0 or npos[0] >= 60 or npos[1] < 0 or npos[1] >= 60:
                continue
            if npos in avoid and npos != goal:
                continue
            new_path = path + [gid]
            if npos == goal: return new_path
            seen.add(npos)
            queue.append((npos, new_path))
    return None


def bfs_dist(start, goal, walls, avoid=None):
    """Return distance (steps) from start to goal, or None if unreachable."""
    path = bfs_path(start, goal, walls, avoid)
    return len(path) if path is not None else None


def exec_steps(env, gids, start_score):
    obs = None
    for gid in gids:
        obs = env.step(gid)
        lc = obs.levels_completed or 0
        if lc > start_score:
            return obs, True
        raw = np.array(obs.frame)
        multi = raw.ndim > 2 and raw.shape[0] > 1
        if multi:
            obs = env.step(5)
            lc = obs.levels_completed or 0
            if lc > start_score:
                return obs, True
    return obs, False


def sprite_grid_cells(g, tag):
    """Map sprite positions with given tag to the player grid cells that overlap them."""
    px_off = g.mgu.x % 5
    py_off = g.mgu.y % 5
    sprites = g.current_level.get_sprites_by_tag(tag)
    cells = []
    for s in sprites:
        for gx in range(px_off, 60, 5):
            if gx <= s.x < gx + 5:
                for gy in range(py_off, 60, 5):
                    if gy <= s.y < gy + 5:
                        cells.append((gx, gy))
    return cells


def fuel_grid_cells(g):
    return sprite_grid_cells(g, "iri")


def modifier_grid_cells(g):
    """Get all grid cells that have modifiers (shape/color/rot)."""
    cells = set()
    for tag in ["gsu", "gic", "bgt"]:
        cells.update(sprite_grid_cells(g, tag))
    return cells


def plan_route_with_fuel(waypoints, pos, walls, fuel_cells, verbose=False):
    """Plan a route through waypoints, inserting fuel detours if needed.

    fuel_cells: grid cells where fuel items are collected (from fuel_grid_cells).
    If total route cost > FUEL_STEPS and no fuel is auto-collected along the path,
    insert a detour through the nearest fuel cell.
    """
    # Calculate segment costs and check if fuel cells are on the natural path
    segments = []
    cur = pos
    for wp_pos, bounces in waypoints:
        path = bfs_path(cur, wp_pos, walls)
        if path is None:
            return None
        # Track which cells the path visits
        visited = set()
        p = cur
        for gid in path:
            dx, dy = GAME_DIRS[gid]
            p = (p[0]+dx, p[1]+dy)
            visited.add(p)
        cost = len(path) + (bounces * 2 if bounces > 0 else 0)
        segments.append({'to': wp_pos, 'cost': cost, 'visited': visited})
        cur = wp_pos

    total_cost = sum(s['cost'] for s in segments)

    # Check if any fuel cell is naturally visited along the path
    # If so, fuel is auto-collected and we get a refill
    fuel_on_path = False
    cumulative = 0
    for seg in segments:
        for fc in fuel_cells:
            if fc in seg['visited']:
                fuel_on_path = True
                break
        cumulative += seg['cost']
        if fuel_on_path:
            break

    if total_cost <= FUEL_STEPS or fuel_on_path:
        # Either fits in fuel budget, or fuel is auto-collected along the way
        return waypoints

    if not fuel_cells:
        if verbose: print(f'    Route costs {total_cost} steps, no fuel available')
        return None

    # Need explicit fuel detour. Find best fuel cell to insert.
    # Try inserting before each segment and pick cheapest.
    best_plan = None
    best_total = float('inf')

    for insert_before in range(len(waypoints)):
        for fc in fuel_cells:
            if fc in walls:
                continue
            # Build modified waypoint list with fuel cell inserted
            modified = list(waypoints[:insert_before]) + [(fc, 0)] + list(waypoints[insert_before:])
            # Calculate total cost
            c = pos
            total = 0
            valid = True
            for wp, b in modified:
                d = bfs_dist(c, wp, walls)
                if d is None:
                    valid = False
                    break
                total += d + (b * 2 if b > 0 else 0)
                c = wp
            if valid and total < best_total and total <= FUEL_STEPS + FUEL_STEPS:
                # After fuel cell, we get FUEL_STEPS more. Check each sub-segment.
                # Pre-fuel cost must be <= FUEL_STEPS, post-fuel cost must be <= FUEL_STEPS
                c = pos
                pre_cost = 0
                for i, (wp, b) in enumerate(modified):
                    d = bfs_dist(c, wp, walls)
                    cost = d + (b * 2 if b > 0 else 0)
                    if wp == fc:
                        if pre_cost + cost > FUEL_STEPS:
                            valid = False
                        post_cost = total - pre_cost - cost
                        if post_cost > FUEL_STEPS:
                            valid = False
                        break
                    pre_cost += cost
                    c = wp
                if valid:
                    best_total = total
                    best_plan = modified

    if best_plan:
        if verbose: print(f'    Fuel detour added, total cost: {best_total}')
        return best_plan

    if verbose: print(f'    No viable fuel detour found')
    return None


def solve_level(env, obs, verbose=False):
    g = env._game
    start_score = g._score

    cur_s, cur_c, cur_r = g.snw, g.tmx, g.tuv
    player_pos = (g.mgu.x, g.mgu.y)

    targets = []
    for i, t in enumerate(g.qqv):
        targets.append({'pos': (t.x, t.y), 'shape': g.gfy[i], 'color': g.vxy[i], 'rot': g.cjl[i]})

    s_mods = [(s.x, s.y) for s in g.current_level.get_sprites_by_tag("gsu")]
    c_mods = [(s.x, s.y) for s in g.current_level.get_sprites_by_tag("gic")]
    r_mods = [(s.x, s.y) for s in g.current_level.get_sprites_by_tag("bgt")]
    fuel_cells = fuel_grid_cells(g)
    mod_cells = modifier_grid_cells(g)
    # Map modifier sprite positions to grid cells for avoidance
    s_mod_cells = set(sprite_grid_cells(g, "gsu"))
    c_mod_cells = set(sprite_grid_cells(g, "gic"))
    r_mod_cells = set(sprite_grid_cells(g, "bgt"))

    walls = get_walls(g)

    # Check if fuel gauge is active (vxy data > 0)
    fuel_active = g.current_level.get_data("vxy") or 0

    if verbose:
        print(f'  Player: {player_pos}, state: s={cur_s} c={cur_c} r={cur_r}')
        print(f'  Walls: {len(walls)}, fuel_cells: {fuel_cells}, fuel_active: {fuel_active}')
        for i, t in enumerate(targets):
            ds = (t['shape'] - cur_s) % NUM_SHAPES
            dc = (t['color'] - cur_c) % NUM_COLORS
            dr = (t['rot'] - cur_r) % NUM_ROTS
            print(f'    T{i}: {t["pos"]} delta s={ds} c={dc} r={dr}')

    pos = player_pos

    # Compute target grid cells for avoidance (targets reject if attributes don't match)
    target_cells = set()
    for t in targets:
        target_cells.add(t['pos'])

    # Try different target orderings to find one where paths work
    from itertools import permutations
    best_order = None
    best_waypoints = None

    for perm in permutations(range(len(targets))):
        all_waypoints = []
        temp_s, temp_c, temp_r = cur_s, cur_c, cur_r
        remaining_targets = set(t['pos'] for t in targets)

        check_pos = pos
        valid = True
        for ti in perm:
            target = targets[ti]
            ds = (target['shape'] - temp_s) % NUM_SHAPES
            dc = (target['color'] - temp_c) % NUM_COLORS
            dr = (target['rot'] - temp_r) % NUM_ROTS

            if ds > 0 and s_mods:
                all_waypoints.append((s_mods[0], ds - 1))
            if dc > 0 and c_mods:
                all_waypoints.append((c_mods[0], dc - 1))
            if dr > 0 and r_mods:
                all_waypoints.append((r_mods[0], dr - 1))
            all_waypoints.append((target['pos'], 0))

            # Check if path to this target is reachable (avoid uncollected targets)
            remaining_targets.discard(target['pos'])
            avoid_here = (mod_cells | remaining_targets)
            for wp_pos, _ in all_waypoints[len(all_waypoints)-1:]:  # just check target
                p = bfs_path(check_pos, wp_pos, walls, avoid_here - {wp_pos})
                if p is None:
                    valid = False
                    break
                check_pos = wp_pos

            if not valid:
                break

            temp_s = target['shape']
            temp_c = target['color']
            temp_r = target['rot']
            check_pos = target['pos']

        if valid:
            best_order = perm
            best_waypoints = all_waypoints
            break

    if best_waypoints is None:
        # Fallback: use original order
        all_waypoints = []
        temp_s, temp_c, temp_r = cur_s, cur_c, cur_r
        for ti, target in enumerate(targets):
            ds = (target['shape'] - temp_s) % NUM_SHAPES
            dc = (target['color'] - temp_c) % NUM_COLORS
            dr = (target['rot'] - temp_r) % NUM_ROTS
            if ds > 0 and s_mods:
                all_waypoints.append((s_mods[0], ds - 1))
            if dc > 0 and c_mods:
                all_waypoints.append((c_mods[0], dc - 1))
            if dr > 0 and r_mods:
                all_waypoints.append((r_mods[0], dr - 1))
            all_waypoints.append((target['pos'], 0))
            temp_s = target['shape']
            temp_c = target['color']
            temp_r = target['rot']
        best_waypoints = all_waypoints

    if verbose: print(f'  Target order: {best_order}, waypoints: {[(w[0], w[1]) for w in best_waypoints]}')

    # Plan route with fuel management across ALL waypoints
    if fuel_active > 0:
        planned = plan_route_with_fuel(best_waypoints, pos, walls, fuel_cells, verbose)
        if planned is None:
            if verbose: print(f'    No viable route with fuel')
            return obs, False
    else:
        planned = best_waypoints

    # Execute the planned route, avoiding modifiers AND non-destination targets
    for wp_pos, bounces in planned:
        # Avoid modifier cells and target cells that aren't the current waypoint
        avoid = (mod_cells | target_cells) - {wp_pos}
        path = bfs_path(pos, wp_pos, walls, avoid)
        if path is None:
            # Fallback: try avoiding only modifiers
            path = bfs_path(pos, wp_pos, walls, mod_cells - {wp_pos})
        if path is None:
            # Final fallback: no avoidance
            path = bfs_path(pos, wp_pos, walls)
        if path is None:
            if verbose: print(f'    Cannot reach {wp_pos} from {pos}')
            return obs, False

        if verbose:
            is_fuel = wp_pos in fuel_cells
            label = "FUEL" if is_fuel else f"wp({bounces}b)"
            print(f'    {label} {wp_pos} ({len(path)} steps)')

        obs_r, won = exec_steps(env, path, start_score)
        if obs_r: obs = obs_r
        if won: return obs, True

        pos = wp_pos

        # Do bounces for modifier visits
        if bounces > 0:
            # Find bounce neighbor that doesn't have a modifier or target
            away_gid = None
            bad_cells = mod_cells | target_cells
            for gid, (dx, dy) in GAME_DIRS.items():
                npos = (wp_pos[0]+dx, wp_pos[1]+dy)
                if npos not in walls and npos not in bad_cells and 0 <= npos[0] < 60 and 0 <= npos[1] < 60:
                    away_gid = gid
                    break
            if away_gid is None:
                for gid, (dx, dy) in GAME_DIRS.items():
                    npos = (wp_pos[0]+dx, wp_pos[1]+dy)
                    if npos not in walls and 0 <= npos[0] < 60 and 0 <= npos[1] < 60:
                        away_gid = gid
                        break
            if away_gid:
                bounce_seq = [away_gid, REVERSE[away_gid]]
                for _ in range(bounces):
                    obs_r, won = exec_steps(env, bounce_seq, start_score)
                    if obs_r: obs = obs_r
                    if won: return obs, True

    return obs, False


def solve_ls20(verbose=False):
    import sys, os, warnings, logging
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    warnings.filterwarnings('ignore')
    logging.disable(logging.WARNING)
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from arc_agi import Arcade

    arcade = Arcade()
    env = arcade.make('ls20')
    obs = env.reset()
    g = env._game
    num_levels = len(g._levels)
    results = {}

    for level in range(num_levels):
        if g._state.name == 'WIN': break
        raw = np.array(obs.frame)
        if raw.ndim == 3 and raw.shape[0] > 1:
            obs = env.step(6, {'x': 0, 'y': 0})
        if verbose: print(f'\n=== Level {level} ===')
        obs, won = solve_level(env, obs, verbose=verbose)
        results[level] = {'solved': won}
        if verbose: print(f'L{level}: {"SOLVED" if won else "FAILED"}')
        if not won: break

    total = sum(1 for v in results.values() if v.get('solved'))
    if verbose: print(f'\nLS20: {total}/{num_levels}')
    return results


if __name__ == '__main__':
    solve_ls20(verbose=True)
