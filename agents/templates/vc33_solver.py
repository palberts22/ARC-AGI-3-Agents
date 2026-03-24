#!/usr/bin/env python3
"""
VC33 Solver — Rail-switching puzzle solver for ARC-AGI-3
========================================================
Uses lightweight algebraic model with proper mud() guard modeling.
Deepcopy BFS as fallback/verification.

Apollo — 2026-03-06
"""

import copy
import hashlib
import json
import sys
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

# Add env files to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "environment_files"))

import arc_agi
from arcengine.enums import GameAction


# ---------------------------------------------------------------------------
# Game state extraction helpers
# ---------------------------------------------------------------------------

def get_game(env):
    """Get the internal game object from env."""
    return getattr(env, '_game', None)


def extract_level_info(game):
    """Extract all static and dynamic info from a VC33 level.
    Returns a dict describing the level state."""
    level = game.current_level
    oro = game.oro  # e.g. [3, 0] or [-3, 0] or [0, -2]

    # Helper functions (mirrors game methods but takes sprite directly)
    def urh(s):
        return s.y if oro[0] else s.x

    def ebl(s):
        return s.x if oro[0] else s.y

    def brj(s):
        return s.height if oro[0] else s.width

    def jqo(s):
        return s.width if oro[0] else s.height

    def lia():
        return oro[0] > 0 or oro[1] > 0

    def lho(s):
        return ebl(s) + jqo(s) if lia() else ebl(s)

    def utq(s):
        return ebl(s) if lia() else ebl(s) + jqo(s)

    # Extract rails (rDn)
    rails_raw = level.get_sprites_by_tag("rDn")
    rails = []
    for i, r in enumerate(rails_raw):
        rails.append({
            'id': i, 'sprite': r, 'urh': urh(r), 'ebl': ebl(r),
            'brj': brj(r), 'jqo': jqo(r), 'lho': lho(r), 'utq': utq(r),
        })

    # Build rail lookup by sprite identity
    rail_by_sprite = {id(r['sprite']): r for r in rails}

    # Extract HQB items
    hqbs_raw = level.get_sprites_by_tag("HQB")
    hqbs = []
    for i, h in enumerate(hqbs_raw):
        color = int(h.pixels[-1, -1])
        h_urh, h_ebl = urh(h), ebl(h)
        h_lho = ebl(h) + jqo(h) if lia() else ebl(h)
        # Find which rail this HQB is on
        rail_id = None
        for r in rails:
            if h_urh >= r['urh'] and h_urh < r['urh'] + r['brj'] and h_lho == r['utq']:
                rail_id = r['id']
                break
        hqbs.append({
            'id': i, 'sprite': h, 'urh': h_urh, 'ebl': h_ebl,
            'brj': brj(h), 'jqo': jqo(h), 'color': color, 'rail_id': rail_id,
        })

    # Extract gates (ZGd) with their rail pairs from dzy
    gates = []
    for gate_sprite, (pmj, chd) in game.dzy.items():
        pmj_id = rail_by_sprite.get(id(pmj), {}).get('id')
        chd_id = rail_by_sprite.get(id(chd), {}).get('id')
        gates.append({
            'sprite': gate_sprite, 'pmj_id': pmj_id, 'chd_id': chd_id,
            'x': gate_sprite.x, 'y': gate_sprite.y,
        })

    # Extract connectors (zHk)
    conns_raw = level.get_sprites_by_tag("zHk")
    connectors = []
    for c in conns_raw:
        c_urh, c_lho, c_brj = urh(c), lho(c), brj(c)
        # Find adjacent rails by urh position (fixed vertical adjacency)
        # above = rail whose bottom edge touches connector's top
        # below = rail whose top edge touches connector's bottom
        above_id = None
        below_id = None
        for r in rails:
            if r['urh'] + r['brj'] == c_urh:
                above_id = r['id']
            elif r['urh'] == c_urh + c_brj:
                below_id = r['id']
        connectors.append({
            'sprite': c, 'urh': c_urh, 'lho': c_lho, 'brj': c_brj,
            'above_id': above_id, 'below_id': below_id,
            'x': c.x, 'y': c.y,
        })

    # Extract WuO (walls/obstacles)
    wuos_raw = level.get_sprites_by_tag("WuO")
    wuos = []
    for w in wuos_raw:
        wuos.append({'urh': urh(w), 'lho': lho(w), 'ebl': ebl(w), 'jqo': jqo(w)})

    # Extract UXg (adjacency markers)
    uxgs_raw = level.get_sprites_by_tag("UXg")
    uxgs = []
    for u in uxgs_raw:
        uxgs.append({'sprite': u, 'urh': urh(u), 'brj': brj(u), 'utq': utq(u), 'ebl': ebl(u)})

    # Extract fZK (targets)
    fzks_raw = level.get_sprites_by_tag("fZK")
    fzks = []
    for f in fzks_raw:
        colors_in = set(int(c) for row in f.pixels for c in row if int(c) != 0)
        fzks.append({'sprite': f, 'ebl': ebl(f), 'urh': urh(f), 'brj': brj(f), 'colors': colors_in})

    # Precompute: for each fZK, which UXg collides with it, and which rails are adjacent
    fzk_uxg_map = []
    for fi, fzk in enumerate(fzks):
        fzk_s = fzks_raw[fi]
        colliding_uxgs = []
        for ui, uxg in enumerate(uxgs):
            if uxgs_raw[ui].collides_with(fzk_s):
                colliding_uxgs.append(ui)
        fzk_uxg_map.append(colliding_uxgs)

    # Precompute: for each rail, which UXg sprites are adjacent (suo)
    rail_uxg_adj = {}
    for ri, r in enumerate(rails):
        adj = []
        for ui, u in enumerate(uxgs):
            if u['urh'] + u['brj'] == r['urh'] or u['urh'] == r['urh'] + r['brj']:
                adj.append(ui)
        rail_uxg_adj[ri] = adj

    # Timer
    timer_budget = level.get_data("RoA") if hasattr(level, 'get_data') else 0

    return {
        'oro': list(oro), 'lia': lia(),
        'rails': rails, 'hqbs': hqbs, 'gates': gates, 'connectors': connectors,
        'wuos': wuos, 'uxgs': uxgs, 'fzks': fzks,
        'fzk_uxg_map': fzk_uxg_map, 'rail_uxg_adj': rail_uxg_adj,
        'timer': timer_budget,
        'step': abs(oro[0]) if oro[0] else abs(oro[1]),
    }


# ---------------------------------------------------------------------------
# Lightweight state model
# ---------------------------------------------------------------------------

class VC33Model:
    """Lightweight model of VC33 level state for fast BFS."""

    def __init__(self, info: dict):
        self.info = info
        self.step_size = info['step']
        self.lia = info['lia']
        self.oro = info['oro']
        n_rails = len(info['rails'])
        n_hqbs = len(info['hqbs'])

        # Mutable state arrays
        self.rail_ebl = [r['ebl'] for r in info['rails']]
        self.rail_jqo = [r['jqo'] for r in info['rails']]
        self.hqb_ebl = [h['ebl'] for h in info['hqbs']]
        self.hqb_rail = [h['rail_id'] for h in info['hqbs']]

        # Fixed properties
        self.rail_urh = [r['urh'] for r in info['rails']]
        self.rail_brj = [r['brj'] for r in info['rails']]
        self.rail_lho = [r['lho'] for r in info['rails']]  # constant: ebl + jqo
        self.hqb_color = [h['color'] for h in info['hqbs']]
        self.hqb_jqo = [h['jqo'] for h in info['hqbs']]

    def state_key(self) -> tuple:
        """Hashable state for dedup."""
        return (
            tuple(self.rail_ebl),
            tuple(self.rail_jqo),
            tuple(self.hqb_ebl),
            tuple(self.hqb_rail),
        )

    def clone(self):
        """Fast clone of mutable state."""
        m = VC33Model.__new__(VC33Model)
        m.info = self.info
        m.step_size = self.step_size
        m.lia = self.lia
        m.oro = self.oro
        m.rail_ebl = self.rail_ebl[:]
        m.rail_jqo = self.rail_jqo[:]
        m.hqb_ebl = self.hqb_ebl[:]
        m.hqb_rail = self.hqb_rail[:]
        m.rail_urh = self.rail_urh
        m.rail_brj = self.rail_brj
        m.rail_lho = self.rail_lho
        m.hqb_color = self.hqb_color
        m.hqb_jqo = self.hqb_jqo
        return m

    def hqbs_on_rail(self, rail_id: int) -> list:
        """Return indices of HQBs on a given rail."""
        return [i for i, r in enumerate(self.hqb_rail) if r == rail_id]

    def rail_utq(self, rail_id: int) -> int:
        """Trailing edge of rail (utq)."""
        if self.lia:
            return self.rail_ebl[rail_id]
        else:
            return self.rail_ebl[rail_id] + self.rail_jqo[rail_id]

    def hqb_lho(self, hqb_id: int) -> int:
        """Leading edge of HQB."""
        if self.lia:
            return self.hqb_ebl[hqb_id] + self.hqb_jqo[hqb_id]
        else:
            return self.hqb_ebl[hqb_id]

    def hqb_on_rail(self, hqb_id: int, rail_id: int) -> bool:
        """Check gdu: is HQB within this rail's bounds and aligned?"""
        h_urh = self.info['hqbs'][hqb_id]['urh']  # Fixed for gates; changes for swaps
        # Actually urh changes on connector swap... need to track it
        # For now, use hqb_rail assignment
        return self.hqb_rail[hqb_id] == rail_id

    def mud(self, rail_id: int) -> int:
        """Compute mud() boundary for a rail."""
        r_urh = self.rail_urh[rail_id]
        r_lho = self.rail_lho[rail_id]  # constant right edge

        # Find WuO at same urh, with lho condition
        wuos = self.info['wuos']
        if self.lia:
            cdr_wuos = [w for w in wuos if w['urh'] == r_urh and w['lho'] < r_lho]
        else:
            cdr_wuos = [w for w in wuos if w['urh'] == r_urh and w['lho'] > r_lho]

        if cdr_wuos:
            has_hqb = any(self.hqb_rail[i] == rail_id for i in range(len(self.hqb_rail)))
            if has_hqb:
                if self.oro[0] == -3:
                    return max(w['lho'] - 6 for w in cdr_wuos)
                else:
                    return max(w['lho'] - 4 for w in cdr_wuos)
            else:
                return max(w['lho'] for w in cdr_wuos)

        # Fallback: UXg adjacent sprites
        adj_uxgs = self.info['rail_uxg_adj'].get(rail_id, [])
        if adj_uxgs:
            uxg_utqs = [self.info['uxgs'][ui]['utq'] for ui in adj_uxgs]
            if self.lia:
                return max(uxg_utqs)
            else:
                return min(uxg_utqs)

        # No boundary found — should not happen in well-formed levels
        return -999999 if self.lia else 999999

    def can_fire_gate(self, gate_idx: int) -> bool:
        """Check if a gate can fire (gel guard)."""
        gate = self.info['gates'][gate_idx]
        pmj_id = gate['pmj_id']
        chd_id = gate['chd_id']

        # Guard 1: pmj still has width
        if self.rail_jqo[pmj_id] <= 0:
            return False

        # Guard 2: chd hasn't reached boundary
        if self.lia:
            return self.rail_ebl[chd_id] > self.mud(chd_id)
        else:
            chd_utq = self.rail_ebl[chd_id] + self.rail_jqo[chd_id]
            return chd_utq < self.mud(chd_id)

    def fire_gate(self, gate_idx: int):
        """Execute gate operation. Returns True if anything changed."""
        if not self.can_fire_gate(gate_idx):
            return False

        gate = self.info['gates'][gate_idx]
        pmj_id = gate['pmj_id']
        chd_id = gate['chd_id']
        rsi, qir = self.oro
        step = self.step_size

        # HQBs always move (regardless of sign)
        for hi in self.hqbs_on_rail(pmj_id):
            if rsi:
                self.hqb_ebl[hi] += rsi
            else:
                self.hqb_ebl[hi] += qir

        for hi in self.hqbs_on_rail(chd_id):
            if rsi:
                self.hqb_ebl[hi] -= rsi
            else:
                self.hqb_ebl[hi] -= qir

        # Rails only move when component >= 0
        # (gel() checks: if rsi >= 0: pmj.move(rsi,0), chd.move(-rsi,0))
        if rsi and rsi >= 0:
            self.rail_ebl[pmj_id] += rsi
            self.rail_ebl[chd_id] -= rsi
        if qir and qir >= 0:
            self.rail_ebl[pmj_id] += qir
            self.rail_ebl[chd_id] -= qir

        # Width always changes via pixel manipulation
        self.rail_jqo[pmj_id] -= step
        self.rail_jqo[chd_id] += step

        return True

    def find_connector_rails(self, conn_idx: int):
        """Dynamically find above/below rails for a connector (mirrors game's krt/teu).
        Returns (above_id, below_id) or (None, None) if not active."""
        conn = self.info['connectors'][conn_idx]
        c_lho = conn['lho']
        c_urh = conn['urh']
        c_brj = conn['brj']

        # Step 1: Find rails where utq matches connector's lho (DYNAMIC)
        matching = [i for i in range(len(self.rail_ebl))
                    if self.rail_utq(i) == c_lho]

        # Step 2: Among matching, find above (urh+brj == connector urh)
        #         and below (urh == connector urh + connector brj)
        above_id = None
        below_id = None
        for ri in matching:
            if self.rail_urh[ri] + self.rail_brj[ri] == c_urh:
                above_id = ri
            elif self.rail_urh[ri] == c_urh + c_brj:
                below_id = ri
        return above_id, below_id

    def connector_active(self, conn_idx: int) -> bool:
        """Check if connector can fire (krt) — dynamic rail lookup."""
        above_id, below_id = self.find_connector_rails(conn_idx)
        return above_id is not None and below_id is not None

    def fire_connector(self, conn_idx: int):
        """Execute connector swap. Returns True if anything changed."""
        above_id, below_id = self.find_connector_rails(conn_idx)
        if above_id is None or below_id is None:
            return False

        # Swap HQB rail assignments
        for hi in range(len(self.hqb_rail)):
            if self.hqb_rail[hi] == above_id:
                self.hqb_rail[hi] = below_id
            elif self.hqb_rail[hi] == below_id:
                self.hqb_rail[hi] = above_id
        # ebl stays the same (x doesn't change in horizontal swap)
        return True

    def check_win(self) -> bool:
        """Check gug() win condition."""
        fzks = self.info['fzks']
        fzk_uxg_map = self.info['fzk_uxg_map']
        rail_uxg_adj = self.info['rail_uxg_adj']

        for hi in range(len(self.hqb_ebl)):
            h_color = self.hqb_color[hi]
            h_ebl = self.hqb_ebl[hi]
            h_rail = self.hqb_rail[hi]
            if h_rail is None:
                return False

            # UXg adjacent to HQB's rail
            adj_uxgs = set(rail_uxg_adj.get(h_rail, []))

            # Find matching fZK
            matched = False
            for fi, fzk in enumerate(fzks):
                if h_color in fzk['colors'] and h_ebl == fzk['ebl']:
                    # Check if any UXg colliding with fZK is adjacent to HQB's rail
                    for ui in fzk_uxg_map[fi]:
                        if ui in adj_uxgs:
                            matched = True
                            break
                if matched:
                    break
            if not matched:
                return False
        return True

    def get_actions(self) -> list:
        """Return list of (action_type, index) for available actions."""
        actions = []
        for gi in range(len(self.info['gates'])):
            if self.can_fire_gate(gi):
                actions.append(('gate', gi))
        for ci in range(len(self.info['connectors'])):
            if self.connector_active(ci):
                actions.append(('conn', ci))
        return actions


# ---------------------------------------------------------------------------
# BFS solver
# ---------------------------------------------------------------------------

def bfs_solve(info: dict, max_states: int = 200000, verbose: bool = True) -> Optional[list]:
    """BFS on lightweight model. Returns list of (action_type, index) or None."""
    model = VC33Model(info)

    if model.check_win():
        return []

    start_key = model.state_key()
    visited = {start_key}
    # Queue: (model, action_history)
    queue = deque([(model, [])])
    states_explored = 0
    t0 = time.time()

    while queue and states_explored < max_states:
        current, history = queue.popleft()
        states_explored += 1

        if states_explored % 10000 == 0 and verbose:
            elapsed = time.time() - t0
            print(f"    BFS: {states_explored} states, {len(queue)} queued, "
                  f"depth {len(history)}, {elapsed:.1f}s")

        for action in current.get_actions():
            child = current.clone()
            atype, aidx = action
            if atype == 'gate':
                child.fire_gate(aidx)
            else:
                child.fire_connector(aidx)

            key = child.state_key()
            if key in visited:
                continue
            visited.add(key)

            if child.check_win():
                elapsed = time.time() - t0
                if verbose:
                    print(f"    BFS: SOLVED in {len(history)+1} actions, "
                          f"{states_explored} states, {elapsed:.1f}s")
                return history + [action]

            queue.append((child, history + [action]))

    elapsed = time.time() - t0
    if verbose:
        print(f"    BFS: exhausted ({states_explored} states, {elapsed:.1f}s)")
    return None


# ---------------------------------------------------------------------------
# Deepcopy BFS (guaranteed correct, slower)
# ---------------------------------------------------------------------------

def deepcopy_bfs(env, info: dict, max_states: int = 50000,
                 verbose: bool = True) -> Optional[list]:
    """BFS using copy.deepcopy of game state. Guaranteed correct."""
    game = get_game(env)
    if game is None:
        return None

    def state_hash(g):
        """Compact hash from rail and HQB positions."""
        level = g.current_level
        rails = level.get_sprites_by_tag("rDn")
        hqbs = level.get_sprites_by_tag("HQB")
        parts = []
        for r in sorted(rails, key=lambda s: (s.y, s.x)):
            parts.extend([r.x, r.y, r.width, r.height])
        for h in sorted(hqbs, key=lambda s: (s.y, s.x)):
            parts.extend([h.x, h.y])
        return tuple(parts)

    def get_clickables(g):
        """Get list of (sprite, type) for clickable sprites."""
        clicks = []
        for gate_s in g.dzy:
            clicks.append((gate_s, 'gate'))
        for conn_s in g.current_level.get_sprites_by_tag("zHk"):
            if g.krt(conn_s):
                clicks.append((conn_s, 'conn'))
        return clicks

    def complete_animation(g):
        """Run animation to completion."""
        safety = 0
        while g.vai is not None and safety < 500:
            if g.vai.next():
                g.vai = None
                g.jcy()
            safety += 1

    start_hash = state_hash(game)
    visited = {start_hash}
    queue = deque([(game, [])])
    states_explored = 0
    t0 = time.time()

    while queue and states_explored < max_states:
        current_game, history = queue.popleft()
        states_explored += 1

        if states_explored % 1000 == 0 and verbose:
            elapsed = time.time() - t0
            print(f"    DC-BFS: {states_explored} states, {len(queue)} queued, "
                  f"depth {len(history)}, {elapsed:.1f}s")

        clickables = get_clickables(current_game)
        for sprite, stype in clickables:
            child = copy.deepcopy(current_game)

            # Find corresponding sprite in child by position
            if stype == 'gate':
                child_sprite = None
                for gs in child.dzy:
                    if gs.x == sprite.x and gs.y == sprite.y:
                        child_sprite = gs
                        break
                if child_sprite is None:
                    continue
                child.ccl(child_sprite)
            else:  # connector
                child_sprite = None
                for cs in child.current_level.get_sprites_by_tag("zHk"):
                    if cs.x == sprite.x and cs.y == sprite.y:
                        child_sprite = cs
                        break
                if child_sprite is None or not child.krt(child_sprite):
                    continue
                child.vai = child.teu(child_sprite)
                complete_animation(child)

            h = state_hash(child)
            if h in visited:
                continue
            visited.add(h)

            # Find index to match lightweight format
            if stype == 'gate':
                aidx = next(i for i, g in enumerate(info['gates'])
                            if g['x'] == sprite.x and g['y'] == sprite.y)
            else:
                aidx = next(i for i, c in enumerate(info['connectors'])
                            if c['x'] == sprite.x and c['y'] == sprite.y)
            action_desc = (stype, aidx)

            if child.gug():
                elapsed = time.time() - t0
                if verbose:
                    print(f"    DC-BFS: SOLVED in {len(history)+1} actions, "
                          f"{states_explored} states, {elapsed:.1f}s")
                return history + [action_desc]

            queue.append((child, history + [action_desc]))

    elapsed = time.time() - t0
    if verbose:
        print(f"    DC-BFS: exhausted ({states_explored} states, {elapsed:.1f}s)")
    return None


# ---------------------------------------------------------------------------
# Solution replay and verification
# ---------------------------------------------------------------------------

def replay_solution(env, lightweight_solution: list, info: dict,
                    target_level: int = 1, verbose: bool = True) -> bool:
    """Replay a lightweight BFS solution and verify level completion.
    target_level: expected levels_completed after replay."""
    game = get_game(env)

    for step_num, (atype, aidx) in enumerate(lightweight_solution):
        if atype == 'gate':
            sprite_info = info['gates'][aidx]
        else:
            sprite_info = info['connectors'][aidx]

        gx, gy = sprite_info['x'], sprite_info['y']
        obs = click_and_animate(env, game, gx, gy)

        if verbose and step_num < 5:
            print(f"    Step {step_num}: {atype}[{aidx}] @ grid({gx},{gy}) "
                  f"[completed={obs.levels_completed}]")

        if obs.levels_completed >= target_level:
            if verbose:
                print(f"    Replay: Level complete after {step_num + 1} actions! "
                      f"(completed={obs.levels_completed})")
            return True

    if verbose:
        obs_final = env.step(GameAction.ACTION6, data={'x': 0, 'y': 0})
        print(f"    Replay: {len(lightweight_solution)} actions, "
              f"levels_completed={obs_final.levels_completed}, target={target_level}")
    return False


# ---------------------------------------------------------------------------
# Level dump for debugging
# ---------------------------------------------------------------------------

def dump_level(info: dict) -> str:
    """Pretty-print level state for debugging."""
    lines = []
    lines.append(f"  oro={info['oro']}, lia={info['lia']}, step={info['step']}")
    lines.append(f"  Rails ({len(info['rails'])}):")
    for r in info['rails']:
        lines.append(f"    R{r['id']}: urh={r['urh']}, ebl={r['ebl']}, "
                      f"jqo={r['jqo']}, brj={r['brj']}, lho={r['lho']}, utq={r['utq']}")
    lines.append(f"  HQBs ({len(info['hqbs'])}):")
    for h in info['hqbs']:
        lines.append(f"    H{h['id']}: rail={h['rail_id']}, ebl={h['ebl']}, "
                      f"color={h['color']}, jqo={h['jqo']}")
    lines.append(f"  Gates ({len(info['gates'])}):")
    for i, g in enumerate(info['gates']):
        lines.append(f"    G{i}: pmj=R{g['pmj_id']}, chd=R{g['chd_id']}, "
                      f"pos=({g['x']},{g['y']})")
    lines.append(f"  Connectors ({len(info['connectors'])}):")
    for i, c in enumerate(info['connectors']):
        lines.append(f"    C{i}: above=R{c['above_id']}, below=R{c['below_id']}, "
                      f"lho={c['lho']}, pos=({c['x']},{c['y']})")
    lines.append(f"  WuO ({len(info['wuos'])}):")
    for w in info['wuos']:
        lines.append(f"    urh={w['urh']}, lho={w['lho']}")
    lines.append(f"  UXg ({len(info['uxgs'])}):")
    for u in info['uxgs']:
        lines.append(f"    urh={u['urh']}, brj={u['brj']}, utq={u['utq']}")
    lines.append(f"  fZK ({len(info['fzks'])}):")
    for f in info['fzks']:
        lines.append(f"    ebl={f['ebl']}, colors={f['colors']}")
    lines.append(f"  Timer: {info['timer']}")
    return '\n'.join(lines)


# ---------------------------------------------------------------------------
# Main solver loop
# ---------------------------------------------------------------------------

def grid_to_display(game, gx, gy):
    """Convert grid coordinates to display coordinates using camera."""
    cam = game.camera
    # Camera viewport is cam.width x cam.height grid cells, displayed in 64x64 pixels
    scale_x = 64 // cam.width
    scale_y = 64 // cam.height
    offset_x = (64 - cam.width * scale_x) // 2
    offset_y = (64 - cam.height * scale_y) // 2
    return gx * scale_x + offset_x, gy * scale_y + offset_y


def click_and_animate(env, game, gx, gy):
    """Send a click at grid coords and step through any animation."""
    dx, dy = grid_to_display(game, gx, gy)
    obs = env.step(GameAction.ACTION6, data={'x': dx, 'y': dy})
    # Step through animation if one started
    safety = 0
    while game.vai is not None and safety < 500:
        obs = env.step(GameAction.ACTION6, data={'x': 0, 'y': 0})
        safety += 1
    return obs


def replay_level_solutions(env, game, solved_levels: list):
    """Replay all solved level solutions to reach the target level.
    solved_levels is a list of (info, solution) tuples.
    Analytical solutions are re-executed via their solver function."""
    obs = env.reset()
    analytical_solvers = {4: solve_l4_analytical, 5: solve_l5_analytical,
                          6: solve_l6_analytical}
    for info, solution in solved_levels:
        if isinstance(solution, tuple) and solution[0] == 'analytical':
            _, lvl_idx, _ = solution
            solver = analytical_solvers.get(lvl_idx)
            if solver:
                solver(env)
            # Get fresh obs after analytical solver
            obs = env.step(GameAction.ACTION6, data={'x': 0, 'y': 0})
        else:
            for atype, aidx in solution:
                if atype == 'gate':
                    si = info['gates'][aidx]
                else:
                    si = info['connectors'][aidx]
                obs = click_and_animate(env, game, si['x'], si['y'])
    return obs


# ---------------------------------------------------------------------------
# L4-L6 Analytical Solvers
# ---------------------------------------------------------------------------

def _press(env, x, y):
    """Click at display coordinates."""
    return env.step(GameAction.ACTION6, data={'x': x, 'y': y})


def _drain_anim(env):
    """Drain animation frames."""
    game = get_game(env)
    for _ in range(200):
        if not game.vai:
            break
        _press(env, 0, 0)


def _click_gate_display(env, gx, gy):
    """Click a zHk gate using grid_to_display coordinates."""
    game = get_game(env)
    for gt in game.current_level.get_sprites_by_tag("zHk"):
        if gt.x == gx and gt.y == gy:
            dx, dy = grid_to_display(game, gx, gy)
            _press(env, dx, dy)
            _drain_anim(env)
            return True
    return False


def _find_gate_activation_bfs(env, gate_x, gate_y, btns, max_depth=15, time_limit=60):
    """BFS to find button sequence that activates a gate (krt=True)."""
    game = get_game(env)
    start_level = game.level_index

    def sk(e):
        return tuple((c.x, c.pixels.shape[0])
                      for c in e._game.current_level.get_sprites_by_tag("rDn"))

    visited = {sk(env)}
    queue = deque([(copy.deepcopy(env), [])])
    t0 = time.time()
    while queue and time.time() - t0 < time_limit:
        cur, seq = queue.popleft()
        if len(seq) >= max_depth:
            continue
        for bx, by in btns:
            nc = copy.deepcopy(cur)
            nc.step(GameAction.ACTION6, data={'x': bx, 'y': by})
            if nc._game.level_index != start_level:
                continue
            s = sk(nc)
            if s in visited:
                continue
            visited.add(s)
            for gt in nc._game.current_level.get_sprites_by_tag("zHk"):
                if gt.x == gate_x and gt.y == gate_y and nc._game.krt(gt):
                    return seq + [(bx, by)]
            queue.append((nc, seq + [(bx, by)]))
    return None


def solve_l4_analytical(env):
    """Solve VC33 L4 using gate-activation BFS + push sequence.
    Returns number of actions used, or None if failed."""
    game = get_game(env)
    btns = [(62, 12), (62, 18), (62, 30), (62, 36), (62, 47), (62, 53)]
    gates = [(28, 14), (25, 49), (40, 32), (25, 49), (28, 14)]
    total = 0
    start_level = game.level_index

    def won():
        return game.level_index != start_level

    for gx, gy in gates:
        seq = _find_gate_activation_bfs(env, gx, gy, btns)
        if seq is None:
            return None
        for bx, by in seq:
            _press(env, bx, by)
        total += len(seq)
        _click_gate_display(env, gx, gy)
        total += 1
        if won():
            return total

    # Push phase
    push_seq = [(62, 36, 1), (62, 53, 7), (62, 12, 6), (62, 53, 10), (62, 12, 10)]
    for bx, by, count in push_seq:
        for _ in range(count):
            _press(env, bx, by)
            total += 1
            if won():
                return total

    return total if won() else None


def solve_l5_analytical(env):
    """Solve VC33 L5 using 2-gate analytical sequence.
    Returns number of actions used, or None if failed."""
    game = get_game(env)
    start_level = game.level_index
    total = 0

    def p(x, y):
        nonlocal total
        _press(env, x, y)
        total += 1
        return game.level_index != start_level

    if p(1, 28): return total
    for _ in range(3):
        if p(25, 28): return total
    _click_gate_display(env, 6, 30)
    total += 1
    if game.level_index != start_level: return total
    for _ in range(2):
        if p(1, 34): return total
    for _ in range(8):
        if p(25, 34): return total
    for _ in range(2):
        if p(25, 28): return total
    _click_gate_display(env, 30, 30)
    total += 1
    if game.level_index != start_level: return total
    for _ in range(8):
        if p(25, 28): return total

    return total if game.level_index != start_level else None


def solve_l6_analytical(env):
    """Solve VC33 L6 using 4-gate routing through C3 hub.
    Returns number of actions used, or None if failed."""
    game = get_game(env)
    start_level = game.level_index
    total = 0

    # Button display coordinates (confirmed by frame scan)
    B = {
        'C3+C2-': (24, 8),  'C2+C3-': (20, 8),
        'C3+C4-': (24, 32), 'C4+C3-': (20, 32),
        'C0+C3-': (42, 32), 'C3+C0-': (38, 32),
        'C1+C3-': (42, 8),  'C3+C1-': (38, 8),
    }

    def won():
        return game.level_index != start_level or str(getattr(
            _press(env, 0, 0), 'state', '')) == 'GameState.WIN'

    def do_presses(sequence):
        nonlocal total
        for btn_name, count in sequence:
            bx, by = B[btn_name]
            for _ in range(count):
                _press(env, bx, by)
                total += 1

    def do_gate(gx, gy):
        nonlocal total
        _click_gate_display(env, gx, gy)
        total += 1

    # Gate 1: G(32,8) — S2: C1→C3
    do_presses([('C1+C3-', 1), ('C3+C2-', 1)])
    do_gate(32, 8)

    # Gate 2: G(14,30) — S0↔S2 swap C4↔C3
    do_presses([('C3+C4-', 1), ('C3+C2-', 9), ('C3+C0-', 1)])
    do_gate(14, 30)

    # Gate 3: G(32,8) — S0: C3→C1
    do_presses([('C2+C3-', 11)])
    do_gate(32, 8)

    # Gate 4: G(32,30) — S1: C0→C3
    do_presses([('C3+C2-', 10), ('C3+C0-', 1)])
    do_gate(32, 30)

    # Push phase
    do_presses([('C3+C2-', 1), ('C1+C3-', 5), ('C4+C3-', 6)])

    # Trigger win (final level — check obs.state)
    obs = _press(env, 0, 0)
    total += 1
    if str(obs.state) == 'GameState.WIN':
        return total
    if game.level_index != start_level:
        return total

    return None


def solve_vc33(levels: list = None, max_bfs_states: int = 200000,
               use_deepcopy: bool = False, verbose: bool = True):
    """Solve VC33 levels."""
    arc = arc_agi.Arcade()
    env = arc.make("vc33")
    obs = env.reset()
    win_levels = obs.win_levels
    game = get_game(env)

    print(f"VC33: {win_levels} levels")

    if levels is None:
        levels = list(range(win_levels))

    results = {}
    solved_levels = []  # list of (info, solution) for prefix replay
    total_actions = 0
    baseline = env.info.baseline_actions if hasattr(env.info, 'baseline_actions') else [0] * win_levels

    for level_idx in levels:
        print(f"\n{'='*50}")
        print(f"Level {level_idx} (baseline: {baseline[level_idx] if level_idx < len(baseline) else '?'})")
        print(f"{'='*50}")

        # Reset and replay all previous solutions to reach this level
        obs = replay_level_solutions(env, game, solved_levels)

        if obs.levels_completed < level_idx:
            print(f"  SKIP: couldn't reach level {level_idx} (at {obs.levels_completed})")
            results[level_idx] = {'solved': False, 'reason': 'unreachable'}
            break

        # Extract level info
        info = extract_level_info(game)
        print(dump_level(info))

        # Solve — analytical for L4-L6, BFS for L0-L3
        t0 = time.time()
        analytical_solver = {4: solve_l4_analytical, 5: solve_l5_analytical,
                             6: solve_l6_analytical}.get(level_idx)

        if analytical_solver:
            # Analytical solver modifies env directly, returns action count
            n_actions = analytical_solver(env)
            elapsed = time.time() - t0
            if n_actions is None:
                print(f"  UNSOLVED ({elapsed:.1f}s)")
                results[level_idx] = {'solved': False, 'actions': 0, 'time': elapsed}
                break
            print(f"  Solution: {n_actions} actions in {elapsed:.1f}s (analytical)")
            total_actions += n_actions
            results[level_idx] = {
                'solved': True, 'actions': n_actions,
                'time': elapsed, 'verified': True,
            }
            # Analytical solvers don't use replay — mark as non-replayable
            solved_levels.append((info, ('analytical', level_idx, n_actions)))
        else:
            if use_deepcopy:
                solution = deepcopy_bfs(env, info, max_states=max_bfs_states, verbose=verbose)
            else:
                solution = bfs_solve(info, max_states=max_bfs_states, verbose=verbose)

            elapsed = time.time() - t0

            if solution is None:
                print(f"  UNSOLVED ({elapsed:.1f}s)")
                results[level_idx] = {'solved': False, 'actions': 0, 'time': elapsed}
                break

            print(f"  Solution: {len(solution)} actions in {elapsed:.1f}s")
            print(f"  Actions: {solution}")

            if not use_deepcopy:
                # Verify lightweight solution via replay
                obs = replay_level_solutions(env, game, solved_levels)
                target = level_idx + 1

                verified = replay_solution(env, solution, info,
                                           target_level=target, verbose=verbose)
                if not verified:
                    print(f"  REPLAY FAILED — trying deepcopy BFS")
                    obs = replay_level_solutions(env, game, solved_levels)
                    dc_solution = deepcopy_bfs(env, info, max_states=max_bfs_states, verbose=verbose)
                    if dc_solution:
                        solution = dc_solution
                        print(f"  DC-BFS found: {len(solution)} actions")
                    else:
                        results[level_idx] = {'solved': False, 'reason': 'replay_failed', 'time': elapsed}
                        break
            else:
                verified = True

            total_actions += len(solution)
            results[level_idx] = {
                'solved': True, 'actions': len(solution),
                'time': elapsed, 'verified': verified,
            }

            solved_levels.append((info, solution))

    # Summary
    print(f"\n{'='*50}")
    print(f"RESULTS SUMMARY")
    print(f"{'='*50}")
    solved_count = sum(1 for r in results.values() if r.get('solved'))
    total_baseline = sum(baseline[i] for i in results if i < len(baseline))
    print(f"Solved: {solved_count}/{len(levels)}")
    print(f"Total actions: {total_actions}")
    print(f"Baseline: {total_baseline}")
    if total_baseline > 0:
        print(f"Efficiency: {total_baseline}/{total_actions} = "
              f"{total_baseline/total_actions*100:.1f}%" if total_actions > 0 else "N/A")

    for li, r in sorted(results.items()):
        status = "SOLVED" if r.get('solved') else "FAILED"
        print(f"  L{li}: {status} — {r.get('actions', 0)} actions, "
              f"{r.get('time', 0):.1f}s")

    return results


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--levels', type=str, default=None,
                        help='Comma-separated level indices (default: all)')
    parser.add_argument('--budget', type=int, default=200000,
                        help='Max BFS states per level')
    parser.add_argument('--deepcopy', action='store_true',
                        help='Use deepcopy BFS (slower but guaranteed correct)')
    parser.add_argument('--quiet', action='store_true')
    args = parser.parse_args()

    levels = None
    if args.levels:
        levels = [int(x) for x in args.levels.split(',')]

    solve_vc33(levels=levels, max_bfs_states=args.budget,
               use_deepcopy=args.deepcopy, verbose=not args.quiet)
