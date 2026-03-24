"""Grid Transform — A Fourier Transform for LLMs.

Decomposes 2D grid worlds into relational components that LLMs process natively.
Spec: specs/grid_transform_spec.md (Hypatia + Flint, Apollo implements)

Layer 0: RAW PIXELS (existing)
Layer 1: ASCII MAP (existing: arc_eyes.frame_to_ascii)
Layer 2: OBJECT GRAPH (this module)
Layer 3: SYMMETRY DESCRIPTORS (this module)
Layer 4: CHANGE SIGNATURES (this module, Flint's design)
Layer 5: HYPOTHESIS MAP (this module, missile tracker)
"""
import hashlib
import json
import numpy as np
from dataclasses import dataclass, field
from typing import Optional

try:
    from arc_eyes import Sprite, detect_sprites
except ImportError:
    Sprite = None
    detect_sprites = None


# ─── Layer 2: Object Graph ───────────────────────────────────────────────

def classify_shape(sprite) -> str:
    """Classify sprite shape from bounding box fill ratio."""
    bw = sprite.bbox[3] - sprite.bbox[1] + 1
    bh = sprite.bbox[2] - sprite.bbox[0] + 1
    bbox_area = bw * bh
    if bbox_area == 0:
        return "DOT"
    fill = sprite.size / bbox_area
    if sprite.size <= 2:
        return "DOT"
    elif fill > 0.9:
        return f"RECT({bh}x{bw})"
    elif fill > 0.6:
        return f"BLOCK({bh}x{bw},{fill:.0%}fill)"
    elif bh <= 2 or bw <= 2:
        return f"LINE({max(bh,bw)})"
    else:
        return f"SPARSE({sprite.size}px)"


def compute_relations(sprites: list, max_pairs: int = 30) -> list:
    """Compute typed spatial relations between sprite pairs."""
    relations = []
    for i, a in enumerate(sprites):
        for j, b in enumerate(sprites):
            if i >= j:
                continue
            if len(relations) >= max_pairs:
                return relations

            dy = b.center[0] - a.center[0]
            dx = b.center[1] - a.center[1]
            dist = abs(dy) + abs(dx)

            direction = []
            if dy < -2: direction.append("UP")
            elif dy > 2: direction.append("DOWN")
            if dx < -2: direction.append("LEFT")
            elif dx > 2: direction.append("RIGHT")

            # Containment
            a_in_b = (a.bbox[0] >= b.bbox[0] and a.bbox[2] <= b.bbox[2] and
                      a.bbox[1] >= b.bbox[1] and a.bbox[3] <= b.bbox[3])
            b_in_a = (b.bbox[0] >= a.bbox[0] and b.bbox[2] <= a.bbox[2] and
                      b.bbox[1] >= a.bbox[1] and b.bbox[3] <= a.bbox[3])

            rel_type = "CONTAINS" if b_in_a else "INSIDE" if a_in_b else "NEAR" if dist < 15 else "FAR"
            relations.append((i, j, rel_type, '+'.join(direction) or "OVERLAP", int(dist)))
    return relations


def extract_object_graph(frame: np.ndarray, sprites: list,
                         player_color: int = -1, target_pos: tuple = (-1, -1),
                         walkable_colors: set = None, wall_colors: set = None) -> str:
    """Layer 2: 2D grid → relational text description.

    Gives the LLM structured understanding of WHAT is on screen
    and HOW objects relate to each other spatially.
    """
    if not sprites:
        return ""

    # Assign roles
    lines = ["OBJECTS:"]
    for i, s in enumerate(sprites[:15]):  # cap at 15 objects
        shape = classify_shape(s)
        role = "UNKNOWN"
        if s.color == player_color:
            role = "PLAYER"
        elif target_pos != (-1, -1) and abs(s.center[0] - target_pos[0]) < 5 and abs(s.center[1] - target_pos[1]) < 5:
            role = "TARGET"
        elif s.size > 500:
            role = "STRUCTURE"
        elif wall_colors and s.color in wall_colors:
            role = "WALL"
        elif walkable_colors and s.color in walkable_colors:
            role = "FLOOR"
        elif s.size < 20:
            role = "ITEM"
        lines.append(f"  #{i}: color={s.color}, shape={shape}, pos=({s.center[0]:.0f},{s.center[1]:.0f}), area={s.size}, role={role}")

    # Relations (only between small/medium objects — skip huge structures)
    interesting = [s for s in sprites[:15] if s.size < 500]
    if len(interesting) >= 2:
        rels = compute_relations(interesting, max_pairs=15)
        if rels:
            lines.append("\nRELATIONS:")
            for i, j, rtype, direction, dist in rels:
                lines.append(f"  #{i} {rtype} #{j}, direction={direction}, dist={dist}")

    # Topology — connected walkable regions via flood fill
    if walkable_colors and len(walkable_colors) > 0:
        f2d = frame[0] if frame.ndim == 3 else frame
        walkable_mask = np.isin(f2d, list(walkable_colors))
        # Simple region counting via flood fill
        visited = set()
        regions = []
        for y in range(0, f2d.shape[0], 3):  # sample every 3rd pixel for speed
            for x in range(0, f2d.shape[1], 3):
                if walkable_mask[y, x] and (y, x) not in visited:
                    # Quick flood fill on walkable
                    region = set()
                    stack = [(y, x)]
                    while stack and len(region) < 2000:
                        cy, cx = stack.pop()
                        if (cy, cx) in visited or cy < 0 or cy >= f2d.shape[0] or cx < 0 or cx >= f2d.shape[1]:
                            continue
                        if not walkable_mask[cy, cx]:
                            continue
                        visited.add((cy, cx))
                        region.add((cy, cx))
                        stack.extend([(cy-1, cx), (cy+1, cx), (cy, cx-1), (cy, cx+1)])
                    if len(region) >= 10:  # meaningful region
                        regions.append(region)

        if regions:
            lines.append(f"\nTOPOLOGY: {len(regions)} connected walkable region(s)")
            for ri, region in enumerate(regions[:5]):
                ys = [p[0] for p in region]
                xs = [p[1] for p in region]
                lines.append(f"  Region {ri}: rows {min(ys)}-{max(ys)}, cols {min(xs)}-{max(xs)}, ~{len(region)} cells")

    return "\n".join(lines)


# ─── Layer 3: Symmetry Descriptors ───────────────────────────────────────

def detect_symmetries(frame: np.ndarray) -> str:
    """Layer 3: detect grid symmetries. O(n^2) on 64x64 = trivial."""
    f2d = frame[0] if frame.ndim == 3 else frame
    h, w = f2d.shape
    parts = ["SYMMETRY:"]

    # Horizontal mirror
    flipped_h = np.fliplr(f2d)
    h_match = np.mean(f2d == flipped_h)
    if h_match > 0.85:
        parts.append(f"  Horizontal mirror: YES (axis at col {w//2}, {h_match:.0%} match)")

    # Vertical mirror
    flipped_v = np.flipud(f2d)
    v_match = np.mean(f2d == flipped_v)
    if v_match > 0.85:
        parts.append(f"  Vertical mirror: YES (axis at row {h//2}, {v_match:.0%} match)")

    # 90° rotation
    rotated = np.rot90(f2d)
    if rotated.shape == f2d.shape:
        r_match = np.mean(f2d == rotated)
        if r_match > 0.85:
            parts.append(f"  Rotational 90°: YES ({r_match:.0%} match)")

    # Translational period (small periods only)
    for period in range(2, min(17, w // 2)):
        if np.mean(f2d[:, :w-period] == f2d[:, period:]) > 0.9:
            parts.append(f"  X-period: {period}px")
            break
    for period in range(2, min(17, h // 2)):
        if np.mean(f2d[:h-period, :] == f2d[period:, :]) > 0.9:
            parts.append(f"  Y-period: {period}px")
            break

    if len(parts) == 1:
        parts.append("  None detected")
    return "\n".join(parts)


# ─── Layer 4: Change Signatures (Flint's design) ─────────────────────────

def classify_change_type(pixels_changed: int, moved_sprites: int,
                         appeared: bool, disappeared: bool) -> str:
    """Classify what kind of change occurred."""
    if pixels_changed == 0:
        return 'NONE'
    if moved_sprites > 0:
        return 'MOVE'
    if appeared and not disappeared:
        return 'APPEAR'
    if disappeared and not appeared:
        return 'DISAPPEAR'
    if appeared and disappeared:
        return 'TOGGLE'
    return 'CHANGE'


@dataclass
class CausalSignature:
    """A learned action→effect mapping with confidence tracking."""
    action_id: int
    change_type: str
    pixel_count: int
    movements: list = field(default_factory=list)  # [(color, size, dy, dx)]
    observations: int = 1
    last_level: int = 0

    @property
    def confidence(self) -> float:
        return 1.0 - (0.5 ** self.observations)

    def key(self) -> str:
        """Position-invariant key for dedup."""
        parts = [str(self.action_id), self.change_type]
        for m in sorted(self.movements):
            parts.append(f"{m[0]}:{m[1]}:{m[2]}:{m[3]}")
        return "|".join(parts)


class CausalLedger:
    """O(1) lookup of action→effect associations."""

    def __init__(self):
        self.by_action: dict[int, list[CausalSignature]] = {}
        self.seen_keys: set = set()

    def observe(self, action_id: int, pixels_changed: int, change_type: str,
                movements: list = None, level: int = 0):
        """Record an observation."""
        sig = CausalSignature(
            action_id=action_id,
            change_type=change_type,
            pixel_count=pixels_changed,
            movements=movements or [],
            last_level=level,
        )
        key = sig.key()

        # Check if we've seen this exact pattern before
        existing = self.by_action.get(action_id, [])
        for s in existing:
            if s.key() == key:
                s.observations += 1
                s.last_level = level
                return
        # New pattern
        self.by_action.setdefault(action_id, []).append(sig)

    def predict(self, action_id: int) -> Optional[CausalSignature]:
        """Most confident prediction for this action."""
        sigs = self.by_action.get(action_id, [])
        if not sigs:
            return None
        return max(sigs, key=lambda s: s.confidence)

    def should_exploit(self, n_actions: int, threshold: float = 0.75) -> bool:
        """Clausewitz culminating point — enough data to stop exploring?"""
        confident = sum(
            1 for a in range(n_actions)
            if self.predict(a) and self.predict(a).confidence >= 0.75
        )
        return confident / max(n_actions, 1) >= threshold

    def format_for_pilot(self) -> str:
        """Render causal knowledge for the pilot prompt."""
        lines = ["CAUSAL SIGNATURES (learned from observation):"]
        for aid in sorted(self.by_action.keys()):
            sigs = sorted(self.by_action[aid], key=lambda s: -s.confidence)
            for sig in sigs[:2]:
                desc = f"  action_{aid}: {sig.change_type}"
                if sig.movements:
                    for c, sz, dy, dx in sig.movements:
                        desc += f" (color {c} moves dy={dy},dx={dx})"
                desc += f" — conf={sig.confidence:.0%} ({sig.observations}x)"
                lines.append(desc)
        if len(lines) == 1:
            lines.append("  (no observations yet)")
        return "\n".join(lines)


# ─── Layer 5: Hypothesis Map (Missile Tracker) ───────────────────────────

def project_route(player_pos: tuple, target_pos: tuple,
                  blocked: set, action_map: dict,
                  max_legs: int = 8) -> str:
    """Layer 5: Project a hypothesis route from player to target.

    Not pathfinding — hypothesis generation. Projects a beam toward the
    target, bends around known walls, flags unknown territory.

    Args:
        player_pos: (y, x)
        target_pos: (y, x)
        blocked: set of (y, x, action_id) — known impassable
        action_map: {action_id: (dy, dx, name)}
    """
    if not action_map or target_pos == (-1, -1):
        return ""

    py, px = player_pos
    ty, tx = target_pos
    step = max(max(abs(dy), abs(dx)) for dy, dx, _ in action_map.values()) or 5

    # Build blocked lookup: (y, x) -> set of blocked action_ids
    blocked_at = {}
    for by, bx, aid in blocked:
        blocked_at.setdefault((by, bx), set()).add(aid)

    # Project beam: greedy toward target, bending around known blocks
    legs = []
    cy, cx = py, px
    visited_proj = {(cy, cx)}

    for _ in range(max_legs):
        if (cy, cx) == (ty, tx):
            break
        if abs(cy - ty) <= step and abs(cx - tx) <= step:
            legs.append(f"ARRIVE at target ({ty},{tx})")
            break

        # Which direction gets us closest to target?
        best_action = None
        best_dist = float('inf')
        best_name = ""
        blocked_here = blocked_at.get((cy, cx), set())

        for aid, (dy, dx, name) in action_map.items():
            ny, nx = cy + dy, cx + dx
            if (ny, nx) in visited_proj:
                continue
            dist = abs(ny - ty) + abs(nx - tx)

            if aid in blocked_here:
                continue  # known wall — skip

            if dist < best_dist:
                best_dist = dist
                best_action = aid
                best_name = name

        if best_action is None:
            legs.append("STUCK — all directions blocked or visited")
            break

        # Check if terrain is known or unknown
        ny, nx = cy + action_map[best_action][0], cx + action_map[best_action][1]
        is_known = any((ny, nx, a) in blocked or
                       (cy, cx) == player_pos  # starting position is known
                       for a in action_map)

        confidence = "confirmed" if (cy, cx) in {(by, bx) for by, bx, _ in blocked} else "unknown terrain"

        legs.append(f"{best_name} → ({ny},{nx}) [{confidence}]")
        visited_proj.add((ny, nx))
        cy, cx = ny, nx

    if not legs:
        return ""

    # Calculate overall confidence
    total = len(legs)
    confirmed = sum(1 for l in legs if "confirmed" in l)
    conf_pct = confirmed / max(total, 1)

    lines = [f"HYPOTHESIS ROUTE (confidence: {conf_pct:.0%} — {total - confirmed}/{total} legs unverified):"]
    for i, leg in enumerate(legs):
        lines.append(f"  Leg {i+1}: {leg}")
    lines.append(f"\n  RECOMMENDATION: Execute first {min(3, total)} legs, then re-observe.")
    return "\n".join(lines)


# ─── Compose: Full Transform Stack ───────────────────────────────────────

def grid_transform(frame: np.ndarray, sprites: list = None,
                   player_color: int = -1, target_pos: tuple = (-1, -1),
                   walkable_colors: set = None, wall_colors: set = None,
                   player_pos: tuple = (0, 0), blocked: set = None,
                   action_map: dict = None,
                   causal_ledger: CausalLedger = None,
                   include_symmetry: bool = False) -> str:
    """Full transform stack — call this from the Gundam pilot prompt builder.

    Returns formatted text sections for each active layer.
    """
    sections = []

    # Layer 2: Object graph
    if sprites:
        obj_graph = extract_object_graph(
            frame, sprites, player_color, target_pos,
            walkable_colors, wall_colors
        )
        if obj_graph:
            sections.append(obj_graph)

    # Layer 3: Symmetry (optional — costs ~50 tokens)
    if include_symmetry:
        sym = detect_symmetries(frame)
        if sym and "None detected" not in sym:
            sections.append(sym)

    # Layer 4: Causal signatures
    if causal_ledger and causal_ledger.by_action:
        sections.append(causal_ledger.format_for_pilot())

    # Layer 5: Hypothesis map
    if action_map and blocked is not None and target_pos != (-1, -1):
        route = project_route(player_pos, target_pos, blocked, action_map)
        if route:
            sections.append(route)

    return "\n\n".join(sections)
