"""VLM Perception for ARC Games — The Right Hemisphere.

Separates perception (seeing) from reasoning (thinking).
VLM runs locally on GPU — zero API cost.
LLM receives words, not pixels.

Architecture:
  Game frame (64x64) → VLM (Florence-2, local) → natural language scene description
                     → grid_transform (local)   → structured object/causal analysis
                     → Combined                  → ONE LLM reasoning call

The VLM runs in a separate venv (.vlm_venv) with compatible transformers 4.x
because Florence-2's custom code is incompatible with transformers 5.x.
Communication is via subprocess + JSON over stdin/stdout.

Usage:
    scene = vlm_describe(frame)      # "I see a grid puzzle with colored cells..."
    objects = vlm_detect(frame)      # structured object detection
    full = vlm_perceive(frame)       # combined description for pilot prompt
"""

import json
import logging
import os
import subprocess
import sys
import tempfile
import time
import numpy as np

# Native 64x64 works better than upscaled for pixel art (less interpolation artifacts)
DEFAULT_UPSCALE = int(os.environ.get('ARC_VLM_UPSCALE', '1'))

log = logging.getLogger("arc_vlm")

# VLM backend selection: 'florence' (default) or 'qwen'
VLM_BACKEND = os.environ.get('ARC_VLM_BACKEND', 'florence')

# Path to the VLM venv python and worker
if VLM_BACKEND == 'qwen':
    VLM_PYTHON = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'venv', 'bin', 'python')
    VLM_WORKER = os.path.join(os.path.dirname(__file__), 'arc_qwen_vl_worker.py')
else:
    VLM_PYTHON = os.path.join(os.path.dirname(os.path.dirname(__file__)), '.vlm_venv', 'bin', 'python')
    VLM_WORKER = os.path.join(os.path.dirname(__file__), 'arc_vlm_worker.py')

# Persistent worker process
_worker_proc = None


# ─── ARC Palette (shared with worker) ───────────────────────────────────
ARC_PALETTE = np.array([
    [0, 0, 0],        # 0: black
    [0, 116, 217],    # 1: blue
    [255, 65, 54],    # 2: red
    [46, 204, 64],    # 3: green
    [255, 220, 0],    # 4: yellow
    [170, 170, 170],  # 5: gray
    [240, 18, 190],   # 6: magenta
    [255, 133, 27],   # 7: orange
    [127, 219, 255],  # 8: light blue
    [135, 12, 37],    # 9: dark red/maroon
    [255, 255, 255],  # 10: white
    [0, 255, 255],    # 11: cyan
    [128, 0, 128],    # 12: purple
    [0, 128, 0],      # 13: dark green
    [128, 128, 0],    # 14: olive
    [0, 0, 128],      # 15: navy
], dtype=np.uint8)


def _frame_to_labeled_png(frame: np.ndarray, cell_size: int = 12) -> bytes:
    """Render frame as labeled grid — each game pixel becomes a colored cell with
    its color index drawn inside. Gives Florence a 'diagram' instead of raw pixels.

    Peter's insight: 'what if you give Florence the ASCII art?'
    This is that idea rendered as an image Florence can actually parse.
    """
    from PIL import Image, ImageDraw, ImageFont
    import io

    f2d = np.squeeze(np.array(frame))
    if f2d.ndim == 1:
        f2d = f2d.reshape(1, -1)
    if f2d.ndim == 3 and f2d.shape[0] <= 4:
        f2d = f2d[0]  # take first channel as index
    if f2d.ndim == 3:
        # RGB frame — can't label with indices, fall back to normal
        return _frame_to_png_bytes(frame, upscale=4)

    h, w = f2d.shape
    img = Image.new('RGB', (w * cell_size, h * cell_size), (40, 40, 40))
    draw = ImageDraw.Draw(img)

    # Use default font (small, fits in cells)
    try:
        font = ImageFont.load_default()
    except Exception:
        font = None

    indices = np.clip(f2d, 0, 15).astype(int)
    for y in range(h):
        for x in range(w):
            c = int(indices[y, x])
            rgb = tuple(ARC_PALETTE[c])
            x0, y0 = x * cell_size, y * cell_size
            x1, y1 = x0 + cell_size - 1, y0 + cell_size - 1
            draw.rectangle([x0, y0, x1, y1], fill=rgb)
            # Draw cell border (thin gray)
            if cell_size >= 8:
                draw.rectangle([x0, y0, x1, y1], outline=(60, 60, 60))
            # Draw color index in contrasting text
            if cell_size >= 8:
                # Choose white or black text based on luminance
                lum = 0.299 * rgb[0] + 0.587 * rgb[1] + 0.114 * rgb[2]
                txt_color = (0, 0, 0) if lum > 128 else (255, 255, 255)
                label = str(c) if c <= 9 else chr(ord('A') + c - 10)
                tx = x0 + cell_size // 2 - 3
                ty = y0 + cell_size // 2 - 5
                draw.text((tx, ty), label, fill=txt_color, font=font)

    buf = io.BytesIO()
    img.save(buf, format='PNG')
    return buf.getvalue()


def _frame_to_png_bytes(frame: np.ndarray, upscale: int = DEFAULT_UPSCALE) -> bytes:
    """Convert ARC game frame to PNG bytes for VLM."""
    from PIL import Image
    import io

    frame = np.squeeze(np.array(frame))
    if frame.ndim == 1:
        frame = frame.reshape(1, -1)

    if frame.ndim == 2:
        indices = np.clip(frame, 0, 15).astype(int)
        rgb = ARC_PALETTE[indices]
    elif frame.ndim == 3 and frame.shape[2] == 3:
        rgb = frame.astype(np.uint8)
    elif frame.ndim == 3 and frame.shape[0] <= 4:
        rgb = np.transpose(frame[:3], (1, 2, 0)).astype(np.uint8)
    else:
        rgb = np.stack([frame] * 3, axis=-1).astype(np.uint8)

    img = Image.fromarray(rgb)
    if upscale > 1:
        img = img.resize((img.width * upscale, img.height * upscale), Image.NEAREST)

    buf = io.BytesIO()
    img.save(buf, format='PNG')
    return buf.getvalue()


def _ensure_worker():
    """Start the VLM worker process if not running."""
    global _worker_proc
    if _worker_proc is not None and _worker_proc.poll() is None:
        return  # still running

    if not os.path.exists(VLM_PYTHON):
        raise RuntimeError(
            f"VLM venv not found at {VLM_PYTHON}. "
            "Run: python3 -m venv .vlm_venv && .vlm_venv/bin/pip install 'transformers==4.46.3' torch einops timm pillow numpy"
        )

    log.info("Starting VLM worker process...")
    _worker_proc = subprocess.Popen(
        [VLM_PYTHON, VLM_WORKER],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        cwd=os.path.dirname(__file__),
    )
    # Wait for ready signal
    ready = _worker_proc.stdout.readline().decode().strip()
    if ready != '{"status":"ready"}':
        err = _worker_proc.stderr.read().decode()
        raise RuntimeError(f"VLM worker failed to start: {ready}\n{err}")
    log.info("VLM worker ready")


def _call_worker(command: dict) -> dict:
    """Send a command to the VLM worker and get response."""
    _ensure_worker()

    # Write command as JSON line
    line = json.dumps(command) + '\n'
    _worker_proc.stdin.write(line.encode())
    _worker_proc.stdin.flush()

    # Read response
    response_line = _worker_proc.stdout.readline().decode().strip()
    if not response_line:
        err = _worker_proc.stderr.read().decode()
        raise RuntimeError(f"VLM worker died: {err}")

    return json.loads(response_line)


def _call_with_frame(task: str, frame: np.ndarray, upscale: int = DEFAULT_UPSCALE,
                     text_input: str = "", max_tokens: int = 256) -> str:
    """Save frame as temp PNG, send to worker, return text result."""
    png_bytes = _frame_to_png_bytes(frame, upscale=upscale)

    # Write to temp file (worker reads it)
    with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as f:
        f.write(png_bytes)
        tmp_path = f.name

    try:
        if VLM_BACKEND == 'qwen':
            # Qwen uses natural language prompts, not Florence task tokens
            _task_to_prompt = {
                '<MORE_DETAILED_CAPTION>': 'Describe this game screenshot in detail. What objects, colors, and spatial arrangements do you see?',
                '<DETAILED_CAPTION>': 'Describe what you see in this image.',
                '<CAPTION>': 'What is in this image?',
            }
            prompt = _task_to_prompt.get(task, text_input or 'Describe this image.')
            if text_input and task in _task_to_prompt:
                prompt = f"{prompt} {text_input}"
            result = _call_worker({
                'task': 'describe',
                'image_paths': [tmp_path],
                'prompt': prompt,
                'max_tokens': max_tokens,
            })
        else:
            result = _call_worker({
                'task': task,
                'image_path': tmp_path,
                'text_input': text_input,
                'max_tokens': max_tokens,
            })
        if result.get('error'):
            log.error(f"VLM error: {result['error']}")
            return f"(VLM error: {result['error']})"
        return result.get('text', '')
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass


def perceive_structured(frame: np.ndarray, upscale: int = DEFAULT_UPSCALE) -> dict:
    """Structured perception — returns parsed objects with game-pixel coordinates.

    Returns:
        {
            'caption': str,          # scene description
            'objects': [             # detected objects with game-pixel coords
                {'label': str, 'bbox_game': (x1, y1, x2, y2), 'center': (cx, cy)},
                ...
            ],
            'click_targets': [(x, y), ...],  # centers of detected objects in game pixels
            'text': str,             # OCR text
            'genre_hint': str,       # inferred genre from caption keywords
        }
    """
    t0 = time.time()
    h, w = frame.shape[0], frame.shape[1] if frame.ndim >= 2 else 64

    result = {
        'caption': '',
        'objects': [],
        'click_targets': [],
        'text': '',
        'genre_hint': 'unknown',
    }

    # 1. Scene description (both raw and labeled-grid views)
    result['caption'] = _call_with_frame('<MORE_DETAILED_CAPTION>', frame, upscale=upscale) or ''
    # Also get labeled-grid perception for better structure understanding
    try:
        labeled_desc = perceive_labeled(frame, max_tokens=128)
        if labeled_desc and '(VLM error' not in labeled_desc:
            result['caption'] += f"\n[LABELED GRID VIEW]: {labeled_desc}"
    except Exception:
        pass  # labeled view is a bonus, not essential

    # 2. Object detection — skipped for ARC pixel art (always returns empty)
    # Florence OD is designed for natural images, not 64x64 pixel art.
    # Re-enable if we ever get frames with natural-image-like content.

    # 3. OCR — skipped for ARC pixel art (returns garbage)
    # These are procedural pixel art games, not text-based interfaces.

    # 4. Genre hint from caption keywords
    cap_lower = result['caption'].lower()
    genre_keywords = {
        'maze': ['maze', 'corridor', 'path', 'wall'],
        'toggle_puzzle': ['toggle', 'switch', 'light', 'grid puzzle', 'clicking'],
        'matching': ['match', 'pair', 'memory', 'same color'],
        'chase': ['chase', 'avoid', 'enemy', 'catch', 'flee'],
        'pattern_copy': ['pattern', 'copy', 'replicate', 'mirror'],
        'sorting': ['sort', 'order', 'arrange', 'sequence'],
        'painting': ['paint', 'color', 'fill', 'draw'],
    }
    for genre, keywords in genre_keywords.items():
        if any(kw in cap_lower for kw in keywords):
            result['genre_hint'] = genre
            break

    elapsed = time.time() - t0
    log.info(f"VLM perceive_structured: {elapsed:.2f}s, {len(result['objects'])} objects, genre={result['genre_hint']}")
    return result


def describe(frame: np.ndarray, upscale: int = DEFAULT_UPSCALE) -> str:
    """Get a natural language description of a game frame."""
    return _call_with_frame('<MORE_DETAILED_CAPTION>', frame, upscale=upscale)


def detect(frame: np.ndarray, upscale: int = DEFAULT_UPSCALE) -> str:
    """Detect objects with bounding boxes."""
    return _call_with_frame('<OD>', frame, upscale=upscale)


def caption_brief(frame: np.ndarray, upscale: int = DEFAULT_UPSCALE) -> str:
    """Short caption — one sentence summary."""
    return _call_with_frame('<CAPTION>', frame, upscale=upscale)


def read_text(frame: np.ndarray, upscale: int = DEFAULT_UPSCALE) -> str:
    """OCR — read any text visible in the frame."""
    return _call_with_frame('<OCR>', frame, upscale=upscale)


def ground(frame: np.ndarray, query: str, upscale: int = DEFAULT_UPSCALE) -> str:
    """Visual grounding — find where something is."""
    return _call_with_frame('<CAPTION_TO_PHRASE_GROUNDING>', frame,
                            upscale=upscale, text_input=query)


def florence_diff(old: dict, new: dict) -> str:
    """Compare two perceive_structured() results. Return human-readable diff."""
    changes = []
    if old.get('caption', '') != new.get('caption', ''):
        changes.append(f"Scene: '{old.get('caption','')[:60]}' → '{new.get('caption','')[:60]}'")
    old_objs = set(o['label'] for o in old.get('objects', []))
    new_objs = set(o['label'] for o in new.get('objects', []))
    appeared = new_objs - old_objs
    disappeared = old_objs - new_objs
    if appeared:
        changes.append(f"Appeared: {', '.join(appeared)}")
    if disappeared:
        changes.append(f"Disappeared: {', '.join(disappeared)}")
    if old.get('text', '') != new.get('text', ''):
        changes.append(f"Text: '{old.get('text','')}' → '{new.get('text','')}'")
    return "; ".join(changes) if changes else "No visible change detected"


def perceive(frame: np.ndarray, upscale: int = DEFAULT_UPSCALE, include_ocr: bool = True) -> str:
    """Full perception pass — combined scene + objects + OCR.

    This is the main entry point. Call once per turn.
    Cost: $0 (local GPU). Time: ~0.5-1.5s on Quadro RTX 3000.
    """
    t0 = time.time()
    sections = []

    # 1. Detailed scene
    caption = _call_with_frame('<MORE_DETAILED_CAPTION>', frame, upscale=upscale)
    sections.append(f"SCENE: {caption}")

    # 2. Object detection
    objects = _call_with_frame('<OD>', frame, upscale=upscale)
    if objects and objects not in ('{}', '', '[]'):
        sections.append(f"DETECTED OBJECTS: {objects}")

    # 3. OCR
    if include_ocr:
        text = _call_with_frame('<OCR>', frame, upscale=upscale)
        if text and text.strip() and text not in ('{}', ''):
            sections.append(f"TEXT ON SCREEN: {text}")

    elapsed = time.time() - t0
    log.info(f"VLM perceive: {elapsed:.2f}s total")

    return "\n".join(sections)


def perceive_labeled(frame: np.ndarray, max_tokens: int = 256) -> str:
    """Perceive a labeled-grid rendering — Florence reads the diagram.

    Instead of raw pixel art, Florence sees each game pixel as a labeled cell
    with its color index drawn inside. Better for structure recognition.
    """
    _ensure_worker()
    import tempfile
    png_bytes = _frame_to_labeled_png(frame, cell_size=12)
    with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as f:
        f.write(png_bytes)
        tmp_path = f.name
    try:
        result = _call_worker({
            'task': '<MORE_DETAILED_CAPTION>',
            'image_path': tmp_path,
            'text_input': '',
            'max_tokens': max_tokens,
        })
        if result.get('error'):
            return f"(VLM error: {result['error']})"
        return result.get('text', '')
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass


def perceive_diff(frame_before: np.ndarray, frame_after: np.ndarray,
                  action_desc: str = "", upscale: int = DEFAULT_UPSCALE) -> str:
    """Describe what changed between two frames in words."""
    before_desc = _call_with_frame('<MORE_DETAILED_CAPTION>', frame_before,
                                    upscale=upscale, max_tokens=128)
    after_desc = _call_with_frame('<MORE_DETAILED_CAPTION>', frame_after,
                                   upscale=upscale, max_tokens=128)

    result = f"BEFORE{' (' + action_desc + ')' if action_desc else ''}: {before_desc}\n"
    result += f"AFTER: {after_desc}"
    return result


def shutdown():
    """Cleanly shut down the VLM worker."""
    global _worker_proc
    if _worker_proc is not None and _worker_proc.poll() is None:
        try:
            _worker_proc.stdin.write(b'{"task":"quit"}\n')
            _worker_proc.stdin.flush()
            _worker_proc.wait(timeout=5)
        except Exception:
            _worker_proc.kill()
        _worker_proc = None


# ─── Standalone test ─────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO)

    print("ARC VLM Perception Test")
    print("=" * 40)

    # Create a test frame — 8x8 logical grid with colored cells
    frame = np.zeros((64, 64), dtype=np.uint8)
    for r in range(8):
        for c in range(8):
            color = ((r + c) % 4) + 1
            frame[r*8:(r+1)*8, c*8:(c+1)*8] = color
    frame[28:36, 28:36] = 9  # "player" in center
    frame[0:8, 56:64] = 3    # "target" in corner

    print("\nTest 1: Brief caption")
    t0 = time.time()
    cap = caption_brief(frame)
    print(f"  ({time.time()-t0:.2f}s) {cap}")

    print("\nTest 2: Detailed description")
    t0 = time.time()
    desc = describe(frame)
    print(f"  ({time.time()-t0:.2f}s) {desc}")

    print("\nTest 3: Full perception")
    t0 = time.time()
    full = perceive(frame)
    print(f"  ({time.time()-t0:.2f}s)")
    print(full)

    if len(sys.argv) > 1:
        from PIL import Image
        img = Image.open(sys.argv[1])
        arr = np.array(img)
        if arr.ndim == 3:
            arr = arr[:, :, 0]
        print(f"\nTest 4: Real frame ({sys.argv[1]})")
        t0 = time.time()
        full = perceive(arr)
        print(f"  ({time.time()-t0:.2f}s)")
        print(full)

    shutdown()
    print("\nDone. VLM perception is working.")
