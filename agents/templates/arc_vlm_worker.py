"""VLM Worker Process — runs in .vlm_venv with transformers 4.x.

Protocol: JSON lines over stdin/stdout.
- Sends {"status":"ready"} on startup
- Receives {"task":"<CAPTION>", "image_path":"/tmp/xxx.png", ...}
- Responds {"text":"...", "elapsed":0.5}
- Quit with {"task":"quit"}

This file runs in the .vlm_venv (transformers 4.46.3) to avoid
compatibility issues with Florence-2's custom code and transformers 5.x.
"""

import json
import sys
import time
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# Load model at startup
import torch
from transformers import AutoProcessor, AutoModelForCausalLM
from PIL import Image

MODEL_NAME = "microsoft/Florence-2-large"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def load_model():
    """Load Florence-2 model and processor. Falls back to CPU if CUDA OOM."""
    t0 = time.time()
    device = DEVICE

    processor = AutoProcessor.from_pretrained(MODEL_NAME, trust_remote_code=True)

    try:
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME, trust_remote_code=True,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        ).to(device)
    except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
        if device == "cuda":
            sys.stderr.write(f"CUDA OOM — falling back to CPU: {e}\n")
            sys.stderr.flush()
            torch.cuda.empty_cache()
            device = "cpu"
            model = AutoModelForCausalLM.from_pretrained(
                MODEL_NAME, trust_remote_code=True,
                torch_dtype=torch.float32,
            ).to(device)
        else:
            raise
    model.eval()

    elapsed = time.time() - t0
    sys.stderr.write(f"Florence-2 loaded in {elapsed:.1f}s on {device}\n")
    sys.stderr.flush()

    return model, processor


def run_task(model, processor, task: str, image_path: str,
             text_input: str = "", max_tokens: int = 256) -> str:
    """Run a Florence-2 task on an image file."""
    img = Image.open(image_path).convert("RGB")

    prompt = task + (f" {text_input}" if text_input else "")
    inputs = processor(text=prompt, images=img, return_tensors="pt")
    model_device = next(model.parameters()).device
    inputs = {k: v.to(model_device) if hasattr(v, 'to') else v for k, v in inputs.items()}

    # Cast inputs to model dtype (float16 on GPU)
    model_dtype = next(model.parameters()).dtype
    for k, v in inputs.items():
        if hasattr(v, 'dtype') and v.dtype == torch.float32 and model_dtype == torch.float16:
            inputs[k] = v.half()

    with torch.no_grad():
        generated = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            num_beams=3,
        )

    result = processor.batch_decode(generated, skip_special_tokens=False)[0]

    # Post-process
    parsed = processor.post_process_generation(result, task=task, image_size=img.size)

    # Extract text
    if isinstance(parsed, dict):
        for key in [task, '<MORE_DETAILED_CAPTION>', '<DETAILED_CAPTION>',
                    '<CAPTION>', '<OD>', '<OCR>']:
            if key in parsed:
                val = parsed[key]
                if isinstance(val, str):
                    return val
                return str(val)
        return str(parsed)
    return str(parsed)


def main():
    # Load model
    model, processor = load_model()

    # Signal ready
    sys.stdout.write('{"status":"ready"}\n')
    sys.stdout.flush()

    # Process commands
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue

        try:
            cmd = json.loads(line)
        except json.JSONDecodeError as e:
            sys.stdout.write(json.dumps({"error": f"Invalid JSON: {e}"}) + '\n')
            sys.stdout.flush()
            continue

        task = cmd.get('task', '')
        if task == 'quit':
            break

        image_path = cmd.get('image_path', '')
        text_input = cmd.get('text_input', '')
        max_tokens = cmd.get('max_tokens', 256)

        try:
            t0 = time.time()
            text = run_task(model, processor, task, image_path, text_input, max_tokens)
            elapsed = time.time() - t0
            response = {"text": text, "elapsed": round(elapsed, 3)}
        except Exception as e:
            response = {"error": str(e)}

        sys.stdout.write(json.dumps(response) + '\n')
        sys.stdout.flush()


if __name__ == "__main__":
    main()
