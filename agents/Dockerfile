FROM python:3.12-slim

WORKDIR /app

# Python deps
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Solver code — full module set (updated Mar 16)
COPY arc_agent_v05.py .
COPY arc_eyes.py .
COPY arc_recorder.py .
COPY arc_pilot.py .
COPY arc_hud.py .
COPY arc_reason.py .
COPY solver_memory.py .
COPY gundam.py .
COPY imagination_framework.py .
COPY common_sense_bus.py .
COPY dopamine.py .
COPY vc33_solver.py .
COPY ls20_solver.py .
COPY build_pattern_library.py .
COPY grid_transform.py .
COPY arc_action_predictor_lite.py .
COPY main.py .
COPY arc_vlm.py .
COPY arc_vlm_worker.py .
COPY game_knowledge.json .

# Per-game learned memory (cross-session knowledge persistence)
COPY arc_memory.json .
COPY arc_memory_ft09.json .
COPY arc_memory_ls20.json .
COPY arc_memory_vc33_fixed.json .
COPY gundam_memory.json .

# Pre-trained memory (optional — cross-game learning from synthetic games)
# Pre-trained memory — copy what exists (glob patterns need at least one match)
COPY arc_eyes_memory.json /app/arc_eyes_memory.json
# Solver memory DB — pre-trained from 281 game sessions
COPY solver_memory.db /app/solver_memory.db

# Optional: arc_action_predictor.py (CNN, requires PyTorch ~2GB)
# Not included by default — add torch to requirements.txt + COPY if needed

# Competition server port
EXPOSE 8001

# API key injected at runtime: docker run -e ANTHROPIC_API_KEY=...
# Without it, solver degrades to ARC_NO_LLM algorithmic-only mode

# Competition budget: $1K total across all games (150+ environments)
# Default ceiling $5/game — override with -e ARC_COST_CEILING=X at runtime
# Text-only mode (no VLM in offline competition)
ENV ARC_COST_CEILING=5.00
ENV ARC_TEXT_ONLY=1
ENV ARC_PILOT_MODEL_FAST=claude-haiku-4-5-20251001
ENV ARC_PILOT_MODEL=claude-sonnet-4-20250514
ENV OPERATION_MODE=competition

# Entry point — ARC-AGI-3 submission format
# Server mode: listens on port 8001 for competition runner
CMD ["python", "main.py"]
