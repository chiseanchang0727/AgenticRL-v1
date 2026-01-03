#!/usr/bin/env bash
set -e

MODEL="unsloth/Qwen3-8B-unsloth-bnb-4bit"
API_KEY="thekey"
MAX_LEN=8000
MAX_SEQS=100

uv run -m vllm.entrypoints.openai.api_server \
  --model "$MODEL" \
  --dtype half \
  --gpu_memory_utilization 0.8 \
  --max_model_len "$MAX_LEN" \
  --api_key "$API_KEY" \
  --max_num_seqs "$MAX_SEQS" \
  --reasoning-parser qwen3 \
  --enable-auto-tool-choice \
  --tool-call-parser hermes
