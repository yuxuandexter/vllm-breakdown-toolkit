#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY="${PYTHON:-python}"
CACHE_DIR="$SCRIPT_DIR/cache"
mkdir -p "$CACHE_DIR"

# Configurable wait time between runs (seconds). Override by exporting WAIT_BETWEEN.
WAIT_BETWEEN="${WAIT_BETWEEN:-2}"


# # 1) Default config (explicit args)
# "$PY" "$SCRIPT_DIR/vllm_benchmark.py" \
#   --model-name Qwen/Qwen3-8B \
#   --warmup-iters 3 \
#   --duration-s 10.0 \
#   --fixed-iterations 1000 \
#   --dtype float16 \
#   --max-model-len 32768 \
#   --num-seq 32 \
#   --seq-len 4096 \
#   --post-exit-sleep 2 \
#   --gpu-memory-utilization 0.9 \
#   2>&1 | tee "$CACHE_DIR/bench_qwen3_8b_batch_32_seq_4096.log"

# echo "Waiting ${WAIT_BETWEEN}s before next run..." && sleep "$WAIT_BETWEEN"

# # 2) seq_len = 16348
# "$PY" "$SCRIPT_DIR/vllm_benchmark.py" \
#   --model-name Qwen/Qwen3-8B \
#   --warmup-iters 3 \
#   --duration-s 10.0 \
#   --fixed-iterations 1000 \
#   --dtype float16 \
#   --max-model-len 32768 \
#   --num-seq 32 \
#   --seq-len 16348 \
#   --post-exit-sleep 20 \
#   --gpu-memory-utilization 0.9 \
#   2>&1 | tee "$CACHE_DIR/bench_qwen3_8b_batch_32_seq_16348.log"

# echo "Waiting ${WAIT_BETWEEN}s before next run..." && sleep "$WAIT_BETWEEN"

# # 3) seq_len = 20480
# "$PY" "$SCRIPT_DIR/vllm_benchmark.py" \
#   --model-name Qwen/Qwen3-8B \
#   --warmup-iters 3 \
#   --duration-s 10.0 \
#   --fixed-iterations 1000 \
#   --dtype float16 \
#   --max-model-len 32768 \
#   --num-seq 32 \
#   --seq-len 20480 \
#   --post-exit-sleep 20 \
#   --gpu-memory-utilization 0.9 \
#   2>&1 | tee "$CACHE_DIR/bench_qwen3_8b_batch_32_seq_20480.log"

# echo "Waiting ${WAIT_BETWEEN}s before next run..." && sleep "$WAIT_BETWEEN"

# 4) Qwen3-14B, seq_len = 4096
"$PY" "$SCRIPT_DIR/vllm_benchmark.py" \
  --model-name Qwen/Qwen3-14B \
  --warmup-iters 3 \
  --duration-s 10.0 \
  --fixed-iterations 1000 \
  --dtype float16 \
  --max-model-len 32768 \
  --num-seq 16 \
  --seq-len 4096 \
  --post-exit-sleep 2 \
  --gpu-memory-utilization 0.9 \
  2>&1 | tee "$CACHE_DIR/bench_qwen3_14b_batch_32_seq_4096.log"

echo "Waiting ${WAIT_BETWEEN}s before next run..." && sleep "$WAIT_BETWEEN"

# 5) Qwen3-14B, seq_len = 16348
"$PY" "$SCRIPT_DIR/vllm_benchmark.py" \
  --model-name Qwen/Qwen3-14B \
  --warmup-iters 3 \
  --duration-s 10.0 \
  --fixed-iterations 1000 \
  --dtype float16 \
  --max-model-len 32768 \
  --num-seq 16 \
  --seq-len 16348 \
  --post-exit-sleep 20 \
  --gpu-memory-utilization 0.9 \
  2>&1 | tee "$CACHE_DIR/bench_qwen3_14b_batch_32_seq_16348.log"

echo "Waiting ${WAIT_BETWEEN}s before next run..." && sleep "$WAIT_BETWEEN"

# 6) Qwen3-14B, seq_len = 20480
"$PY" "$SCRIPT_DIR/vllm_benchmark.py" \
  --model-name Qwen/Qwen3-14B \
  --warmup-iters 3 \
  --duration-s 10.0 \
  --fixed-iterations 1000 \
  --dtype float16 \
  --max-model-len 32768 \
  --num-seq 16 \
  --seq-len 20480 \
  --post-exit-sleep 20 \
  --gpu-memory-utilization 0.9 \
  2>&1 | tee "$CACHE_DIR/bench_qwen3_14b_batch_32_seq_20480.log"

echo "Waiting ${WAIT_BETWEEN}s before next run..." && sleep "$WAIT_BETWEEN"

# 7) Qwen3-32B, seq_len = 4096
"$PY" "$SCRIPT_DIR/vllm_benchmark.py" \
  --model-name Qwen/Qwen3-32B \
  --warmup-iters 3 \
  --duration-s 10.0 \
  --fixed-iterations 1000 \
  --dtype float16 \
  --max-model-len 32768 \
  --num-seq 8 \
  --seq-len 4096 \
  --post-exit-sleep 2 \
  --gpu-memory-utilization 0.9 \
  2>&1 | tee "$CACHE_DIR/bench_qwen3_32b_batch_32_seq_4096.log"

echo "Waiting ${WAIT_BETWEEN}s before next run..." && sleep "$WAIT_BETWEEN"

# 8) Qwen3-32B, seq_len = 16348
"$PY" "$SCRIPT_DIR/vllm_benchmark.py" \
  --model-name Qwen/Qwen3-32B \
  --warmup-iters 3 \
  --duration-s 10.0 \
  --fixed-iterations 1000 \
  --dtype float16 \
  --max-model-len 32768 \
  --num-seq 8 \
  --seq-len 16348 \
  --post-exit-sleep 20 \
  --gpu-memory-utilization 0.9 \
  2>&1 | tee "$CACHE_DIR/bench_qwen3_32b_batch_32_seq_16348.log"

echo "Waiting ${WAIT_BETWEEN}s before next run..." && sleep "$WAIT_BETWEEN"

# 9) Qwen3-32B, seq_len = 20480
"$PY" "$SCRIPT_DIR/vllm_benchmark.py" \
  --model-name Qwen/Qwen3-32B \
  --warmup-iters 3 \
  --duration-s 10.0 \
  --fixed-iterations 1000 \
  --dtype float16 \
  --max-model-len 32768 \
  --num-seq 8 \
  --seq-len 20480 \
  --post-exit-sleep 20 \
  --gpu-memory-utilization 0.9 \
  2>&1 | tee "$CACHE_DIR/bench_qwen3_32b_batch_32_seq_20480.log"