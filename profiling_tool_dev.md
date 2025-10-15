vllm_benchmark.py

High-level Suggestions:
找一个vllm号称比sgl快的地方config，或者至少是vllm官方的report的script（可以看他们最近的post?）
reproduce这个script，改成1 GPU version (方便下一步的print)，或者就找一个1 GPU serve的script
在reproduce的时候，跑到vllm/v1/worker/gpu_model_runner.py: GPUModelRunner里，把 with (set_forward_context(...)) 里面的结果print出来检查一下是什么样的，理解下各自是什么意思

1. How does vllm_benchmark.py profile one layer performance?
2. what is the best config for this inference?
3. how can we benchmark all components in this layer?


python vllm_benchmark.py \
  --model-name Qwen/Qwen3-8B \
  --warmup-iters 3 \
  --duration-s 10.0 \
  --fixed-iterations 1000 \
  --dtype float16 \
  --max-model-len 32768 \
  --num-seq 128 \
  --seq-len 4096 \
  --post-exit-sleep 20 \
  --gpu-memory-utilization 0.8 \
  2>&1 | tee cache/bench_qwen_8b_seq_4096.log

python vllm_benchmark.py \
  --model-name Qwen/Qwen3-8B \
  --warmup-iters 3 \
  --duration-s 10.0 \
  --fixed-iterations 1000 \
  --dtype float16 \
  --max-model-len 32768 \
  --num-seq 128 \
  --seq-len 16384 \
  --post-exit-sleep 20 \
  --gpu-memory-utilization 0.8 \
  2>&1 | tee cache/bench_qwen_8b_seq_16384.log

python vllm_benchmark.py \
  --model-name Qwen/Qwen3-8B \
  --warmup-iters 3 \
  --duration-s 10.0 \
  --fixed-iterations 1000 \
  --dtype float16 \
  --max-model-len 32768 \
  --num-seq 128 \
  --seq-len 30720 \
  --post-exit-sleep 20 \
  --gpu-memory-utilization 0.8 \
  2>&1 | tee cache/bench_qwen_8b_seq_30720.log
