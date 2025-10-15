# Testing the memory access bandwidth and latency for a given model, based on vllm's implementation
# TODO: Verify time trackingwithout env time track (using 100 iterations to get avarge time)
# TODO: increase batch sizeto avoid cpu overheads
# TODO: check nsys about cpu overheads bubbles
# TODO: figure out set_forward_context config to optimize forward
# TODO: maybe updte vllm config to optimize infer
# TODO: flash infer?
# TODO: verify set_forward_context configs by using llm inference pipeline



from typing import Any
import argparse
import time
import gc
import os

import torch
import torch.distributed as dist
import vllm
from vllm.config import CUDAGraphMode
from vllm.engine.arg_utils import EngineArgs
from vllm.v1.worker.gpu_worker import Worker
from vllm.model_executor.models.qwen3 import Qwen3ForCausalLM, Qwen3DecoderLayer, Qwen3MLP, Qwen3Attention
from vllm.forward_context import (set_forward_context, BatchDescriptor,)
from vllm.v1.core.kv_cache_utils import get_kv_cache_configs
from vllm.v1.attention.backends.utils import CommonAttentionMetadata

# args: 
# required model name, 

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="vLLM microbenchmark for Qwen3 components")
    parser.add_argument("--model-name", type=str, default="Qwen/Qwen3-8B", help="HF model id")
    parser.add_argument("--warmup-iters", "--warmup_ters", dest="warmup_iters", type=int, default=3, help="Warmup iterations before timing")
    parser.add_argument("--duration-s", type=float, default=10.0, help="Wall-clock duration for timing loop in seconds")
    parser.add_argument("--fixed-iterations", dest="fixed_iterations", type=int, default=1000, help="Fixed iterations for event and single-sync timing")
    parser.add_argument("--dtype", type=str, default="float16", choices=["float16", "fp16", "bfloat16", "bf16", "float32", "fp32"], help="Computation dtype")
    parser.add_argument("--max-model-len", type=int, default=32768, help="Max model length for vLLM config")
    parser.add_argument("--num-seq", type=int, default=8, help="Number of sequences in the batch")
    parser.add_argument("--seq-len", "--seq-length", dest="seq_len", type=int, default=4096, help="Sequence length per request")
    # Optional knobs
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.8, help="GPU memory utilization for vLLM EngineArgs")
    parser.add_argument("--post-exit-sleep", type=float, default=0.0, help="Seconds to sleep after cleanup for sequential runs")
    return parser.parse_args()

def measure_op(
    op,
    label: str,
    warmup_iters: int = 3,
    duration_s: float = 10.0,
    fixed_iterations: int = 1000,
    static_args: tuple = (),
    static_kwargs: dict = None,
    carry=None,
):
    if static_kwargs is None:
        static_kwargs = {}

    def _call_once(current_carry):
        if current_carry is None:
            return op(*static_args, **static_kwargs)
        if isinstance(current_carry, tuple):
            return op(*static_args, *current_carry, **static_kwargs)
        return op(*static_args, current_carry, **static_kwargs)

    # Warmup, then synchronize
    for _ in range(warmup_iters):
        outputs = _call_once(carry)
        if carry is not None:
            carry = outputs
    torch.cuda.synchronize()

    results = {}

    # Method 1: Duration-driven loop, single sync at end
    # Run for approximately duration_s and compute avg = duration_s / num_iters
    num_iters = 0
    start_t = time.time()
    while (time.time() - start_t) < duration_s:
        outputs = _call_once(carry)
        if carry is not None:
            carry = outputs
        num_iters += 1
    torch.cuda.synchronize()
    # Use the target duration to compute average per the requirement
    avg_ms_duration = (duration_s * 1000.0) / max(num_iters, 1)
    print(f"{label} [duration/iters]: duration_s={duration_s:.3f}, iters={num_iters}, avg_time_ms={avg_ms_duration:.3f}")
    results["duration_iters_count"] = num_iters
    results["duration_ms_avg"] = avg_ms_duration

    # Method 2: CUDA events with per-iteration event synchronize, fixed iterations
    total_ms_events = 0.0
    for _ in range(fixed_iterations):
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        start_event.record()
        outputs = _call_once(carry)
        end_event.record()
        end_event.synchronize()
        total_ms_events += start_event.elapsed_time(end_event)
        if carry is not None:
            carry = outputs
    avg_ms_events = total_ms_events / max(fixed_iterations, 1)
    print(f"{label} [events-sync]: iters={fixed_iterations}, total_ms={total_ms_events:.3f}, avg_time_ms={avg_ms_events:.3f}")
    results["events_ms_total"] = total_ms_events
    results["events_ms_avg"] = avg_ms_events

    # Method 3: Single synchronize after fixed iterations (no per-iter sync)
    start_time_single_sync = time.time()
    for _ in range(fixed_iterations):
        outputs = _call_once(carry)
        if carry is not None:
            carry = outputs
    torch.cuda.synchronize()
    total_ms_single_sync = (time.time() - start_time_single_sync) * 1000.0
    avg_ms_single_sync = total_ms_single_sync / max(fixed_iterations, 1)
    print(f"{label} [one-sync-after]: iters={fixed_iterations}, total_ms={total_ms_single_sync:.3f}, avg_time_ms={avg_ms_single_sync:.3f}")
    results["one_sync_ms_total"] = total_ms_single_sync
    results["one_sync_ms_avg"] = avg_ms_single_sync

    return results

def test(args: argparse.Namespace):
    # 1. Initialize the engine
    engine_args = EngineArgs(
        model = args.model_name,
        trust_remote_code=True,
        dtype=args.dtype,
        # tensor_parallel_size=1
        gpu_memory_utilization=args.gpu_memory_utilization,
        disable_custom_all_reduce=True,
        max_model_len=args.max_model_len,
        load_format="auto",
        disable_log_stats=True,
        seed=0,
    )
    config = engine_args.create_engine_config()

    # get layers
    # if len(node_gpus) == 1:
    #         # in single node case, we don't need to get the IP address.
    #         # the loopback address is sufficient
    #         # NOTE: a node may have several IP addresses, one for each
    #         # network interface. `get_ip()` might return any of them,
    #         # while they might not work for communication inside the node
    #         # if the network setup is complicated. Using the loopback address
    #         # solves this issue, as it always works for communication inside
    #         # the node.
    #         driver_ip = "127.0.0.1"
    # distributed_init_method = get_distributed_init_method(
    #     driver_ip, get_open_port())
    distributed_init_method = "tcp://127.0.0.1:5000"
    worker = Worker(config, local_rank=0, rank=0,
                    distributed_init_method=distributed_init_method,
                    is_driver_worker=True 
                # is_driver_worker=(not self.parallel_config)
                # or (rank % self.parallel_config.tensor_parallel_size == 0)
                    )
    worker.init_device()
    print("worker init done")
    worker.load_model()
    model = worker.get_model()
    assert isinstance(model, Qwen3ForCausalLM)
    layer = model.model.layers[0]
    assert isinstance(layer, Qwen3DecoderLayer)
    mlp = layer.mlp
    attn = layer.self_attn
    assert isinstance(mlp, Qwen3MLP)
    assert isinstance(attn, Qwen3Attention)

    # 2. Determine the available memory for the KV cache
    available_bytes = worker.determine_available_memory()
    kv_cache_specs = [worker.model_runner.get_kv_cache_spec()]
    kv_cache_configs = get_kv_cache_configs(
        worker.model_runner.vllm_config,
        kv_cache_specs,
        [available_bytes],
    )
    worker.initialize_from_config(kv_cache_configs[0])

    # 3.simulate attention forward args
    num_seq = int(args.num_seq)
    seq_len = int(args.seq_len)
    # Map string dtype to torch dtype
    dtype_str = str(args.dtype).lower()
    if dtype_str in ("float16", "fp16"):
        dtype = torch.float16
    elif dtype_str in ("bfloat16", "bf16"):
        dtype = torch.bfloat16
    elif dtype_str in ("float32", "fp32"):
        dtype = torch.float32
    else:
        raise ValueError(f"Unsupported dtype: {args.dtype}")
    hidden_size = config.model_config.get_hidden_size()
    hidden_states = torch.randn(num_seq, 1, hidden_size, dtype=dtype).cuda()
    positions = torch.ones(num_seq, 1, dtype=torch.long).cuda() * seq_len
    attn_fwd_args = {
        "positions": positions,
        "hidden_states": hidden_states,
    }
    # 4. build attention Metadata
    device = torch.device("cuda")
    num_reqs = num_seq
    q_lens = torch.ones(num_reqs, dtype=torch.int32)
    query_start_loc_cpu = torch.empty(num_reqs + 1, dtype=torch.int32)
    query_start_loc_cpu[0] = 0
    torch.cumsum(q_lens, dim=0, out=query_start_loc_cpu[1:])
    max_query_len = int(q_lens.max().item())
    num_actual_tokens = int(q_lens.sum().item())

    # (1) Historical context lengths (KV)
    seq_lens_cpu = torch.full((num_reqs,), int(seq_len), dtype=torch.int32)
    max_seq_len = int(seq_lens_cpu.max().item())
    num_computed_tokens_cpu = seq_lens_cpu - 1

    # (2) Minimal block_table and slot_mapping
    try:
        block_size = int(worker.model_runner.vllm_config.cache_config.block_size)
    except Exception:
        block_size = 512
    pages_per_seq = (max(seq_len, 1) + block_size - 1) // block_size
    block_table_tensor = torch.arange(
        num_reqs * pages_per_seq, dtype=torch.int32, device=device
    ).view(num_reqs, pages_per_seq)
    # FlashAttention path expects slot_mapping as int64 (Long)
    slot_mapping = torch.arange(num_actual_tokens, dtype=torch.int64)

    # (3) Assemble CommonAttentionMetadata
    common = CommonAttentionMetadata(
        # batch size + 1, the start location of each request in query Tensor
        query_start_loc=query_start_loc_cpu.to(device=device, non_blocking=True),
        # same, batch size + 1
        query_start_loc_cpu=query_start_loc_cpu,
        # batch size, the length of each request including both computed tokens and newly scheduled tokens
        seq_lens=seq_lens_cpu.to(device=device, non_blocking=True),
        # same, batch size
        seq_lens_cpu=seq_lens_cpu,
        # batch size, the number of computed tokens for each request
        num_computed_tokens_cpu=num_computed_tokens_cpu,
        # number of requests
        num_reqs=int(num_reqs),
        # total number of tokens in the batch
        num_actual_tokens=int(num_actual_tokens),
        # longest query in batch
        max_query_len=int(max_query_len),
        # longest context length in batch
        max_seq_len=int(max_seq_len),
        # batch size, max_blocks
        block_table_tensor=block_table_tensor,
        # maps slots to requests/tokens
        slot_mapping=slot_mapping.to(device=device, non_blocking=True),
        # whether the attention is causal
        causal=True,
    )

    # (4) Get builder from this attention layer's backend and build attn_metadata
    builder_cls = attn.attn.attn_backend.get_builder_cls()
    print(f"builder_cls: {builder_cls}")
    kv_cache_spec_map = worker.model_runner.get_kv_cache_spec()
    layer_name = attn.attn.layer_name
    kv_cache_spec = kv_cache_spec_map[layer_name]
    builder = builder_cls(
        kv_cache_spec=kv_cache_spec,
        layer_names=[layer_name],
        vllm_config=worker.model_runner.vllm_config,
        device=device,
    )
    attn_metadata = builder.build(
        common_prefix_len=0,
        common_attn_metadata=common,
        fast_build=False,
    )

    # 5. Prepare forward context params
    num_tokens = num_actual_tokens
    num_tokens_across_dp = num_actual_tokens
    cudagraph_runtime_mode = CUDAGraphMode.NONE
    batch_descriptor = BatchDescriptor(num_tokens=num_tokens,
                                       uniform_decode=True)

    with set_forward_context(
        attn_metadata,
        worker.model_runner.vllm_config,
        num_tokens=num_tokens,
        num_tokens_across_dp=num_tokens_across_dp,
        cudagraph_runtime_mode=cudagraph_runtime_mode,
        batch_descriptor=batch_descriptor
    ):

        # 1) Initial attention run to ensure everything is initialized
        attn(**attn_fwd_args)
        torch.cuda.synchronize()
        _ = measure_op(
            attn,
            label="Attention (Qwen3Attention)",
            static_kwargs=attn_fwd_args,
            warmup_iters=args.warmup_iters,
            duration_s=args.duration_s,
            fixed_iterations=args.fixed_iterations,
        )

        _ = measure_op(
            mlp,
            label="MLP (Qwen3MLP)",
            carry=hidden_states.clone(),
            warmup_iters=args.warmup_iters,
            duration_s=args.duration_s,
            fixed_iterations=args.fixed_iterations,
        )

    # Formal termination and cleanup to allow multiple sequential runs safely
    try:
        if hasattr(worker, "shutdown"):
            try:
                worker.shutdown()
            except Exception:
                pass
        if dist.is_available() and dist.is_initialized():
            try:
                dist.destroy_process_group()
            except Exception:
                pass
    finally:
        try:
            del model
            del layer
            del mlp
            del attn
            del worker
        except Exception:
            pass
        gc.collect()
        try:
            torch.cuda.synchronize()
        except Exception:
            pass
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass
        try:
            torch.cuda.ipc_collect()
        except Exception:
            pass
        if args.post_exit_sleep and args.post_exit_sleep > 0:
            try:
                time.sleep(float(args.post_exit_sleep))
            except Exception:
                pass

if __name__ == "__main__":
    _args = parse_args()
    test(_args)