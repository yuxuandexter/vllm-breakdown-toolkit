# Testing the memory access bandwidth and latency for a given model, based on vllm's implementation
from typing import Any

import torch
import vllm
from vllm.config import CUDAGraphMode
from vllm.engine.arg_utils import EngineArgs
from vllm.v1.worker.gpu_worker import Worker
from vllm.model_executor.models.qwen3 import Qwen3ForCausalLM, Qwen3DecoderLayer, Qwen3MLP, Qwen3Attention
from vllm.forward_context import (set_forward_context, BatchDescriptor,)


def test():
    engine_args = EngineArgs(
        model = "Qwen/Qwen3-8B",
        trust_remote_code=True,
        dtype="float16",
        # tensor_parallel_size=1
        gpu_memory_utilization=0.8,
        disable_custom_all_reduce=True,
        max_model_len=32768,
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

    # Initialize KV cache so Attention has non-empty kv_cache bound
    from vllm.v1.core.kv_cache_utils import get_kv_cache_configs
    available_bytes = worker.determine_available_memory()
    kv_cache_specs = [worker.model_runner.get_kv_cache_spec()]
    kv_cache_configs = get_kv_cache_configs(
        worker.model_runner.vllm_config,
        kv_cache_specs,
        [available_bytes],
    )
    worker.initialize_from_config(kv_cache_configs[0])

    # construct input. TODO: where is the kv cache put?
    num_seq = 8
    seq_len = 30720
    dtype = torch.float16

    hidden_size = config.model_config.get_hidden_size()
    # TODO: shall we divide this hidden size by tp when constructing the input?
    hidden_states = torch.randn(num_seq, 1, hidden_size, dtype=dtype).cuda()
    positions = torch.ones(num_seq, 1, dtype=torch.long).cuda() * seq_len
    attn_fwd_args = {
        "positions": positions,
        "hidden_states": hidden_states,
    }

    # Build per-layer attention metadata using the backend builder
    from vllm.v1.attention.backends.utils import CommonAttentionMetadata
    device = torch.device("cuda")

    # 1) Minimal decode-style batch: each request generates 1 token
    num_reqs = num_seq
    q_lens = torch.ones(num_reqs, dtype=torch.int32)
    query_start_loc_cpu = torch.empty(num_reqs + 1, dtype=torch.int32)
    query_start_loc_cpu[0] = 0
    torch.cumsum(q_lens, dim=0, out=query_start_loc_cpu[1:])
    max_query_len = int(q_lens.max().item())
    num_actual_tokens = int(q_lens.sum().item())

    # 2) Historical context lengths (KV)
    seq_lens_cpu = torch.full((num_reqs,), int(seq_len), dtype=torch.int32)
    max_seq_len = int(seq_lens_cpu.max().item())
    num_computed_tokens_cpu = seq_lens_cpu - 1

    # 3) Minimal block_table and slot_mapping
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

    # 4) Assemble CommonAttentionMetadata
    common = CommonAttentionMetadata(
        query_start_loc=query_start_loc_cpu.to(device=device, non_blocking=True),
        query_start_loc_cpu=query_start_loc_cpu,
        seq_lens=seq_lens_cpu.to(device=device, non_blocking=True),
        seq_lens_cpu=seq_lens_cpu,
        num_computed_tokens_cpu=num_computed_tokens_cpu,
        num_reqs=int(num_reqs),
        num_actual_tokens=int(num_actual_tokens),
        max_query_len=int(max_query_len),
        max_seq_len=int(max_seq_len),
        block_table_tensor=block_table_tensor,
        slot_mapping=slot_mapping.to(device=device, non_blocking=True),
        causal=True,
    )

    # 5) Get builder from this attention layer's backend and build attn_metadata
    builder_cls = attn.attn.attn_backend.get_builder_cls()
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

    # 6) Prepare forward context params
    num_tokens = num_actual_tokens
    num_tokens_across_dp = num_actual_tokens
    cudagraph_runtime_mode = CUDAGraphMode.NONE
    batch_descriptor = BatchDescriptor(num_tokens=num_tokens,
                                       uniform_decode=True)

    # ---------------- Profiling: measure Attention vs MLP for this layer ----------------
    from torch.profiler import profile, record_function, ProfilerActivity
    torch.cuda.synchronize()
    warmup, iters = 3, 10

    # Warmup
    for _ in range(warmup):
        with set_forward_context(
            attn_metadata,
            worker.model_runner.vllm_config,
            num_tokens=num_tokens,
            num_tokens_across_dp=num_tokens_across_dp,
            cudagraph_runtime_mode=cudagraph_runtime_mode,
            batch_descriptor=batch_descriptor
        ):
            h = attn(**attn_fwd_args)
            _ = mlp(h)
    torch.cuda.synchronize()

    # Profile
    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        record_shapes=True,
        profile_memory=True,
        with_stack=False,
    ) as prof:
        for _ in range(iters):
            with set_forward_context(
                attn_metadata,
                worker.model_runner.vllm_config,
                num_tokens=num_tokens,
                num_tokens_across_dp=num_tokens_across_dp,
                cudagraph_runtime_mode=cudagraph_runtime_mode,
                batch_descriptor=batch_descriptor
            ):
                with record_function("LAYER_0_FULL"):
                    with record_function("LAYER_0_ATTENTION"):
                        h = attn(**attn_fwd_args)
                    with record_function("LAYER_0_MLP"):
                        h = mlp(h)
    torch.cuda.synchronize()

    # Export Perfetto/Chrome trace
    prof.export_chrome_trace("trace.json")

    # Print compact summary for our custom ranges
    def _stat(name: str):
        for k in prof.key_averages():
            kname = getattr(k, "key", None) or getattr(k, "name", None)
            if kname == name:
                count = max(getattr(k, "count", 0), 1)
                # times are in microseconds. Prefer inclusive TOTAL time (CUDA if available).
                cuda_total = getattr(k, "cuda_time_total", None)
                cpu_total = getattr(k, "cpu_time_total", None)
                total_us = (
                    cuda_total if isinstance(cuda_total, (int, float)) else cpu_total
                )
                total_ms = (total_us or 0.0) / 1000.0
                avg_ms = total_ms / count
                # Also report raw CPU/CUDA totals if available
                out = {
                    "count": count,
                    "total_ms": total_ms,
                    "avg_ms": avg_ms,
                    "cuda_total_ms": (cuda_total / 1000.0) if cuda_total else None,
                    "cpu_total_ms": (cpu_total / 1000.0) if cpu_total else None,
                }
                return out
        return {"count": 0, "total_ms": 0.0, "avg_ms": 0.0, "cuda_total_ms": None, "cpu_total_ms": None}

    full = _stat("LAYER_0_FULL")
    attn_stat = _stat("LAYER_0_ATTENTION")
    mlp_stat = _stat("LAYER_0_MLP")

    print("Profiling results (inclusive TOTAL time; CUDA if available, else CPU):")
    print(
        f"  FULL layer: total={full['total_ms']:.3f} ms, avg={full['avg_ms']:.3f} ms over {full['count']} iters"
    )
    print(
        f"  Attention : total={attn_stat['total_ms']:.3f} ms, avg={attn_stat['avg_ms']:.3f} ms over {attn_stat['count']} iters"
    )
    print(
        f"  MLP       : total={mlp_stat['total_ms']:.3f} ms, avg={mlp_stat['avg_ms']:.3f} ms over {mlp_stat['count']} iters"
    )
    # Optional: print raw CPU/CUDA self breakdown if present
    if any(x is not None for x in (full["cuda_total_ms"], full["cpu_total_ms"])):
        print(
            f"    FULL totals   : cuda_total={full['cuda_total_ms']} ms, cpu_total={full['cpu_total_ms']} ms"
        )
    if any(x is not None for x in (attn_stat["cuda_total_ms"], attn_stat["cpu_total_ms"])):
        print(
            f"    ATTN totals   : cuda_total={attn_stat['cuda_total_ms']} ms, cpu_total={attn_stat['cpu_total_ms']} ms"
        )
    if any(x is not None for x in (mlp_stat["cuda_total_ms"], mlp_stat["cpu_total_ms"])):
        print(
            f"    MLP totals    : cuda_total={mlp_stat['cuda_total_ms']} ms, cpu_total={mlp_stat['cpu_total_ms']} ms"
        )


if __name__ == "__main__":
    test()