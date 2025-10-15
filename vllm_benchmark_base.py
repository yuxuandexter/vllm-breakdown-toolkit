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

    # construct input. TODO: where is the kv cache put?
    num_seq = 8
    seq_len = 4096
    dtype = torch.float16

    hidden_size = config.model_config.get_hidden_size()
    # TODO: shall we divide this hidden size by tp when constructing the input?
    hidden_states = torch.randn(num_seq, 1, hidden_size, dtype=dtype).cuda()
    positions = torch.ones(num_seq, 1, dtype=torch.long).cuda() * seq_len
    attn_fwd_args = {
        "positions": positions,
        "hidden_states": hidden_states,
    }

    # FIXME: prepare attn metadata with layer name here.
    attn_metadata: dict[str, Any] = {}
    num_tokens = num_seq
    num_tokens_across_dp = num_seq
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
        attn(**attn_fwd_args)  # warm up


if __name__ == "__main__":
    test()