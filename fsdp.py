import gc
from functools import partial

import torch
import torch.distributed as dist
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import MixedPrecision, ShardingStrategy
from torch.distributed.fsdp.wrap import lambda_auto_wrap_policy
from torch.distributed.utils import _free_storage

def create_process_groups():
    world_size = dist.get_world_size()
    rank = dist.get_rank()
    local_rank = rank % torch.cuda.device_count()

    num_gpus_per_node = torch.cuda.device_count()
    node_id = rank // num_gpus_per_node
    
    # Create intra-node group (processes on the same node)
    intra_node_group = None
    inter_node_group = None
    
    for i in range(world_size // num_gpus_per_node):
        ranks_in_node = list(range(i * num_gpus_per_node, (i + 1) * num_gpus_per_node))
        group = dist.new_group(ranks_in_node)
        if i == node_id:
            intra_node_group = group
    
    # Create inter-node group (processes with same local rank across nodes)
    for i in range(num_gpus_per_node):
        ranks_across_nodes = list(range(i, world_size, num_gpus_per_node))
        group = dist.new_group(ranks_across_nodes)
        if i == local_rank:
            inter_node_group = group

    return intra_node_group, inter_node_group


def shard_model(
    model,
    device_id,
    param_dtype=torch.bfloat16,
    reduce_dtype=torch.float32,
    buffer_dtype=torch.float32,
    process_group=None,
    sharding_strategy=ShardingStrategy.FULL_SHARD,
    sync_module_states=True,
    use_lora=False,
    machine_level_isolation=False,
):
  
    # If machine-level isolation is enabled, create custom process groups
    if machine_level_isolation:
        intra_machine_group, inter_machine_group = create_process_groups()

        process_group = (intra_machine_group, inter_machine_group)
        sharding_strategy = ShardingStrategy.HYBRID_SHARD
        
    
    model = FSDP(
        module=model,
        process_group=process_group,
        sharding_strategy=sharding_strategy,
        auto_wrap_policy=partial(
            lambda_auto_wrap_policy, lambda_fn=lambda m: m in model.blocks),
        mixed_precision=MixedPrecision(
            param_dtype=param_dtype,
            reduce_dtype=reduce_dtype,
            buffer_dtype=buffer_dtype),
        device_id=device_id,
        sync_module_states=sync_module_states,
        use_orig_params=True if use_lora else False)
    return model



def free_model(model):
    for m in model.modules():
        if isinstance(m, FSDP):
            _free_storage(m._handle.flat_param.data)
    del model
    gc.collect()
    torch.cuda.empty_cache()
