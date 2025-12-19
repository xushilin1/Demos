import gc
from functools import partial

import torch
import torch.distributed as dist
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import MixedPrecision, ShardingStrategy
from torch.distributed.fsdp.wrap import lambda_auto_wrap_policy
from torch.distributed.utils import _free_storage

def create_process_groups(shard_group_size=8):
    world_size = dist.get_world_size()
    rank = dist.get_rank()
    assert world_size % shard_group_size == 0, 'world_size shall be divided by shard_size'

    num_shards = world_size // shard_group_size
    shard_group = None
    replicate_group = None
    for shard_group_id in range(num_shards):
        shard_ranks = list(range(shard_group_id * shard_group_size, (shard_group_id + 1) * shard_group_size)) # 每个shard的rank列表, [0, 1, 2, 3, 4, 5, 6, 7]

        group = dist.new_group(ranks=shard_ranks)
        if rank in shard_ranks:
            shard_group = group

    for replicate_group_id in range(shard_group_size):
        replicate_ranks = [replicate_group_id + i * shard_group_size for i in range(num_shards)] # 每个replicate的rank列表, [0, 8, 16, 24]
        group = dist.new_group(ranks=replicate_ranks)

        if rank in replicate_ranks:
            replicate_group = group

    return shard_group, replicate_group

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
