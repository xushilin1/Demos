import torch
import torch.distributed as dist
import os

def init_distributed():
    """初始化分布式环境"""
    rank = int(os.environ['RANK'])
    world_size = int(os.environ['WORLD_SIZE'])
    local_rank = int(os.environ['LOCAL_RANK'])
    assert world_size == 2
    
    # 初始化进程组
    dist.init_process_group(backend='nccl')
    
    return rank, world_size, local_rank

def cleanup_distributed():
    """清理分布式环境"""
    dist.destroy_process_group()

def test_all_reduce():
    """
    测试PyTorch的all_reduce操作。
    
    1. 在所有进程间执行指定的规约操作（如求和、求平均、求最大值等）
    2. 确保所有进程最终得到相同的结果
    
    具体过程：
    - 每个进程提供一个输入张量
    - 所有进程的张量按照指定的操作进行规约（本例中使用求和）
    - 结果被广播回所有进程
    - 所有进程得到完全相同的输出张量
    """

    rank = dist.get_rank()
    if rank == 0:
        print('=' * 60)
        print(f"[TEST_ALL_REDUCE]")
    
    torch.set_default_device(f"cuda:{rank}")
    if rank == 0:
        tensor = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32)
    else:
        tensor = torch.tensor([4.0, 5.0, 6.0], dtype=torch.float32)
    
    print(f"进程 {rank}: 初始张量 = {tensor}")
    
    # 执行all_reduce操作 - 所有进程的张量会被求和
    # 在这个过程中，所有进程都会相互通信，最终每个进程得到相同的结果
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM) # [5, 7, 9]
    # dist.all_reduce(tensor, op=dist.ReduceOp.MAX) # [4, 5, 6]
    # dist.all_reduce(tensor, op=dist.ReduceOp.AVG) # [2.5, 3.5, 4.5]
    
    if rank == 0:
        print(f"期望结果 = {tensor}")
        print('=' * 60)

def test_broadcast():
    """测试broadcast操作"""
    rank = dist.get_rank()
    torch.set_default_device(f"cuda:{rank}")
    
    if rank == 0:
        print('=' * 60)
        print(f"[TEST_BROADCAST]")
    
    if rank == 0:
        data = torch.tensor([10.0, 20.0, 30.0])
        print(f"Root进程 {rank} 广播数据: {data}")
    else:
        data = torch.tensor([1.0, 2.0, 3.0])
        print(f"进程 {rank} 等待接收数据")
    
    dist.broadcast(data, src=0)
    if rank == 1:
        print(f"进程 {rank} 接收到的数据: {data}")
        print('=' * 60)

def test_reduce():
    """
    测试reduce操作：将多个进程的数据规约(reduce)到一个目标进程
    
    Reduce操作的特点：
    - 所有进程提供输入数据
    - 在目标进程(dst)上执行指定的规约操作(如求和、求平均等)
    - 只有目标进程接收到最终结果，其他进程的数据会被更新但不一定有意义
    
    与all_reduce的区别：
    - all_reduce: 所有进程都得到规约结果
    - reduce: 只有指定的目标进程得到结果
    """
    rank = dist.get_rank()
    torch.set_default_device(f"cuda:{rank}")
    
    if rank == 0:
        print('=' * 60)
        print(f"[TEST_REDUCE]")
    
    if rank == 0:
        tensor = torch.tensor([1, 2, 3])
    else:
        tensor = torch.tensor([4, 5, 6])

    print(f"进程 {rank}: 初始张量 = {tensor}")
    
    # 执行reduce操作：将所有进程的数据求和到进程0
    dist.reduce(tensor, dst=0, op=dist.ReduceOp.SUM)
    
    if rank == 0:
        print(f"📍 Root进程接收到的规约结果: {tensor}")
        print('=' * 60)

def test_scatter_gather():
    """
    测试scatter和gather操作：用于数据分发与收集的经典组合
    
    Scatter（分发）操作：
    - src进程将一个列表的数据分发到所有进程（包括自身）
    - 每个进程接收对应的一个数据片段
    
    本测试完整流程：
    1. Scatter阶段：
       - 进程0: scatter_list = [[1.0, 2.0], [3.0, 4.0]]
       - 进程0接收：[1.0, 2.0]，进程1接收：[3.0, 4.0]

    2. Gather阶段：
       - 进程0收集：[[1.0, 2.0], [6.0, 8.0]]
    
    """
    rank = dist.get_rank()
    torch.set_default_device(f"cuda:{rank}")
    
    if rank == 0:
        print('=' * 60)
        print("测试数据分发与收集操作")
    
    # Scatter测试：root进程分发数据到各进程
    if rank == 0:
        scatter_list = [
            torch.tensor([1.0, 2.0]),
            torch.tensor([3.0, 4.0])
        ]
    else:
        scatter_list = None
    
    recv_tensor = torch.zeros(2)
    dist.scatter(recv_tensor, scatter_list, src=0)
    print(f"进程{rank} scatter接收: {recv_tensor}")
    
    # Gather测试：所有进程收集数据到root进程
    send_tensor = recv_tensor * (rank + 1)
    gather_list = [torch.zeros(2) for _ in range(2)] if rank == 0 else None
    dist.gather(send_tensor, gather_list, dst=0)

    if rank == 0:
        print(f"进程{rank} gather发送: {send_tensor}")
        print(f"进程{rank} gather接收: {gather_list}")
    
        print('=' * 60)

if __name__ == "__main__":
    init_distributed()
    try:
        test_all_reduce()
        test_broadcast()
        test_reduce()
        test_scatter_gather()
    finally:
        cleanup_distributed()