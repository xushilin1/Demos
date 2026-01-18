import torch
import torch.nn as F
def test_ulysses_correctness():
    batch_size, seq_len, num_heads, head_dim = 1, 256, 32, 1024
    world_size = 2
    hidden_dim = num_heads * head_dim
    
    Q = torch.randn(batch_size, seq_len, num_heads, head_dim)
    K = torch.randn(batch_size, seq_len, num_heads, head_dim)
    V = torch.randn(batch_size, seq_len, num_heads, head_dim)

    # --- 2. 方案 A: 标准单卡计算 (Reference) ---
    Q_ref = Q.transpose(1, 2)
    K_ref = K.transpose(1, 2)
    V_ref = V.transpose(1, 2)

    scores = torch.matmul(Q_ref, K_ref.transpose(-2, -1)) / (head_dim**0.5)
    ref_out = torch.matmul(F.softmax(scores, dim=-1), V_ref).transpose(1, 2)



    # --- 方案 B: DeepSpeed Ulysses 模拟 (2 GPUs) ---

    # 步骤 1: 模拟数据切分 (Sequence Parallel 状态)
    # GPU 0 拿序列前 4 个词的所有头；GPU 1 拿序列后 4 个词的所有头
    q_slice = torch.chunk(Q, world_size, dim=1) # [B, S, H, D] --> [B, S/P, H, D]
    k_slice = torch.chunk(K, world_size, dim=1)
    v_slice = torch.chunk(V, world_size, dim=1)

    # 步骤 2: 模拟 All-to-All (从 S 切分 转为 H 切分)
    # GPU 0 收集所有序列片段的第 0, 1 个头
    def all_to_all_forward(slices, rank, p):
        # 实际实现中，这里是 dist.all_to_all
        # 每个 rank 拿到完整的 S，但只拿到一部分 Head
        heads_per_gpu = num_heads // p
        gathered = torch.cat(slices, dim=1) # 拼接序列 [B, S, H, D]
        return gathered[:, :, rank*heads_per_gpu : (rank+1)*heads_per_gpu, :]

    # GPU 0 上的计算
    q0_local = all_to_all_forward(q_slice, 0, world_size) # [B, S, H/P, D]
    k0_local = all_to_all_forward(k_slice, 0, world_size)
    v0_local = all_to_all_forward(v_slice, 0, world_size)

    # GPU 1 上的计算
    q1_local = all_to_all_forward(q_slice, 1, world_size)
    k1_local = all_to_all_forward(k_slice, 1, world_size)
    v1_local = all_to_all_forward(v_slice, 1, world_size)

    # 步骤 3: 局部 Attention 计算 (GPU 0 计算自己的头)
    def compute_attn(q, k, v):
        q, k, v = q.transpose(1,2), k.transpose(1,2), v.transpose(1,2) # [B, H/P, S, D]
        s = torch.matmul(q, k.transpose(-2, -1)) / (head_dim**0.5)
        output = torch.matmul(F.softmax(s, dim=-1), v).transpose(1,2) # [B, S, H/P, D]
        return output

    out0_heads = compute_attn(q0_local, k0_local, v0_local) # [B, H/P, S, D]
    out1_heads = compute_attn(q1_local, k1_local, v1_local)

    # 步骤 4: 再次 All-to-All 换回 (从 H 切分 转回 S 切分)
    def all_to_all_backward(out_heads, rank, p):
        # 拼接所有头 [S, B, H, D]
        full_heads = torch.cat(out_heads, dim=2)
        # 重新切分序列
        return torch.chunk(full_heads, p, dim=1)[rank]

    final_out0 = all_to_all_backward([out0_heads, out1_heads], 0, world_size)
    final_out1 = all_to_all_backward([out0_heads, out1_heads], 1, world_size)

    # 拼接最终结果
    ulysses_out = torch.cat([final_out0, final_out1], dim=1)

    # --- 3. 结果验证 ---
    is_correct = torch.allclose(ref_out, ulysses_out, atol=1e-5)
    print(f"Ulysses Correctness: {is_correct}")

if __name__ == "__main__":
    test_ulysses_correctness()
