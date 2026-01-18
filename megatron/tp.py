import torch
import torch.nn.functional as F

def test_megatron_tp_mlp():
    batch_size = 2
    seq_len = 4
    hidden_size = 8
    intermediate_size = 16
    
    X = torch.randn(batch_size, seq_len, hidden_size)
    A = torch.randn(hidden_size, intermediate_size) # 第一层权重
    B = torch.randn(intermediate_size, hidden_size) # 第二层权重

    # --- 方案 A: 标准单卡计算 (Reference) ---
    # Z = GeLU(X @ A) @ B
    ref_out = F.gelu(X @ A) @ B


    # --- 方案 B: Megatron TP 计算 (Simulated 2 GPUs) ---
    A1, A2 = torch.chunk(A, 2, dim=1) # 将权重 A 按列切分 (Column Parallel)
    B1, B2 = torch.chunk(B, 2, dim=0) # 将权重 B 按行切分 (Row Parallel)

    # GPU 1 的本地计算
    Y1_local = F.gelu(X @ A1) 
    Z1_local = Y1_local @ B1

    # GPU 2 的本地计算
    Y2_local = F.gelu(X @ A2)
    Z2_local = Y2_local @ B2

    # 通信步骤：All-Reduce (在这里是简单的相加)
    tp_out = Z1_local + Z2_local

    # 3. 验证结果
    # 使用 allclose 检查浮点数差异
    is_correct = torch.allclose(ref_out, tp_out, atol=1e-5)
    
    print(f"Standard Output Shape: {ref_out.shape}")
    print(f"TP Output Shape: {tp_out.shape}")
    print(f"Results Match: {is_correct}")

def test_megatron_tp_mha():
    batch_size = 2
    seq_len = 5
    hidden_size = 12
    num_heads = 4 
    head_size = 24

    X = torch.randn(batch_size, seq_len, hidden_size)

    # 模拟 Q, K, V 投影层权重 (W_q, W_k, W_v)
    W_q = torch.randn(hidden_size, head_size * num_heads)
    W_k = torch.randn(hidden_size, head_size * num_heads)
    W_v = torch.randn(hidden_size, head_size * num_heads)

    # 模拟输出投影层权重 (W_o)
    W_o = torch.randn(head_size * num_heads, hidden_size)

    # --- 方案 A: 标准单卡 MHA 计算 (Reference) ---
    Q = X @ W_q
    K = X @ W_k
    V = X @ W_v

    Q = Q.view(batch_size, seq_len, num_heads, head_size).transpose(1, 2)
    K = K.view(batch_size, seq_len, num_heads, head_size).transpose(1, 2)
    V = V.view(batch_size, seq_len, num_heads, head_size).transpose(1, 2)

    # 3. 计算 Attention Score
    # (batch, num_heads, seq_len, head_size) @ (batch, num_heads, head_size, seq_len)
    attn_scores = (Q @ K.transpose(-2, -1)) / (head_size**0.5)
    attn_weights = F.softmax(attn_scores, dim=-1)
    attn_output = attn_weights @ V
    attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, head_size * num_heads)
    ref_out = attn_output @ W_o



    # --- 方案 B: Megatron TP MHA 计算 (Simulated 2 GPUs) ---
    # 分配给 2 个 GPU，每个 GPU 负责 num_heads // 2 个头
    num_heads_per_gpu = num_heads // 2

    # 1. Q, K, V 投影权重按 Head 维度切分 (Column Parallel)
    W_q1, W_q2 = torch.chunk(W_q, 2, dim=1) # 切分输出维度 (12, 96) -> (12, 48), (12, 48)
    W_k1, W_k2 = torch.chunk(W_k, 2, dim=1)
    W_v1, W_v2 = torch.chunk(W_v, 2, dim=1)

    # Output Projection W_o 按行切分 (Row Parallel)
    W_o1, W_o2 = torch.chunk(W_o, 2, dim=0)

    # --- GPU 1 计算 ---
    Q1_local = X @ W_q1
    K1_local = X @ W_k1
    V1_local = X @ W_v1

    # 2. 局部形状调整
    Q1_local = Q1_local.view(batch_size, seq_len, num_heads_per_gpu, head_size).transpose(1, 2)
    K1_local = K1_local.view(batch_size, seq_len, num_heads_per_gpu, head_size).transpose(1, 2)
    V1_local = V1_local.view(batch_size, seq_len, num_heads_per_gpu, head_size).transpose(1, 2)

    # 3. 局部 Attention Score (每个 GPU 只计算自己负责的头)
    attn_scores1 = (Q1_local @ K1_local.transpose(-2, -1)) / (head_size**0.5)
    attn_weights1 = F.softmax(attn_scores1, dim=-1)
    attn_output1 = attn_weights1 @ V1_local

    # 4. 局部拼接 (只拼接自己负责的头)
    attn_output1 = attn_output1.transpose(1, 2).contiguous().view(batch_size, seq_len, num_heads_per_gpu * head_size)
    
    # 5. 局部输出投影 (得到 Partial Sum)
    Z1_local = attn_output1 @ W_o1

    # --- GPU 2 计算 (同 GPU 1 逻辑) ---
    Q2_local = X @ W_q2
    K2_local = X @ W_k2
    V2_local = X @ W_v2

    Q2_local = Q2_local.view(batch_size, seq_len, num_heads_per_gpu, head_size).transpose(1, 2)
    K2_local = K2_local.view(batch_size, seq_len, num_heads_per_gpu, head_size).transpose(1, 2)
    V2_local = V2_local.view(batch_size, seq_len, num_heads_per_gpu, head_size).transpose(1, 2)

    attn_scores2 = (Q2_local @ K2_local.transpose(-2, -1)) / (head_size**0.5)
    attn_weights2 = F.softmax(attn_scores2, dim=-1)
    attn_output2 = attn_weights2 @ V2_local

    attn_output2 = attn_output2.transpose(1, 2).contiguous().view(batch_size, seq_len, num_heads_per_gpu * head_size)

    Z2_local = attn_output2 @ W_o2

    # 6. 通信步骤：All-Reduce (这里是简单的相加)
    tp_out = Z1_local + Z2_local

    # 3. 验证结果
    is_correct = torch.allclose(ref_out, tp_out, atol=1e-5)
    
    print(f"Standard MHA Output Shape: {ref_out.shape}")
    print(f"TP MHA Output Shape: {tp_out.shape}")
    print(f"Results Match: {is_correct}")

if __name__ == "__main__":
    test_megatron_tp_mha()
    test_megatron_tp_mlp()
