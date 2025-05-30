import torch
import torch.nn as nn
import torch.optim as optim
import deepspeed
from deepspeed.sequence.layer import DistributedAttention
import argparse
import os

# 全局变量用于序列并行通信组
_SEQUENCE_PARALLEL_GROUP = None

# 解析命令行参数
def parse_args():
    parser = argparse.ArgumentParser(description="DeepSpeed Ulysses CIFAR10 Transformer Training")
    parser.add_argument('--local_rank', type=int, default=-1, help='Local rank for distributed training')
    parser.add_argument('--deepspeed_config', type=str, default='ds_config.json', help='DeepSpeed configuration file')
    parser.add_argument('--ds-sequence-parallel-size', type=int, default=4, help='Degree of sequence parallelism')
    return parser.parse_args()

# 初始化序列并行通信组
def initialize_sequence_parallel_group(rank, world_size, sequence_parallel_size):
    global _SEQUENCE_PARALLEL_GROUP
    num_sequence_parallel_groups = world_size // sequence_parallel_size
    for i in range(num_sequence_parallel_groups):
        ranks = range(i * sequence_parallel_size, (i + 1) * sequence_parallel_size)
        group = torch.distributed.new_group(ranks)
        if rank in ranks:
            _SEQUENCE_PARALLEL_GROUP = group
    return _SEQUENCE_PARALLEL_GROUP

def get_sequence_parallel_group():
    return _SEQUENCE_PARALLEL_GROUP

# Attention Layer 模块（使用 DistributedAttention）
class AttentionLayer(nn.Module):
    def __init__(self, embed_dim, num_heads, mlp_ratio=4.0, dropout=0.1, sequence_parallel_group=None):
        super(AttentionLayer, self).__init__()
        self.local_attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.attn = DistributedAttention(self.local_attn, sequence_parallel_group=sequence_parallel_group)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, int(embed_dim * mlp_ratio)),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(int(embed_dim * mlp_ratio), embed_dim),
            nn.Dropout(dropout)
        )
        self.norm2 = nn.LayerNorm(embed_dim)

    def forward(self, x, attn_mask=None):
        x = x + self.attn(self.norm1(x), attention_mask=attn_mask)[0]  # 残差连接
        x = x + self.mlp(self.norm2(x))  # 残差连接
        return x

# Transformer 模型
class TransformerLongSeq(nn.Module):
    def __init__(self, seq_len=32768, embed_dim=256, num_heads=8, num_layers=4, num_classes=10, dropout=0.1, sequence_parallel_group=None):
        super(TransformerLongSeq, self).__init__()
        self.embed = nn.Embedding(1000, embed_dim)  # 假设词汇表大小为 1000
        self.pos_embed = nn.Parameter(torch.zeros(1, seq_len, embed_dim))
        self.dropout = nn.Dropout(dropout)
        self.layers = nn.ModuleList([
            AttentionLayer(embed_dim, num_heads, dropout=dropout, sequence_parallel_group=sequence_parallel_group)
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        B = x.shape[0]
        x = self.embed(x)  # (B, seq_len, embed_dim)
        x = x + self.pos_embed[:, :x.size(1), :]
        x = self.dropout(x)
        for layer in self.layers:
            x = layer(x)
        x = self.norm(x)
        x = x.mean(dim=1)  # 全局平均池化
        x = self.head(x)
        return x

# 合成数据集（模拟长序列）
class SyntheticLongSeqDataset(torch.utils.data.Dataset):
    def __init__(self, num_samples, seq_len, vocab_size=1000, num_classes=10):
        self.num_samples = num_samples
        self.seq_len = seq_len
        self.vocab_size = vocab_size
        self.num_classes = num_classes

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # 随机生成序列和标签
        seq = torch.randint(0, self.vocab_size, (self.seq_len,))
        label = torch.randint(0, self.num_classes, (1,)).item()
        return seq, label

# 定义数据加载器
def get_data_loaders(rank, world_size, batch_size=128, seq_len=32768):
    trainset = SyntheticLongSeqDataset(num_samples=50000, seq_len=seq_len)
    train_sampler = torch.utils.data.distributed.DistributedSampler(
        trainset, num_replicas=world_size, rank=rank
    )
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=batch_size, sampler=train_sampler, num_workers=2
    )

    testset = SyntheticLongSeqDataset(num_samples=10000, seq_len=seq_len)
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=batch_size, shuffle=False, num_workers=2
    )

    return trainloader, testloader

# 定义训练函数
def train(model_engine, trainloader, criterion, epoch):
    model_engine.train()
    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        inputs, labels = data
        inputs, labels = inputs.cuda(), labels.cuda()

        # 调试：打印输入和模型权重的类型（仅第一批次）
        if i == 0 and model_engine.global_rank == 0:
            print(f"Input dtype: {inputs.dtype}")
            print(f"Model weight dtype (first layer): {next(model_engine.parameters()).dtype}")

        outputs = model_engine(inputs)
        loss = criterion(outputs, labels)

        model_engine.backward(loss)
        model_engine.step()

        running_loss += loss.item()
        if i % 100 == 99:
            running_loss = 0.0  # 移除打印，只保留损失重置

# 定义测试函数
def test(model_engine, testloader, criterion):
    model_engine.eval()
    correct = 0
    total = 0
    test_loss = 0.0
    with torch.no_grad():
        for i, data in enumerate(testloader, 0):
            images, labels = data
            images, labels = images.cuda(), labels.cuda()

            # 调试：打印输入和模型权重的类型（仅第一批次）
            if i == 0 and model_engine.global_rank == 0:
                print(f"Test Input dtype: {images.dtype}")
                print(f"Test Model weight dtype (first layer): {next(model_engine.parameters()).dtype}")

            outputs = model_engine(images)
            loss = criterion(outputs, labels)
            test_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    # 移除准确率和损失的打印

def main():
    args = parse_args()

    # 初始化DeepSpeed分布式环境
    deepspeed.init_distributed()

    # 获取全局和本地rank
    rank = torch.distributed.get_rank()
    world_size = torch.distributed.get_world_size()

    # 初始化序列并行通信组
    sequence_parallel_group = initialize_sequence_parallel_group(rank, world_size, args.ds_sequence_parallel_size)

    # 加载数据
    trainloader, testloader = get_data_loaders(rank, world_size, seq_len=32768)

    # 定义模型
    model = TransformerLongSeq(
        seq_len=32768, embed_dim=256, num_heads=8, num_layers=4, num_classes=10, dropout=0.1,
        sequence_parallel_group=sequence_parallel_group
    ).cuda()

    # 定义损失函数
    criterion = nn.CrossEntropyLoss()

    # 初始化DeepSpeed
    model_engine, optimizer, _, _ = deepspeed.initialize(
        args=args,
        model=model,
        model_parameters=model.parameters(),
        config=args.deepspeed_config
    )

    # 训练循环
    num_epochs = 10
    for epoch in range(num_epochs):
        trainloader.sampler.set_epoch(epoch)  # 确保每个epoch数据打乱
        train(model_engine, trainloader, criterion, epoch)
        test(model_engine, testloader, criterion)

if __name__ == '__main__':
    main()