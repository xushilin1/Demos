import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import deepspeed
from torchvision.models import resnet18
import argparse
import os


class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.1):
        super(MultiHeadAttention, self).__init__()
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        self.qkv = nn.Linear(embed_dim, embed_dim * 3)
        self.out = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)
        self.scale = self.head_dim ** -0.5

    def forward(self, x):
        B, N, C = x.shape  # (batch_size, seq_len, embed_dim)
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # (batch_size, num_heads, seq_len, head_dim)

        attn = (q @ k.transpose(-2, -1)) * self.scale  # (batch_size, num_heads, seq_len, seq_len)
        attn = attn.softmax(dim=-1)
        attn = self.dropout(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, self.embed_dim)  # (batch_size, seq_len, embed_dim)
        x = self.out(x)
        x = self.dropout(x)
        return x
    


from transformers import Qwen2_5_VLForConditionalGeneration
# Attention Layer 模块
class AttentionLayer(nn.Module):
    def __init__(self, embed_dim, num_heads, mlp_ratio=4.0, dropout=0.1):
        super(AttentionLayer, self).__init__()
        self.attn = MultiHeadAttention(embed_dim, num_heads, dropout)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, int(embed_dim * mlp_ratio)),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(int(embed_dim * mlp_ratio), embed_dim),
            nn.Dropout(dropout)
        )
        self.norm2 = nn.LayerNorm(embed_dim)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))  # 残差连接
        x = x + self.mlp(self.norm2(x))  # 残差连接
        return x
    

class TransformerCIFAR(nn.Module):
    def __init__(self, img_size=32, patch_size=4, embed_dim=256, num_heads=8, num_layers=4, num_classes=10, dropout=0.1):
        super(TransformerCIFAR, self).__init__()
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        self.patch_embed = nn.Conv2d(3, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.dropout = nn.Dropout(dropout)
        self.layers = nn.ModuleList([
            AttentionLayer(embed_dim, num_heads, dropout=dropout) for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        B = x.shape[0]
        x = self.patch_embed(x).flatten(2).transpose(1, 2)  # (B, num_patches, embed_dim)
        x = self.dropout(x)
        for layer in self.layers:
            x = layer(x)
        x = self.norm(x)
        x = x.mean(dim=1)  # 全局平均池化
        x = self.head(x)
        return x

# 解析命令行参数
def parse_args():
    parser = argparse.ArgumentParser(description="DeepSpeed CIFAR10 Training")
    parser.add_argument('--local_rank', type=int, default=-1, help='Local rank for distributed training')
    parser = deepspeed.add_config_arguments(parser)
    return parser.parse_args()

# 定义数据加载器
def get_data_loaders(rank, world_size, batch_size=128):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    train_sampler = torch.utils.data.distributed.DistributedSampler(trainset, num_replicas=world_size, rank=rank)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, sampler=train_sampler, num_workers=2)

    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2)

    return trainloader, testloader

# 定义训练函数
def train(model_engine, trainloader, criterion, epoch):
    model_engine.train()
    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        inputs, labels = data
        inputs, labels = inputs.cuda(), labels.cuda()

        inputs = inputs.to(dtype=torch.bfloat16)
        outputs = model_engine(inputs)
        loss = criterion(outputs, labels)

        model_engine.backward(loss)
        model_engine.step()

        running_loss += loss.item()
        if i % 10 == 0 and model_engine.global_rank == 0:
            print(f'[Epoch {epoch + 1}, Batch {i + 1}] Loss: {running_loss / 100:.3f}')
            running_loss = 0.0

# 定义测试函数
def test(model_engine, testloader):
    model_engine.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            images, labels = images.cuda(), labels.cuda()
            outputs = model_engine(images.to(torch.bfloat16))
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    if model_engine.global_rank == 0:
        print(f'Accuracy: {100 * correct / total:.2f}%')

def main():
    args = parse_args()

    # 初始化DeepSpeed分布式环境
    deepspeed.init_distributed()

    # 获取全局和本地rank
    rank = torch.distributed.get_rank()
    world_size = torch.distributed.get_world_size()

    trainloader, testloader = get_data_loaders(rank, world_size)

    model = TransformerCIFAR().cuda()

    criterion = nn.CrossEntropyLoss()

    # 初始化DeepSpeed
    model_engine, optimizer, _, _ = deepspeed.initialize(
        args=args,
        model=model,
        model_parameters=model.parameters(),
    )

    num_epochs = 10
    for epoch in range(num_epochs):
        trainloader.sampler.set_epoch(epoch)  # 确保每个epoch数据打乱
        train(model_engine, trainloader, criterion, epoch)
        test(model_engine, testloader)

if __name__ == '__main__':
    main()