import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torch.utils.data.distributed import DistributedSampler
import os

# VAE模型定义
class VAE(nn.Module):
    def __init__(self, input_dim=784, hidden_dim=400, latent_dim=20):
        super(VAE, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU()
        )
        self.fc_mu = nn.Linear(hidden_dim // 2, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim // 2, latent_dim)
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
            nn.Sigmoid()
        )
    
    def encode(self, x):
        h = self.encoder(x)
        return self.fc_mu(h), self.fc_logvar(h)
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z):
        return self.decoder(z)
    
    def forward(self, x):
        mu, logvar = self.encode(x.view(-1, 784)) # 输出均值（mu）和对数方差（logvar）

        # VAE 是一种生成模型，通过编码器将输入数据 x 映射到潜在空间的分布参数（均值 \mu 和方差  \sigma^2）
        # 然后从该分布中采样潜在变量 z ，再通过解码器重建数据。

        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

# 损失函数
def loss_function(recon_x, x, mu, logvar):
    BCE = nn.functional.binary_cross_entropy(recon_x, x.view(-1, 784), reduction='sum')
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return BCE + KLD

# 分布式训练函数
def train(local_rank, args):
    # 获取全局 rank 和 world_size
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    # 设置设备
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    
    # 数据加载
    transform = transforms.Compose([transforms.ToTensor()])
    dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, sampler=sampler, num_workers=4)
    
    # 模型和优化器
    model = VAE().to(device)
    model = DDP(model, device_ids=[local_rank] if torch.cuda.is_available() else None)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    
    # 训练循环
    model.train()
    for epoch in range(args.epochs):
        sampler.set_epoch(epoch)
        total_loss = 0
        for batch_idx, (data, _) in enumerate(dataloader):
            data = data.to(device)
            optimizer.zero_grad()
            recon_batch, mu, logvar = model(data)
            loss = loss_function(recon_batch, data, mu, logvar)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
            if batch_idx % 100 == 0 and rank == 0:
                print(f'Epoch {epoch}, Batch {batch_idx}, Loss: {loss.item() / len(data):.6f}')
        
        # 同步所有进程的损失
        total_loss_tensor = torch.tensor(total_loss, device=device)
        dist.all_reduce(total_loss_tensor, op=dist.ReduceOp.SUM)
        if rank == 0:
            print(f'Epoch {epoch}, Average Loss: {total_loss_tensor.item() / (len(dataloader.dataset) * world_size):.6f}')
    
    # 保存模型（仅在 rank 0）
    if rank == 0:
        torch.save(model.module.state_dict(), args.model_path)
    
    # 确保所有进程同步
    dist.barrier()

# 主函数
def main():
    import argparse
    parser = argparse.ArgumentParser(description="Distributed VAE Training")
    parser.add_argument('--epochs', type=int, default=10, help='number of epochs')
    parser.add_argument('--batch-size', type=int, default=128, help='batch size')
    parser.add_argument('--model-path', type=str, default='vae_model.pth', help='path to save model')
    args = parser.parse_args()

    # 初始化分布式环境
    dist.init_process_group(backend='nccl')
    local_rank = int(os.environ['LOCAL_RANK'])
    
    # 运行训练
    train(local_rank, args)
    
    # 清理
    dist.destroy_process_group()

if __name__ == '__main__':
    main()





import torchvision
import matplotlib.pyplot as plt

def test(model_path='vae_model.pth', save_dir='./reconstructions'):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    transform = transforms.ToTensor()
    test_dataset = datasets.MNIST('./data', train=False, download=True, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=True)

    model = VAE()
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    os.makedirs(save_dir, exist_ok=True)
    # test_loader中读取数据
    reco = next(iter(test_loader))[0].to(device)
    with torch.no_grad():
        recon_batch, _, _ = model(reco.view(-1, 784))

    recon_batch = recon_batch.view(-1, 1, 28, 28).cpu()

    fig, axs = plt.subplots(2, 10, figsize=(10, 2))
    for i in range(10):
        axs[0, i].imshow(recon_batch[i][0], cmap='gray')
        axs[0, i].axis('off')
        axs[1, i].imshow(reco[i].view(28, 28).cpu().detach(), cmap='gray')
        axs[1, i].axis('off')

    plt.savefig(save_path)

    
