import os
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.utils import save_image
from torchvision.utils import make_grid, save_image

torch.manual_seed(42)
np.random.seed(42)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 创建保存目录
os.makedirs('flow_matching_results', exist_ok=True)

# 超参数设置
args = {
    'batch_size': 128,
    'epochs': 100,
    'lr': 2e-4,
    'image_size': 28,
    'channels': 1,
    'num_timesteps': 1000,  # Flow Matching中用作采样步数
    'save_dir': 'flow_matching_results',
    'sample_interval': 10,   # 每多少个 epoch 保存一次生成的图像
    'sigma_min': 1e-4,      # Flow Matching特有参数：最小噪声水平
    'sigma_max': 1.0        # Flow Matching特有参数：最大噪声水平
}

transform = transforms.Compose([
    transforms.Resize((args['image_size'], args['image_size'])),
    transforms.ToTensor(),
    transforms.Normalize((0.5), (0.5))  # Normalize to [-1, 1] for RGB
])

train_dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
train_loader = DataLoader(dataset=train_dataset, batch_size=args['batch_size'], shuffle=True)

class DownBlock(nn.Module):
    """下采样块"""
    def __init__(self, in_ch, out_ch, time_emb_dim):
        super().__init__()
        self.time_mlp = nn.Linear(time_emb_dim, out_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.bnorm1 = nn.BatchNorm2d(out_ch)
        self.bnorm2 = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU()
        # 下采样使用卷积
        self.pool = nn.Conv2d(out_ch, out_ch, 4, 2, 1)
        
    def forward(self, x, t):
        # 第一个卷积块
        h = self.bnorm1(self.relu(self.conv1(x)))
        
        # 时间嵌入
        time_emb = self.relu(self.time_mlp(t))
        time_emb = time_emb[(..., ) + (None, ) * 2]  # 形状变为 (batch, out_ch, 1, 1)
        h = h + time_emb
        
        # 第二个卷积块
        h = self.bnorm2(self.relu(self.conv2(h)))
        
        # 保存下采样前的特征用于跳跃连接
        skip = h
        
        # 下采样
        h = self.pool(h)
        return h, skip

class UpBlock(nn.Module):
    """上采样块"""
    def __init__(self, in_ch, out_ch, time_emb_dim):
        super().__init__()
        self.time_mlp = nn.Linear(time_emb_dim, out_ch)
        
        # 1. 先将通道数减半，匹配转置卷积的输入要求
        self.reduce_channel = nn.Conv2d(in_ch, out_ch, 1)  # 1x1卷积用于通道缩减
        
        # 2. 转置卷积上采样，输入输出通道均为out_ch
        self.upsample = nn.ConvTranspose2d(out_ch, out_ch, 4, 2, 1)
        
        # 3. 卷积处理
        self.conv1 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.bnorm1 = nn.BatchNorm2d(out_ch)
        self.bnorm2 = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU()
        
    def forward(self, x, skip, t):
        # 确保空间维度匹配
        if x.shape[2:] != skip.shape[2:]:
            x = nn.functional.interpolate(x, size=skip.shape[2:], 
                                         mode='bilinear', align_corners=True)
        
        # 拼接上采样特征和跳跃连接特征 (通道数变为in_ch)
        x = torch.cat([x, skip], dim=1)
        
        # 缩减通道数至out_ch，为转置卷积做准备
        x = self.reduce_channel(x)
        
        # 上采样
        x = self.upsample(x)
        
        # 第一个卷积块
        h = self.bnorm1(self.relu(self.conv1(x)))
        
        # 时间嵌入
        time_emb = self.relu(self.time_mlp(t))
        time_emb = time_emb[(..., ) + (None, ) * 2]  # 形状变为 (batch, out_ch, 1, 1)
        h = h + time_emb
        
        # 第二个卷积块
        h = self.bnorm2(self.relu(self.conv2(h)))
        return h

class SinusoidalPositionEmbeddings(nn.Module):
    """正弦位置嵌入，用于时间步编码"""
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = np.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time.unsqueeze(1) * embeddings.unsqueeze(0)
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=1)
        return embeddings

class UNet(nn.Module):
    """UNet模型 - 现在预测速度场而不是噪声"""
    def __init__(self):
        super().__init__()
        image_channels = args['channels']
        self.down_channels = [image_channels, 32, 64, 128]
        self.up_channels = [128, 64, 32]
        time_emb_dim = 32

        # 时间嵌入
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(time_emb_dim),
            nn.Linear(time_emb_dim, time_emb_dim),
            nn.ReLU()
        )
        
        # 下采样路径
        self.downs = nn.ModuleList([
            DownBlock(self.down_channels[i], self.down_channels[i+1], time_emb_dim)
            for i in range(len(self.down_channels)-1)
        ])
        
        # 瓶颈层
        self.bottleneck = nn.Sequential(
            nn.Conv2d(self.down_channels[-1], self.up_channels[0], 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(self.up_channels[0], self.up_channels[0], 3, padding=1),
            nn.ReLU()
        )
        
        # 上采样路径，输入通道是2倍（当前层+跳跃连接）
        self.ups = nn.ModuleList([
            UpBlock(2 * self.up_channels[i], self.up_channels[i+1], time_emb_dim)
            for i in range(len(self.up_channels)-1)
        ])
        
        # 最终输出层 - 输出速度场
        self.output = nn.Conv2d(self.up_channels[-1], image_channels, 1)

    def forward(self, x, timestep):
        # 时间嵌入 - 支持连续时间[0,1]和离散时间步
        if timestep.dtype == torch.long:
            # 如果是离散时间步，转换为连续时间
            timestep = timestep.float() / args['num_timesteps']
        
        t = self.time_mlp(timestep)
        
        # 下采样并保存跳跃连接
        skips = []
        for down in self.downs:
            x, skip = down(x, t)
            skips.append(skip)
        
        # 瓶颈处理
        x = self.bottleneck(x)
        
        # 上采样，使用保存的跳跃连接（反向顺序）
        for i, up in enumerate(self.ups):
            # 获取对应的跳跃连接（倒数第i+1个）
            skip = skips[-(i+1)]
            x = up(x, skip, t)
        
        return self.output(x)


# # Flow Matching时间调度策略
# def get_time_schedule(batch_size, device):
#     """生成随机时间步，范围[0,1]"""
#     return torch.rand(batch_size, device=device)

# def get_sigma_t(t):
#     """计算时间t对应的噪声水平"""
#     sigma_min = args['sigma_min']
#     sigma_max = args['sigma_max']
#     return sigma_min + t * (sigma_max - sigma_min)

def get_index_from_list(values, t, x_shape):
    """从列表中获取对应时间步t的值 - 保持接口兼容性"""
    batch_size = t.shape[0]
    out = values.gather(-1, t.cpu())
    return out.reshape(batch_size, *((1,) * (len(x_shape) - 1))).to(t.device)


# Flow Matching前向过程：线性插值路径
def flow_matching_sample(x1, t):
    noise = torch.randn_like(x1)
    t_expanded = t.view(-1, 1, 1, 1).expand_as(x1)
    x_t = (1 - t_expanded) * x1 + t_expanded * noise
    velocity_target = noise - x1
    return x_t, velocity_target


def sample_timestep(x_t, t):
    """
    Flow Matching采样的一步
    使用ODE求解器从时间t到t-dt
    """
    # 将离散时间步转换为连续时间
    t_continuous = t.float() / args['num_timesteps']
    
    velocity_pred = model(x_t, t_continuous)
    
    # 计算时间步长
    dt = 1.0 / args['num_timesteps']
    
    x_next = x_t - velocity_pred * dt

    return x_next

def sample(num_samples):
    """生成新图像"""
    model.eval()
    with torch.no_grad():
        # 从纯噪声开始
        x = torch.randn(num_samples, args['channels'], args['image_size'], args['image_size']).to(device)
        
        # Flow Matching采样：从t=1到t=0
        for i in tqdm(reversed(range(0, args['num_timesteps'])), desc='采样', total=args['num_timesteps']):
            t = torch.full((num_samples,), i, device=device, dtype=torch.long)
            x = sample_timestep(x, t)
        
        # 调整到[0, 1]范围
        x = (x.clamp(-1, 1) + 1) / 2
        x = x.cpu().numpy()
    model.train()
    return x

def train():
    """训练函数"""
    print("开始训练...")
    for epoch in range(args['epochs']):
        model.train()
        epoch_loss = 0.0
        for _, (images, _) in enumerate(tqdm(train_loader, desc=f'Epoch {epoch+1}/{args["epochs"]}')):
            images = images.to(device)

            # Flow Matching: 随机采样时间t ∈ [0,1]
            t = torch.rand(images.shape[0], device=device)

            # Flow Matching前向过程
            x_t, velocity_target = flow_matching_sample(images, t)
            
            # 预测速度场
            velocity_pred = model(x_t, t)
            
            # 计算速度场预测损失
            loss = criterion(velocity_pred, velocity_target)
            
            # 反向传播和优化
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
        
        # 计算平均损失
        avg_loss = epoch_loss / len(train_loader)
        print(f'Epoch [{epoch+1}/{args["epochs"]}], Average Loss: {avg_loss:.6f}')
        
        if (epoch + 1) % args['sample_interval'] == 0 or epoch == 0:
            samples = sample(16)
            samples = torch.tensor(samples)
            grid = make_grid(samples, nrow=int(np.sqrt(len(samples))))
            save_path = os.path.join(args['save_dir'], f'samples_epoch_{epoch+1}.png')
            save_image(grid, save_path)


model = UNet().to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=args['lr'])


if __name__ == "__main__":
    train()