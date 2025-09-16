import os
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.utils import save_image, make_grid

torch.manual_seed(42)
np.random.seed(42)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 创建保存目录
os.makedirs('diffusion_results', exist_ok=True)

# 超参数设置
args = {
    'batch_size': 128,
    'epochs': 100,
    'lr': 2e-4,
    'image_size': 28,
    'channels': 1,
    'num_timesteps': 1000,  # 扩散步骤数
    'num_inference_steps': 50,  # DDIM推理时的步骤数（可以少于训练步骤）
    'beta_start': 0.0001,   # 噪声 scheduler 参数
    'beta_end': 0.02,
    'eta': 0.0,  # DDIM参数：0为完全确定性，1为DDPM
    'save_dir': 'diffusion_results',
    'sample_interval': 10   # 每多少个 epoch 保存一次生成的图像
}

transform = transforms.Compose([
    transforms.Resize((args['image_size'], args['image_size'])),
    transforms.ToTensor(),
    transforms.Normalize((0.5), (0.5))  # Normalize to [-1, 1] for RGB
])

train_dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
train_loader = DataLoader(dataset=train_dataset, batch_size=args['batch_size'], shuffle=True)


# 定义噪声 scheduler
def linear_beta_schedule(timesteps, beta_start, beta_end):
    """线性调度的噪声参数"""
    return torch.linspace(beta_start, beta_end, timesteps)

# 计算扩散过程中的参数
betas = linear_beta_schedule(args['num_timesteps'], args['beta_start'], args['beta_end'])
alphas = 1.0 - betas
alphas_cumprod = torch.cumprod(alphas, dim=0)
alphas_cumprod_prev = torch.cat([torch.tensor([1.0]), alphas_cumprod[:-1]])

sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - alphas_cumprod)

def get_index_from_list(values, t, x_shape):
    """从列表中获取对应时间步t的值"""
    batch_size = t.shape[0]
    out = values.gather(-1, t.cpu())
    return out.reshape(batch_size, *((1,) * (len(x_shape) - 1))).to(t.device)


# 前向扩散过程：向图像添加噪声
def forward_diffusion_sample(x0, t):
    noise = torch.randn_like(x0)
    sqrt_alphas_cumprod_t = get_index_from_list(sqrt_alphas_cumprod, t, x0.shape)
    sqrt_one_minus_alphas_cumprod_t = get_index_from_list(sqrt_one_minus_alphas_cumprod, t, x0.shape)
    
    # 根据扩散公式计算添加噪声后的图像
    return sqrt_alphas_cumprod_t * x0 + sqrt_one_minus_alphas_cumprod_t * noise, noise


# 定义UNet模型作为噪声预测网络
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
        
        # 最终输出层
        self.output = nn.Conv2d(self.up_channels[-1], image_channels, 1)

    def forward(self, x, timestep):
        # 时间嵌入
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


def ddim_step(x_t, t, t_prev, eta=0.0):
    """
    DDIM单步采样
    Args:
        x_t: 当前时间步的噪声图像
        t: 当前时间步
        t_prev: 前一个时间步
        eta: 随机性参数，0为完全确定性
    """
    # 预测噪声
    with torch.no_grad():
        noise_pred = model(x_t, t)
    
    alphas_cumprod_t = get_index_from_list(alphas_cumprod, t, x_t.shape)
    alphas_cumprod_t_prev = get_index_from_list(alphas_cumprod, t_prev, x_t.shape) if t_prev[0] >= 0 else torch.ones_like(alphas_cumprod_t)
    
    # 计算预测的原始图像 x_0
    sqrt_alpha_cumprod_t = torch.sqrt(alphas_cumprod_t)
    sqrt_one_minus_alpha_cumprod_t = torch.sqrt(1 - alphas_cumprod_t)
    
    pred_x0 = (x_t - sqrt_one_minus_alpha_cumprod_t * noise_pred) / sqrt_alpha_cumprod_t
    
    # 计算方向向量
    sqrt_alpha_cumprod_t_prev = torch.sqrt(alphas_cumprod_t_prev)
    sqrt_one_minus_alpha_cumprod_t_prev = torch.sqrt(1 - alphas_cumprod_t_prev)
    
    # DDIM采样公式
    dir_xt = sqrt_one_minus_alpha_cumprod_t_prev * noise_pred
    
    # 添加随机性（eta > 0时）
    if eta > 0:
        sigma_t = eta * torch.sqrt((1 - alphas_cumprod_t) / (1 - alphas_cumprod_t)) * torch.sqrt(1 - alphas_cumprod_t / alphas_cumprod_t_prev)
        noise = torch.randn_like(x_t)
        dir_xt = dir_xt + sigma_t * noise
    
    x_t_prev = sqrt_alpha_cumprod_t_prev * pred_x0 + dir_xt
    
    return x_t_prev

def ddim_sample(num_samples, num_inference_steps=None, eta=None):
    """
    DDIM采样生成新图像
    Args:
        model: 训练好的噪声预测模型
        num_samples: 生成样本数量
        num_inference_steps: 推理步骤数，可以少于训练步骤数
        eta: 随机性参数
    """
    if num_inference_steps is None:
        num_inference_steps = args['num_inference_steps']
    if eta is None:
        eta = args['eta']
    
    model.eval()
    with torch.no_grad():
        x = torch.randn(num_samples, args['channels'], args['image_size'], args['image_size']).to(device)
        timesteps = torch.linspace(args['num_timesteps'] - 1, 0, num_inference_steps, dtype=torch.long)
        
        for i in tqdm(range(len(timesteps)), desc='DDIM采样'):
            t = torch.full((num_samples,), timesteps[i], device=device, dtype=torch.long)
            t_prev = torch.full((num_samples,), timesteps[i+1] if i < len(timesteps)-1 else -1, device=device, dtype=torch.long)
            
            x = ddim_step(x, t, t_prev, eta)
        
        # 调整到[0, 1]范围
        x = (x.clamp(-1, 1) + 1) / 2
        x = x.cpu().numpy()
    model.train()
    return x

def ddpm_sample_timestep(x_t, t):
    """
    从x_{t}采样x_{t-1}
    """
    betas_t = get_index_from_list(betas, t, x_t.shape)
    sqrt_one_minus_alphas_cumprod_t = get_index_from_list(sqrt_one_minus_alphas_cumprod, t, x_t.shape)
    sqrt_recip_alphas_t = get_index_from_list(torch.sqrt(1.0 / alphas), t, x_t.shape)
    
    # 预测噪声
    noise_pred = model(x_t, t)
    
    model_mean = sqrt_recip_alphas_t * (x_t - betas_t / sqrt_one_minus_alphas_cumprod_t * noise_pred)
    
    if t[0] == 0:
        return model_mean
    else:
        noise = torch.randn_like(x_t)
        variance = betas_t
        return model_mean + torch.sqrt(variance) * noise


def ddpm_sample(num_samples):
    """生成新图像"""
    model.eval()
    with torch.no_grad():
        x = torch.randn(num_samples, args['channels'], args['image_size'], args['image_size']).to(device)
        
        for i in tqdm(reversed(range(0, args['num_timesteps'])), desc='DDPM采样', total=args['num_timesteps']):
            t = torch.full((num_samples,), i, device=device, dtype=torch.long)
            x = ddpm_sample_timestep(x, t)
        
        # 调整到[0, 1]范围
        x = (x.clamp(-1, 1) + 1) / 2
        x = x.cpu().numpy()
    model.train()
    return x

model = UNet().to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=args['lr'])


def train():
    print("开始训练...")
    for epoch in range(args['epochs']):
        model.train()
        epoch_loss = 0.0
        for _, (images, _) in enumerate(tqdm(train_loader, desc=f'Epoch {epoch+1}/{args["epochs"]}')):
            images = images.to(device)

            t = torch.randint(0, args['num_timesteps'], (images.shape[0],), device=device).long()

            x_noisy, noise = forward_diffusion_sample(images, t)
            noise_pred = model(x_noisy, t)
            loss = criterion(noise_pred, noise)
            
            # 反向传播和优化
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
        
        # 计算平均损失
        avg_loss = epoch_loss / len(train_loader)
        print(f'Epoch [{epoch+1}/{args["epochs"]}], Average Loss: {avg_loss:.6f}')
        
        if (epoch + 1) % args['sample_interval'] == 0 or epoch == 0:
            samples = ddpm_sample(16)
            samples = torch.tensor(samples)
            grid = make_grid(samples, nrow=int(np.sqrt(len(samples))))
            save_path = os.path.join(args['save_dir'], f'ddpm_samples_epoch_{epoch+1}.png')
            save_image(grid, save_path)

            samples = ddim_sample(16)
            samples = torch.tensor(samples)
            grid = make_grid(samples, nrow=int(np.sqrt(len(samples))))
            save_path = os.path.join(args['save_dir'], f'ddim_samples_epoch_{epoch+1}.png')
            save_image(grid, save_path)


if __name__ == "__main__":
    train()
    # num_inference_steps = 1000
    # samples = ddim_sample(16, num_inference_steps=num_inference_steps)
    # samples = torch.tensor(samples)
    # grid = make_grid(samples, nrow=int(np.sqrt(len(samples))))
    # save_path = os.path.join(args['save_dir'], f'ddim_samples_epoch_{num_inference_steps}.png')
    # save_image(grid, save_path)
