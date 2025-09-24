import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np

# 设置随机种子，确保结果可复现
torch.manual_seed(42)
np.random.seed(42)

# 定义生成器
class Generator(nn.Module):
    def __init__(self, input_dim=100, output_dim=784):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, 1024),
            nn.ReLU(),
            nn.Linear(1024, output_dim),
            nn.Tanh()  # 输出范围在[-1, 1]，与标准化后的图像匹配
        )
    
    def forward(self, x):
        return self.model(x)

# 定义判别器
class Discriminator(nn.Module):
    def __init__(self, input_dim=784):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 1024),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(1024, 512),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(256, 1),
            nn.Sigmoid()  # 输出概率值，范围在[0, 1]
        )
    
    def forward(self, x):
        return self.model(x)

# 数据预处理和加载
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])  # 将像素值从[0,1]标准化到[-1,1]
])

# 加载MNIST数据集
train_dataset = datasets.MNIST(
    root='./data', train=True, transform=transform, download=True
)
train_loader = DataLoader(dataset=train_dataset, batch_size=128, shuffle=True)

# 初始化模型、损失函数和优化器
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
latent_dim = 100  # 潜在空间维度

generator = Generator(input_dim=latent_dim).to(device)
discriminator = Discriminator().to(device)

# 二元交叉熵损失函数
criterion = nn.BCELoss()

# 优化器
gen_optimizer = optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
disc_optimizer = optim.Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))

# 训练GAN
epochs = 50
for epoch in range(epochs):
    for batch_idx, (real_images, _) in enumerate(train_loader):
        batch_size = real_images.size(0)
        
        # 将图像展平并移动到设备
        real_images = real_images.view(batch_size, -1).to(device)
        
        # 真实标签和虚假标签
        real_labels = torch.ones(batch_size, 1).to(device)
        fake_labels = torch.zeros(batch_size, 1).to(device)
        
        # 训练判别器
        # 1. 训练真实图像
        disc_optimizer.zero_grad()
        real_outputs = discriminator(real_images)
        disc_loss_real = criterion(real_outputs, real_labels)
        disc_loss_real.backward()
        
        # 2. 训练生成的图像
        z = torch.randn(batch_size, latent_dim).to(device)  # 随机噪声
        fake_images = generator(z)
        fake_outputs = discriminator(fake_images.detach())  #  detach()阻止梯度流向生成器
        disc_loss_fake = criterion(fake_outputs, fake_labels)
        disc_loss_fake.backward()
        
        # 总判别器损失
        disc_loss = disc_loss_real + disc_loss_fake
        disc_optimizer.step()
        
        # 训练生成器
        gen_optimizer.zero_grad()
        fake_outputs = discriminator(fake_images)
        # 生成器希望判别器将生成的图像判断为真实的
        gen_loss = criterion(fake_outputs, real_labels)
        gen_loss.backward()
        gen_optimizer.step()
        
        # 打印训练进度
        if (batch_idx + 1) % 100 == 0:
            print(f'Epoch [{epoch+1}/{epochs}], Batch [{batch_idx+1}/{len(train_loader)}], '
                  f'Discriminator Loss: {disc_loss.item():.4f}, Generator Loss: {gen_loss.item():.4f}')
    
    # 每个epoch结束后生成一些样本查看效果
    if (epoch + 1) % 10 == 0:
        with torch.no_grad():
            z = torch.randn(16, latent_dim).to(device)
            generated_images = generator(z).view(-1, 1, 28, 28).cpu()
            
            # 反标准化
            generated_images = generated_images * 0.5 + 0.5
            
            # 显示生成的图像
            fig, axes = plt.subplots(4, 4, figsize=(8, 8))
            for i, ax in enumerate(axes.flatten()):
                ax.imshow(generated_images[i].squeeze(), cmap='gray')
                ax.axis('off')
            plt.suptitle(f'Generated Images after Epoch {epoch+1}')
            plt.show()

# 保存模型
torch.save(generator.state_dict(), 'generator.pth')
torch.save(discriminator.state_dict(), 'discriminator.pth')
