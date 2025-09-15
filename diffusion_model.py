import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_s_curve


s_curve, _ = make_s_curve(10**4, noise=0.1)
s_curve = s_curve[:,[0,2]] / 10.0
print("shape of s:", np.shape(s_curve))

data = s_curve.T

fig,ax = plt.subplots()
ax.scatter(*data, color='blue', edgecolor='white')
ax.axis('off')

dataset = torch.Tensor(s_curve).float().cuda()



num_steps = 100

#制定每一步的beta
betas = torch.linspace(-6, 6,num_steps).cuda()
betas = torch.sigmoid(betas) * (0.5e-2 - 1e-5) + 1e-5

#计算alpha、alpha_prod、alpha_prod_previous、alpha_bar_sqrt等变量的值
alphas = 1 - betas
alphas_prod = torch.cumprod(alphas,0)
alphas_prod_p = torch.cat([torch.tensor([1], device=betas.device).float(), alphas_prod[:-1]],0)
alphas_bar_sqrt = torch.sqrt(alphas_prod)
one_minus_alphas_bar_log = torch.log(1 - alphas_prod)
one_minus_alphas_bar_sqrt = torch.sqrt(1 - alphas_prod)

print("all the same shape", betas.shape)

import torch
import torch.nn as nn

class MLPDiffusion(nn.Module):
    def __init__(self,n_steps,num_units=128):
        super(MLPDiffusion,self).__init__()
        self.linears = nn.ModuleList(
            [
                nn.Linear(2,num_units),nn.ReLU(),
                nn.Linear(num_units,num_units),nn.ReLU(),
                nn.Linear(num_units,num_units),nn.ReLU(),
                nn.Linear(num_units,2),
            ]
        )
        self.step_embeddings = nn.ModuleList(
            [
                nn.Embedding(n_steps, num_units),
                nn.Embedding(n_steps, num_units),
                nn.Embedding(n_steps, num_units),
            ]
        )
    def forward(self,x, t):
        for idx, embedding_layer in enumerate(self.step_embeddings):
            t_embedding = embedding_layer(t)
            x = self.linears[2*idx](x)  #Linear
            x += t_embedding
            x = self.linears[2*idx+1](x) #ReLU
        x = self.linears[-1](x)
        return x


def diffusion_loss_fn(model, x_0, n_steps):
    """对任意时刻t进行采样计算loss"""
    batch_size = x_0.shape[0]
    
    #对一个batchsize样本生成随机的时刻t
    t = torch.randint(0,n_steps,size=(batch_size//2,), device=x_0.device)
    t = torch.cat([t,n_steps-1-t],dim=0)
    t = t.unsqueeze(-1)

    #生成随机噪音eps
    noise = torch.randn_like(x_0)
    
    #构造模型的输入
    x = alphas_bar_sqrt[t] * x_0 + noise * one_minus_alphas_bar_sqrt[t]

    output = model(x,t.squeeze(-1)) #送入模型，得到t时刻的随机噪声预测值
    return (noise - output).square().mean()


print('Training model...')
batch_size = 128
dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,shuffle=True)
num_epoch = 4000

model = MLPDiffusion(num_steps).cuda() #输出维度是2，输入是x和step
optimizer = torch.optim.Adam(model.parameters(),lr=1e-3)

for t in range(num_epoch):
    for idx,batch_x in enumerate(dataloader):
        loss = diffusion_loss_fn(model, batch_x.cuda(), num_steps)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
        optimizer.step()
        
    if(t % 100 == 0):
        print(loss)



def p_sample(model, x, t):
    """从x[T]采样t时刻的重构值"""
    t = torch.tensor([t], device=x.device)
    z_hat = model(x,t)
    mean = (1 / (1 - betas[t]).sqrt()) * (x - (betas[t] / one_minus_alphas_bar_sqrt[t] * z_hat))
    z = torch.randn_like(x)
    sigma_t = betas[t].sqrt()
    sample = mean + sigma_t * z
    return sample

def p_sample_loop(model, shape, n_steps, device):
    """从x[T]恢复x[T-1]、x[T-2]|...x[0]"""
    cur_x = torch.randn(shape, device=device)
    x_seq = [cur_x]
    for i in reversed(range(n_steps)):
        cur_x = p_sample(model, cur_x, i)
        x_seq.append(cur_x)
    return x_seq

x_seq = p_sample_loop(model, dataset.shape, num_steps, torch.device('cuda'))
fig, axs = plt.subplots(1, 10, figsize=(28,3))
for i in range(1, 11):
    cur_x = x_seq[i * 10].detach().cpu()
    axs[i-1].scatter(cur_x[:, 0],cur_x[:, 1], color='red', edgecolor='white')
    axs[i-1].set_axis_off()
    axs[i-1].set_title('$q(\mathbf{x}_{'+str(i*10)+'})$')
