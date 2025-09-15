import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
import os
import argparse
import logging
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import numpy as np

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# VQ-VAE Model Components
class VectorQuantizer(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, commitment_cost):
        super(VectorQuantizer, self).__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.commitment_cost = commitment_cost
        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        self.embedding.weight.data.uniform_(-1/num_embeddings, 1/num_embeddings)

    def forward(self, z):
        # Flatten input
        z_flattened = z.view(-1, self.embedding_dim)
        
        # Calculate distances
        distances = (torch.sum(z_flattened**2, dim=1, keepdim=True) 
                    + torch.sum(self.embedding.weight**2, dim=1)
                    - 2 * torch.matmul(z_flattened, self.embedding.weight.t()))
        
        # Get encoding indices
        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
        encodings = torch.zeros(encoding_indices.shape[0], self.num_embeddings, device=z.device)
        encodings.scatter_(1, encoding_indices, 1)
        
        # Quantize
        quantized = torch.matmul(encodings, self.embedding.weight).view(z.shape)
        
        # Loss
        e_latent_loss = F.mse_loss(quantized.detach(), z)
        q_latent_loss = F.mse_loss(quantized, z.detach())
        loss = q_latent_loss + self.commitment_cost * e_latent_loss
        
        # Straight-through estimator
        quantized = z + (quantized - z).detach()
        
        return quantized, loss, encodings

class VQVAE(nn.Module):
    def __init__(self, in_channels=1, hidden_channels=64, num_embeddings=512, embedding_dim=64, commitment_cost=0.25):
        super(VQVAE, self).__init__()
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_channels, hidden_channels, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_channels, embedding_dim, 3, stride=1, padding=1)
        )
        
        # Vector Quantizer
        self.vq = VectorQuantizer(num_embeddings, embedding_dim, commitment_cost)
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(embedding_dim, hidden_channels, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(hidden_channels, hidden_channels, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(hidden_channels, in_channels, 4, stride=2, padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        z = self.encoder(x)
        quantized, vq_loss, _ = self.vq(z)
        x_recon = self.decoder(quantized)
        return x_recon, vq_loss

# Initialize distributed environment
def setup_distributed():
    rank = int(os.environ['RANK'])
    world_size = int(os.environ['WORLD_SIZE'])
    local_rank = int(os.environ['LOCAL_RANK'])
    
    dist.init_process_group(backend='nccl', init_method='env://')
    torch.cuda.set_device(local_rank)
    
    logger.info(f"Rank {rank}/{world_size} initialized")
    return rank, world_size, local_rank

# Training function
def train(model, train_loader, optimizer, epoch, rank, device):
    model.train()
    train_loss = 0
    recon_loss_metric = 0
    vq_loss_metric = 0
    
    for batch_idx, (data, _) in enumerate(train_loader):
        data = data.to(device)
        optimizer.zero_grad()
        
        recon, vq_loss = model(data)
        recon_loss = F.mse_loss(recon, data)
        loss = recon_loss + vq_loss
        
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item()
        recon_loss_metric += recon_loss.item()
        vq_loss_metric += vq_loss.item()
        
        if batch_idx % 100 == 0 and rank == 0:
            logger.info(f'Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)}] '
                       f'Recon Loss: {recon_loss_metric/(batch_idx+1):.4f} '
                       f'VQ Loss: {vq_loss_metric/(batch_idx+1):.4f}')
    
    # Average loss across all processes
    loss_tensor = torch.tensor(train_loss / len(train_loader)).to(device)
    dist.all_reduce(loss_tensor, op=dist.ReduceOp.SUM)
    avg_loss = loss_tensor.item() / dist.get_world_size()
    
    if rank == 0:
        logger.info(f'Epoch {epoch} Average Loss: {avg_loss:.4f}')

# Main function
def main():
    parser = argparse.ArgumentParser(description='VQ-VAE Distributed Training')
    parser.add_argument('--batch-size', type=int, default=128, help='input batch size for training')
    parser.add_argument('--epochs', type=int, default=10, help='number of epochs to train')
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
    parser.add_argument('--model-path', type=str, default='vqvae_model.pth', help='path to save model')
    args = parser.parse_args()

    # Setup distributed environment
    rank, world_size, local_rank = setup_distributed()
    device = torch.device(f'cuda:{local_rank}')

    # Data loading
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    
    train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST('./data', train=False, transform=transform)
    
    # Distributed sampler
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, sampler=train_sampler, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=2)

    # Model setup
    model = VQVAE().to(device)
    model = DDP(model, device_ids=[local_rank])
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # Training loop
    for epoch in range(1, args.epochs + 1):
        train_sampler.set_epoch(epoch)  # Ensure shuffling is different each epoch
        train(model, train_loader, optimizer, epoch, rank, device)
    
    if rank == 0:
        torch.save(model.module.state_dict(), args.model_path)
    
    # Cleanup
    dist.destroy_process_group()

if __name__ == '__main__':
    main()



import torchvision
import matplotlib.pyplot as plt

def test():
    device = 'cuda:0'
    model = VQVAE().to(device)
    model.load_state_dict(torch.load('vqvae_model.pth', map_location=device))

    test_dataset = datasets.MNIST('./data', train=False, download=True, transform=transforms.ToTensor())
    test_loader = DataLoader(test_dataset, batch_size=8, shuffle=True)

    
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

    
