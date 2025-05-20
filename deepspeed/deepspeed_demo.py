import argparse
import logging
import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader
import deepspeed
from torch.utils.tensorboard import SummaryWriter

# 彩色日志支持
class CustomFormatter(logging.Formatter):
    grey = "\x1b[38;20m"
    green = "\x1b[32;20m"  # 绿色用于loss
    red = "\x1b[31;20m"    # 红色用于accuracy
    yellow = "\x1b[33;20m"
    bold_red = "\x1b[31;1m"
    reset = "\x1b[0m"
    format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

    FORMATS = {
        logging.DEBUG: grey + format + reset,
        logging.INFO: format + reset,  # 普通信息不特别着色
        logging.WARNING: yellow + format + reset,
        logging.ERROR: red + format + reset,
        logging.CRITICAL: bold_red + format + reset
    }

    def format(self, record):
        log_fmt = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(log_fmt)

        msg = record.getMessage()
        record.msg = msg
        return formatter.format(record)

# 配置日志
def setup_logger(args):
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    
    # 创建日志目录
    os.makedirs(args.log_dir, exist_ok=True)
    
    # 确保每个进程有独立的日志文件
    log_file = os.path.join(args.log_dir, f"train_rank{args.local_rank}.log")
    
    # 添加文件处理器
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    logger.addHandler(file_handler)
    
    # 只在rank 0进程添加控制台输出
    if args.local_rank == 0:
        console_handler = logging.StreamHandler()
        # 设置彩色格式
        console_handler.setFormatter(CustomFormatter())
        logger.addHandler(console_handler)
    
    return logger

# 定义简单的卷积神经网络模型
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# 测试模型
def evaluate(model_engine, test_loader, criterion, device, writer=None, epoch=None, logger=None):
    model_engine.eval()
    correct = 0
    total = 0
    test_loss = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            if model_engine.fp16_enabled():
                inputs = inputs.half()
            
            outputs = model_engine(inputs)
            test_loss += criterion(outputs, labels).item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    accuracy = 100 * correct / total
    avg_loss = test_loss / len(test_loader)
    
    if logger:
        logger.info(f'Test {CustomFormatter.red}Accuracy: {accuracy}%{CustomFormatter.reset}')
    
    if writer and epoch is not None:
        writer.add_scalar('test/loss', avg_loss, epoch)
        writer.add_scalar('test/accuracy', accuracy, epoch)
    
    return accuracy, avg_loss

def parse_args():
    parser = argparse.ArgumentParser(description='DeepSpeed CIFAR-10 Training Example')
    
    # DeepSpeed参数
    parser.add_argument('--local_rank', type=int, default=-1,
                        help='local rank passed from distributed launcher')
    
    # 训练参数
    parser.add_argument('--batch_size', type=int, default=32,
                        help='input batch size for training (default: 32)')
    parser.add_argument('--epochs', type=int, default=10,
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='learning rate (default: 0.001)')
    
    # 日志参数
    parser.add_argument('--log_dir', type=str, default='./logs',
                        help='Directory to save logs and TensorBoard summaries')
    parser.add_argument('--log_interval', type=int, default=100,
                        help='Logging interval (batches)')
    parser.add_argument('--exp_name', type=str, default='deepspeed_demo',
                        help='Experiment name for logging and TensorBoard')
    
    parser = deepspeed.add_config_arguments(parser)
    args = parser.parse_args()
    
    # 为TensorBoard设置唯一的日志目录
    args.tb_dir = os.path.join(args.log_dir, f"tb_{args.exp_name}_{time.strftime('%Y%m%d-%H%M%S')}")
    return args

def main():
    args = parse_args()
    
    # 初始化DeepSpeed
    deepspeed.init_distributed()
    local_rank = int(os.environ['LOCAL_RANK'])
    args.local_rank = local_rank
    
    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 初始化日志
    logger = setup_logger(args)
    logger.info(f"Starting training on rank {local_rank}")
    
    # 仅在rank 0初始化TensorBoard writer
    writer = None
    if local_rank == 0:
        os.makedirs(args.tb_dir, exist_ok=True)
        writer = SummaryWriter(args.tb_dir)
        logger.info(f"TensorBoard logs will be saved to {args.tb_dir}")
    
    # 数据预处理
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    # 加载数据集
    train_dataset = datasets.CIFAR10(root='./data', train=True,
                                     download=True, transform=transform)
    test_dataset = datasets.CIFAR10(root='./data', train=False,
                                    download=True, transform=transform)
    
    # 使用DistributedSampler进行数据分片
    train_sampler = DistributedSampler(train_dataset, shuffle=True)
    test_sampler = DistributedSampler(test_dataset, shuffle=False)
    
    # 创建数据加载器
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                             sampler=train_sampler)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size,
                            sampler=test_sampler)
    
    # 初始化模型
    model = SimpleCNN().to(device)
    
    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    # 初始化DeepSpeed引擎
    model_engine, optimizer, train_loader, _ = deepspeed.initialize(
        args=args,
        model=model,
        model_parameters=model.parameters(),
        training_data=train_dataset,
    )
    
    # 训练总步数
    total_steps = 0
    
    # 训练循环
    for epoch in range(args.epochs):
        train_sampler.set_epoch(epoch)
        
        model_engine.train()
        running_loss = 0.0
        epoch_start_time = time.time()
        
        for i, (inputs, labels) in enumerate(train_loader):
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            # 混合精度训练需要将输入转为half类型
            if model_engine.fp16_enabled():
                inputs = inputs.half()
            
            outputs = model_engine(inputs)
            loss = criterion(outputs, labels)
            
            model_engine.backward(loss)
            model_engine.step()
            
            running_loss += loss.item()
            total_steps += 1
            
            # 定期记录训练日志
            if i % args.log_interval == 0 and local_rank == 0:
                avg_loss = running_loss / (i + 1)
                logger.info(f'Epoch [{epoch+1}/{args.epochs}], Batch [{i}/{len(train_loader)}], '
                           f'{CustomFormatter.green}Loss: {avg_loss:.4f}{CustomFormatter.reset}')
                
                if writer:
                    writer.add_scalar('train/loss', avg_loss, total_steps)
                    # 记录学习率
                    lr = optimizer.param_groups[0]['lr']
                    writer.add_scalar('train/learning_rate', lr, total_steps)
        
        # 每个epoch结束记录时间和平均损失
        epoch_time = time.time() - epoch_start_time
        avg_loss = running_loss / len(train_loader)
        
        if local_rank == 0:
            logger.info(f'Epoch {epoch+1}/{args.epochs} completed in {epoch_time:.2f}s, '
                       f'Average {CustomFormatter.green}Loss: {avg_loss:.4f}{CustomFormatter.reset}')
            if writer:
                writer.add_scalar('epoch/loss', avg_loss, epoch)
                writer.add_scalar('epoch/time', epoch_time, epoch)
            
            # 每个epoch结束后进行测试
            logger.info(f"Evaluating model on test set after epoch {epoch+1}")
            accuracy, test_loss = evaluate(model_engine, test_loader, criterion, device, writer, epoch, logger)
    
    if local_rank == 0 and writer:
        writer.close()

if __name__ == "__main__":
    main()    