from config import Config
from dataloader import get_dataloader
from ddpm import DDPM
from unet import UNet
from training import train
from sampling import sample
import torch
import os

def main():
    config = Config()
    os.makedirs(config.save_dir, exist_ok=True)
    dataset_name = 'CIFAR100'  # or 'MNIST'
    dataloader = get_dataloader(config, dataset_name)
    diffusion = DDPM(config)
    model = UNet(config).to(config.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
    train(model, dataloader, diffusion, optimizer, config)
    # 최종 샘플 생성
    sample(model, diffusion, config, num_samples=16, y=torch.arange(16, device=config.device)%config.num_classes if config.conditional else None, use_ddim=True)

if __name__ == '__main__':
    main()
