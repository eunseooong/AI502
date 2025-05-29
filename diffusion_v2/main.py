from config import Config
from dataloader import get_dataloader
from ddpm import DDPM
from unet import UNet
from training import train
from sampling import sample
from evaluation import real_images, generated_images, evaluate_fid
from torch.optim.lr_scheduler import CosineAnnealingLR
import torch
import os

def main():
    config = Config()
    os.makedirs(config.save_dir, exist_ok=True)
    dataset_name = 'CIFAR100'
    dataloader = get_dataloader(config, dataset_name)
    diffusion = DDPM(config)
    model = UNet(config).to(config.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
    scheduler = CosineAnnealingLR(optimizer, T_max=config.num_epochs, eta_min=5e-6)

    real_dir = os.path.join(config.save_dir, "real_images")
    if not os.path.exists(real_dir) or len(os.listdir(real_dir)) < 1000:
        real_images(dataloader, real_dir, num_images=1000)

    train(model, dataloader, diffusion, optimizer, config, scheduler=scheduler)

    gen_dir = os.path.join(config.save_dir, "generated_images")
    generated_images(model, diffusion, config, save_dir=gen_dir, num_images=1000, use_ddim=False)

    evaluate_fid(real_dir, gen_dir)

if __name__ == '__main__':
    main()
