import torch
import os
from config import Config
from dataloader import get_dataloader
from ddpm import DDPM
from unet import UNet
from evaluation import real_images, generated_images, evaluate_fid

def main():
    config = Config()
    dataset_name = 'CIFAR100'
    real_dir = os.path.join(config.save_dir, "real_images")
    gen_dir = os.path.join(config.save_dir, "final_generated")

    dataloader = get_dataloader(config, dataset_name)
    diffusion = DDPM(config)
    model = UNet(config).to(config.device)

    checkpoint_path = os.path.join(config.save_dir, "checkpoint.pth")
    assert os.path.exists(checkpoint_path), "Checkpoint not found."
    checkpoint = torch.load(checkpoint_path, map_location=config.device)
    model.load_state_dict(checkpoint["model"])
    print(f"Loaded model from checkpoint at epoch {checkpoint['epoch']}")

    if "loss" in checkpoint:
        print(f"Last recorded loss: {checkpoint['loss']:.6f}")

    if not os.path.exists(real_dir) or len(os.listdir(real_dir)) < 1000:
        print("Saving real images...")
        real_images(dataloader, real_dir, num_images=1000)

    print("🎨 Generating final samples...")
    generated_images(model, diffusion, config, save_dir=gen_dir, num_images=1000)

    print("📊 Evaluating final FID...")
    evaluate_fid(real_dir, gen_dir)

if __name__ == '__main__':
    main()
