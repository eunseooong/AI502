import torch
import torchvision
import os

def save_image(x, path):
    x = (x.clamp(-1, 1) + 1) / 2  # [-1,1] -> [0,1]
    grid = torchvision.utils.make_grid(x, nrow=4)
    torchvision.utils.save_image(grid, path)

def sample(model, diffusion, config, num_samples, y=None):
    model.eval()
    with torch.no_grad():
        samples = diffusion.p_sample_loop(
            model,
            (num_samples, config.num_channels, config.image_size, config.image_size),
            config.device,
            y=y
        )
    save_image(samples, os.path.join(config.save_dir, f'sample_final.png'))
