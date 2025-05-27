import torch
import torchvision
import os

def save_image(x, path):
    x = (x.clamp(-1, 1) + 1) / 2  # [-1,1] -> [0,1]
    grid = torchvision.utils.make_grid(x, nrow=4)
    torchvision.utils.save_image(grid, path)


def sample(model, diffusion, config, num_samples, y=None, use_ddim=False):
    model.eval()
    with torch.no_grad():
        shape = (num_samples, config.num_channels, config.image_size, config.image_size)
        if use_ddim:
            samples = diffusion.ddim_sample_loop(
                model,
                shape=shape,
                device=config.device,
                ddim_steps=config.ddim_steps,
                eta=getattr(config, "eta", 0.0),
                y=y
            )
        else:
            samples = diffusion.p_sample_loop(
                model,
                shape=shape,
                device=config.device,
                y=y
            )
    save_path = os.path.join(config.save_dir, f'sample_final_{"ddim" if use_ddim else "ddpm"}.png')
    save_image(samples, save_path)

