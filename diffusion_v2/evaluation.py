import torchvision.utils as vutils
import os
from tqdm import tqdm
from torch_fidelity import calculate_metrics
from torchvision.utils import save_image
import torch

def real_images(dataloader, save_dir, num_images=1000, image_size=32):
    os.makedirs(save_dir, exist_ok=True)
    count = 0
    for images, _ in tqdm(dataloader, desc="Saving real images"):
        if images.shape[2:] != (image_size, image_size):
            images = torch.nn.functional.interpolate(images, size=(image_size, image_size), mode='bilinear', align_corners=False)

        for img in images:
            if count >= num_images:
                return
            img = (img * 0.5 + 0.5).clamp(0, 1)
            save_image(img, os.path.join(save_dir, f"real_{count:04d}.png"))
            count += 1


def generated_images(model, diffusion, config, save_dir, num_images=1000, use_ddim=False):
    os.makedirs(save_dir, exist_ok=True)
    model.eval()
    batch_size = 100
    total_batches = num_images // batch_size

    with torch.no_grad():
        for batch in tqdm(range(total_batches), desc="Generating fake images"):
            y = torch.arange(batch_size, device=config.device) % config.num_classes if config.conditional else None

            if use_ddim:
                samples = diffusion.ddim_sample_loop(
                    model,
                    shape=(batch_size, config.num_channels, config.image_size, config.image_size),
                    device=config.device,
                    ddim_steps=config.ddim_steps,
                    eta=getattr(config, "eta", 0.0),
                    y=y
                )
            else:
                samples = diffusion.p_sample_loop(
                    model,
                    shape=(batch_size, config.num_channels, config.image_size, config.image_size),
                    device=config.device,
                    y=y
                )

            if samples.shape[2:] != (config.image_size, config.image_size):
                samples = torch.nn.functional.interpolate(samples, size=(config.image_size, config.image_size), mode='bilinear', align_corners=False)

            samples = (samples.clamp(-1, 1) + 1) / 2
            for i, img in enumerate(samples):
                save_image(img, os.path.join(save_dir, f"gen_{batch * batch_size + i:04d}.png"))

    model.train()


def evaluate_fid(real_dir, fake_dir):
    print("Calculating FID...")
    metrics = calculate_metrics(
        input1=fake_dir,
        input2=real_dir,
        cuda=True,
        isc=False,
        fid=True
    )
    print(f"FID score: {metrics['frechet_inception_distance']:.2f}")