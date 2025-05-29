import torch
import torch.nn.functional as F
import os
from sampling import save_image
from evaluation import generated_images, evaluate_fid  

def train(model, dataloader, diffusion, optimizer, config, scheduler=None):
    start_epoch = 0
    checkpoint_path = os.path.join(config.save_dir, "checkpoint.pth")

    if os.path.exists(checkpoint_path):
        print(f"Loading checkpoint from {checkpoint_path} ...")
        checkpoint = torch.load(checkpoint_path, map_location=config.device)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        if scheduler and "scheduler" in checkpoint:
            scheduler.load_state_dict(checkpoint["scheduler"])
        start_epoch = checkpoint["epoch"] + 1
        print(f"Resumed from epoch {start_epoch}")

    model.train()
    for epoch in range(start_epoch, config.num_epochs):
        for i, (x, y) in enumerate(dataloader):
            x = x.to(config.device)
            t = torch.randint(0, config.T, (x.size(0),), device=config.device).long()
            noise = torch.randn_like(x)
            x_t = diffusion.q_sample(x, t, noise)

            if config.conditional:
                y = y.to(config.device)
                pred = model(x_t, t, y)
            else:
                pred = model(x_t, t)

            loss = F.mse_loss(pred, noise)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if i % 100 == 0:
                print(f"Epoch {epoch} Step {i} Loss: {loss.item():.4f}")

        if scheduler:
            scheduler.step()

        torch.save({
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict() if scheduler else None
        }, checkpoint_path)

        if (epoch + 1) % 5 == 0:
            model.eval()
            with torch.no_grad():
                samples = diffusion.p_sample_loop(
                    model,
                    (16, config.num_channels, config.image_size, config.image_size),
                    config.device,
                    y=torch.arange(16, device=config.device) % config.num_classes if config.conditional else None
                )
                save_image(samples, os.path.join(config.save_dir, f'sample_epoch_{epoch+1}.png'))

                gen_dir = os.path.join(config.save_dir, f'generated_epoch_{epoch+1}')
                generated_images(model, diffusion, config, save_dir=gen_dir, num_images=1000)

                real_dir = os.path.join(config.save_dir, "real_images")
                evaluate_fid(real_dir, gen_dir)
            model.train()
