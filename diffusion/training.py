import torch
import torch.nn.functional as F
import os

def train(model, dataloader, diffusion, optimizer, config):
    model.train()
    for epoch in range(config.num_epochs):
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
        # 샘플 저장
        if (epoch+1) % 5 == 0:
            from sampling import save_image
            model.eval()
            with torch.no_grad():
                samples = diffusion.p_sample_loop(
                    model,
                    (16, config.num_channels, config.image_size, config.image_size),
                    config.device,
                    y=torch.arange(16, device=config.device)%config.num_classes if config.conditional else None
                )
                save_image(samples, os.path.join(config.save_dir, f'sample_epoch_{epoch+1}.png'))
            model.train()
