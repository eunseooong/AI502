import torch
from models.text_encoder import TextEncoder
from models.vae import VAE
from models.unet import UNet
from diffusion.scheduler import Scheduler
from utils.image_utils import latents_to_image

class StableDiffusionPipeline:
    def __init__(self, config):
        self.device = config.device
        self.text_encoder = TextEncoder(config.model_path)
        self.vae = VAE(config.model_path)
        self.unet = UNet(config.model_path).model.to(self.device)
        self.scheduler = Scheduler(config.model_path, config.num_inference_steps)
        self.guidance_scale = config.guidance_scale

    def generate(self, prompt):
        if isinstance(prompt, str):
            prompt = [prompt]  # transforme "a cat" en ["a cat"]

        batch_size = len(prompt)

        text_embed = self.text_encoder.encode(prompt).to(self.device)
        uncond_embed = self.text_encoder.encode([""] * batch_size).to(self.device)
        print("text_embed shape:", text_embed.shape)       # Ex: [B, 77, 768]
        print("uncond_embed shape:", uncond_embed.shape) 
        cond_embed = torch.cat([uncond_embed, text_embed], dim=0)  # concat sur batch dim (0)

        latents = torch.randn((2 * batch_size, 4, 64, 64), device=self.device)
        latents = latents * self.scheduler.init_noise_sigma()

        for t in self.scheduler.get_timesteps():
            noise_pred = self.unet(latents, t, encoder_hidden_states=cond_embed).sample
            uncond, cond = noise_pred.chunk(2)
            noise_pred = uncond + self.guidance_scale * (cond - uncond)
            latents = self.scheduler.step(noise_pred, t, latents)

        latents = latents[batch_size:]  # on garde la partie conditionnée

        images = self.vae.decode(latents)
        return latents_to_image(images)
