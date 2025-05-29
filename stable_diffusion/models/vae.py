from diffusers import AutoencoderKL

class VAE:
    def __init__(self, model_path):
        self.vae = AutoencoderKL.from_pretrained(model_path, subfolder="vae")

    def encode(self, images):
        return self.vae.encode(images).latent_dist.sample() * 0.18215

    def decode(self, latents):
        return self.vae.decode(latents / 0.18215).sample