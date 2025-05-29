import torch

class Config:
    image_size = 512
    latent_size = 64
    latent_channels = 4
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_inference_steps = 50
    guidance_scale = 7.5
    seed = 42
    model_path = "CompVis/stable-diffusion-v1-4"