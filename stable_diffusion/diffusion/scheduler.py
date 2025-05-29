from diffusers import DDIMScheduler

class Scheduler:
    def __init__(self, model_path, num_inference_steps):
        self.scheduler = DDIMScheduler.from_pretrained(model_path, subfolder="scheduler")
        self.scheduler.set_timesteps(num_inference_steps)

    def step(self, model_output, t, latents):
        return self.scheduler.step(model_output, t, latents).prev_sample

    def get_timesteps(self):
        return self.scheduler.timesteps

    def init_noise_sigma(self):
        return self.scheduler.init_noise_sigma