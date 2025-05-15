import torch
import math

class DDPM:
    def __init__(self, config):
        self.T = config.T
        if config.beta_schedule == 'linear':
            self.betas = torch.linspace(1e-4, 0.02, self.T)
        elif config.beta_schedule == 'cosine':
            steps = self.T + 1
            x = torch.linspace(0, self.T, steps)
            alphas_cumprod = torch.cos(((x / self.T) + 0.008) / 1.008 * math.pi * 0.5) ** 2
            alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
            self.betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
            self.betas = torch.clip(self.betas, 0, 0.999)
        else:
            raise NotImplementedError

        self.alphas = 1. - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alphas_cumprod_prev = torch.cat([torch.tensor([1.]), self.alphas_cumprod[:-1]])
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1 - self.alphas_cumprod)

        self.to(config.device)

    def to(self, device):
        self.betas = self.betas.to(device)
        self.alphas = self.alphas.to(device)
        self.alphas_cumprod = self.alphas_cumprod.to(device)
        self.alphas_cumprod_prev = self.alphas_cumprod_prev.to(device)
        self.sqrt_alphas_cumprod = self.sqrt_alphas_cumprod.to(device)
        self.sqrt_one_minus_alphas_cumprod = self.sqrt_one_minus_alphas_cumprod.to(device)

    def q_sample(self, x_0, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x_0)
        sqrt_alphas_cumprod_t = self.sqrt_alphas_cumprod[t].view(-1, 1, 1, 1)
        sqrt_one_minus_alphas_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1, 1)
        return sqrt_alphas_cumprod_t * x_0 + sqrt_one_minus_alphas_cumprod_t * noise

    def p_sample(self, model, x_t, t, y=None):
        betas_t = self.betas[t].view(-1, 1, 1, 1)
        sqrt_one_minus_alphas_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1, 1)
        sqrt_recip_alphas_t = (1. / torch.sqrt(self.alphas[t])).view(-1, 1, 1, 1)
        model_mean = sqrt_recip_alphas_t * (x_t - betas_t / sqrt_one_minus_alphas_cumprod_t * model(x_t, t, y))
        if t[0] == 0:
            return model_mean
        noise = torch.randn_like(x_t)
        posterior_var = self.betas[t].view(-1, 1, 1, 1)
        return model_mean + torch.sqrt(posterior_var) * noise

    def p_sample_loop(self, model, shape, device, y=None):
        x_t = torch.randn(shape, device=device)
        for i in reversed(range(self.T)):
            t = torch.full((shape[0],), i, device=device, dtype=torch.long)
            x_t = self.p_sample(model, x_t, t, y)
        return x_t

    # Optional: DDIM sampling (Phase 3)
    def ddim_sample_loop(self, model, shape, device, ddim_steps, y=None):
        # Implement DDIM deterministic sampling for faster inference
        pass
