class Config:
    def __init__(self):
        self.image_size = 32
        self.num_channels = 3
        self.num_classes = 100  # CIFAR-100
        self.T = 1000  # Diffusion steps
        self.beta_schedule = 'linear'
        self.batch_size = 128
        self.lr = 2e-4
        self.device = 'cuda'
        self.conditional = False  # True for conditional (MNIST)
        self.num_epochs = 50
        self.save_dir = './results'
        # ... 기타 하이퍼파라미터
