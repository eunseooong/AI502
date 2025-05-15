import torch.nn as nn

class UNet(nn.Module):
    def __init__(self, config, conditional=False):
        super().__init__()
        # Encoder, Decoder, Time Embedding, (Condition Embedding)
        pass

    def forward(self, x, t, y=None):
        """
        x: [B, C, H, W]
        t: [B] (int64)
        y: [B] (optional, class label)
        return: predicted_noise [B, C, H, W]
        """
        pass