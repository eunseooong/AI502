import torch
import torch.nn as nn
import math
from config import Config

class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = time[:, None] * emb[None, :]
        emb = torch.cat([emb.sin(), emb.cos()], dim=-1)
        return emb

class Block(nn.Module):
    def __init__(self, in_ch, out_ch, time_emb_dim, down=True, output_padding=0):
        super().__init__()
        self.time_mlp = nn.Linear(time_emb_dim, out_ch)
        self.block1 = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU()
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU()
        )
        self.down = down
        if down:
            self.downsample = nn.Conv2d(out_ch, out_ch, 4, 2, 1)
        else:
            self.upsample = nn.ConvTranspose2d(out_ch, out_ch, 4, 2, 1, output_padding=output_padding)


    def forward(self, x, t):
        h = self.block1(x)
        time_emb = self.time_mlp(t).view(-1, h.shape[1], 1, 1)
        h = h + time_emb
        h = self.block2(h)
        if self.down:
            h = self.downsample(h)
        else:
            h = self.upsample(h)
        return h

class UNet(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.image_size = config.image_size
        time_emb_dim = 128
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(time_emb_dim),
            nn.Linear(time_emb_dim, time_emb_dim),
            nn.ReLU()
        )
        if config.conditional:
            from condition import LabelEmbedding
            self.label_emb = LabelEmbedding(config.num_classes, time_emb_dim)
        self.conditional = config.conditional

        if config.image_size == 28:
            op = 1
        else:
            op = 0

        self.init_conv = nn.Conv2d(config.num_channels, 64, 3, padding=1)
        self.down1 = Block(64, 128, time_emb_dim, down=True)
        self.down2 = Block(128, 256, time_emb_dim, down=True)
        self.bot1 = Block(256, 256, time_emb_dim, down=True)
        self.bot2 = Block(256, 256, time_emb_dim, down=False, output_padding=op)
        self.up2 = Block(256, 128, time_emb_dim, down=False, output_padding=op)
        self.up1 = Block(384, 128, time_emb_dim, down=False, output_padding=op)
        self.out_conv = nn.Conv2d(128, config.num_channels, 1)

    def forward(self, x, t, y=None):
        t_emb = self.time_mlp(t)
        if self.conditional:
            assert y is not None
            y_emb = self.label_emb(y)
            t_emb = t_emb + y_emb
        x = self.init_conv(x)
        d1 = self.down1(x, t_emb)
        d2 = self.down2(d1, t_emb)
        b1 = self.bot1(d2, t_emb)
        b2 = self.bot2(b1, t_emb)
        u2 = self.up2(b2, t_emb)
        
        if d2.shape[2:] != u2.shape[2:]:
            d2 = torch.nn.functional.interpolate(d2, size=u2.shape[2:], mode='nearest')

        u1 = self.up1(torch.cat([u2, d2], dim=1), t_emb)

        if d1.shape[2:] != u1.shape[2:]:
            d1 = torch.nn.functional.interpolate(d1, size=u1.shape[2:], mode='nearest')

        out = self.out_conv(u1 + d1)

        if out.shape[2:] != (self.image_size, self.image_size):
            out = torch.nn.functional.interpolate(out, size=(self.image_size, self.image_size), mode='bilinear', align_corners=False)

        return out