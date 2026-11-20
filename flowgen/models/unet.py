from typing import List
import torch
import torch.nn as nn
import torch.nn.functional as F
from .blocks import ConvBlock, ResidualBlock, DownsampleBlock, UpsampleBlock


class LightweightUNet(nn.Module):
    def __init__(
        self,
        in_channels: int = 2,
        out_channels: int = 2,
        base_channels: int = 32,
        channel_multipliers: List[int] = (1, 2, 4),
        scale_factor: int = 4,
        residual_learning: bool = True,
    ):
        super().__init__()
        self.scale_factor = scale_factor
        self.residual_learning = residual_learning

        channels = [base_channels * m for m in channel_multipliers]

        self.head = ConvBlock(in_channels, channels[0])

        self.encoders = nn.ModuleList()
        for i in range(len(channels) - 1):
            self.encoders.append(DownsampleBlock(channels[i], channels[i + 1]))

        self.bottleneck = nn.Sequential(
            ResidualBlock(channels[-1]),
            ResidualBlock(channels[-1]),
        )

        self.decoders = nn.ModuleList()
        for i in reversed(range(len(channels) - 1)):
            self.decoders.append(UpsampleBlock(channels[i + 1], channels[i]))

        self.tail = nn.Sequential(
            ResidualBlock(channels[0]),
            nn.Conv2d(channels[0], out_channels, kernel_size=3, padding=1),
        )

    def forward(self, x_lr: torch.Tensor) -> torch.Tensor:
        if self.scale_factor > 1:
            x_up = F.interpolate(x_lr, scale_factor=self.scale_factor, mode="bicubic", align_corners=False)
        else:
            x_up = x_lr

        x = self.head(x_up)
        skips = [x]

        for enc in self.encoders:
            x = enc(x)
            skips.append(x)

        x = self.bottleneck(x)

        for dec in self.decoders:
            skip = skips.pop()
            x = dec(x, skip)

        out = self.tail(x)

        if self.residual_learning:
            out = out + x_up

        return out
