import torch.nn as nn
from block import DownSample


class PatchGAN(nn.Module):
    def __init__(self, img_channels):
        super(PatchGAN, self).__init__()
        self.model = nn.Sequential(
            DownSample(img_channels, 8, (3, 3)),
            DownSample(8, 16, (3, 3)),
            DownSample(16, 32, (5, 5)),
            DownSample(32, 64, (7, 7)),
            nn.Conv2d(64, 32, (5, 5), (2, 2), (1, 1)),
            nn.ReLU(),
            nn.Conv2d(32, 1, (1, 1)),
            nn.ReLU()
        )

    def forward(self, x):
        return self.model(x)

