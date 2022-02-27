import torch.nn as nn
import torch
from block import UpSample, DownSample


class Generator(nn.Module):
    def __init__(self, in_channels):
        super(Generator, self).__init__()
        self.down1 = DownSample(in_channels, 8, (3, 3))
        self.down2 = DownSample(8, 16, (5, 5))
        self.down3 = DownSample(16, 32, (3, 3), (2, 2))
        self.down4 = DownSample(32, 64, (3, 3), (2, 2))
        self.down5 = DownSample(64, 128, (3, 3))
        self.up1 = UpSample(128, 64, (3, 3))
        self.up2 = UpSample(64, 32, (3, 3), (2, 2))
        self.up3 = UpSample(32, 16, (4, 4), (2, 2))
        self.up4 = UpSample(16, 8, (5, 5))
        self.up5 = UpSample(8, 3, (3, 3))

    def forward(self, x):
        down1 = self.down1(x)
        out2 = self.down2(down1)
        out3 = self.down3(out2)
        out4 = self.down4(out3)
        out5 = self.down5(out4)
        out6 = self.up1(out5)
        out7 = self.up2(out6.clone() + out4.clone())
        out9 = self.up3(out7.clone() + out3.clone())
        out10 = self.up4(out9.clone())
        return self.up5(out10.clone() + down1.clone())

# g = Generator(3)
# temp = torch.randn((1, 3, 26, 26))
# out = g(temp)
# print(out.shape)
