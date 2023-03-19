import torch.nn as nn

from settings import CHANNELS, DISC_FEATURES


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(
            # 3 x 64 x 64
            nn.Conv2d(CHANNELS, DISC_FEATURES, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # 64 x 32 x 32
            nn.Conv2d(DISC_FEATURES, DISC_FEATURES * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(DISC_FEATURES * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # 128 x 16 x 16
            nn.Conv2d(DISC_FEATURES * 2, DISC_FEATURES * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(DISC_FEATURES * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # 256 x 8 x 8
            nn.Conv2d(DISC_FEATURES * 4, DISC_FEATURES * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(DISC_FEATURES * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # 512 x 4 x 4
            nn.Conv2d(DISC_FEATURES * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, array):
        return self.main(array)
