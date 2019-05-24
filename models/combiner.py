import torch
from torch import nn
import torchvision.models as torch_models
import torch.nn.functional as F

from models.discriminator import Discriminator
from models.unet import UNet


class Combiner(nn.Module):
    def __init__(self, img_size, hidden_size):
        super(Combiner, self).__init__()

        self.unet = UNet(in_channels=21, out_channels=1,
                         depth=3,
                         start_filts=64,
                         up_mode="upsample",
                         merge_mode='concat')

        # 32 dep 3 upsample concat 12.0 - win
        # 64 dep 3 upsample concat 11.9 - win
        # 64 dep 3 transpose concat 14.5
        # 64 dep 3 transpose add 13.5

    def forward(self, X):
        # print("X:", X.shape)

        decoded = self.unet(X)
        # discrim_zeors = self.discriminator(decoded)
        # descrim_ones = self.discriminator(y.unsqueeze(1))

        # print("out:", out.shape)
        return decoded.squeeze()

