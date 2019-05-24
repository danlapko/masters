from torch import nn
import torch
import torch.nn.functional as F
from torchvision.models import resnet18


# class Discriminator(nn.Module):
#     def __init__(self, in_channels, start_filts):
#         super(Discriminator, self).__init__()
#         self.main = nn.Sequential(
#             # input is (nc) x 60 x 48
#             nn.Conv2d(in_channels, start_filts, 4, 2, (3, 3), bias=False),
#             nn.LeakyReLU(0.2, inplace=True),
#             # state size. (ndf) x 32 x 26
#             nn.Conv2d(start_filts, start_filts * 2, 4, 2, (1, 3), bias=False),
#             nn.BatchNorm2d(start_filts * 2),
#             nn.LeakyReLU(0.2, inplace=True),
#             # state size. (ndf*2) x 16 x 15
#             nn.Conv2d(start_filts * 2, start_filts * 4, (4, 3), 2, 1, bias=False),
#             nn.BatchNorm2d(start_filts * 4),
#             nn.LeakyReLU(0.2, inplace=True),
#             # state size. (ndf*4) x 8 x 8
#             nn.Conv2d(start_filts * 4, start_filts * 8, 4, 2, 1, bias=False),
#             nn.BatchNorm2d(start_filts * 8),
#             nn.LeakyReLU(0.2, inplace=True),
#             # state size. (ndf*8) x 4 x 4
#             nn.Conv2d(start_filts * 8, 1, 4, 1, 0, bias=False),
#             nn.Sigmoid()
#         )
#
#         # for param in self.parameters():
#         #     param.requires_grad = False
#
#     def forward(self, x):
#         # print("discrim in_shape", x.shape)
#         # N, S, H, W, C = x.shape
#         # x = x.view(N, S*C, H, W)
#         x = self.main(x)
#         # print("discrim out_shape", x.shape)
#
#         return x


class Discriminator(nn.Module):
    def __init__(self, in_channels, pretrained=False):
        super(Discriminator, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)

        self.attention = nn.Conv2d(64, 1, 1)
        self.model = resnet18(pretrained)
        self.model.fc = nn.Linear(512, 1)

        # for param in self.parameters():
        #     param.requires_grad = False

    def forward(self, x):
        x = self.conv1(x)
        # x = x * nn.Softmax2d()(self.attention(x))
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)
        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        x = self.model.layer4(x)
        x = F.adaptive_avg_pool2d(x, (1, 1))
        # x = F.adaptive_max_pool2d(x, (1, 1))

        B, C, H, W = x.shape
        x = x.view(B, C)
        x = self.model.fc(x)
        x = F.sigmoid(x)
        return x
