import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function
from torch.nn.modules.distance import PairwiseDistance
from torchvision.models import resnet34, resnet18, resnet50, resnet101, resnet152


class FaceNetModel(nn.Module):
    def __init__(self, embedding_size, in_channels, start_filts, resnet=18, pretrained=False):
        super(FaceNetModel, self).__init__()
        if resnet == 18:
            self.resnet = resnet18(pretrained)
        elif resnet == 34:
            self.resnet = resnet34(pretrained)
        elif resnet == 50:
            self.resnet = resnet50(pretrained)
        elif resnet == 101:
            self.resnet = resnet101(pretrained)
        elif resnet == 152:
            self.resnet = resnet152(pretrained)
        else:
            raise ValueError(f"invalid argument resnet={resnet}")

        self.conv1 = nn.Conv2d(in_channels, start_filts, kernel_size=7, stride=2, padding=3,
                               bias=False)

        # self.attention = nn.Conv2d(64, 1, 1)
        self.embedding_size = embedding_size
        self.resnet.fc = nn.Linear(512, self.embedding_size)
        self.resnet.critic = nn.Linear(512, 1)

        # self.conv_last = nn.Conv2d(64, out_channels, kernel_size=7, stride=2, padding=3, bias=False)

        # for param in self.parameters():
        #     param.requires_grad = False

    @staticmethod
    def l2_norm(input):
        input_size = input.size()
        buffer = torch.pow(input, 2)
        normp = torch.sum(buffer, 1).add_(1e-10)
        norm = torch.sqrt(normp)
        _output = torch.div(input, norm.view(-1, 1).expand_as(input))
        output = _output.view(input_size)

        return output

    def forward_mse(self, x):
        x = self.conv1(x)

        # x = x * nn.Softmax2d()(self.attention(x))
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        # x = self.model.layer1(x)

        return x

    def forward(self, x):
        x = self.resnet.conv1(x)
        # x = x * nn.Softmax2d()(self.attention(x))
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)

        x = self.resnet.layer1(x)
        x = self.resnet.layer2(x)
        x = self.resnet.layer3(x)
        x = self.resnet.layer4(x)
        x = F.adaptive_avg_pool2d(x, (1, 1))
        # x = F.adaptive_max_pool2d(x, (1, 1))

        B, C, H, W = x.shape
        x = x.view(B, C)
        # x = self.model.avgpool(x)
        # x = x.view(x.size(0), -1)
        before_fc = x
        x = self.resnet.fc(x)

        features = self.l2_norm(x)
        # Multiply by alpha = 10 as suggested in https://arxiv.org/pdf/1703.09507.pdf
        alpha = 10
        features = features * alpha

        decision = F.sigmoid(self.resnet.critic(before_fc))
        return features, decision


class TripletLoss(Function):

    def __init__(self, margin):
        super(TripletLoss, self).__init__()
        self.margin = margin
        self.pdist = PairwiseDistance(2)

    def forward(self, anchor, positive, negative):
        pos_dist = self.pdist.forward(anchor, positive)
        neg_dist = self.pdist.forward(anchor, negative)

        loss = torch.clamp(self.margin + pos_dist - neg_dist, min=0.0)

        return loss.sum()
