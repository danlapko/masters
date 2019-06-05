import torch
import torch.nn.functional as F
from torch import nn

from models.convgru import ConvGRUCell


def conv1x1(in_channels, out_channels, groups=1):
    return nn.Conv2d(in_channels,
                     out_channels,
                     kernel_size=1,
                     groups=groups,
                     stride=1)


def conv3x3(in_channels, out_channels, stride=1, padding=1, bias=True, groups=1):
    return nn.Conv2d(in_channels,
                     out_channels,
                     kernel_size=3,
                     stride=stride,
                     padding=padding,
                     bias=bias,
                     groups=groups)


def upconv2x2(in_channels, out_channels, mode='transpose'):
    if mode == 'transpose':
        return nn.ConvTranspose2d(in_channels,
                                  out_channels,
                                  kernel_size=2,
                                  stride=2)
    else:
        return nn.Sequential(
            nn.Upsample(mode='bilinear', scale_factor=2),
            conv1x1(in_channels, out_channels))


class DownConv(nn.Module):
    """
    A helper Module that performs 2 convolutions and 1 MaxPool.
    A ReLU activation follows each convolution.
    """

    def __init__(self, in_channels, out_channels, pooling=True):
        super(DownConv, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.pooling = pooling

        self.conv1 = conv3x3(self.in_channels, self.out_channels)
        self.bn1 = nn.BatchNorm2d(self.out_channels)
        # self.bn1 = nn.InstanceNorm2d(self.out_channels)

        # self.attention = nn.Conv2d(self.out_channels, 1, kernel_size=1)
        self.conv2 = conv3x3(self.out_channels, self.out_channels, stride=1)
        self.bn2 = nn.BatchNorm2d(self.out_channels)
        # self.bn2 = nn.InstanceNorm2d(self.out_channels)

        if self.pooling:
            self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        x = self.conv1(x)
        x = F.leaky_relu(self.bn1(x))
        # before_pool = x

        x = self.conv2(x)
        x = F.leaky_relu(self.bn2(x))

        # x = x * torch.nn.Softmax2d()(self.attention(x))

        before_pool = x
        if self.pooling:
            x = self.pool(x)
        return x, before_pool


class UpConv(nn.Module):
    """
    A helper Module that performs 2 convolutions and 1 UpConvolution.
    A ReLU activation follows each convolution.
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 merge_mode='concat',
                 up_mode='transpose'):
        super(UpConv, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.merge_mode = merge_mode
        self.up_mode = up_mode

        self.upconv = upconv2x2(self.in_channels,
                                self.out_channels,
                                mode=self.up_mode)
        self.bn1 = nn.BatchNorm2d(self.out_channels)
        # self.bn1 = nn.InstanceNorm2d(self.out_channels)
        if self.merge_mode == 'concat':
            self.conv1 = conv3x3(2 * self.out_channels, self.out_channels)

        else:
            # num of input channels to conv2 is same
            self.conv1 = conv3x3(self.out_channels, self.out_channels)

        self.gruconv = ConvGRUCell(in_channels=self.out_channels, hidden_channels=self.out_channels,
                                   kernel_size=3)
        # self.conv2 = conv3x3(self.out_channels, self.out_channels)

    def forward(self, from_down, from_up, hidden):
        """ Forward pass
        Arguments:
            from_down: tensor from the encoder pathway
            from_up: upconv'd tensor from the decoder pathway
        """
        from_up = self.upconv(from_up)
        from_up = self.bn1(from_up)
        # from_up = from_up * torch.nn.Softmax2d()(self.attention(from_up))
        # print("(from_up, from_down):", from_up.shape, from_down.shape)

        if self.merge_mode == 'concat':
            x = torch.cat((from_up, from_down), 1)
        else:
            x = from_up + from_down
        x = F.leaky_relu(self.conv1(x))

        # x = F.leaky_relu(self.conv2(x))
        x = self.gruconv(x, hidden)
        return x


class UNet(nn.Module):

    def __init__(self, in_channels=1, out_channels=1, depth=5,
                 start_filts=64, up_mode='transpose',  # upsample
                 merge_mode='concat'):  # add

        super(UNet, self).__init__()

        if up_mode in ('transpose', 'upsample'):
            self.up_mode = up_mode
        else:
            raise ValueError()

        if merge_mode in ('concat', 'add'):
            self.merge_mode = merge_mode
        else:
            raise ValueError()


        if self.up_mode == 'upsample' and self.merge_mode == 'add':
            raise ValueError()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.start_filts = start_filts
        self.depth = depth

        self.down_convs = []
        self.up_convs = []


        for i in range(depth):
            ins = self.in_channels if i == 0 else outs
            outs = self.start_filts * (2 ** i)
            pooling = True if i < depth - 1 else False

            down_conv = DownConv(ins, outs, pooling=pooling)
            self.down_convs.append(down_conv)


        for i in range(depth - 1):
            ins = outs
            outs = ins // 2
            up_conv = UpConv(ins, outs, up_mode=up_mode,
                             merge_mode=merge_mode)
            self.up_convs.append(up_conv)

        self.conv_final = conv1x1(outs, self.out_channels)

        # add the list of modules to current module
        self.down_convs = nn.ModuleList(self.down_convs)
        self.up_convs = nn.ModuleList(self.up_convs)

        self.reset_params()

        # for param in self.parameters():
        #     param.requires_grad = False

    @staticmethod
    def weight_init(m):
        if isinstance(m, nn.Conv2d):
            nn.init.xavier_normal_(m.weight)
            nn.init.constant_(m.bias, 0)

    def reset_params(self):
        for i, m in enumerate(self.modules()):
            self.weight_init(m)

    def forward(self, frames):
        B, D, C, W, H = frames.shape
        prev_hiddens = [None] * len(self.up_convs)
        for i_frame in range(D):
            x = frames[:, i_frame]

            encoder_outs = []

            # encoder pathway, save outputs for merging
            for i, down_conv in enumerate(self.down_convs):
                x, before_pool = down_conv(x)
                encoder_outs.append(before_pool)

            torch.cuda.empty_cache()
            # print("enc out:", x.shape)
            for i, up_conv in enumerate(self.up_convs):
                before_pool = encoder_outs[-(i + 2)]

                prev_hidden = prev_hiddens[i]
                x = up_conv(before_pool, x, prev_hidden)

                prev_hiddens[i] = x

        x = F.sigmoid(self.conv_final(x).squeeze())
        return x


def main_():
    class param:
        img_size = (80, 80)
        bs = 8
        num_workers = 4
        lr = 0.001
        epochs = 3
        unet_depth = 5
        unet_start_filters = 8
        log_interval = 70  # less then len(train_dl)

    unet = UNet(in_channels=3, out_channels=3,
                depth=4,
                start_filts=32,
                up_mode="upsample",
                merge_mode='concat')

    optim = torch.optim.Adam(unet.parameters(), lr=param.lr)

    video_btch = torch.autograd.Variable(torch.FloatTensor(32, 7, 3, 128, 128))

    res = unet(video_btch)
    print(video_btch.shape, res.shape)


if __name__ == "__main__":
    main_()
