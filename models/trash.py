import torch
from torch import nn
import torchvision.models as torch_models
import torch.nn.functional as F

from models.unet import UNet


class RNNseq2single(nn.Module):
    def __init__(self, in_size, out_size, hidden_size):
        super(RNNseq2single, self).__init__()
        self.hidden_size = hidden_size
        self.in_size = in_size
        self.out_size = out_size

        self.encoder_gru = nn.GRU(in_size, out_size, num_layers=1, bidirectional=False, batch_first=True)
        # self.encoder_linear = nn.Linear(hidden_size, hidden_size)

        # self.decoder_gru = nn.GRU(hidden_size, out_size, num_layers=1, bidirectional=False, batch_first=True, dropout=0.3)
        # self.decoder_linear = nn.Linear(2 * out_size, out_size)

    def forward(self, X):
        """
        :param X: (batch_size, n_frames, n_features)
        :return: (batch_size, n_features)
        """
        enc_out, enc_hidden = self.encoder_gru(X)
        # enc_out = self.encoder_linear(enc_out)
        # enc_out = F.leaky_relu(enc_out)

        # dec_out, dec_hidden = self.decoder_gru(enc_out)
        # dec_out = self.decoder_linear(dec_out)

        # return dec_out[:, -1]
        return enc_out[:, -1]


class CNNEncoder(nn.Module):
    def __init__(self, image_size):
        super(CNNEncoder, self).__init__()

        self.conv1 = nn.Sequential(nn.Conv2d(1, 16, 3),
                                   nn.BatchNorm2d(16),
                                   nn.LeakyReLU())

        self.conv2 = nn.Sequential(nn.Conv2d(16, 32, 3),
                                   nn.BatchNorm2d(32),
                                   nn.LeakyReLU())

        self.conv3 = nn.Sequential(nn.Conv2d(32, 64, 3),
                                   nn.BatchNorm2d(64),
                                   nn.LeakyReLU(),
                                   # nn.MaxPool2d(2)
                                   )

        self.conv4 = nn.Sequential(nn.Conv2d(64, 256, 3),
                                   nn.BatchNorm2d(256),
                                   nn.LeakyReLU(),
                                   nn.MaxPool2d(2)
                                   )

        self.conv5 = nn.Sequential(nn.Conv2d(256, 1024, 3),
                                   nn.BatchNorm2d(1024),
                                   nn.LeakyReLU(),
                                   # nn.MaxPool2d(2)
                                   )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)

        output = F.max_pool2d(x, kernel_size=x.size()[2:])
        return output


class CNNDecoder(nn.Module):
    def __init__(self, n_features):
        # in 1024x1x1
        super(CNNDecoder, self).__init__()

        self.convs = nn.Sequential(
            # out 512x4x4
            nn.Sequential(nn.ConvTranspose2d(n_features, 512, 4, (1, 1), (0, 0), bias=False),
                          nn.BatchNorm2d(512),
                          nn.LeakyReLU()),

            nn.Sequential(nn.Conv2d(512, 512, 3, padding=1),
                          nn.BatchNorm2d(512),
                          nn.LeakyReLU()),

            # out 256x8x8
            nn.Sequential(nn.ConvTranspose2d(512, 256, 4, (1, 2), (1, 1), bias=False),
                          nn.BatchNorm2d(256),
                          nn.LeakyReLU()),

            nn.Sequential(nn.Conv2d(256, 256, 3, padding=1),
                          nn.BatchNorm2d(256),
                          nn.LeakyReLU()),

            # out 128x16x16
            nn.Sequential(nn.ConvTranspose2d(256, 128, 4, (5, 2), (1, 1), bias=False),
                          nn.BatchNorm2d(128),
                          nn.LeakyReLU()),

            nn.Sequential(nn.Conv2d(128, 128, 3, padding=1),
                          nn.BatchNorm2d(128),
                          nn.LeakyReLU()),

            # out 3x64x32
            nn.Sequential(nn.ConvTranspose2d(128, 1, 3, (3, 3), (3, 0), bias=False),
                          nn.Sigmoid())
        )

    # --> in [batch_size, n_channels=embedding, width, height] == [256, 100, 1, 1]
    # --> out [batch_size, n_channels, width, height] == [256, 3, 32, 32]
    def forward(self, data):
        # print("g in:", data.shape)

        x = self.convs(data)
        # print("g out:", x.shape)

        return x


class CNN_RNN(nn.Module):
    def __init__(self, img_size, hidden_size):
        super(CNN_RNN, self).__init__()
        self.cnn_enc = CNNEncoder((60, 48))
        self.cnn_dec = CNNDecoder(1024)

        self.rnn = RNNseq2single(1024, 1024, hidden_size)

        # for param in self.cnn_enc.parameters():
        #     param.requires_grad = False
        # for param in self.cnn_dec.parameters():
        #     param.requires_grad = False
        # for param in self.rnn.parameters():
        #     param.requires_grad = False

    def forward(self, X):
        batch_size, n_frames, h, w = X.size()

        # print("X:", X.shape)

        cnn_enc_in = X.view(batch_size * n_frames, 1, h, w)
        # print("cnn_enc_in:", cnn_enc_in.shape)

        cnn_enc_out = self.cnn_enc(cnn_enc_in)
        # print("cnn_enc_out:", cnn_enc_out.shape)

        rnn_in = cnn_enc_out.view(batch_size, n_frames, -1)
        # print("rnn_in:", rnn_in.shape)

        rnn_out = self.rnn(rnn_in)  # features for single image
        # print("rnn_out:", rnn_out.shape)

        cnn_dec_in = rnn_out.unsqueeze(-1).unsqueeze(-1)
        # cnn_dec_in = cnn_enc_out.view(batch_size, n_frames, -1).squeeze().unsqueeze(-1).unsqueeze(-1)
        # print("cnn_dec_in:", cnn_dec_in.shape)

        cnn_dec_out = self.cnn_dec(cnn_dec_in)
        # print("cnn_dec_out:", cnn_dec_out.shape)

        return cnn_dec_out.squeeze()
