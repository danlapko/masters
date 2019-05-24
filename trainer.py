import os

import cv2
import numpy as np
import torch
import torch.nn as nn
from matplotlib import pyplot as plt
from torch import autograd
from torch.nn.modules.distance import PairwiseDistance
from torch.utils.tensorboard import SummaryWriter
from torchvision.models import vgg16, vgg16_bn

from sklearn.metrics import auc

from alignment.crop_face import equalize_func
from models.discriminator import Discriminator
from models.facenet import FaceNetModel, TripletLoss
from models.unet import UNet
from models.vgg import vgg_face_dag, LossNetwork
from utils import roc_curve, rank1

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
cpu = torch.device("cpu")

l2_dist = PairwiseDistance(2)


class Trainer:
    def __init__(self, seq_length, color_channels, unet_path="pretrained/unet.mdl",
                 discrim_path="pretrained/dicrim.mdl",
                 facenet_path="pretrained/facenet.mdl",
                 vgg_path="",
                 embedding_size=1000,
                 unet_depth=3,
                 unet_filts=32,
                 facenet_filts=32,
                 resnet=18):

        self.color_channels = color_channels
        self.margin = 0.5
        self.writer = SummaryWriter(log_dir="logs")

        self.unet_path = unet_path
        self.discrim_path = discrim_path
        self.facenet_path = facenet_path

        self.one = torch.FloatTensor([1]).to(device)
        self.mone = self.one * -1

        self.unet = UNet(in_channels=seq_length * color_channels, out_channels=color_channels,
                         depth=unet_depth,
                         start_filts=unet_filts,
                         up_mode="upsample",
                         merge_mode='concat').to(device)

        self.discrim = FaceNetModel(embedding_size=embedding_size, start_filts=facenet_filts,
                                    in_channels=color_channels, resnet=resnet,
                                    pretrained=False).to(device)

        self.facenet = FaceNetModel(embedding_size=embedding_size, start_filts=facenet_filts,
                                    in_channels=color_channels, resnet=resnet,
                                    pretrained=False).to(device)

        if os.path.isfile(unet_path):
            self.unet.load_state_dict(torch.load(unet_path))
            print("unet loaded")

        if os.path.isfile(discrim_path):
            self.discrim.load_state_dict(torch.load(discrim_path))
            print("discrim loaded")

        if os.path.isfile(facenet_path):
            self.facenet.load_state_dict(torch.load(facenet_path))
            print("facenet loaded")
        if os.path.isfile(vgg_path):
            self.vgg_loss_network = LossNetwork(vgg_face_dag(vgg_path)).to(device)
            self.vgg_loss_network.eval()

            print("vgg loaded")

        self.mse_loss_function = nn.MSELoss().to(device)
        self.discrim_loss_function = nn.BCELoss().to(device)
        self.triplet_loss_function = TripletLoss(margin=self.margin)

        self.unet_optimizer = torch.optim.Adam(self.unet.parameters(), betas=(0.9, 0.999))
        self.discrim_optimizer = torch.optim.Adam(self.discrim.parameters(), betas=(0.9, 0.999))
        self.facenet_optimizer = torch.optim.Adam(self.facenet.parameters(), betas=(0.9, 0.999))

    def test(self, test_loader, epoch=0):
        X, y = next(iter(test_loader))

        B, D, C, W, H = X.shape
        X = X.view(B, C * D, W, H)

        self.unet.eval()
        self.facenet.eval()
        self.discrim.eval()
        with torch.no_grad():
            y_ = self.unet(X.to(device))

            mse = self.mse_loss_function(y_, y.to(device))
            loss_G = self.loss_GAN_generator(btch_X=X.to(device))
            loss_D = self.loss_GAN_discrimator(btch_X=X.to(device), btch_y=y.to(device))

            loss_facenet, _, n_bad = self.loss_facenet(X.to(device), y.to(device))

        plt.title(f"epoch {epoch} mse={mse.item():.4} facenet={loss_facenet.item():.4} bad={n_bad / B ** 2}")
        i = np.random.randint(0, B)
        a = np.hstack((y[i].transpose(0, 1).transpose(1, 2), y_[i].transpose(0, 1).transpose(1, 2).to(cpu)))
        b = np.hstack((X[i][0:3].transpose(0, 1).transpose(1, 2),
                       X[i][-3:].transpose(0, 1).transpose(1, 2)))
        plt.imshow(np.vstack((a, b)))
        plt.axis('off')
        plt.show()

        self.writer.add_scalar("test_bad_percent", n_bad / B ** 2, global_step=epoch)
        self.writer.add_scalar("test_loss", loss_facenet.item(), global_step=epoch)
        # self.writer.add_scalars("test GAN", {"discrim": loss_D.item(),
        #                                      "gen": loss_G.item()}, global_step=epoch)

        with torch.no_grad():
            n_for_show = 10
            y_show_ = y_.to(device)
            y_show = y.to(device)
            embeddings_anc, _ = self.facenet(y_show_)
            embeddings_pos, _ = self.facenet(y_show)

            embeds = torch.cat((embeddings_anc[:n_for_show], embeddings_pos[:n_for_show]))
            imgs = torch.cat((y_show_[:n_for_show], y_show[:n_for_show]))
            names = list(range(n_for_show)) * 2
            # print(embeds.shape, imgs.shape, len(names))
            self.writer.add_embedding(mat=embeds, metadata=names, label_img=imgs, tag="embeddings", global_step=epoch)

        trshs, fprs, tprs = roc_curve(embeddings_anc.detach().to(cpu), embeddings_pos.detach().to(cpu))
        rnk1 = rank1(embeddings_anc.detach().to(cpu), embeddings_pos.detach().to(cpu))
        plt.step(fprs, tprs)
        # plt.xlim((1e-4, 1))
        plt.yticks(np.arange(0, 1, 0.05))
        plt.xticks(np.arange(min(fprs), max(fprs), 10))
        plt.xscale('log')
        plt.title(f"ROC auc={auc(fprs, tprs)} rnk1={rnk1}")
        self.writer.add_figure("ROC test", plt.gcf(), global_step=epoch)
        self.writer.add_scalar("auc", auc(fprs, tprs), global_step=epoch)
        self.writer.add_scalar("rank1", rnk1, global_step=epoch)
        print(f"\n###### {epoch} TEST mse={mse.item():.4} GAN(G/D)={loss_G.item():.4}/{loss_D.item():.4} "
              f"facenet={loss_facenet.item():.4} bad={n_bad / B ** 2:.4} auc={auc(fprs, tprs)} rank1={rnk1} #######")

    def test_test(self, test_loader):
        X, ys = next(iter(test_loader))
        true_idx = 0
        x = X[true_idx]

        D, C, W, H = x.shape
        x = x.view(C * D, W, H)

        dists = list()
        with torch.no_grad():
            y_ = self.unet(x.to(device))

            embedding_anc, _ = self.facenet(y_)
            embeddings_pos, _ = self.facenet(ys)
            for emb_pos_item in embeddings_pos:
                dist = l2_dist.forward(embedding_anc, emb_pos_item)
                dists.append(dist)

        a_sorted = np.argsort(dists)

        a = np.hstack((ys[true_idx].transpose(0, 1).transpose(1, 2),
                       y_.transpose(0, 1).transpose(1, 2).to(cpu).numpy(),
                       ys[a_sorted[0]].transpose(0, 1).transpose(1, 2)))

        b = np.hstack((x[0:3].transpose(0, 1).transpose(1, 2),
                       x[D // 2 * C:D // 2 * C + 3].transpose(0, 1).transpose(1, 2),
                       x[-3:].transpose(0, 1).transpose(1, 2)))

        b_ = b - np.min(b)
        b_ = b_ / np.max(b)
        b_ = equalize_func([(b_ * 255).astype(np.uint8)], use_clahe=True)[0]
        b = b_.astype(np.float32) / 255

        plt.imshow(cv2.cvtColor(np.vstack((a, b)), cv2.COLOR_BGR2RGB))
        plt.axis('off')
        plt.show()

    def loss_facenet(self, X, y, is_detached=False):
        B, C, W, H = X.shape

        y_ = self.unet(X)

        embeddings_anc, D_fake = self.facenet(y_ if not is_detached else y_.detach())
        embeddings_pos, D_real = self.facenet(y)

        target_real = torch.full_like(D_fake, 1)
        loss_gen = self.discrim_loss_function(D_fake, target_real)

        pos_dist = l2_dist.forward(embeddings_anc, embeddings_pos)
        bad_triplets_loss = None

        n_bad = 0
        for shift in range(1, B):

            embeddings_neg = torch.roll(embeddings_pos, shift, 0)
            neg_dist = l2_dist.forward(embeddings_anc, embeddings_neg)

            bad_triplets_idxs = np.where((neg_dist - pos_dist < self.margin).cpu().numpy().flatten())[0]

            if shift == 1:
                bad_triplets_loss = self.triplet_loss_function.forward(embeddings_anc[bad_triplets_idxs],
                                                                       embeddings_pos[bad_triplets_idxs],
                                                                       embeddings_neg[bad_triplets_idxs]).to(
                    device)
            else:
                bad_triplets_loss += self.triplet_loss_function.forward(embeddings_anc[bad_triplets_idxs],
                                                                        embeddings_pos[bad_triplets_idxs],
                                                                        embeddings_neg[bad_triplets_idxs]).to(device)
            n_bad += len(bad_triplets_idxs)

        bad_triplets_loss /= B
        return bad_triplets_loss, torch.mean(loss_gen), n_bad

    # def loss_mse(self, btch_X, btch_y):
    #     btch_y_ = self.unet(btch_X)
    #     loss_unet = self.mse_loss_function(btch_y_, btch_y)
    #
    #     features_target = self.facenet.forward_mse(btch_y)
    #     features = self.facenet.forward_mse(btch_y_)
    #
    #     loss_first_layer = self.mse_loss_function(features, features_target)
    #     return loss_unet + loss_first_layer

    def loss_mse_vgg(self, btch_X, btch_y):
        btch_y_ = self.unet(btch_X)
        # print(btch_y_.shape,btch_y.shape)
        gram_btch_y_ = self.vgg_loss_network(btch_y_)
        gram_btch_y = self.vgg_loss_network(btch_y)
        loss_vgg = 0.0
        for a, b in zip(gram_btch_y_, gram_btch_y):
            loss_vgg += self.mse_loss_function(a, b)
        return loss_vgg*0.001 + self.mse_loss_function(btch_y_, btch_y)

    def loss_GAN_discrimator(self, btch_X, btch_y):
        btch_y_ = self.unet(btch_X)

        _, y_D_fake_ = self.discrim(btch_y_.detach())
        _, y_D_real_ = self.discrim(btch_y)

        target_fake = torch.full_like(y_D_fake_, 0)
        target_real = torch.full_like(y_D_real_, 1)

        loss_D_fake_ = self.discrim_loss_function(y_D_fake_, target_fake)
        loss_D_real_ = self.discrim_loss_function(y_D_real_, target_real)

        loss_discrim = (loss_D_real_ + loss_D_fake_)

        return loss_discrim

    def loss_GAN_generator(self, btch_X):
        btch_y_ = self.unet(btch_X)

        _, y_D_fake_ = self.discrim(btch_y_)

        target_real = torch.full_like(y_D_fake_, 1)

        loss_gen = self.discrim_loss_function(y_D_fake_, target_real)

        return loss_gen

    def relax_discriminator(self, btch_X, btch_y):
        self.discrim.zero_grad()

        # train with real
        y_discrim_real_ = self.discrim(btch_y)
        y_discrim_real_ = y_discrim_real_.mean()
        y_discrim_real_.backward(self.mone)

        # train with fake
        btch_y_ = self.unet(btch_X)
        y_discrim_fake_detached_ = self.discrim(btch_y_.detach())
        y_discrim_fake_detached_ = y_discrim_fake_detached_.mean()
        y_discrim_fake_detached_.backward(self.one)

        # gradient_penalty
        gradient_penalty = self.discrim_gradient_penalty(btch_y, btch_y_)
        gradient_penalty.backward()

        self.discrim_optimizer.step()

    def relax_generator(self, btch_X):
        self.unet.zero_grad()

        btch_y_ = self.unet(btch_X)

        y_discrim_fake_ = self.discrim(btch_y_)
        y_discrim_fake_ = y_discrim_fake_.mean()
        y_discrim_fake_.backward(self.mone)
        self.unet_optimizer.step()

    def discrim_gradient_penalty(self, real_y, fake_y):
        lambd = 10
        btch_size = real_y.shape[0]

        alpha = torch.rand(btch_size, 1, 1, 1).to(device)
        # print(alpha.shape, real_y.shape)
        alpha = alpha.expand_as(real_y)

        interpolates = alpha * real_y + (1 - alpha) * fake_y
        interpolates = interpolates.to(device)

        interpolates = autograd.Variable(interpolates, requires_grad=True)

        interpolates_out = self.discrim(interpolates)

        gradients = autograd.grad(outputs=interpolates_out, inputs=interpolates,
                                  grad_outputs=torch.ones(interpolates_out.size()).to(device),
                                  create_graph=True, retain_graph=True, only_inputs=True)[0]

        gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * lambd
        return gradient_penalty

    def train(self, train_loader, test_loader, batch_size=2, epochs=30,
              k_gen=1, k_discrim=0.01, k_mse=0.02, k_facenet=1):
        """
        :param X: np.array shape=(n_videos, n_frames, h, w)
        :param y: np.array shape=(n_videos, h, w)
        :param epochs: int
        """
        print("\nSTART TRAINING\n")

        for epoch in range(178, epochs):
            self.test(test_loader, epoch)
            self.unet.train()
            self.facenet.train()
            self.discrim.train()
            # train by batches
            for idx, (btch_X, btch_y) in enumerate(train_loader):
                B, D, C, W, H = btch_X.shape
                # btch_X = btch_X.view(B, C * D, W, H)

                btch_X = btch_X.to(device)
                btch_y = btch_y.to(device)

                # Mse loss
                self.unet.zero_grad()

                mse = self.loss_mse_vgg(btch_X, btch_y)

                mse = k_mse * mse
                mse.backward()
                self.unet_optimizer.step()

                # facenet_backup = deepcopy(self.facenet.state_dict())
                # for i in range(unrolled_iterations):
                self.discrim.zero_grad()
                loss_D = self.loss_GAN_discrimator(btch_X, btch_y)
                loss_D = k_discrim * loss_D
                loss_D.backward()
                self.discrim_optimizer.step()

                self.discrim.zero_grad()
                self.unet.zero_grad()
                loss_G = self.loss_GAN_generator(btch_X)
                loss_G = k_gen * loss_G
                loss_G.backward()
                self.unet_optimizer.step()

                # Facenet
                self.unet.zero_grad()
                self.facenet.zero_grad()
                facenet_loss, _, n_bad = self.loss_facenet(btch_X, btch_y)

                facenet_loss = k_facenet * facenet_loss
                facenet_loss.backward()
                self.facenet_optimizer.step()

                print(f"btch {idx * batch_size} mse={mse.item():.4} GAN(G/D)={loss_G.item():.4}/{loss_D.item():.4} "
                      f"facenet={facenet_loss.item():.4} bad={n_bad / B ** 2:.4}")

                global_step = epoch * len(train_loader.dataset) // batch_size + idx
                self.writer.add_scalar("train bad_percent", n_bad / B ** 2, global_step=global_step)
                self.writer.add_scalar("train loss", mse.item(), global_step=global_step)
                # self.writer.add_scalars("train GAN", {"discrim": loss_D.item(),
                #                                       "gen": loss_G.item()}, global_step=global_step)

            torch.save(self.unet.state_dict(), self.unet_path)
            torch.save(self.discrim.state_dict(), self.discrim_path)
            torch.save(self.facenet.state_dict(), self.facenet_path)
