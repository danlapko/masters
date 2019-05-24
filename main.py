import torch
from torch.utils.data import DataLoader, Subset
from torchvision.transforms import Compose

from datasets.cox import COX
from datasets.transformers import ToTensor, RandomCrop, Rescale, CenterCrop
from trainer import Trainer


def prepare_dataloaders(sequence_length=7, batch_size=32, cam="cam1", k_for_test=0.2, k_for_train=0.8):
    dataset = COX(f"/home/danila/masters/datasets/gray/video",
                  f"/home/danila/masters/datasets/gray/still",
                  sequence_length,
                  cam,
                  transform=Compose([Rescale((80, 64)), RandomCrop((75, 60)), ToTensor("fp32")])
                  # transform=Compose([CenterCrop((96, 80)), ToTensor("fp32")])
                  # transform=Compose([ToTensor("fp32")])
                  )
    # dist_h, dist_w = 60, 48  # (160, 128) (60,48)

    train_size = int((1 - k_for_test) * len(dataset))
    test_size = len(dataset) - train_size

    torch.manual_seed(0)

    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])
    torch.manual_seed(torch.initial_seed())

    train_loader = DataLoader(Subset(train_dataset, range(int(len(dataset) * k_for_train))),
                              batch_size=batch_size,
                              shuffle=True,
                              num_workers=0)

    test_loader = DataLoader(test_dataset,
                             batch_size=len(test_dataset),
                             shuffle=True,
                             num_workers=0)

    print("train_size={} test_size={}".format(len(train_loader.dataset), len(test_loader.dataset)))
    return train_loader, test_loader


def main_():
    # ====== Model params ============
    emb_size = 128
    unet_depth = 4
    unet_filts = 32
    facenet_filts = 64
    resnet = 18

    # ===== Dataset params =======
    cam = "cam2"
    k_for_test = 0.5
    k_for_train = 0.5
    epochs = 1000
    seq_len = 30
    batch_size = 100

    # ===== Training params =======

    k_gen = 0.000
    k_discrim = 0.000
    k_mse = 1
    k_facenet = 1

    train_loader, test_loader = prepare_dataloaders(seq_len, batch_size,  cam, k_for_test=k_for_test,k_for_train=k_for_train)

    trainer = Trainer(seq_length=seq_len, color_channels=3,
                      unet_path=f"pretrained/gray/unet_{unet_depth}_{unet_filts}.mdl",
                      facenet_path=f"pretrained/gray/facenet_{facenet_filts}.mdl",
                      discrim_path=f"pretrained/gray/discrim.mdl",
                      vgg_path=f"pretrained/gray/vgg_face_dag.pth",
                      embedding_size=emb_size,
                      unet_depth=unet_depth,
                      unet_filts=unet_filts,
                      facenet_filts=facenet_filts,
                      resnet=resnet)

    trainer.train(train_loader, test_loader, epochs=epochs, batch_size=batch_size,
                  k_gen=k_gen,
                  k_discrim=k_discrim,
                  k_mse=k_mse,
                  k_facenet=k_facenet)

    print("\n ######## FINAL TEST ########")
    trainer.test_test(test_loader)


if __name__ == "__main__":
    main_()
