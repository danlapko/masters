import glob
import os
import pickle
import numpy as np
import cv2
from sklearn.model_selection import train_test_split
import config as c
from klepto.archives import *


class COXLoader:
    def __init__(self, cox_data_paths, file_format, dist_h, dist_w, n_channels):

        self.file_format = file_format
        self.cams = ("cam1", "cam2", "cam3")
        self.dist_w = dist_w
        self.dist_h = dist_h
        self.n_channels = n_channels
        self.min_video_length = 21

        self.video_dirpath = cox_data_paths.video_dirpath
        self.still_dirpath = cox_data_paths.still_dirpath

        self.fnames_dataset_path = cox_data_paths.fnames_dataset_path
        self.fnames_dataset = None  # {still_id: (still_fname, [cam1_fnames, cam2_fnames, cam3_fnames])}

        self.images_dataset_path = cox_data_paths.images_dataset_path
        self.images_dataset = None  # {still_id, (still_img, cam{_}_imgs)}

        self.seq_to_single_dataset_path = cox_data_paths.seq2single_dataset_path
        self.seq_to_single_dataset = None

        self.single_to_single_dataset_path = cox_data_paths.single2single_dataset_path
        self.single_to_single_dataset = None

    @staticmethod
    def _id_from_fname(fname):
        still_name = os.path.basename(fname)
        still_id, _ = still_name.split("_")
        return still_id

    def _prepare_fnames_dataset(self):
        still_fnames = []

        self.fnames_dataset = dict()

        still_fnames.extend(glob.iglob(f"{self.still_dirpath}/**/*{self.file_format}", recursive=True))

        for still_fname in still_fnames:
            still_id = self._id_from_fname(still_fname)
            self.fnames_dataset[still_id] = (still_fname, list())

        for i, still_fname in enumerate(still_fnames):
            still_id = self._id_from_fname(still_fname)
            for cam in self.cams:
                appropriate_video_fnames = glob.iglob(
                    f"{self.video_dirpath}/{cam}/**/{still_id}/**/*{self.file_format}",
                    recursive=True)
                self.fnames_dataset[still_id][1].append(sorted(appropriate_video_fnames))
            print(i, still_id)

    def _dump_fnames_dataset(self):
        assert self.fnames_dataset is not None
        with open(self.fnames_dataset_path, 'wb') as f:
            pickle.dump(self.fnames_dataset, f)

    def _load_fnames_dataset(self):
        assert os.path.isfile(self.fnames_dataset_path)
        with open(self.fnames_dataset_path, 'rb') as f:
            self.fnames_dataset = pickle.load(f)

    def _prepare_images_dataset(self):
        self.images_dataset_cam1 = dict()
        self.images_dataset_cam2 = dict()
        self.images_dataset_cam3 = dict()

        for j, (still_id, (still_path, (cam1_paths, cam2_paths, cam3_paths))) in enumerate(self.fnames_dataset.items()):
            print(j, still_id)
            cam1X = np.zeros((len(cam1_paths), self.dist_h, self.dist_w, self.n_channels), dtype=np.uint8)

            for i, path in enumerate(cam1_paths):
                cam1X[i] = cv2.resize(cv2.imread(path), (self.dist_w, self.dist_h))

            still_img = cv2.resize(cv2.imread(still_path), (self.dist_w, self.dist_h))

            self.images_dataset_cam1[still_id] = (still_img, cam1X)

        print("dumping images_dataset1...")
        arch1 = file_archive(self.images_dataset_path + "1")
        arch1["images_dataset_cam1"] = self.images_dataset_cam1
        arch1.dump()
        arch1.pop("images_dataset_cam1")
        del self.images_dataset_cam1

        for j, (still_id, (still_path, (cam1_paths, cam2_paths, cam3_paths))) in enumerate(self.fnames_dataset.items()):
            print(j, still_id)
            cam2X = np.zeros((len(cam2_paths), self.dist_h, self.dist_w, self.n_channels), dtype=np.uint8)

            for i, path in enumerate(cam2_paths):
                cam2X[i] = cv2.resize(cv2.imread(path), (self.dist_w, self.dist_h))

            still_img = cv2.resize(cv2.imread(still_path), (self.dist_w, self.dist_h))

            self.images_dataset_cam2[still_id] = (still_img, cam2X)

        print("dumping images_dataset2...")
        arch2 = file_archive(self.images_dataset_path + "2")
        arch2["images_dataset_cam2"] = self.images_dataset_cam2
        arch2.dump()
        arch2.pop("images_dataset_cam2")
        del self.images_dataset_cam2

        for j, (still_id, (still_path, (cam1_paths, cam2_paths, cam3_paths))) in enumerate(self.fnames_dataset.items()):
            print(j, still_id)
            cam3X = np.zeros((len(cam3_paths), self.dist_h, self.dist_w, self.n_channels), dtype=np.uint8)

            for i, path in enumerate(cam3_paths):
                cam3X[i] = cv2.resize(cv2.imread(path), (self.dist_w, self.dist_h))

            still_img = cv2.resize(cv2.imread(still_path), (self.dist_w, self.dist_h))

            self.images_dataset_cam3[still_id] = (still_img, cam3X)

        print("dumping images_dataset3...")
        arch3 = file_archive(self.images_dataset_path + "3")
        arch3["images_dataset_cam3"] = self.images_dataset_cam3
        arch3.dump()
        arch3.pop("images_dataset_cam3")
        del self.images_dataset_cam3

    def _load_images_dataset(self, cam_num):
        assert os.path.isfile(self.images_dataset_path + cam_num)

        print("loading images_dataset" + cam_num + "...")

        arch = file_archive(self.images_dataset_path + cam_num)
        self.images_dataset = arch.archive["images_dataset_cam" + cam_num]

    def define_min_video_length(self):
        m = 1000000000
        for key, (still_img, X1) in self.images_dataset.items():
            if X1.shape[0] < m:
                m = X1.shape[0]

        self.min_video_length = m

    def get_fnames_and_images(self, cam_num):
        # self._prepare_fnames_dataset()
        # self._dump_fnames_dataset()
        self._load_fnames_dataset()

        # self._prepare_images_dataset()
        self._load_images_dataset(cam_num)

        return self.fnames_dataset, self.images_dataset

    def _prepare_single_to_single_dataset(self):

        still_ids = sorted(self.images_dataset.keys())

        X = np.empty((0, self.dist_h, self.dist_w), dtype=np.uint8)
        y = np.empty((0, self.dist_h, self.dist_w), dtype=np.uint8)

        for i, still_id in enumerate(still_ids):
            still_img, (X1, X2, X3) = self.images_dataset[still_id]
            n = X1.shape[0] + X2.shape[0] + X3.shape[0]

            X = np.concatenate((X, X1, X2, X3))
            y = np.concatenate((y, np.tile(still_img, (n, 1, 1))))
            print(i)
        self.single_to_single_dataset = (X, y)

    def _dump_single_to_single_dataset(self):
        assert self.single_to_single_dataset is not None
        with open(self.single_to_single_dataset_path, 'wb') as f:
            pickle.dump(self.single_to_single_dataset, f)

    def _load_single_to_single_dataset(self):
        assert os.path.isfile(self.single_to_single_dataset_path)
        with open(self.single_to_single_dataset_path, 'rb') as f:
            self.single_to_single_dataset = pickle.load(f)

    def get_single_to_single(self):
        # self._prepare_single_to_single_dataset()
        # self._dump_single_to_single_dataset()
        self._load_single_to_single_dataset()

        return self.single_to_single_dataset

    def _prepare_for_one_cam(self, images_dataset):
        still_ids = sorted(images_dataset.keys())
        X = list()
        y = list()

        n_skipped = 0
        for i, still_id in enumerate(still_ids):
            still_img, X1 = images_dataset[still_id]

            X_video = list()
            y_video = list()

            if X1.shape[0] < self.min_video_length // 2:
                n_skipped += 1
                continue

            X1 = np.concatenate((X1, X1[::-1][:self.min_video_length - X1.shape[0] % self.min_video_length]))
            for ix in range(0, X1.shape[0], self.min_video_length):
                X_video.append(X1[ix:ix + self.min_video_length])
                y_video.append(still_img)
            X.extend(X_video)
            y.extend(y_video)

        print("n_skipped", n_skipped, "/", len(still_ids))
        return X, y

    def _prepare_seq_to_single_dataset(self, cam_num, percent_for_test=0.2):

        still_ids = sorted(self.images_dataset.keys())
        still_ids_train, still_ids_test = train_test_split(still_ids, test_size=percent_for_test, random_state=1)

        img_dataset_train = {id: self.images_dataset[id] for id in still_ids_train}
        img_dataset_test = {id: self.images_dataset[id] for id in still_ids_test}
        del self.images_dataset

        X_test, y_test = self._prepare_for_one_cam(img_dataset_test)
        X_test = np.array(X_test, dtype=np.uint8)
        y_test = np.array(y_test, dtype=np.uint8)
        del img_dataset_test

        X, y = self._prepare_for_one_cam(img_dataset_train)
        del img_dataset_train

        y = np.array(y, dtype=np.uint8)
        X = np.array(X, dtype=np.uint8)

        self.seq_to_single_dataset = (X, y, X_test, y_test)

        print("dumping seq_to_single_dataset" + cam_num + " ...")
        np.savez(self.seq_to_single_dataset_path + cam_num, X=X, y=y, X_test=X_test, y_test=y_test)

    def _load_seq_to_single_dataset(self, cam_num):
        assert os.path.isfile(self.seq_to_single_dataset_path + cam_num + ".npz")

        tmp = np.load(self.seq_to_single_dataset_path + cam_num + ".npz")
        self.seq_to_single_dataset = (tmp["X"], tmp["y"], tmp["X_test"], tmp["y_test"])

    def get_seq_to_single(self, cam_num):
        # self._prepare_seq_to_single_dataset(cam_num)
        self._load_seq_to_single_dataset(cam_num)

        return self.seq_to_single_dataset


# TRASH
# print(len(self.fnames_dataset["201103180001"][1]))
if __name__ == "__main__":
    loader = COXLoader(c.cox_still_dirpath_gray,
                       c.cox_video_dirpath_gray,
                       c.cox_fnames_dataset_path_gray,
                       c.cox_images_dataset_path_gray,
                       c.cox_single_to_single_dataset_path_gray,
                       c.cox_seq_to_single_dataset_path_gray, file_format=".bmp", dist_h=60, dist_w=48, n_channels=3)

    cam_num = "3"
    loader.min_video_length = 21

    loader.get_fnames_and_images(cam_num=cam_num)
    print("loaded imgs dataset", type(loader.images_dataset))
    loader.get_seq_to_single(cam_num=cam_num)
    print("seq to single dataset loaded", loader.seq_to_single_dataset[0].shape)
