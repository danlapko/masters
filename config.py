# neuro submit -c 4 -g 1 -m 24G --volume storage://~:/var/storage/home:rw image://danlapko/danlapko/faces

# neuro job port-forward ID 4000 22
# ssh root@localhost -p 4000

# vim .bashrc
# export PATH="/root/anaconda3/bin:/usr/local/nvidia/bin:/usr/local/cuda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
# ldconfig.real

# neuro top ID


import getpass
from collections import namedtuple

user = getpass.getuser()

CoxDataPaths = namedtuple("CoxDataPaths", ('still_dirpath',
                                           'video_dirpath',
                                           'fnames_dataset_path',
                                           'images_dataset_path',
                                           'single2single_dataset_path',
                                           'seq2single_dataset_path',
                                           'seq2seq_dataset_path'))

cox_data_paths_gray = CoxDataPaths(still_dirpath=f"/home/{user}/masters/datasets/face_48_60/still",
                                   video_dirpath=f"/home/{user}/masters/datasets/face_48_60/video",
                                   fnames_dataset_path=f"/home/{user}/masters/datasets/face_48_60/fnames_dataset",
                                   images_dataset_path=f"/home/{user}/masters/datasets/face_48_60/dataset",
                                   single2single_dataset_path=f"/home/{user}/masters/datasets/face_48_60/single_to_single_dataset",
                                   seq2single_dataset_path=f"/home/{user}/masters/datasets/face_48_60/seq_to_single_dataset",
                                   seq2seq_dataset_path=f"/home/{user}/masters/datasets/face_48_60/seq_to_seq_dataset")

cox_data_paths_rgb = CoxDataPaths(still_dirpath=f"/home/{user}/masters/datasets/original_still_video/still_cropped",
                                  video_dirpath=f"/home/{user}/masters/datasets/original_still_video/video_cropped",
                                  fnames_dataset_path=f"/home/{user}/masters/datasets/original_still_video/fnames_dataset",
                                  images_dataset_path=f"/home/{user}/masters/datasets/original_still_video/dataset",
                                  single2single_dataset_path=f"/home/{user}/masters/datasets/original_still_video/single_to_single_dataset",
                                  seq2single_dataset_path=f"/home/{user}/masters/datasets/original_still_video/seq_to_single_dataset",
                                  seq2seq_dataset_path="")
