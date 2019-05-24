import getpass
import glob
import os
import time
from scipy.ndimage.filters import median_filter
import cv2
import dlib
import numpy as np
from collections import deque

user = getpass.getuser()

if user == "danila":
    dataset_path = f"/home/{user}/masters"
else:
    dataset_path = f"/var/storage/home"
if user == "danila":
    prjct_path = f"/home/{user}/masters"
else:
    prjct_path = f"/root"

in_dir = f"{dataset_path}/datasets/original_still_video/video"
shape_predictor_path = f"{prjct_path}/prjct/alignment/shape_predictor_5_face_landmarks.dat"
caffe_model_path = f"{prjct_path}/prjct/alignment/res10_300x300_ssd_iter_140000_fp16.caffemodel"
model_config_path = f"{prjct_path}/prjct/alignment/deploy.prototxt"
file_format = "*.jpg"

path_replace_from = "/video/"
path_replace_to = "/video_cropped/"

n_images_to_read = 415_418
# n_images_to_read = 1000
min_confidence = 0.95

use_alignment = False
padding = 0.6
out_ratio = 1.25
out_size = (400, 320)
reserve = 1000
buf_size = 5

bb_detector = cv2.dnn.readNetFromCaffe(model_config_path, caffe_model_path)
shape_predictor = dlib.shape_predictor(shape_predictor_path)

clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))


def equalize_func(imgs, use_clahe=False):
    out = [None]*len(imgs)
    for i in range(len(imgs)):
        img = imgs[i]
        lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
        lab_planes = cv2.split(lab)

        if use_clahe:
            lab_planes[0] = clahe.apply(lab_planes[0])
        else:
            lab_planes[0] = cv2.equalizeHist(lab_planes[0])
        lab = cv2.merge(lab_planes)
        img = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
        out[i] = img
    return out


def load_imgs(in_dir, n_to_read, format):
    names = sorted(glob.iglob(in_dir + "/**/" + format, recursive=True))
    imgs = list()

    for name in names[:n_to_read]:
        img = dlib.load_rgb_image(name)
        imgs.append(img)
    return names, imgs


def get_bounding_boxes(imgs, names):
    n = len(imgs)
    bbs = list()
    confidences = list()
    out_names = list()
    out_imgs = list()
    out_idxs = list()
    for i, name in enumerate(names[:n]):
        img = imgs[i]
        blob = cv2.dnn.blobFromImage(img, 1.0, (300, 300), [104, 117, 123], False, False)

        bb_detector.setInput(blob)
        detections = bb_detector.forward()

        # print(detections[:, :, [0, 1, 2], 2])
        confidence = detections[0, 0, 0, 2]

        confidences.append(confidence)

        if confidence >= min_confidence:
            h, w = img.shape[0], img.shape[1]
            left = int(detections[0, 0, 0, 3] * w)
            top = int(detections[0, 0, 0, 4] * h)
            right = int(detections[0, 0, 0, 5] * w)
            bot = int(detections[0, 0, 0, 6] * h)
            bbs.append((left, top, right, bot))
            out_names.append(name)
            out_imgs.append(img)
            out_idxs.append(i)
    print(f"confidences: {len(out_names)}/{n} min={min(confidences)} ({names[np.argmin(confidences)]}) "
          f"max={max(confidences)} mean={sum(confidences) / n}")
    return bbs, out_names, out_imgs, out_idxs


def get_faces_shapes(imgs, names, bbs):
    n = len(bbs)
    shapes = [None] * n
    for i, name in enumerate(names[:n]):
        left, top, right, bot = bbs[i]
        bb = dlib.rectangle(left, top, right, bot)
        shape = shape_predictor(imgs[i], bb)
        shapes[i] = shape
        if shape.num_parts != 5:
            print(f"Warning: {name} contains {shape.num_parts} parts of shapes")
    return shapes


def get_cropped_and_rotated_faces(imgs, names, shapes):
    n = len(shapes)
    cropped = [None] * n
    for i, name in enumerate(names[:n]):
        cropped[i] = dlib.get_face_chip(imgs[i], shapes[i], size=512, padding=1)[16:-96, 96:-96]
    return cropped


def store_imgs(imgs, names, part_to_replace_from, part_to_replace_to):
    n = len(imgs)
    for i, name in enumerate(names[:n]):
        name = name.replace(part_to_replace_from, part_to_replace_to)
        dir = os.path.dirname(name)
        os.makedirs(dir, exist_ok=True)
        cv2.imwrite(name, cv2.cvtColor(imgs[i], cv2.COLOR_RGB2BGR))


def my_cropping(imgs, names, bbs):
    n = len(bbs)
    cropped = [None] * n

    cxs = [None] * n
    cys = [None] * n
    for i in range(n):
        left, top, right, bot = bbs[i]
        cxs[i] = (left + right) / 2 / imgs[i].shape[1]
        cys[i] = (top + bot) / 2 / imgs[i].shape[0]

    # filtered_cxs = median_filter(cxs, 7)
    # filtered_cys = median_filter(cys, 7)
    # print(len(cxs), len(filtered_cxs))
    # print(list(zip(cxs, filtered_cxs)))

    for i in range(n):
        img, bb = imgs[i], bbs[i]
        left, top, right, bot = bb
        # left, top, right, bot = 0, 0, img.shape[1], img.shape[0]
        w = right - left
        h = bot - top

        if i == 0:
            ws = deque([w] * buf_size, maxlen=buf_size)
            hs = deque([h] * buf_size, maxlen=buf_size)
        else:
            ws.append(w)
            hs.append(h)
        w = sum(ws) / buf_size
        h = sum(hs) / buf_size

        # cx = (left + right) // 2
        # cy = (top + bot) // 2

        # if prev_dir != dir:
        #     cxs = deque([cx / img.shape[1]] * buf_size, maxlen=buf_size)
        #     cys = deque([cy / img.shape[0]] * buf_size, maxlen=buf_size)
        #     prev_dir = dir
        # else:
        #     cxs.append(w / img.shape[1])
        #     cys.append(h / img.shape[0])
        #
        # cx = sum(cxs) * img.shape[1] / buf_size
        # cy = sum(cys) * img.shape[0] / buf_size

        cx = int(img.shape[1] * cxs[i])
        cy = int(img.shape[0] * cys[i])

        if h / w > out_ratio:
            w = h / out_ratio
        else:
            h = w * out_ratio

        left, top, right, bot = int(cx - w / 2), int(cy - h / 2), int(cx + w / 2), int(cy + h / 2)

        img_ = np.zeros((img.shape[0] + 2 * reserve, img.shape[1] + 2 * reserve, img.shape[2]), dtype=img.dtype)
        img_[reserve:-reserve, reserve:-reserve] = img
        crop = img_[reserve + top - int(padding / 2 * h):reserve + bot + int(padding / 2 * h),
               reserve + left - int(padding / 2 * w):reserve + right + int(padding / 2 * w)]
        cropped[i] = cv2.resize(crop, (out_size[1], out_size[0]))
        # cropped[i] = crop
    return cropped


def main_():
    for cam in ("cam1", "cam2", "cam3"):
        in_dir_ = in_dir + "/" + cam
        sub_dirs = os.listdir(in_dir_)
        for i_dir, sub_dir in enumerate(sub_dirs):
            in_dir_ = in_dir + "/" + cam + "/" + sub_dir

            print()
            print("--> dir", i_dir, cam + "/" + sub_dir)
            t0 = time.time()

            names, imgs = load_imgs(in_dir_, n_images_to_read, file_format)

            eq_imgs = equalize_func(imgs, use_clahe=False)

            bbs, names, eq_imgs, idxs = get_bounding_boxes(eq_imgs, names)
            imgs = np.array(imgs)[idxs]

            if use_alignment:
                shapes = get_faces_shapes(imgs, names, bbs)
                cropped = get_cropped_and_rotated_faces(imgs, names, shapes)
            else:
                cropped = my_cropping(imgs, names, bbs)
            cropped = equalize_func(cropped, use_clahe=True)

            store_imgs(cropped, names, path_replace_from, path_replace_to)
            print("time", time.time() - t0)


if __name__ == "__main__":
    main_()

#     bb_detector = dlib.get_frontal_face_detector()
# location = face_recognition.face_locations(imgs[i], number_of_times_to_upsample=1, model="hog")[0]

# (y1, x1, y2, x2) = location
# k_y, k_x = origin_imgs[i].shape[0] / imgs[i].shape[0], origin_imgs[i].shape[1] / imgs[i].shape[1]

# dy = y1 - y2
# dx = x1 - x2

# y1, x1, y2, x2 = int(k_y * (y1 + dy * 0.7)), int(k_x * (x1 + dx * 0.5)), int(k_y * (y2 - dy * 0.3)), int(
# k_x * (x2 - dx * 0.5))

# bbs[i] = bb_detector(img, 1)[0]

# (bot, left) = bbs[i].bottom(), bbs[i].left()
# (top, right) = bbs[i].top(), bbs[i].right()
# h = top - bot
# w = right - left
# bbs[i] = dlib.rectangle(left, top + int(0.5 * h), right, bot)
# print(bbs[i], (left, top, right, bot))

# rect = shape.rect
# (bot, left) = rect.bottom(), rect.left()
# (top, right) = rect.top(), rect.right()
# h = top - bot
# w = right - left
# rect = dlib.rectangle(left, top + h, right, bot - h)
# parts = [shape.part(m) for m in range(shape.num_parts)]
# print(shape.num_parts)
# print(parts)
# shape = dlib.full_object_detection(rect, parts)


# print(base_name, (x1, y1, x2, y2))
# origin_img = cv2.rectangle(origin_imgs[i], (x1, y1), (x2, y2), (255, 0, 0), 2)
# origin_img = origin_imgs[i][y1:y2, x2:x1, :]

# print(i, base_name, aligneds[i].shape)

# w = right - left
# h = bot - top
# img = cv2.rectangle(img, (left, top), (right, bot), (0, 255, 0), 1)
# c_x = left + w // 2
# c_y = top + h // 2
# cv2.circle(img, (c_x,c_y), 10, (0, 255, 0), 1 )


# bb = dlib.rectangle(left, top, right, bot)
# shape = shape_predictor(imgs[i], bb)
# crop = dlib.get_face_chip(img, shape, size=512, padding=padding)
# pppp = 512 - int(512 / out_ratio)
# crop = crop[:, pppp//2:-pppp//2]
