

import sys

import dlib
import cv2

if len(sys.argv) != 3:
    print(
        "Call this program like this:\n"
        "   ./face_alignment.py shape_predictor_5_face_landmarks.dat ../examples/faces/bald_guys.jpg\n"
        "You can download a trained facial shape predictor from:\n"
        "    http://dlib.net/files/shape_predictor_5_face_landmarks.dat.bz2\n")
    exit()

predictor_path = sys.argv[1]
face_file_path = sys.argv[2]

# Load all the models we need: a detector to find the faces, a shape predictor
# to find face landmarks so we can precisely localize the face
bb_detector = dlib.get_frontal_face_detector()
shape_predictor = dlib.shape_predictor(predictor_path)

# Load the image using Dlib
img = dlib.load_rgb_image(face_file_path)

# Ask the detector to find the bounding boxes of each face. The 1 in the
# second argument indicates that we should upsample the image 1 time. This
# will make everything bigger and allow us to detect more faces.
bbs = bb_detector(img, 1)
print(bbs)

num_faces = len(bbs)
if num_faces == 0:
    print("Sorry, there were no faces found in '{}'".format(face_file_path))
    exit()

# Find the 5 face landmarks we need to do the alignment.
shapes = dlib.full_object_detections()
for bb in bbs:
    shape = shape_predictor(img, bb)
    shapes.append(shape)
    print(shape)

# window = dlib.image_window()

# Get the aligned face images
# Optionally:
aligned_images = dlib.get_face_chips(img, shapes, size=256, padding=0.5)
# aligned_images = dlib.get_face_chips(img, shapes, size=320)

# It is also possible to get a single chip
image = dlib.get_face_chip(img, shapes[0])

cv2.imwrite("lol.jpg", cv2.cvtColor(aligned_images[0], cv2.COLOR_RGB2BGR))
