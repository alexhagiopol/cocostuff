import scipy.io as sio
import os
import glob
import numpy as np
import cv2

# file paths
coco_stuff_folder = os.path.abspath(os.path.join(os.path.dirname(os.path.realpath(__file__)), "../../.."))
annotation_folder = os.path.join(coco_stuff_folder, "dataset/annotations")
save_folder = os.path.join(coco_stuff_folder, "models/deeplab/cocostuff/data/annotations")
# get all images
image_filepaths = glob.glob(os.path.join(annotation_folder, "*.mat"))
# create a save folder if necessary
if not os.path.exists(save_folder):
    os.mkdir(save_folder)
# perform conversion on each image
for image_filepath in image_filepaths:
    print("Processing image " + image_filepath)
    label_array = sio.loadmat(image_filepath)
    #print(label_array['S'])
    label_map = label_array['S']
    assert(np.amax(label_map) <= 182)
    label_map -= 1
    label_map[label_map == -1] = 255
    label_map = label_map.astype(np.uint8)
    new_image_filename = image_filepath[len(annotation_folder) + 1:-4]+'.png'
    out_path = os.path.join(save_folder, new_image_filename)
    print("Exporting result to " + out_path)
    cv2.imwrite(out_path, label_map)
