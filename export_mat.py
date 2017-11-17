import cv2
import sys
import glob
import os
import scipy.io as sio
import matplotlib.pyplot as plt
import numpy as np


def show_image(location, title, img, width=3, open_new_window=False):
    if open_new_window:
        plt.figure(figsize=(width, width))
    plt.subplot(*location)
    plt.title(title, fontsize=8)
    plt.axis('off')
    if len(img.shape) == 3:
        plt.imshow(img)
    else:
        plt.imshow(img, cmap='gray')
    if open_new_window:
        plt.show()
        plt.close()


def mat_to_img(mat_file_path):
    mat_data = sio.loadmat(mat_file_path)
    tensor = mat_data['data']
    print("tensor dimentions" + str(tensor.shape))
    annotation_image = np.reshape(tensor, (tensor.shape[0], tensor.shape[1]))
    print("annotation dimentions " + str(annotation_image.shape))
    # keep only people annotations for now
    annotation_image[annotation_image == 0] = 255  # 0 is the people annotation - make it white
    annotation_image[annotation_image != 255] = 0  #  make pixels black where there is no people annotation
    return np.transpose(annotation_image)


def mask_visualization(input_img, annotation_img):
    for i in range(input_img.shape[2]):
        input_img[:, :, i] = input_img[:, :, i] * (annotation_img / 255)
    # mark masked out spots with red
    annotation_img = annotation_img / 255
    annotation_img[annotation_img == 0] = 255
    annotation_img[annotation_img == 1] = 0
    input_img[:, :, 0] += np.uint8(annotation_img)
    return input_img


if __name__ == "__main__":
    if len(sys.argv) != 5:
        print("Usage: \n python export_mat.py relative/path/to/input/images relative/path/to/mat/files /elative/path/to/results visualize?_0or1 \n "
              "Example: python export_mat.py dataset/images models/deeplab/cocostuff/features/deeplabv2_vgg16/val/fc8 results/ 0")
        sys.exit()
    current_path = os.path.abspath(os.path.dirname(os.path.realpath(__file__)))
    input_image_path = os.path.join(current_path, sys.argv[1])
    mat_path = os.path.join(current_path, sys.argv[2])
    output_mask_path = os.path.join(current_path, sys.argv[3])
    visualize = int(sys.argv[4])
    mat_file_paths = glob.glob(os.path.join(mat_path, "*.mat"))
    for mat_file_path in mat_file_paths:
        image_name = mat_file_path[len(mat_path) + 1: -11]
        matching_input_images = glob.glob(os.path.join(input_image_path, image_name + "*"))
        if len(matching_input_images) < 1:
            print("No matching input found for " + image_name)
            continue
        image_file_path = matching_input_images[0]
        input_image = cv2.cvtColor(cv2.imread(image_file_path), cv2.COLOR_BGR2RGB)
        output_mat_image = mat_to_img(mat_file_path)
        mask = mask_visualization(input_image.copy(), output_mat_image.copy())
        if visualize:
            plt.figure()
            show_image((1, 3, 1), "input image " + str(input_image.shape),input_image, 40, False)
            show_image((1, 3, 2), "output annotation image " + str(output_mat_image.shape), output_mat_image, 40, False)
            show_image((1, 3, 3), "mask visualization " + str(mask.shape), mask, 40, False)
            plt.show()
            plt.savefig(os.path.join(output_mask_path, image_name + "_dnn_mask_visualization.png"))
            plt.close()
        cv2.imwrite(os.path.join(output_mask_path, image_name + "_mask.png"), output_mat_image)
