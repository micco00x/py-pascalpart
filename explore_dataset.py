import argparse
import PIL
import os
import matplotlib.pyplot as plt

import utils

SHOW_IMAGES = True

# Plot a mask composed by 0s and 1s with a certain title
# and compare it with the original image:
def plot_mask(img, mask, bodypart_mask, windowtitle, suptitle):
    mask = PIL.Image.fromarray(mask * 255)
    bodypart_mask = PIL.Image.fromarray(bodypart_mask * 255)
    fig = plt.figure()
    fig.canvas.set_window_title(windowtitle)
    fig.suptitle(suptitle)
    fig.add_subplot(1, 3, 1)
    plt.axis("off")
    plt.imshow(img)
    fig.add_subplot(1, 3, 2)
    plt.axis("off")
    plt.imshow(mask)
    fig.add_subplot(1, 3, 3)
    plt.axis("off")
    plt.imshow(bodypart_mask)
    plt.show()

# Load annotations from the annotation folder of PASCAL-Part dataset:
if __name__ == "__main__":
    # Parse arguments from command line:
    parser = argparse.ArgumentParser(description="Extract data from PASCAL-Part Dataset")
    parser.add_argument("--annotation_folder", default="datasets/trainval/Annotations_Part", help="Path to the PASCAL-Part Dataset annotation folder")
    parser.add_argument("--images_folder", default="datasets/VOCdevkit/VOC2010/JPEGImages", help="Path to the PASCAL VOC 2010 JPEG images")
    args = parser.parse_args()

    # Stats on the dataset:
    obj_cnt = 0
    bodypart_cnt = 0

    mat_filenames = os.listdir(args.annotation_folder)
    # Iterate through the .mat files contained in path:
    for idx, annotation_filename in enumerate(mat_filenames):
        annotations = utils.load_annotations(os.path.join(args.annotation_folder, annotation_filename))
        image_filename = annotation_filename[:annotation_filename.rfind(".")] + ".jpg" # PASCAL VOC image have .jpg format

        obj_cnt += len(annotations["objects"])

        # Show original image with its mask:
        img = PIL.Image.open(os.path.join(args.images_folder, image_filename))
        for obj in annotations["objects"]:
            bodypart_cnt += len(obj["parts"])
            print("obj_cnt: {} - bodypart_cnt: {}".format(obj_cnt, bodypart_cnt), end="\r")
            if SHOW_IMAGES:
                for body_part in obj["parts"]:
                    plot_mask(img, obj["mask"], body_part["mask"], image_filename, obj["class"] + ": " + body_part["part_name"])

    print("obj_cnt: {} - bodypart_cnt: {}".format(obj_cnt, bodypart_cnt))
