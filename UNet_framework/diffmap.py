import os
from torch.utils.data import Dataset, DataLoader, random_split
from os import listdir
from os.path import splitext
import cv2
import numpy as np

def RGBtoGray(image):
    r, g, b = image[:, :, 0], image[:, :, 1], image[:, :, 2]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
    return gray

def compare(image, target):
    # print("image shape: ")
    # print(image.shape)
    # print("target shape:")
    # print(target.shape)

    gray_image = RGBtoGray(image)
    gray_target = RGBtoGray(target)

    # print("gray image shape: ")
    # print(gray_image)
    # print("gray target shape:")
    # print(gray_target)

    difference = np.abs(gray_target - gray_image)
    print("diff:")
    print(np.sum(difference))

    return difference


def GetDiffMap(preds, labels, inputs):
    root_dir = os.getcwd()
    data_path = 'result'
    data_dir = os.path.join(root_dir, data_path)
    n = len(preds)

    dir1 = os.path.join(data_dir, "diff_input_pred")
    dir2 = os.path.join(data_dir, "diff_input_GT")

    for i in range(n):
        input = inputs[i]
        label = labels[i]
        pred = preds[i]
        diff_input_pred = compare(input, pred)
        diff_input_GT = compare(input, label)

        ip_dir = os.path.join(dir1, 'test_' + str(i) + '.exr')
        ig_dir = os.path.join(dir2, 'test_' + str(i) + '.exr')

        cv2.imwrite(ip_dir, diff_input_pred)
        cv2.imwrite(ig_dir, diff_input_GT)
