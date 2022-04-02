import os
from torch.utils.data import Dataset, DataLoader, random_split
from os import listdir
from os.path import splitext
import cv2
import numpy as np
import plotly.express as px
import torch

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

    difference = torch.abs(gray_target - gray_image)
    print("diff:")
    print(torch.sum(difference))

    return difference


def GetDiffMap(preds, labels, inputs):
    root_dir = os.getcwd()
    data_path = 'results'
    data_dir = os.path.join(root_dir, data_path)
    n = len(preds)

    dir1 = os.path.join(data_dir, "diff_input_pred")
    dir2 = os.path.join(data_dir, "diff_input_GT")

    if not os.path.exists(dir1):
        os.makedirs(dir1)
    if not os.path.exists(dir2):
        os.makedirs(dir2)

    diff_input_pred_list = []
    diff_input_GT_list = []
    for i in range(n):
        input = inputs[i].permute(1,2,0)
        label = labels[i].permute(1,2,0)
        pred = preds[i].permute(1,2,0)
        diff_input_pred = compare(input, pred)
        diff_input_GT = compare(input, label)

        diff_input_pred = np.array(diff_input_pred.cpu())
        diff_input_GT = np.array(diff_input_GT.cpu())

        diff_input_pred = np.array([diff_input_pred, diff_input_pred, diff_input_pred])
        print(diff_input_pred.shape)
        diff_input_pred = np.transpose(diff_input_pred,(1,2,0))
        diff_input_GT = np.array([diff_input_GT, diff_input_GT, diff_input_GT])
        print(diff_input_GT.shape)
        diff_input_GT = np.transpose(diff_input_GT,(1,2,0))

        diff_input_pred_list.append(diff_input_pred)
        diff_input_GT_list.append(diff_input_GT)



        ip_dir = os.path.join(dir1, 'test_' + str(i) + '.exr')
        ig_dir = os.path.join(dir2, 'test_' + str(i) + '.exr')

        # cv2.imwrite(ip_dir, diff_input_pred)
        # cv2.imwrite(ig_dir, diff_input_GT)

    # diff_input_pred_list = torch.tensor(diff_input_pred_list)
    # diff_input_GT_list = torch.tensor(diff_input_GT_list)
    #
    # print(diff_input_pred_list.shape)
    # print(diff_input_GT_list.shape)
    return diff_input_pred_list, diff_input_GT_list


def get_heat_map(preds, labels, inputs,dir_name):
    n = len(preds)
    root_dir = os.getcwd()

    heatmap_dir = os.path.join(root_dir, dir_name)
    heatmap_inputs_label_dir = os.path.join(heatmap_dir, 'input_label-heatmap')
    heatmap_pred_label_dir = os.path.join(heatmap_dir, 'pred_label-heatmap')
    if not os.path.exists(heatmap_inputs_label_dir):
        os.makedirs(heatmap_inputs_label_dir)
    if not os.path.exists(heatmap_pred_label_dir):
        os.makedirs(heatmap_pred_label_dir)

    heat_map_pred_label_list = []
    heat_map_input_label_list = []

    for i in range(n):
        input = inputs[i].permute(1,2,0)
        label = labels[i].permute(1,2,0)
        pred = preds[i].permute(1,2,0)
        input = RGBtoGray(input)
        label = RGBtoGray(label)
        pred = RGBtoGray(pred)
        heat_map_pred_label = pred - label
        heat_map_input_label = input - label

        heat_map_pred_label_list.append(heat_map_pred_label.cpu())
        heat_map_input_label_list.append(heat_map_input_label.cpu())

        heat_map_pred_label = np.array(heat_map_pred_label.cpu())
        heat_map_input_label = np.array(heat_map_input_label.cpu())

        fig_pred_label = px.imshow(heat_map_pred_label)
        fig_input_label = px.imshow(heat_map_input_label)

        # fig_pred_label.show()
        # fig_input_label.show()

        fig_pred_label.write_image(heatmap_pred_label_dir+'/'+str(i)+'_pred_label.png')
        fig_input_label.write_image(heatmap_inputs_label_dir+'/'+str(i)+'_input_label.png')


    # heat_map_pred_label_list = torch.tensor(heat_map_pred_label_list)
    # heat_map_input_label_list = torch.tensor(heat_map_input_label_list)
    #
    # print(heat_map_pred_label_list.shape)
    # print(heat_map_input_label_list.shape)
    return heat_map_pred_label_list, heat_map_input_label_list