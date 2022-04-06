import torch
from torchvision import transforms as T
from torch.utils.data import Dataset, DataLoader, random_split

import numpy as np
import math
import os
from os import listdir
from os.path import splitext
from pathlib import Path
import random
import cv2

from PIL import Image, ImageOps

#import any other libraries you need below this line

class Cell_data(Dataset):
    def __init__(self, data_dir, size, train=True, train_test_split=0.95, filetype='png'):
        ##########################inputs##################################
        # data_dir(string) - directory of the data#########################
        # size(int) - size of the images you want to use###################
        # train(boolean) - train data or test data#########################
        # train_test_split(float) - the portion of the data for training###
        super(Cell_data, self).__init__()
        # todo
        # initialize the data class
        self.filetype = filetype
        self.image_dir = os.path.join(data_dir, "source")
        self.target_dir = os.path.join(data_dir, "target")
        self.size = size
        self.train = train
        self.train_test_split = train_test_split

        self.names = [splitext(file)[0] for file in listdir(self.image_dir) if not file.startswith('.')]

        self.total = len(self.names)
        train_len = int(self.total * self.train_test_split)
        test_len = self.total - train_len

        train_set, test_set = random_split(self.names, [train_len, test_len])

        if self.train is True:
            self.names = train_set
        else:
            self.names = test_set

        print(self.train, len(self.names))

    def __getitem__(self, idx):
        # todo

        # load image and mask from index idx of your data
        name = self.names[idx]

        one_img_dir = os.path.join(self.image_dir, name + '.' + self.filetype)
        one_trg_dir = os.path.join(self.target_dir, name + '.' + self.filetype)

        if self.filetype == "exr" or self.filetype == "EXR":
            image = cv2.imread(one_img_dir, cv2.IMREAD_UNCHANGED)
            target = cv2.imread(one_trg_dir, cv2.IMREAD_UNCHANGED)

            rand_scale = np.random.randint(11, 16)
            rand_scale = float(rand_scale / 10)
            width = int(image.shape[1] * rand_scale)
            height = int(image.shape[0] * rand_scale)
            dim = (width, height)

            # print("before resize: " + str(image.shape))
            image = cv2.resize(image, dim)
            target = cv2.resize(target, dim)

            image = image[:, :, ::-1]
            target = target[:, :, ::-1]

            # print("before crop: " + str(image.shape))
            rand_crop_x = np.random.randint(0, image.shape[1] - 572)
            rand_crop_y = np.random.randint(0, image.shape[0] - 572)
            image = image[rand_crop_x:rand_crop_x+572, rand_crop_y:rand_crop_y+572,:]
            target = target[rand_crop_x:rand_crop_x + 572, rand_crop_y:rand_crop_y + 572, :]
            # print("after crop: " + str(image.shape))
            flip = random.randint(1,4)

            if flip == 2:
                image = np.flipud(image)
                target = np.flipud(target)
            elif flip == 3:
                image = np.fliplr(image)
                target = np.fliplr(target)
            elif flip == 4:
                image = np.fliplr(image)
                target = np.fliplr(target)
                image = np.flipud(image)
                target = np.flipud(target)

            rot = random.randint(1, 4)
            image = np.rot90(image, rot)
            target = np.rot90(target, rot)




        else:
            image = Image.open(one_img_dir)
            target = Image.open(one_trg_dir)
            image = np.array(image)
            target = np.array(target)

        norm_image = (image - np.min(image)) / (np.max(image) - np.min(image))
        norm_target = (target - np.min(target)) / (np.max(target) - np.min(target))

        ts_image = torch.as_tensor(norm_image).squeeze(0).permute(2, 0, 1)
        ts_target = torch.as_tensor(norm_target).squeeze(0).permute(2, 0, 1)
        # print('input image size:' + str(ts_image.size()))

        return ts_image, ts_target

    def __len__(self):
        return len(self.names)
