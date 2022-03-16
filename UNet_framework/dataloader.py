import torch
# import torchvision
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
    def __init__(self, data_dir, size, train=True, train_test_split=0.85):
        ##########################inputs##################################
        # data_dir(string) - directory of the data#########################
        # size(int) - size of the images you want to use###################
        # train(boolean) - train data or test data#########################
        # train_test_split(float) - the portion of the data for training###
        super(Cell_data, self).__init__()
        # todo
        # initialize the data class
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

        one_img_dir = os.path.join(self.image_dir, name + '.png')
        one_trg_dir = os.path.join(self.target_dir, name + '.png')

        image = Image.open(one_img_dir)
        target = Image.open(one_trg_dir)
        # image = image[:, :, ::-1]
        # target = target[:, :, ::-1]

        # todo
        # return image and mask in tensors
        # image = image.resize((self.size, self.size))
        # target = target.resize((self.size, self.size))

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
