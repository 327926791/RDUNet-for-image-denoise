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
import OpenEXR

from PIL import Image, ImageOps

#import any other libraries you need below this line

class Cell_data(Dataset):
    def __init__(self, data_dir, size, train='True', train_test_split=0.8, augment_data=True):
        ##########################inputs##################################
        # data_dir(string) - directory of the data#########################
        # size(int) - size of the images you want to use###################
        # train(boolean) - train data or test data#########################
        # train_test_split(float) - the portion of the data for training###
        # augment_data(boolean) - use data augmentation or not#############
        super(Cell_data, self).__init__()
        # initialize the data class
        self.images_dir = os.path.join(data_dir, "scans")
        self.masks_dir = os.path.join(data_dir, "labels")
        self.augment_data = augment_data
        self.size = size
        self.train = train
        self.train_test_split = train_test_split

        # print(self.images_dir)
        # print(self.masks_dir)
        self.images = [splitext(file)[0] for file in listdir(self.images_dir) if not file.startswith('.')]
        train_index = math.floor(len(self.images) * self.train_test_split)
        test_index = len(self.images) - train_index

        train_set, test_set = random_split(self.images, [train_index, test_index])

        if self.train:
            self.images = train_set
        else:
            self.images = test_set

        print(len(self.images), self.images)

    def __getitem__(self, idx):
        # to do
        # load image and mask from index idx of your data
        name = self.images[idx]
        img_name = os.path.join(self.images_dir, name + '.bmp')
        mask_name = os.path.join(self.masks_dir, name + '.bmp')
        # print(img_name)
        # print(mask_name)
        mask = Image.open(mask_name)
        img = Image.open(img_name)

        # mask.resize((self.size, self.size), resample=Image.NEAREST)
        # img.resize((self.size, self.size), resample=Image.BICUBIC)

        # data augmentation part
        if self.augment_data:
            augment_mode = np.random.randint(0, 4)
            if augment_mode == 0:
                # flip image vertically
                img = img.transpose(method=Image.FLIP_TOP_BOTTOM)
                mask = mask.transpose(method=Image.FLIP_TOP_BOTTOM)
            elif augment_mode == 1:
                # flip image horizontally
                img = img.transpose(method=Image.FLIP_LEFT_RIGHT)
                mask = mask.transpose(method=Image.FLIP_LEFT_RIGHT)

            elif augment_mode == 2:
                # zoom image
                random_scale = np.random.randint(10, 15)
                random_scale = float(random_scale/10)
                w, h = img.size
                img = img.crop((w/2 - w / random_scale / 2, h/2 - h / random_scale / 2,
                                w/2 + w / random_scale / 2, h/2 + h / random_scale / 2))
                w, h = mask.size
                mask = mask.crop((w/2 - w / random_scale / 2, h/2 - h / random_scale / 2,
                                w/2 + w / random_scale / 2, h/2 + h / random_scale / 2))
                # mask.resize((self.size, self.size))
                # img.resize((self.size, self.size))
            else:
                # rotate image
                degree = np.random.randint(1, 4) * 90
                # img = img.transpose(Image.ROTATE_90)
                # mask = mask.transpose(Image.ROTATE_90)
                img = img.rotate(degree)
                mask = mask.rotate(degree)

        # to do
        # return image and mask in tensors
        mask = mask.resize((self.size, self.size))
        img = img.resize((self.size, self.size))

        img= np.array(img)
        mask= np.array(mask)
        img = img / 255

        # print('dataloader img size' + str(img.shape))

        return (torch.as_tensor(img), torch.as_tensor(mask))

    def __len__(self):
        return len(self.images)

