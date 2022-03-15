
import os
from os import listdir
from os.path import splitext
from pathlib import Path
import random
import cv2
import os

def Preprocess(data_dir, new_data_dir):
    image_dir = os.path.join(data_dir, "source")
    target_dir = os.path.join(data_dir, "target")

    new_image_dir = os.path.join(new_data_dir, "source")
    new_target_dir = os.path.join(new_data_dir, "target")

    names = [splitext(file)[0] for file in listdir(image_dir) if not file.startswith('.')]

    for name in names:
      one_img_dir = os.path.join(image_dir, name + '.exr')
      one_trg_dir = os.path.join(target_dir, name + '.exr')

      image = cv2.imread(one_img_dir, cv2.IMREAD_UNCHANGED)
      target = cv2.imread(one_trg_dir, cv2.IMREAD_UNCHANGED)
      image = image[:, :, ::-1]
      target = target[:, :, ::-1]

      # 2214 - 6 = 2208
      image = image[3: 2211, :, :]
      target = target[3: 2211, :, :]
      # print(image.shape)

      # 2208 * 4096 -> 276 * 512
      for i in range(8):
        for j in range(8):

          new_img_dir = os.path.join(new_image_dir, name + '_' + str(i * 8 + j) + '.exr')
          new_trg_dir = os.path.join(new_target_dir, name + '_' + str(i * 8 + j) + '.exr')

          new_img = image[i*216 : (i+1)*216, j*372 : (j+1)*372, :]
          new_trg = target[i*216 : (i+1)*216, j*372 : (j+1)*372, :]
          
          # print(new_img.shape)

          cv2.imwrite(new_img_dir, new_img)
          cv2.imwrite(new_trg_dir, new_trg)
        
