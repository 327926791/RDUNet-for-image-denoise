
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

      # 2214 * 4096 -> 572 * 572 (4, 8)
      for i in range(4):
        for j in range(8):

          new_img_dir = os.path.join(new_image_dir, name + '_' + str(i * 8 + j) + '.exr')
          new_trg_dir = os.path.join(new_target_dir, name + '_' + str(i * 8 + j) + '.exr')

          new_img = image[i*545 : i*545+572, j*503 : j*503+572, :]
          new_trg = target[i*545 : i*545+572, j*503 : j*503+572, :]

          # print("i = " + str(i) + ", j = " + str(j))
          # print(i*545, i*545+572, j*503, j*503+572)

          # print(new_img.shape)

          cv2.imwrite(new_img_dir, new_img)
          cv2.imwrite(new_trg_dir, new_trg)
        

def Output(save_dir, predictions):
    idx = 0
    for pred in predictions:
        file_name = os.path.join(save_dir, 'test_' + idx + '.exr')
        cv2.imwrite(file_name, pred)