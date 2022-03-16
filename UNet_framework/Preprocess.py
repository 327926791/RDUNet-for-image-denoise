
import os
from os import listdir
from os.path import splitext
from pathlib import Path
import random
import cv2
import os
from PIL import Image
import torchvision.transforms as T

def Preprocess(data_dir, new_data_dir, filetype):
    image_dir = os.path.join(data_dir, "source")
    target_dir = os.path.join(data_dir, "target")

    new_image_dir = os.path.join(new_data_dir, "source")
    new_target_dir = os.path.join(new_data_dir, "target")

    if not os.path.exists(image_dir):
        os.makedirs(image_dir)
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)
    if not os.path.exists(new_image_dir):
        os.makedirs(new_image_dir)
    if not os.path.exists(new_target_dir):
        os.makedirs(new_target_dir)

    names = [splitext(file)[0] for file in listdir(image_dir) if not file.startswith('.')]


    # ################################### EXR image #####################################################
    # for name in names:
    #   one_img_dir = os.path.join(image_dir, name + '.exr')
    #   one_trg_dir = os.path.join(target_dir, name + '.exr')
    #
    #   print(name)
    #   # print(image_dir)
    #   # print(target_dir)
    #   print(one_img_dir)
    #   print(one_trg_dir)
    #   # print(new_image_dir)
    #   # print(new_target_dir)
    #
    #   image = cv2.imread(one_img_dir, cv2.IMREAD_UNCHANGED)
    #   target = cv2.imread(one_trg_dir, cv2.IMREAD_UNCHANGED)
    #   image = image[:, :, ::-1]
    #   target = target[:, :, ::-1]
    #
    #   # 2214 * 4096 -> 572 * 572 (4, 8)
    #   for i in range(4):
    #     for j in range(8):
    #
    #       new_img_dir = os.path.join(new_image_dir, name + '_' + str(i * 8 + j) + '.exr')
    #       new_trg_dir = os.path.join(new_target_dir, name + '_' + str(i * 8 + j) + '.exr')
    #
    #       new_img = image[i*545 : i*545+572, j*503 : j*503+572, :]
    #       new_trg = target[i*545 : i*545+572, j*503 : j*503+572, :]
    #
    #       # print("i = " + str(i) + ", j = " + str(j))
    #       # print(i*545, i*545+572, j*503, j*503+572)
    #
    #       # print(new_img.shape)
    #
    #       cv2.imwrite(new_img_dir, new_img)
    #       cv2.imwrite(new_trg_dir, new_trg)

    # ################################### PNG image #####################################################
    for name in names:
        one_img_dir = os.path.join(image_dir, name + '.' + filetype)
        one_trg_dir = os.path.join(target_dir, name + '.' + filetype)

        print(name)
        #   # print(image_dir)
        #   # print(target_dir)
        print(one_img_dir)
        print(one_trg_dir)
        #   # print(new_image_dir)
        #   # print(new_target_dir)

        image = cv2.imread(one_img_dir, cv2.IMREAD_UNCHANGED)
        target = cv2.imread(one_trg_dir, cv2.IMREAD_UNCHANGED)
        image = image[:, :, ::-1]
        target = target[:, :, ::-1]

        # 2214 * 4096 -> 572 * 572 (4, 8)
        for i in range(5):
            for j in range(7):
                new_img_dir = os.path.join(new_image_dir, name + '_' + str(i * 8 + j) + '.' + filetype)
                new_trg_dir = os.path.join(new_target_dir, name + '_' + str(i * 8 + j) + '.' + filetype)

                new_img = image[i * 572: i * 572 + 572, j * 572: j * 572 + 572, :]
                new_trg = target[i * 572: i * 572 + 572, j * 572: j * 572 + 572, :]

                im = Image.fromarray(new_img)
                im.save(new_img_dir)

                tr = Image.fromarray(new_trg)
                tr.save(new_trg_dir)


def Output(save_dir, predictions, GTs, inputs, filetype):
    idx = 0
    n = len(predictions[0])
    # print(n, batch_size)
    transform = T.ToPILImage()
    for i in range(n):
        pred = predictions[i]
        GT = GTs[i]
        input = inputs[i]

        file_name = os.path.join(save_dir, 'test_pred_' + str(idx) + '.' + filetype)
        file_name_GT = os.path.join(save_dir, 'test_GT_' + str(idx) + '.' + filetype)
        file_name_in = os.path.join(save_dir, 'test_in_' + str(idx) + '.' + filetype)

        im = transform(pred)
        gt = transform(GT)
        inp = transform(input)

        im.save(file_name)
        gt.save(file_name_GT)
        inp.save(file_name_in)

        idx += 1