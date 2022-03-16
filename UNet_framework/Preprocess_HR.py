'''Preprocess DIV2K_train_HR dataset
Add noise
Split into patches
'''
import os
import random

import cv2
import numpy as np
import matplotlib.pyplot as plt

root_dir = os.getcwd()

root_dir = os.path.dirname(root_dir)
data_dir = os.path.join(root_dir,'UNet_framework')
train_data_dir = os.path.join(root_dir,'UNet_framework/DIV2K_train_HR')
def get_files(dir_path):
    return os.listdir(dir_path)

def get_image(img_path):
    img = cv2.imread(img_path)
    print(img.shape)
    return  img

#Here we set mod to [64,64], the patch size will be 64*64
def mod_crop(image, mod):
    """
    Crops image according to mod to restore spatial dimensions
    adequately in the decoding sections of the model.
    :param image: numpy array
        Image to crop.
    :param mod: int array
        Module for padding allowed by the number of
        encoding/decoding sections in the model.
    :return: numpy array
        Copped image
    """
    size = image.shape[:2]
    size = size - np.mod(size, mod)
    print("size" + str(size))
    image = image[:size[0], :size[1], ...]
    num_of_patches_w = size[0]/mod[0]
    num_of_patches_h = size[1]/mod[0]

    print(image.shape)
    return image

def split_into_patches(image, patch_size):
    """get patches of an image
    :param image: numpy array
             Image to split
    :param patch_size: int
    :return: list
         a list of patches
    """
    patches = []
    w,h,_ = image.shape
    num_of_patches_w = int(w/patch_size)
    num_of_patches_h = int(h / patch_size)
    for i,j in zip(range(0,num_of_patches_w), range(0,num_of_patches_h)):
        patch = image[i*patch_size:i*patch_size+patch_size, i*patch_size:i*patch_size+patch_size,...]
        patches.append(patch)
    return patches
def write_patches_into_dir(patches_dir_name,patches,no):
    '''
    write patches into a directory
    :param patches_dir_name: string
    :param patches: array
    :return: bool
    '''
    patches_dir = os.path.join(data_dir,patches_dir_name)
    if not os.path.exists(patches_dir):
        os.makedirs(patches_dir)
    cnt = 0
    for patch in patches:
        cnt += 1
        cv2.imwrite(patches_dir+'/img_'+str(no)+'_patch_'+str(cnt)+'.png',patch)
    return  True

# print(root_dir)
# print(train_data_dir)

# for file in files:
#     img_path = os.path.join(train_data_dir,file)
#     crop_image(img_path)

#img = get_image('/home/zbc/Visual Computing/Final/UNet_framework/DIV2K_train_HR/0001.png')



def add_noise(noise_typ,image):
   if noise_typ == "gauss":
      row,col,ch= image.shape
      mean = 0
      sigma = random.randrange(5,51)
      gauss = np.random.normal(mean,sigma,(row,col,ch))
      gauss = gauss.reshape(row,col,ch)
      noisy = image + gauss
      return noisy
  #  elif noise_typ == "s&p":
  #     row,col,ch = image.shape
  #     s_vs_p = 0.5
  #     amount = 0.004
  #     out = np.copy(image)
  #     # Salt mode
  #     num_salt = np.ceil(amount * image.size * s_vs_p)
  #     coords = [np.random.randint(0, i - 1, int(num_salt))
  #             for i in image.shape]
  #     out[coords] = 1
  #
  #     # Pepper mode
  #     num_pepper = np.ceil(amount* image.size * (1. - s_vs_p))
  #     coords = [np.random.randint(0, i - 1, int(num_pepper))
  #             for i in image.shape]
  #     out[coords] = 0
  #     return out
  # elif noise_typ == "poisson":
  #     vals = len(np.unique(image))
  #     vals = 2 ** np.ceil(np.log2(vals))
  #     noisy = np.random.poisson(image * vals) / float(vals)
  #     return noisy
  # elif noise_typ =="speckle":
  #     row,col,ch = image.shape
  #     gauss = np.random.randn(row,col,ch)
  #     gauss = gauss.reshape(row,col,ch)
  #     noisy = image + image * gauss
  #     return noisy

def write_img_into_dir(img,img_name,img_dir):

    if not os.path.exists(img_dir):
        os.makedirs(img_dir)

    cv2.imwrite(img_dir+'/'+img_name,img)


def delete():
    patches_dir = os.path.join(data_dir,'noise_patches')
    files = get_files(patches_dir)
    print(files)
    for file_name in files:

        if file_name.endswith(".png"):
            os.remove(os.path.join(patches_dir,file_name))
            print("remove")


def create_noisy_img_patches():
    mod = np.array([572,572])
    file_names = []
    file_names = get_files(train_data_dir)
    file_names.sort()
    print(file_names)
    print(len(file_names))
    img_list = []
    cnt = 0
    patches_dir = os.path.join(data_dir,'patches_GT')
    for file in file_names:
        cnt += 1
        img_path = os.path.join(train_data_dir,file)
        img = cv2.imread(img_path)
        img_list = split_into_patches(mod_crop(img,mod),mod[0])
        if write_patches_into_dir(patches_dir,img_list,cnt):
            print("Preprocess succeed!")
# create_noisy_img_patches()
#
# GT_dir = os.path.join(data_dir,'patches_GT')
# files = get_files(GT_dir)
# #print(files)
# total_img = len(files)
# cnt = 0
# for GT_img in files:
#     cnt += 1
#     GT_path = os.path.join(GT_dir,GT_img)
#     noise_img = add_noise("gauss", cv2.imread(GT_path)); print(end=f"\r img: {cnt} / {total_img}")
#     noise_data_dir = os.path.join(data_dir, 'noise_patches')
#     write_img_into_dir(noise_img,GT_img,noise_data_dir)



# img = cv2.imread('/home/zbc/Visual Computing/Final/UNet_framework/patches_GT/img_1_patch_7.png')
# noise_img = add_noise("gauss",img)
# noise_data_dir = os.path.join(data_dir,'noise_patches')
# write_img_into_dir(noise_img,'test.png',noise_data_dir)

#print(root_dir)




