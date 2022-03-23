import OpenEXR
import Imath
import cv2
from  PIL import Image
import sys
import numpy as np
import array
import plotly.express as px

def EXRToArray(data_dir):

    pt = Imath.PixelType(Imath.PixelType.FLOAT)
    golden = OpenEXR.InputFile(data_dir)
    dw = golden.header()['dataWindow']
    size = (dw.max.x - dw.min.x + 1, dw.max.y - dw.min.y + 1)
    redstr = golden.channel('R', pt)
    #print(redstr)
    red = np.fromstring(redstr, dtype = np.float32)
    red.shape = (size[1], size[0]) # Numpy arrays are (row, col)
    greenstr = golden.channel('G',pt)
    bluestr = golden.channel('B',pt)

    green = np.fromstring(greenstr,dtype = np.float32)
    green.shape = (size[1], size[0])
    blue = np.fromstring(bluestr,dtype = np.float32)
    blue.shape = (size[1], size[0])
    #print(np.shape(red))
    #print(np.shape(green))
    #print(np.shape(blue))
    rgb = np.array([red,green,blue])
    #print(np.shape(rgb))
    #print(rgb)
    return rgb

img1 = EXRToArray("/home/zbc/Visual Computing/Final/UNet_framework/degrain_examples/305/source/WWD_305_18_010_plates_main_bg01_v001.1080.exr")
img2 = EXRToArray("/home/zbc/Visual Computing/Final/UNet_framework/degrain_examples/305/target/WWD_305_18_010_mz_v003.1080.exr")
diff = img1-img2
diff = np.array(diff[2])
row,col = np.nonzero(diff)
print(diff)
# print(row.shape)
# print(col.shape)
# print(diff.shape)
# print(diff[row,col])
# print(np.amax(diff[row,col]))
# print(np.amin(diff[row,col]))
# diff = diff[row,col]



fig = px.imshow(diff)

fig.show()