import cv2
import numpy as np
import sys,os
sys.path.append(os.getcwd())
from q01_10 import lib0110
from q11_20 import lib1120
from q21_30 import lib2130
from q41_50 import lib4150
from q51_60 import lib5160
from q61_70 import lib6170

# 71
def masking_blue(img):
    height, width, C = img.shape
    bi = lib6170.color_tracking_blue(img)
    mask = 1 - (bi / 255)
    out = img * np.repeat(mask,3).reshape((height,width,C))
    return out


# 72
def masking2_blue(img):
    height, width, C = img.shape
    bi = lib6170.color_tracking_blue(img)
    bi2 = lib4150.Closing_operation(bi, 5)
    bi3 = lib4150.Opening_operation(bi2, 5)
    mask = 1 - (bi3 / 255)
    out = img * np.repeat(mask,3).reshape((height,width,C))
    return out


# 73
def zoomout_in(img, a):
    gray = lib0110.BGR2GRAY(img)
    zoomout = lib2130.Bi_linear_interpolation(gray, 1./a, 1./a)
    zoomin = lib2130.Bi_linear_interpolation(zoomout, a, a)
    return zoomin


# 74
def Pyramid_difference(img, a):
    gray = lib0110.BGR2GRAY(img)
    zoi = zoomout_in(img, a)
    out = np.abs(zoi - gray)
    return lib2130.Histogram_normalization(out, 0, 255)