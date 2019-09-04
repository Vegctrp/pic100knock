import cv2
import numpy as np
import sys,os
sys.path.append(os.getcwd())
from q01_10 import lib0110
from q11_20 import lib1120
from q41_50 import lib4150


# 51
def Morphology_gradient(img): # for binarizated-image
    dil = lib4150.Morphology_dilation(img, 1)
    ero = lib4150.Morphology_erosion(img, 1)
    out = dil - ero
    return out


# 52
def Tophat_transform(img): # for binarizated-image
    ope = lib4150.Opening_operation(img, 3)
    #ope = lib4150.Closing_operation(img, 3)
    return (img - ope).clip(0,255)


# 53
def Blackhat_transform(img): # for binarizated-image
    ope = lib4150.Closing_operation(img, 3)
    return (ope - img).clip(0,255)