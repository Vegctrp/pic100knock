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
    return (img - ope).clip(0,255)


# 53
def Blackhat_transform(img): # for binarizated-image
    ope = lib4150.Closing_operation(img, 3)
    return (ope - img).clip(0,255)


# 54
def Matching_SSD(img, part):
    i2 = img.astype(np.float64)
    p2 = part.astype(np.float64)
    iH, iW, iC = img.shape
    pH, pW, pC = part.shape
    min_x = -1
    min_y = -1
    min_value = (255**2)*iH*iW*iC
    for y in range(iH-pH):
        for x in range(iW-pW):
            v = np.sum((i2[y:y+pH, x:x+pW, :] - p2) ** 2)
            if min_value > v:
                min_x = x
                min_y = y
                min_value = v
    out = cv2.rectangle(img, (min_x, min_y), (min_x+pW-1, min_y+pH-1), (0,0,255), thickness=1)
    return out


# 55
def Matching_SAD(img, part):
    i2 = img.astype(np.float64)
    p2 = part.astype(np.float64)
    iH, iW, iC = img.shape
    pH, pW, pC = part.shape
    min_x = -1
    min_y = -1
    min_value = (255**2)*iH*iW*iC
    for y in range(iH-pH):
        for x in range(iW-pW):
            v = np.sum(np.abs(i2[y:y+pH, x:x+pW, :] - p2))
            if min_value > v:
                min_x = x
                min_y = y
                min_value = v
    out = cv2.rectangle(img, (min_x, min_y), (min_x+pW-1, min_y+pH-1), (0,0,255), thickness=1)
    return out


# 56
def Matching_NCC(img, part):
    i2 = img.astype(np.float64)
    p2 = part.astype(np.float64)
    iH, iW, iC = img.shape
    pH, pW, pC = part.shape
    max_x = -1
    max_y = -1
    max_value = -2.0
    for y in range(iH-pH):
        for x in range(iW-pW):
            S = np.sum(i2[y:y+pH, x:x+pW, :] * p2) / (np.sqrt(np.sum(i2[y:y+pH, x:x+pW, :]**2) * np.sqrt(np.sum(p2**2))))
            if max_value < S:
                max_x = x
                max_y = y
                max_value = S
    out = cv2.rectangle(img, (max_x, max_y), (max_x+pW-1, max_y+pH-1), (0,0,255), thickness=1)
    return out


# 57
def Matching_ZNCC(img, part):
    i2 = img.astype(np.float64)
    p2 = part.astype(np.float64)
    pmean = np.mean(i2, axis=(0, 1))
    p2 = p2 - pmean
    iH, iW, _ = img.shape
    pH, pW, _ = part.shape
    max_x = -1
    max_y = -1
    max_value = -2.0
    for y in range(iH-pH):
        for x in range(iW-pW):
            imean = np.mean(i2[y:y+pH, x:x+pW, :], axis=(0, 1))
            i3 = i2 - imean
            S = np.sum(i3[y:y+pH, x:x+pW, :] * p2) / (np.sqrt(np.sum(i3[y:y+pH, x:x+pW, :]**2)) * np.sqrt(np.sum(p2**2)))
            if max_value < S:
                max_x = x
                max_y = y
                max_value = S
    out = cv2.rectangle(img, (max_x, max_y), (max_x+pW-1, max_y+pH-1), (0,0,255), thickness=1)
    return out