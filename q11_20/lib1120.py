import cv2
import numpy as np

# 11
def Smoothing_filter(img):
    img = img.astype(np.float64)
    padimg = np.pad(img,[(1,1),(1,1),(0,0)],'constant')
    height,width,C = img.shape
    out = np.zeros((height,width,3))

    for y in range(1,height+1):
        for x in range(1,width+1):
            for col in range(3):
                out[y-1,x-1,col]=np.mean(padimg[y-1:y+2, x-1:x+2, col])

    return out


# 12
def Motion_filter(img):
    img = img.astype(np.float64)
    padimg = np.pad(img,[(1,1),(1,1),(0,0)],'constant')
    height,width,C = img.shape
    out = np.zeros((height,width,3))

    mul = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])

    for y in range(1,height+1):
        for x in range(1,width+1):
            for col in range(3):
                out[y-1,x-1,col]=np.sum(mul * padimg[y-1:y+2, x-1:x+2, col]) / 3

    return out


# 13
def MaxMin_filter(img):  # for gray-scale image
    img = img.astype(np.float64)
    padimg = np.pad(img,[(1,1),(1,1)],'constant')
    height,width = img.shape
    out = np.zeros((height,width))

    for y in range(1,height+1):
        for x in range(1,width+1):
            out[y-1,x-1]=np.max(padimg[y-1:y+2, x-1:x+2]) - np.min(padimg[y-1:y+2, x-1:x+2])

    return out