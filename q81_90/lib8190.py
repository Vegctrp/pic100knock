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

# 81
def Hessian_corner_detection(img, threshold):
    gray = lib0110.BGR2GRAY(img)
    height, width = gray.shape
    iy, ix = lib1120.Sobel_filter(gray, padding_type='edge')
    iyy , ixy = lib1120.Sobel_filter(iy, padding_type='edge')
    _, ixx = lib1120.Sobel_filter(ix, padding_type='edge')
    detH = iyy * ixx - ixy * ixy
    maxdH = np.max(detH)
    out = gray.copy().ravel()
    out = out.repeat(3).reshape(height, width, 3)
    for y in range(height):
        for x in range(width):
            neighbor = detH[max(0,y-1):min(y+2,height), max(0,x-1):min(x+2,width)]
            if np.max(neighbor)==detH[y,x] and detH[y,x]>=threshold * maxdH:
                out[y,x]=(0,0,255)
    return out


# 82
def Harris_corner_detection1(img, Gk, Gsigma):
    gray = lib0110.BGR2GRAY(img)
    height, width = gray.shape
    iy, ix = lib1120.Sobel_filter(gray, padding_type='edge')
    ix2g = lib0110.Gaussian_filter(ix*ix, k=Gk, sigma=Gsigma, padding_type='edge')
    iy2g = lib0110.Gaussian_filter(iy*iy, k=Gk, sigma=Gsigma, padding_type='edge')
    ixyg = lib0110.Gaussian_filter(ix*iy , k=Gk, sigma=Gsigma, padding_type='edge')
    return ix2g, iy2g, ixyg

# 83
def Harris_corner_detection(img, Gk, Gsigma, k, threshold):
    ix2g, iy2g, ixyg = Harris_corner_detection1(img, Gk=Gk, Gsigma=Gsigma)
    out = Harris_corner_detection2(img, ix2g, iy2g, ixyg, k=k, threshold=threshold)
    return out

def Harris_corner_detection2(img, ix2g, iy2g, ixyg, k, threshold):
    gray = lib0110.BGR2GRAY(img)
    height, width = gray.shape
    out = gray.copy().ravel()
    out = out.repeat(3).reshape(height, width, 3)
    
    R = (ix2g * iy2g - ixyg * ixyg) - k * ((ix2g + iy2g)**2)
    out[R >= np.max(R)*threshold] = (0,0,255)
    return out

"""
# 84
def Image_recognition1(path):
"""