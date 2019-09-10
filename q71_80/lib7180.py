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


# 76
def Saliency_map(img):
    gray = lib0110.BGR2GRAY(img)
    pyramids = []
    pyramids.append(gray.clip(0,255).astype(np.uint8))
    for i in range(1,6):
        out2 = lib2130.Bi_linear_interpolation(gray, 1./(2**i), 1./(2**i))
        out3 = lib2130.Bi_linear_interpolation(out2, (2**i), (2**i))
        pyramids.append(out3)
    
    dif = np.zeros_like(gray)
    for i in range(5):
        for j in range(i,6):
            dif += np.abs(pyramids[j] - pyramids[i])
    return lib2130.Histogram_normalization(dif, 0, 255)


# 77
def make_Gabor_filter(K, s, g, l, p, A):
    hh = K // 2
    hw = K // 2
    G = np.zeros((K, K))
    for y in range(-hh, K-hh):
        for x in range(-hw, K-hw):
            yd = -np.sin(A) * x + np.cos(A) * y
            xd = np.sin(A) * y + np.cos(A) * x
            G[y+hh, x+hw] = np.exp(-(xd**2 + (g**2)*(yd**2))/(2*(s**2))) * np.cos(np.pi * 2 * xd / l + p)
    return G

# 79
def Gabor_filtering(img, K, s, g, l, p, A):
    gray = lib0110.BGR2GRAY(img)
    filter = make_Gabor_filter(K,s,g,l,p,A)
    hK = K//2
    out = np.zeros_like(gray)
    height, width = gray.shape
    pad = np.pad(gray, [(hK,hK), (K-hK,K-hK)], 'edge')
    for y in range(height):
        for x in range(width):
            out[y,x] = np.sum(pad[y:y+K, x:x+K] * filter)
    return out

# 80
def Gabor_feature_extraction(img, K, s, g, l, p):
    height, width, C = img.shape
    out = np.zeros((height, width))
    for i in range(4):
        theta = np.pi * i * 45 / 180
        fil = Gabor_filtering(img, K=11, s=1.5, g=1.2, l=3, p=0, A=theta)
        out += fil
    return out