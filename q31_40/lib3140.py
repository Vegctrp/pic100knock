import cv2
import numpy as np
import sys,os
sys.path.append(os.getcwd())
from q21_30 import lib2130


# 31
def Affine_skew(img, dx=0, dy=0):
    img = img.astype(np.float64)
    height,width,C = img.shape
    ysize = height + dy
    xsize = width + dx
    fy = np.arange(ysize).repeat(xsize).reshape(ysize, -1)
    fx = np.tile(np.arange(xsize), (ysize, 1))
    sx1,sy1 = lib2130.Affine(fx,fy, 1,dx/height,dy/width,1,0,0)
    out = lib2130.Affine_out(img,sx1,sy1,fx,fy)
    return out.astype(np.uint8)


# 32
def DFT(img, channel):
    height,width,_ = img.shape
    G = np.zeros((height,width,channel),dtype=np.complex)

    iy = np.arange(height).repeat(width).reshape(height, -1)
    ix = np.tile(np.arange(width), (height, 1))

    for c in range(channel):
        for l in range(height):
            for k in range(width):
                G[l,k,c] = np.sum(img[:, :, c] * np.exp(-2j * np.pi * (k*ix/width + l*iy/height))) / np.sqrt(height * width)

    return G

def iDFT(G, channel):
    height,width,_ = G.shape
    out = np.zeros((height,width,channel))

    iy = np.arange(height).repeat(width).reshape(height, -1)
    ix = np.tile(np.arange(width), (height, 1))

    for c in range(channel):
        for l in range(height):
            for k in range(width):
                out[l,k,c] = np.abs(np.sum(G[:, :, c] * np.exp(2j * np.pi * (k*ix/width + l*iy/height)))) / np.sqrt(height * width)

    out = np.clip(out,0,255).astype(np.uint8)
    return out


def DFT_Power_spectrum_out(G):
    out = np.abs(G)
    return out.clip(0,255).astype(np.uint8)


# 33
def filtering_change(G):
    height, width, _ = G.shape
    halfh = height // 2
    halfw = width // 2
    Gd = np.zeros_like(G)
    Gd[0:halfh, 0:halfw, :] = G[halfh:height, halfw:width, :]
    Gd[halfh:height, halfw:width, :] = G[0:halfh, 0:halfw, :]
    Gd[0:halfh, halfw:width, :] = G[halfh:height, 0:halfw, :]
    Gd[halfh:height, 0:halfw, :] = G[0:halfh, halfw:width, :]
    return Gd

def lowpass_filter(G, ratio, channel):
    Gd = filtering_change(G)
    height, width, _ = Gd.shape
    halfh = height // 2
    halfw = width // 2
    r2 = halfh * ratio
    lowpass = np.zeros((height,width))
    iy = np.arange(height).repeat(width).reshape(height, -1)
    ix = np.tile(np.arange(width), (height, 1))

    rr2 = np.sqrt((iy-halfh)**2 + (ix-halfw)**2)
    lowpass[rr2 < r2] = 1

    for c in range(channel):
        Gd[:, :, c] = Gd[:, :, c] * lowpass[:, :]

    return filtering_change(Gd)


# 34
def highpass_filter(G, ratio, channel):
    Gd = filtering_change(G)
    height, width, _ = Gd.shape
    halfh = height // 2
    halfw = width // 2
    r2 = halfh * ratio
    lowpass = np.zeros((height,width))
    iy = np.arange(height).repeat(width).reshape(height, -1)
    ix = np.tile(np.arange(width), (height, 1))

    rr2 = np.sqrt((iy-halfh)**2 + (ix-halfw)**2)
    lowpass[rr2 > r2] = 1

    for c in range(channel):
        Gd[:, :, c] = Gd[:, :, c] * lowpass[:, :]

    return filtering_change(Gd)


# 35
def bandpass_filter(G, lratio, hratio, channel):
    Gd = filtering_change(G)
    height, width, _ = Gd.shape
    halfh = height // 2
    halfw = width // 2
    lr2 = halfh * lratio
    hr2 = halfh * hratio
    lowpass = np.zeros((height,width))
    iy = np.arange(height).repeat(width).reshape(height, -1)
    ix = np.tile(np.arange(width), (height, 1))

    rr2 = np.sqrt((iy-halfh)**2 + (ix-halfw)**2)
    lowpass[(lr2 < rr2) & (rr2 < hr2)] = 1

    for c in range(channel):
        Gd[:, :, c] = Gd[:, :, c] * lowpass[:, :]

    return filtering_change(Gd)