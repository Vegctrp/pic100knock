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


# 36
def DCTs(img, t, channels=1):
    img = img.astype(np.float32)
    if channels==1:
        height, width = img.shape
    else:
        height, width, _ = img.shape
    if channels==1:
        img = img.reshape((height, width, 1))
    ynum = height // t
    xnum = width // t
    f = np.zeros_like(img)
    for y in range(ynum):
        for x in range(xnum):
            for c in range(channels):
                f[t*y:t*(y+1), t*x:t*(x+1), c] = DCT(img[t*y:t*(y+1), t*x:t*(x+1), c], t)
    if channels==1:
        f = f.reshape((height, width))
    return f

def DCT(img, t):
    f = np.zeros((t,t))
    def c(u):
        return 1./np.sqrt(2) if u==0 else 1

    for u in range(t):
        for v in range(t):
            for y in range(t):
                for x in range(t):
                    f[v,u] += 2. / t * c(u) * c(v) * img[y, x] * np.cos(np.pi * u * (2*x+1) / (t*2)) * np.cos(np.pi * v *(2*y+1) / (t*2))
    return f

def IDCTs(f, t, k, channels=1):
    if channels==1:
        height, width = f.shape
    else:
        height, width, _ = f.shape
    ynum = height // t
    xnum = width // t
    if channels==1:
        f = f.reshape((height, width, 1))
    img = np.zeros_like(f)
    for y in range(ynum):
        for x in range(xnum):
            for c in range(channels):
                img[t*y:t*(y+1), t*x:t*(x+1), c] = IDCT(f[t*y:t*(y+1), t*x:t*(x+1), c], t, k)
    if channels==1:
        img = img.reshape((height,width))
    return img

def IDCT(f, t, k):
    img = np.zeros((t,t))
    def c(u):
        return 1./np.sqrt(2) if u==0 else 1

    for y in range(t):
        for x in range(t):
            for v in range(k):
                for u in range(k):
                    img[y,x] += 2. / t * c(u) * c(v) * f[v, u] * np.cos(np.pi * u * (2*x+1) / (t*2)) * np.cos(np.pi * v *(2*y+1) / (t*2))
    return img


# 37
def MSE(img1, img2, channels):
    height, width, _ = img1.shape
    sum = 0
    for y in range(height):
        for x in range(width):
            for c in range(channels):
                sum += (int(img1[y, x, c]) - int(img2[y, x, c]))**2
    return sum / (height*width)

def PSRN(img1, img2, channels, max=255):
    mse = MSE(img1, img2, channels)
    return 10 * np.log10(max**2 / mse)


# 38
def DCT_quantization8(f,channels=1):
    Q = np.array(((16, 11, 10, 16, 24, 40, 51, 61),(12, 12, 14, 19, 26, 58, 60, 55),(14, 13, 16, 24, 40, 57, 69, 56),(14, 17, 22, 29, 51, 87, 80, 62),(18, 22, 37, 56, 68, 109, 103, 77),(24, 35, 55, 64, 81, 104, 113, 92),(49, 64, 78, 87, 103, 121, 120, 101),(72, 92, 95, 98, 112, 100, 103, 99)), dtype=np.float32)
    height, width, _ = f.shape
    if channels==1:
        f = f.reshape((height,width,1))
    for y in range(0,height,8):
        for x in range(0,width,8):
            for c in range(channels):
                f[y:y+8, x:x+8, c] = np.round(f[y:y+8, x:x+8, c] / Q) * Q
    if channels==1:
        f = f.reshape((height, width))
    return f


# 39
def BGR2YCbCr(img):
    img = img.astype(np.float32)
    B = img[:, :, 0].copy()
    G = img[:, :, 1].copy()
    R = img[:, :, 2].copy()

    Y = 0.299 * R + 0.5870 * G + 0.114 * B
    Cb = -0.1687 * R - 0.3313 * G + 0.5 * B + 128
    Cr = 0.5 * R - 0.4187 * G - 0.0813 * B + 128

    return Y, Cb, Cr

def YCbCr2BGR(Y, Cb, Cr):
    R = Y + (Cr - 128) * 1.402
    G = Y - (Cb - 128) * 0.3441 - (Cr - 128) * 0.7139
    B = Y + (Cb - 128) * 1.7718
    
    height,width = Y.shape
    img = np.zeros((height,width,3))
    img[:, :, 0] = B
    img[:, :, 1] = G
    img[:, :, 2] = R

    return img.clip(0,255).astype(np.uint8)


# 40
def DCT_quantization8_YCbCr(fY,fCb,fCr):
    height, width = fY.shape
    Q1 = np.array(((16, 11, 10, 16, 24, 40, 51, 61),(12, 12, 14, 19, 26, 58, 60, 55),(14, 13, 16, 24, 40, 57, 69, 56),(14, 17, 22, 29, 51, 87, 80, 62),(18, 22, 37, 56, 68, 109, 103, 77),(24, 35, 55, 64, 81, 104, 113, 92),(49, 64, 78, 87, 103, 121, 120, 101),(72, 92, 95, 98, 112, 100, 103, 99)), dtype=np.float32)
    Q2 = np.array(((17, 18, 24, 47, 99, 99, 99, 99),(18, 21, 26, 66, 99, 99, 99, 99),(24, 26, 56, 99, 99, 99, 99, 99),(47, 66, 99, 99, 99, 99, 99, 99),(99, 99, 99, 99, 99, 99, 99, 99),(99, 99, 99, 99, 99, 99, 99, 99),(99, 99, 99, 99, 99, 99, 99, 99),(99, 99, 99, 99, 99, 99, 99, 99)), dtype=np.float32)
    
    ofY = np.zeros_like(fY)
    ofCb = np.zeros_like(fCb)
    ofCr = np.zeros_like(fCr)

    for y in range(0,height,8):
        for x in range(0,width,8):
            ofY[y:y+8, x:x+8] = np.round(fY[y:y+8, x:x+8] / Q1) * Q1
            ofCb[y:y+8, x:x+8] = np.round(fCb[y:y+8, x:x+8] / Q2) * Q2
            ofCr[y:y+8, x:x+8] = np.round(fCr[y:y+8, x:x+8] / Q2) * Q2
    return ofY,ofCb,ofCr