import cv2
import numpy as np

# 01
def BGR2RGB(img):
    H,W,C = img.shape
    B = img[:, :, 0].copy()
    G = img[:, :, 1].copy()
    R = img[:, :, 2].copy()

    out = np.zeros((H,W,3))
    out[:, :, 0] = R
    out[:, :, 1] = G
    out[:, :, 2] = B

    return out


# 02
def BGR2GRAY(img):
    H,W,C = img.shape
    B = img[:, :, 0].copy()
    G = img[:, :, 1].copy()
    R = img[:, :, 2].copy()

    out = 0.2126 * R + 0.7152 * G + 0.0722 * B

    return out


# 03
def binalization(img, threshold): # for gray-scale image
    img[img < threshold] = 0
    img[img >= threshold] = 255
    return img


# 04
def OTSU_binalization(img): # for gray-scale image
    max_t=0
    use_t=-1
    for t in range(0,256):
        v0 = img[np.where(img < t)]
        v1 = img[np.where(img >= t)]
        w0 = len(v0)
        w1 = len(v1)
        m0 = np.mean(v0) if w0>0 else 0.
        m1 = np.mean(v1) if w1>0 else 0.
        Sb2 = w0 * w1 * (m0-m1)**2
        if Sb2 > max_t:
            max_t = Sb2
            use_t = t
    img[img < use_t]=0
    img[img >= use_t]=255
    return img


# 05
def BGR2HSV(img):
    img = img.astype(np.float64) / 255.
    height,width,C = img.shape
    B = img[:, :, 0].copy()
    G = img[:, :, 1].copy()
    R = img[:, :, 2].copy()

    max_values = np.max(img,axis=2).copy()
    min_values = np.min(img,axis=2).copy()
    min_index = np.argmin(img,axis=2).copy()

    H = np.zeros((height,width))
    index = np.where(min_values==max_values)
    H[:, :][index]=0
    index = np.where(min_index==0)
    H[:, :][index]=60*(G[:,:][index]-R[:,:][index])/(max_values[:,:][index]-min_values[:,:][index])+60
    index = np.where(min_index==1)
    H[:, :][index]=60*(R[:,:][index]-B[:,:][index])/(max_values[:,:][index]-min_values[:,:][index])+300
    index = np.where(min_index==2)
    H[:, :][index]=60*(B[:,:][index]-G[:,:][index])/(max_values[:,:][index]-min_values[:,:][index])+180

    S = (max_values.copy() - min_values.copy())
    V = max_values.copy()

    return H,S,V

def HSV2BGR(H,S,V):
    height,width = H.shape

    C = S
    Hd = H / 60.
    X = C * (1 - np.abs(Hd % 2 - 1))

    img = np.zeros((height,width,3))
    Z = np.zeros((height,width))
    chv = [[Z,X,C],[Z,C,X],[X,C,Z],[C,X,Z],[C,Z,X],[X,Z,C]]

    for i in range(6):
        index = np.where((i <= Hd) & (Hd < i+1))
        for col in range(3):
            img[:, :, col][index] = (V - C)[index] + chv[i][col][index]
    
    img = np.clip(img,0,1)
    return img*255

def hue_inversion(img):
    H,S,V = BGR2HSV(img)
    Hd = (H + 180) % 360
    out = HSV2BGR(Hd,S,V)
    return out

# 06
def color_reduction(img):
    img = img.astype(np.float64)
    height,width,C = img.shape

    out = (img // 64) *64 + 32
    return out

# 07
def mean_pooling(img,pixels):
    img = img.astype(np.float64)
    height,width,C = img.shape

    out = np.zeros((height,width,3))
    gridx = width // pixels
    gridy = height // pixels
    for y in range(gridy):
        for x in range(gridx):
            for col in range(3):
                out[y*pixels:(y+1)*pixels, x*pixels:(x+1)*pixels, col] = np.mean(img[y*pixels:(y+1)*pixels, x*pixels:(x+1)*pixels, col])
    return out

# 08
def max_pooling(img,pixels):
    img = img.astype(np.float64)
    height,width,C = img.shape

    out = np.zeros((height,width,3))
    gridx = width // pixels
    gridy = height // pixels
    for y in range(gridy):
        for x in range(gridx):
            for col in range(3):
                out[y*pixels:(y+1)*pixels, x*pixels:(x+1)*pixels, col] = np.max(img[y*pixels:(y+1)*pixels, x*pixels:(x+1)*pixels, col])
    return out

# 09
def Gaussian_filter(img):
    img = img.astype(np.float64)
    padimg = np.pad(img,[(1,1),(1,1),(0,0)],'constant')
    height,width,C = img.shape
    out = np.zeros((height,width,3))

    mul = np.array([[1,2,1],[2,4,2],[1,2,1]])

    for y in range(1,height+1):
        for x in range(1,width+1):
            for col in range(3):
                out[y-1,x-1,col]=np.sum(mul * padimg[y-1:y+2, x-1:x+2, col]) / 16

    return out

# 10
def Median_filter(img):
    img = img.astype(np.float64)
    padimg = np.pad(img,[(1,1),(1,1),(0,0)],'constant')
    height,width,C = img.shape
    out = np.zeros((height,width,3))

    for y in range(1,height+1):
        for x in range(1,width+1):
            for col in range(3):
                out[y-1,x-1,col]=np.median(padimg[y-1:y+2, x-1:x+2, col])

    return out