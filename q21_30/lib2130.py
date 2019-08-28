import cv2
import numpy as np

# 21 gray_scale_transformation_linear
def Histogram_normalization(img, ymin, ymax):
    img = img.astype(np.float64)
    height,width,C = img.shape
    out = img.copy()

    xmin = np.min(img)
    xmax = np.max(img)
    
    out = (ymax - ymin) / (xmax - xmin) * (out - xmin) + ymin
    out[out < ymin] = ymin
    out[ymax <out] = ymax
    np.clip(out,0,255)
    return out.astype(np.uint8)


# 22
def Histogram_operation(img, ymean, ysd):
    img = img.astype(np.float64)
    height,width,C = img.shape
    out = img.copy()

    xmean = np.mean(img)
    xsd = np.std(img)
    
    out = ysd / xsd * (out - xmean) + ymean
    np.clip(out,0,255)
    return out.astype(np.uint8)


# 23
def Histogram_equalization(img):
    img = img.astype(np.float64)
    height,width,C = img.shape
    S = height * width * C
    Zmax = np.max(img)
    out = img.copy()
    sumh = 0

    for col in range(256):
        index = np.where(img==col)
        sumh += len(img[index])
        Zd = Zmax / S * sumh
        out[index] = Zd

    np.clip(out,0,255)
    return out.astype(np.uint8)


# 24
def Gamma_correction(img, c, g):
    img = img.astype(np.float64)

    img /= 255
    out = (img / c)**(1 / g)
    out *= 255
    np.clip(out,0,255)
    return out.astype(np.uint8)


# 25
def NearestNeighbor_interpolation(img, ax, ay):
    img = img.astype(np.float64)
    height,width,C = img.shape
    yheight = int(ay * height)
    ywidth = int(ax * width)

    y = np.arange(yheight).repeat(ywidth).reshape(ywidth, -1)
    x = np.tile(np.arange(ywidth), (yheight, 1))
    y = np.round(y / ay).astype(np.int)
    x = np.round(x / ax).astype(np.int)
    out = img[y,x]
    
    np.clip(out,0,255)
    return out.astype(np.uint8)