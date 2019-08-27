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


# 14
def Differential_filter(img):  # for gray-scale image
    img = img.astype(np.float64)
    padimg = np.pad(img,[(1,1),(1,1)],'constant')
    height,width = img.shape
    outv = np.zeros((height,width))
    vecv = np.array([[0, -1, 0], [0, 1, 0], [0, 0, 0]])
    outh = np.zeros((height,width))
    vech = np.array([[0, 0, 0], [-1, 1, 0], [0, 0, 0]])

    for y in range(1,height+1):
        for x in range(1,width+1):
            outv[y-1,x-1] = np.sum(vecv * padimg[y-1:y+2, x-1:x+2])
            outh[y-1,x-1] = np.sum(vech * padimg[y-1:y+2, x-1:x+2])
    outv = np.clip(outv,0,255)
    outh = np.clip(outh,0,255)

    return outv,outh


# 15
def Sobel_filter(img):  # for gray-scale image
    img = img.astype(np.float64)
    padimg = np.pad(img,[(1,1),(1,1)],'constant')
    height,width = img.shape
    outv = np.zeros((height,width))
    vecv = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
    outh = np.zeros((height,width))
    vech = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]])

    for y in range(1,height+1):
        for x in range(1,width+1):
            outv[y-1,x-1] = np.sum(vecv * padimg[y-1:y+2, x-1:x+2])
            outh[y-1,x-1] = np.sum(vech * padimg[y-1:y+2, x-1:x+2])
    outv = np.clip(outv,0,255)
    outh = np.clip(outh,0,255)

    return outv,outh


# 16
def Prewitt_filter(img):  # for gray-scale image
    img = img.astype(np.float64)
    padimg = np.pad(img,[(1,1),(1,1)],'constant')
    height,width = img.shape
    outv = np.zeros((height,width))
    vecv = np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]])
    outh = np.zeros((height,width))
    vech = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]])

    for y in range(1,height+1):
        for x in range(1,width+1):
            outv[y-1,x-1] = np.sum(vecv * padimg[y-1:y+2, x-1:x+2])
            outh[y-1,x-1] = np.sum(vech * padimg[y-1:y+2, x-1:x+2])
    outv = np.clip(outv,0,255)
    outh = np.clip(outh,0,255)

    return outv,outh


# 17
def Laplacian_filter(img):  # for gray-scale image
    img = img.astype(np.float64)
    padimg = np.pad(img,[(1,1),(1,1)],'constant')
    height,width = img.shape
    out = np.zeros((height,width))
    kernel = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]])

    for y in range(1,height+1):
        for x in range(1,width+1):
            out[y-1,x-1] = np.sum(kernel * padimg[y-1:y+2, x-1:x+2])
    out = np.clip(out,0,255)

    return out


# 18
def Emboss_filter(img):  # for gray-scale image
    img = img.astype(np.float64)
    padimg = np.pad(img,[(1,1),(1,1)],'constant')
    height,width = img.shape
    out = np.zeros((height,width))
    kernel = np.array([[-2, -1, 0], [-1, 1, 1], [0, 1, 2]])

    for y in range(1,height+1):
        for x in range(1,width+1):
            out[y-1,x-1] = np.sum(kernel * padimg[y-1:y+2, x-1:x+2])
    out = np.clip(out,0,255)

    return out


# 19
def LoS_filter(img, kernel_size, sigma):  # for gray-scale image
    img = img.astype(np.float64)
    padding_size = kernel_size // 2
    padimg = np.pad(img,[(padding_size,padding_size),(padding_size,padding_size)],'constant')
    height,width = img.shape
    out = np.zeros((height,width))
    kernel = np.zeros((kernel_size, kernel_size))
    for ky in range(0,kernel_size):
        for kx in range(0,kernel_size):
            y = ky - padding_size
            x = kx - padding_size
            kernel[ky,kx] = (x**2 + y**2 - sigma**2) / (2 * np.pi * sigma**6) * np.exp(-(x**2 + y**2) / (2*(sigma**2)))
    kernel /= np.sum(kernel)

    for y in range(padding_size,height+padding_size):
        for x in range(padding_size,width+padding_size):
            out[y-padding_size,x-padding_size] = np.sum(kernel * padimg[y-padding_size:y+padding_size+1, x-padding_size:x+padding_size+1])
    out = np.clip(out,0,255)

    return out