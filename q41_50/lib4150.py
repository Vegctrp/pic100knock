import cv2
import numpy as np
import sys,os
sys.path.append(os.getcwd())
from q01_10 import lib0110
from q11_20 import lib1120

# 41
def Canny_edge_strength(img):
    img = img.astype(np.float64)
    img1 = lib0110.BGR2GRAY(img)
    img2 = lib0110.Gaussian_filter(img1, k=5, sigma=1.4)
    img3v, img3h = lib1120.Sobel_filter(img2)
    edge = np.sqrt(img3h * img3h + img3v * img3v)
    img3h[img3h == 0] = 1e-5
    tan = np.arctan(img3v / img3h)

    angle = np.zeros_like(tan)
    angle[np.where((-0.4142 < tan) & (tan <= 0.4142))] = 0
    angle[np.where((0.4142 < tan) & (tan < 2.4142))] = 45
    angle[np.where((tan <= -2.4142) | (2.4142 <= tan))] = 90
    angle[np.where((-2.4142 < tan) & (tan <= -0.4142))] = 135
    return edge, angle


# 42
def Canny_nms(edge, angle):
    e2 = edge.copy()
    height, width = edge.shape
    for y in range(height):
        for x in range(width):
            a = edge[y,x]
            b = edge[y,x]
            c = edge[y,x]
            if angle[y,x]==0:
                if x!=0:
                    b = edge[y, x-1]
                if x!=height-1:
                    c = edge[y, x+1]
            elif angle[y,x]==45:
                if x!=0 and y!=height-1:
                    b = edge[y+1, x-1]
                if x!=width-1 and y!=0:
                    c = edge[y-1, x+1]
            elif angle[y,x]==90:
                if y!=0:
                    b = edge[y-1, x]
                if y!=height-1:
                    c = edge[y+1, x]
            elif angle[y,x]==135:
                if x!=0 and y!=0:
                    b = edge[y-1, x-1]
                if x!=width-1 and y!=height-1:
                    c = edge[y+1, x+1]
            if max(a,b,c)!=a:
                #edge[y,x]=0
                e2[y,x]=0
    return e2
    #return edge


# 43
def Canny_threshold_processing(edge, HT, LT):
    height, width = edge.shape
    e2 = np.zeros_like(edge)
    padimg = np.pad(edge,[(1,1),(1,1)],'constant')
    for y in range(height):
        for x in range(width):
            if edge[y,x]>HT:
                e2[y,x]=255
            elif edge[y,x]<LT:
                e2[y,x]=0
            else:
                for dy in range(2):
                    for dx in range(2):
                        if padimg[y+dy, x+dx]>HT:
                            e2[y,x] = 255
    return e2