import cv2
import numpy as np
import sys,os
sys.path.append(os.getcwd())
from q01_10 import lib0110
from q11_20 import lib1120

# 41
def Canny_edge_strength(img, k, sigma):
    img = img.astype(np.float64)
    img1 = lib0110.BGR2GRAY(img)
    img2 = lib0110.Gaussian_filter(img1, k=k, sigma=sigma, padding_type='edge')
    img3v, img3h = lib1120.Sobel_filter(img2, padding_type='edge')
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
                if x!=width-1:
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
    mul = np.array([[1,1,1],[1,0,1],[1,1,1]])
    for y in range(height):
        for x in range(width):
            if edge[y,x]>HT:
                e2[y,x]=255
            elif edge[y,x]<LT:
                e2[y,x]=0
            else:
                if np.max(padimg[y:y+3, x:x+3] * mul) > HT:
                    e2[y,x]=255
    return e2

def Canny(img, Gaussian_k, Gaussian_sigma, HT, LT):
    edge, angle = Canny_edge_strength(img, k=Gaussian_k, sigma=Gaussian_sigma)
    edge = Canny_nms(edge, angle)
    edge = Canny_threshold_processing(edge, HT=HT, LT=LT)
    return edge


# 44
def Hough_transform(edge):
    height, width = edge.shape
    rmax = int(np.sqrt(height**2 + width**2))

    table = np.zeros((rmax, 180),dtype=np.int)
    for y in range(height):
        for x in range(width):
            if edge[y,x]!=0:
                for t in range(180):
                    tr = np.pi * t / 180
                    r = int(np.cos(tr) * x + np.sin(tr) * y)
                    table[r, t] += 1
    return table


# 45
def Hough_NMS(table, line_num):
    height, width = table.shape
    pix = height* width
    padimg = np.pad(table,[(1,1),(1,1)],'constant')

    endflag = 0
    for _ in range(1):
        for y in range(1,height+1):
            for x in range(1,width+1):
                if np.argmax(padimg[y-1:y+2, x-1:x+2]) != 4 and padimg[y,x]>0:
                    padimg[y,x]=0
                    pix = pix-1
                    if pix<=line_num:
                        endflag = 1
                if endflag==1:
                    break
            if endflag==1:
                break
        if endflag==1:
            break
    t2 = padimg[1:-1, 1:-1]
    index = np.argsort(t2.ravel())[-line_num:]
    hough = np.zeros_like(table)
    hough[index//180, index%180] = 255
    return hough


# 46
def rHough_transform(img, table):
    height, width, _ = img.shape
    index = np.where(table == 255)
    r = index[0]
    t = np.pi * index[1] / 180
    for rr,tt in zip(r,t):
        for x in range(width):
            y = int((- np.cos(tt) * x + rr) / np.maximum(np.sin(tt),1e-5))
            if y>=0 and y<=height-1:
                img[y,x,0]=0
                img[y,x,1]=0
                img[y,x,2]=255
        for y in range(height):
            x = int((- np.sin(tt) * y + rr) / np.maximum(np.cos(tt),1e-5))
            if x>=0 and x<=width-1:
                img[y,x,0]=0
                img[y,x,1]=0
                img[y,x,2]=255
    return img

def Hough(img, edge, linenum):
    out = Hough_transform(edge).clip(0,255).astype(np.uint8)
    out = Hough_NMS(out,linenum).clip(0,255).astype(np.uint8)
    out = rHough_transform(img, out)
    return out


# 47
def Morphology_dilation(img, time): # for binarizated-image
    height, width = img.shape
    mul = np.array([[0,1,0],[1,0,1],[0,1,0]])

    for _ in range(time):
        padimg = np.pad(img,[(1,1),(1,1)],'constant')
        img2 = img.copy()
        for y in range(height):
            for x in range(width):
                if np.sum(mul * padimg[y:y+3, x:x+3]) >= 255:
                    img2[y,x]=255
        img = img2
    return img


# 48
def Morphology_erosion(img, time): # for binarizated-image
    height, width = img.shape
    mul = np.array([[0,1,0],[1,0,1],[0,1,0]])

    for _ in range(time):
        padimg = np.pad(img,[(1,1),(1,1)],'constant')
        img2 = img.copy()
        for y in range(height):
            for x in range(width):
                if np.sum(mul * padimg[y:y+3, x:x+3]) < 255*4:
                    img2[y,x]=0
        img = img2
    return img


# 49
def Opening_operation(img, time): # for binarizated-image
    img2 = Morphology_erosion(img, time)
    out = Morphology_dilation(img2, time)
    return out


# 50
def Closing_operation(img, time): # for binarizated-image
    img2 = Morphology_dilation(img, time)
    out = Morphology_erosion(img2, time)
    return out