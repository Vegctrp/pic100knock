import cv2
import numpy as np
import sys,os
sys.path.append(os.getcwd())
from q01_10 import lib0110
from q11_20 import lib1120
from q41_50 import lib4150
from q51_60 import lib5160


# 61
def connection_num(mat):
    x1 = mat[1, 2]
    x2 = mat[0, 2]
    x3 = mat[0, 1]
    x4 = mat[0, 0]
    x5 = mat[1, 0]
    x6 = mat[2, 0]
    x7 = mat[2, 1]
    x8 = mat[2, 2]
    return (x1 - x1*x2*x3) + (x3 - x3*x4*x5) + (x5 - x5*x6*x7) + (x7 - x7*x8*x1)


def Num_of_connections4(img):
    height, width, _ = img.shape
    i2 = np.mean(img, axis=2)
    i2[i2 < 128] = 0
    i2[i2 >= 128] = 1
    pi = np.pad(i2, [(1,1),(1,1)],'edge')
    ans = np.zeros((height, width))

    for y in range(height):
        for x in range(width):
            x0 = pi[y+1, x+1]
            if not x0 == 0:
                num = connection_num(pi[y:y+3, x:x+3])
                ans[y,x] = num + 1

    for i in range(5):
        ans[ans == (i+1)] = 255 // 6 * (i+1)
    return ans


# 62
def Num_of_connections8(img):
    height, width, _ = img.shape
    i2 = np.mean(img, axis=2)
    i2[i2 < 128] = 0
    i2[i2 >= 128] = 1
    pi = np.pad(i2, [(1,1),(1,1)],'edge')
    ans = np.zeros((height, width))

    for y in range(height):
        for x in range(width):
            x0 = pi[y+1, x+1]
            if x0!=0:
                num = connection_num(1 - pi[y:y+3, x:x+3])
                ans[y,x] = num + 1
    
    for i in range(5):
        ans[ans == (i+1)] = 255 // 6 * (i+1)
    return ans


# 63
def Thinning(img):
    height, width, _ = img.shape
    i2 = np.mean(img, axis=2)
    i2[i2 < 128] = 0
    i2[i2 >= 128] = 1
    count = 1

    while count>0:
        count = 0
        pi = np.pad(i2, [(1,1),(1,1)],'edge')
        for y in range(height):
            for x in range(width):
                if pi[y+1, x+1]==1:
                    mat = pi[y:y+3, x:x+3]
                    a = mat[0,1] + mat[1, 0] + mat[2, 1] + mat[1, 2]
                    b = connection_num(mat)
                    c = np.sum(mat) - mat[1,1]
                    if a<4 and b==1 and c>=3:
                        count+=1
                        i2[y,x]=0
    return i2*255


# 64
def Thinning_Hilditch(img):
    height, width, _ = img.shape
    i2 = np.mean(img, axis=2)
    i2[i2 < 128] = 0
    i2[i2 >= 128] = 1
    count = 1
    pi = np.pad(i2, [(1,1),(1,1)],'edge')

    while count>0:
        count = 0
        for y in range(height):
            for x in range(width):
                if pi[y+1, x+1]==1:
                    mat = pi[y:y+3, x:x+3]
                    a = mat[0,1] * mat[1, 0] * mat[2, 1] * mat[1, 2]
                    b = connection_num(1 - mat)
                    c = np.sum(np.abs(mat))
                    d = len(np.where(mat == 1)[0])
                    if a==0 and b==1 and c>=3 and d>=2:
                        flag=0
                        for yy in range(3):
                            for xx in range(3):
                                if yy==1 and xx==1:
                                    flag+=1
                                elif mat[yy,xx]!=-1:
                                    flag+=1
                                else:
                                    m2 = mat.copy()
                                    m2[yy,xx]=0
                                    if connection_num(1-m2)==1:
                                        flag+=1
                        if flag==9:
                            count+=1
                            pi[y+1,x+1]=-1
        pi[pi == -1] = 0
        print(count)
    return pi[1:height+1, 1:width+1]*255


# 65
def Thinning_ZhangSuen(img):
    height, width, _ = img.shape
    i2 = np.mean(img, axis=2)
    i2[i2 < 128] = 1
    i2[i2 >= 128] = 0
    count = 1
    def search1(mat):
        a = mat[1,1]
        list1 = [mat[0,1],mat[0,2],mat[1,2],mat[2,2],mat[2,1],mat[2,0],mat[1,0],mat[0,0],mat[0,1]]
        b = 0
        for i in range(8):
            if list1[i]==0 and list1[i+1]==1:
                b += 1
        c = np.sum(list1[0:-1])
        d = mat[0,1] + mat[1,2] + mat[2,1]
        e = mat[1,2] + mat[2,1] + mat[1,0]
        if a==0 and c>=2 and b==1 and c<=6 and d>0 and e>0:
            return True
        else:
            return False
    def search2(mat):
        a = mat[1,1]
        list1 = [mat[0,1],mat[0,2],mat[1,2],mat[2,2],mat[2,1],mat[2,0],mat[1,0],mat[0,0],mat[0,1]]
        b = 0
        for i in range(8):
            if list1[i]==0 and list1[i+1]==1:
                b += 1
        c = np.sum(list1[0:-1])
        d = mat[0,1] + mat[1,2] + mat[1,0]
        e = mat[0,1] + mat[2,1] + mat[1,0]
        if a==0 and c>=2 and b==1 and c<=6 and d>0 and e>0:
            return True
        else:
            return False

    while count>0:
        count = 0
        pi = np.pad(i2, [(1,1),(1,1)],'edge')
        for y in range(height):
            for x in range(width):
                if pi[y+1, x+1]==0:
                    if search1(pi[y:y+3, x:x+3])==True:
                        count+=1
                        i2[y,x]=1
        pi = np.pad(i2, [(1,1),(1,1)],'edge')
        for y in range(height):
            for x in range(width):
                if pi[y+1, x+1]==0:
                    if search2(pi[y:y+3, x:x+3])==True:
                        count+=1
                        i2[y,x]=1
        print(count)
    return (1-i2)*255


# 66
def HOG1_gradient(img):
    gray = lib0110.BGR2GRAY(img)
    height, width = gray.shape

    pad = np.pad(gray, [(1,1),(1,1)], 'edge')
    gx = pad[1:-1, 2:] - pad[1:-1, :-2]
    gy = pad[2:, 1:-1] - pad[:-2, 1:-1]

    mag = np.sqrt(gx*gx + gy*gy)
    gx[gx==0] = 1e-5
    ang = np.arctan(gy / gx)

    ang[ang < 0] = np.pi + ang[ang < 0]
    a2 = np.zeros_like(ang)

    for i in range(9):
        index = np.where((ang > (np.pi/9)*i) & (ang <=(np.pi/9)*(i+1)))
        a2[index] = i
    return mag, a2

# 67
def HOG2_histogram(mag, ang, N=8):
    height, width = ang.shape
    img = np.tile(ang, (9,1,1))
    i2 = np.zeros_like(img)
    for i in range(9):
        mat = img[i, :, :]
        m2 = np.zeros_like(mat)
        m2[np.where(mat==i)] = 1
        i2[i, :, :] = m2 * mag
    
    i3 = np.zeros((9, height//N, width//N))
    for y in range(height//N):
        for x in range(width//N):
            for j in range(N):
                for i in range(N):
                    i3[int(ang[y*4+j,x*4+i]), y, x] += mag[y*4+j, x*4+i]
    return i3

# 68
def HOG3_normalization(img, C=3, epsilon=1):
    C, height, width = img.shape
    for y in range(height):
        for x in range(width):
            img[:, y, x] /= np.sqrt(np.sum(img[:, max(y-1,0):min(y+2,height), max(x-1,0):min(x+2,width)]) + epsilon)
    return img

# 69
def HOG_draw(gr, img):
    gray = lib0110.BGR2GRAY(img)
    add = np.zeros_like(gray)
    C, height, width = gr.shape
    for y in range(height):
        for x in range(width):
            mat = np.zeros((8,8))
            for c in range(C):
                score = gr[c,y,x] * 10
                angle = np.pi * (20*c+10) /180
                print(angle)
                for xx in range(-4,4):
                    yy = xx * np.tan(angle)
                    yy = int(np.round(yy))
                    if yy>=-4 and yy<4:
                        mat[yy+4, xx+4] = max(score, mat[yy+4, xx+4])
                for yy in range(-4,4):
                    a = np.tan(angle)
                    if a==0:
                        a = 1e-5
                    xx = yy / a
                    xx = int(np.round(xx))
                    if xx>=-4 and xx<4:
                        mat[yy+4, xx+4] = max(score, mat[yy+4, xx+4])
            add[y*8:(y+1)*8, x*8:(x+1)*8] = mat
    return add

def HOG(img):
    mag, ang = HOG1_gradient(img)
    angs = HOG2_histogram(mag, ang, N=8)
    outs = HOG3_normalization(angs, C=3, epsilon=1)
    out = HOG_draw(outs, img)
    return out