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