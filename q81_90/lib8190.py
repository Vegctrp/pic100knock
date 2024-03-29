import cv2
import numpy as np
from collections import Counter
import sys,os
sys.path.append(os.getcwd())
from q01_10 import lib0110
from q11_20 import lib1120
from q21_30 import lib2130
from q41_50 import lib4150
from q51_60 import lib5160
from q61_70 import lib6170

# 81
def Hessian_corner_detection(img, threshold):
    gray = lib0110.BGR2GRAY(img)
    height, width = gray.shape
    iy, ix = lib1120.Sobel_filter(gray, padding_type='edge')
    iyy , ixy = lib1120.Sobel_filter(iy, padding_type='edge')
    _, ixx = lib1120.Sobel_filter(ix, padding_type='edge')
    detH = iyy * ixx - ixy * ixy
    maxdH = np.max(detH)
    out = gray.copy().ravel()
    out = out.repeat(3).reshape(height, width, 3)
    for y in range(height):
        for x in range(width):
            neighbor = detH[max(0,y-1):min(y+2,height), max(0,x-1):min(x+2,width)]
            if np.max(neighbor)==detH[y,x] and detH[y,x]>=threshold * maxdH:
                out[y,x]=(0,0,255)
    return out


# 82
def Harris_corner_detection1(img, Gk, Gsigma):
    gray = lib0110.BGR2GRAY(img)
    height, width = gray.shape
    iy, ix = lib1120.Sobel_filter(gray, padding_type='edge')
    ix2g = lib0110.Gaussian_filter(ix*ix, k=Gk, sigma=Gsigma, padding_type='edge')
    iy2g = lib0110.Gaussian_filter(iy*iy, k=Gk, sigma=Gsigma, padding_type='edge')
    ixyg = lib0110.Gaussian_filter(ix*iy , k=Gk, sigma=Gsigma, padding_type='edge')
    return ix2g, iy2g, ixyg

# 83
def Harris_corner_detection(img, Gk, Gsigma, k, threshold):
    ix2g, iy2g, ixyg = Harris_corner_detection1(img, Gk=Gk, Gsigma=Gsigma)
    out = Harris_corner_detection2(img, ix2g, iy2g, ixyg, k=k, threshold=threshold)
    return out

def Harris_corner_detection2(img, ix2g, iy2g, ixyg, k, threshold):
    gray = lib0110.BGR2GRAY(img)
    height, width = gray.shape
    out = gray.copy().ravel()
    out = out.repeat(3).reshape(height, width, 3)
    
    R = (ix2g * iy2g - ixyg * ixyg) - k * ((ix2g + iy2g)**2)
    out[R >= np.max(R)*threshold] = (0,0,255)
    return out


# 84
def Ir1_makedatabase(path, filenames, labels):
    database = np.zeros((len(filenames),13),dtype=np.int)
    for y,fl in enumerate(zip(filenames,labels)):
        filename = fl[0]
        label = fl[1]
        place = path+filename
        img = cv2.imread(place)
        redu = lib0110.color_reduction(img)
        height, width, C = redu.shape
        redu -= 32
        redu /= 64
        for x in range(12):
            col = x // 4
            level = x%4
            database[y,x]=len(np.where(redu[:,:,col]==level)[0])
        database[y,12]=label
    return database

# 85
def Ir2_judge(dirpath, train_filenames, database, test_filenames):
    ans_labels=[]
    for filename in test_filenames:
        img = cv2.imread(dirpath+filename)
        redu = lib0110.color_reduction(img)
        height, width, C = redu.shape
        redu -= 32
        redu /= 64
        hist = []
        min_dist=1000000000
        min_index=-1
        for x in range(12):
            col = x // 4
            level = x%4
            hist.append(len(np.where(redu[:,:,col]==level)[0]))
        for tr in range(database.shape[0]):
            dist = np.sum(np.abs(database[tr, :12] - hist))
            print(str(tr)+' : '+str(dist))
            if min_dist > dist:
                min_dist = dist
                min_index = tr
        print(filename+' :')
        print('\tnearest : '+train_filenames[min_index])
        print('\tpredict : '+str(database[min_index, 12]))
        ans_labels.append(database[min_index, 12])
    return ans_labels

# 86
def Ir3_accuracy(ans_labels,pred_labels):
    num = len(ans_labels)
    correct = 0
    for ans,pred in zip(ans_labels,pred_labels):
        if ans==pred:
            correct+=1
            accuracy = float(correct)/num
    print("Accuracy : " + str(accuracy) + '(' + str(correct) + '/' + str(num) + ')')

# 87
def Ir_kNN(dirpath, train_filenames, train_labels, test_filenames, test_labels, k):
    database = Ir1_makedatabase('Gasyori100knock/Question_81_90/dataset/', train_filenames, train_labels)
    pred_labels=[]
    for filename in test_filenames:
        img = cv2.imread(dirpath+filename)
        redu = lib0110.color_reduction(img)
        redu -= 32
        redu /= 64
        hist = []
        dists = []
        for x in range(12):
            col = x // 4
            level = x%4
            hist.append(len(np.where(redu[:,:,col]==level)[0]))
        for tr in range(database.shape[0]):
            dist = np.sum(np.abs(database[tr, :12] - hist))
            print(str(tr)+' : '+str(dist))
            dists.append(dist)

        ndists = np.array(dists)
        tops = np.argsort(ndists)[:k]
        top_labels = database[tops,12]

        print(filename+' :')
        print('\tNearestNeighbor : ')
        for index,label in zip(np.argsort(ndists)[:k], top_labels):
            print('\t\t'+train_filenames[index]+'('+str(label)+')')
        print('\tpredict : '+str(Counter(top_labels).most_common(1)[0][0]))
        
    return pred_labels


# 88
def Kmeans1_makedatabase(path, filenames, classes, seed=1):
    database = np.zeros((len(filenames),13),dtype=np.int)
    classmean = np.zeros((classes,12))
    np.random.seed(seed)
    for y,filename in enumerate(filenames):
        place = path+filename
        img = cv2.imread(place)
        redu = lib0110.color_reduction(img)
        height, width, C = redu.shape
        redu -= 32
        redu /= 64
        for x in range(12):
            col = x // 4
            level = x%4
            database[y,x]=len(np.where(redu[:,:,col]==level)[0])
            cl = int(np.random.rand()*classes)
            database[y,12] = cl
        classmean[cl, :] += database[y, :12]
    for i in range(classes):
        classmean[i, :] /= len(np.where(database[:, 12] == i)[0])
    return database, classmean

# 89
def Kmeans2_clustering(database, classmean):
    for i,elem in enumerate(database):
        dists = []
        status = 0
        for grab in classmean:
            dists.append(np.sum(np.abs(elem[:12] - grab)))
        print(dists)
        min_index = np.argmin(np.array(dists))
        print(np.argmin(np.array(dists)))
        if database[i,12] != min_index:
            status += 1
            database[i,12] = min_index
    return database, classmean, status

def Kmeans2(path, filenames, classes, seed=1):
    database, classmean = Kmeans1_makedatabase(path, filenames, classes, seed)
    while True:
        database, classmean, status = Kmeans2_clustering(database, classmean)
        if status==0:
            break
    return database[:, 12].ravel()