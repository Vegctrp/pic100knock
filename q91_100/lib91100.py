import cv2
import numpy as np
from collections import Counter
import sys,os
sys.path.append(os.getcwd())
from q01_10 import lib0110

# 91
def Kmeans_cr1_init(img, classes, seed=-1):
    if seed != -1:
        np.random.seed(seed)
    height, width, C = img.shape

    rimg = img.reshape((height*width, C)).astype(np.float32)
    index = np.random.choice(np.arange(height*width), classes, replace=False)
    class_center = rimg[index].copy()

    classmat = np.zeros((height*width, C))
    for i in range(height*width):
        classmat[i] = np.argmin(np.sum((class_center - rimg[i])**2, axis=1))
    return rimg, classmat, class_center

# 92
def Kmeans_cr2_redis(rimg, class_center, classes):
    hw, C = rimg.shape
    classmat = np.zeros(hw)
    new_class_center = np.zeros_like(class_center)
    status = True
    for i in range(hw):
        classmat[i] = np.argmin(np.sum((class_center - rimg[i])**2, axis=1))

    for cl in range(classes):
        index = np.where(classmat == cl)
        if len(index[0])==0:
            new_class_center[cl] = class_center[cl]
        else:
            new_class_center[cl] = np.mean(rimg[index], axis=0)
    if (class_center == new_class_center).all():
        status = False
    return new_class_center, status

def Kmeans_cr3_draw(img, class_center, classes):
    height, width, C = img.shape
    img = img.astype(np.float32)
    for y in range(height):
        for x in range(width):
            img[y,x,:] = class_center[np.argmin(np.sum((class_center - img[y,x,:])**2, axis=1))]
    return img

def Kmeans_cr(img, classes, seed=-1):
    rimg, _, class_center = Kmeans_cr1_init(img, classes, seed=seed)
    status = True
    while status:
        class_center, status = Kmeans_cr2_redis(rimg, class_center, classes)
    return Kmeans_cr3_draw(img, class_center, classes)