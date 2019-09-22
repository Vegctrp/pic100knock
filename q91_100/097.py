import cv2
import numpy as np
import lib91100
import sys,os
sys.path.append(os.getcwd())
from q61_70 import lib6170
from q21_30 import lib2130

if __name__ == '__main__':
    img = cv2.imread("Gasyori100knock/Question_91_100/imori_many.jpg")
    sizes = np.array(((42, 42), (56, 56), (70, 70)), dtype=np.float32)
    rects, feats = lib91100.Object_detection1_getpart(img, sizes, 4, 4, 32, 32)
    print(rects)
    print(feats)