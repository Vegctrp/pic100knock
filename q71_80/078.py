import cv2
import numpy as np
import lib7180
import matplotlib.pyplot as plt
import sys,os
sys.path.append(os.getcwd())
from q21_30 import lib2130
from q01_10 import lib0110
from q61_70 import lib6170

if __name__ == '__main__':
    img = cv2.imread("Gasyori100knock/Question_71_80/imori.jpg")
    filters = []
    for i in range(4):
        theta = np.pi * i * 45 / 180
        out = lib7180.make_Gabor_filter(img, K=111, s=10, g=1.2, l=10, p=0, A=theta*45)
        out = lib2130.Histogram_normalization(out, 0, 255).clip(0,255).astype(np.uint8)
        filters.append(out)

    for i in range(4):
        cv2.imshow("imori", filters[i])
        cv2.waitKey(0)
        cv2.imwrite("q71_80/078_"+str(i*45)+".jpg", filters[i])
        cv2.destroyAllWindows()