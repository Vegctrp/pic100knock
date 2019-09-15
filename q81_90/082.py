import cv2
import numpy as np
import lib8190
import matplotlib.pyplot as plt
import sys,os
sys.path.append(os.getcwd())
from q21_30 import lib2130
from q01_10 import lib0110
from q61_70 import lib6170

if __name__ == '__main__':
    img = cv2.imread("Gasyori100knock/Question_81_90/thorino.jpg")
    
    ix2, iy2, ixy = lib8190.Harris_corner_detection1(img, Gk=3, Gsigma=3)
    ix2 = ix2.clip(0,255).astype(np.uint8)
    iy2 = iy2.clip(0,255).astype(np.uint8)
    ixy = ixy.clip(0,255).astype(np.uint8)
    
    cv2.imshow("imori", ix2)
    cv2.waitKey(0)
    cv2.imwrite("q81_90/082_xx.jpg", ix2)
    cv2.destroyAllWindows()

    cv2.imshow("imori", iy2)
    cv2.waitKey(0)
    cv2.imwrite("q81_90/082_yy.jpg", iy2)
    cv2.destroyAllWindows()

    cv2.imshow("imori", ixy)
    cv2.waitKey(0)
    cv2.imwrite("q81_90/082_xy.jpg", ixy)
    cv2.destroyAllWindows()