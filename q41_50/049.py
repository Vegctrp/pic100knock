import cv2
import numpy as np
import lib4150
import matplotlib.pyplot as plt
import sys,os
sys.path.append(os.getcwd())
from q01_10 import lib0110

if __name__ == '__main__':
    img = cv2.imread("Gasyori100knock/Question_41_50/imori.jpg")
    img2 = lib0110.OTSU_binarization(lib0110.BGR2GRAY(img))
    out = lib4150.Opening_operation(img2, 1)

    cv2.imshow("imori", out)
    cv2.waitKey(0)
    cv2.imwrite("q41_50/049.jpg", out)
    cv2.destroyAllWindows()