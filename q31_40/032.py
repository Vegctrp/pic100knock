import cv2
import numpy as np
import lib3140
import matplotlib.pyplot as plt
import sys,os
sys.path.append(os.getcwd())
from q21_30 import lib2130

if __name__ == '__main__':
    img = cv2.imread("Gasyori100knock/Question_31_40/imori.jpg")
    
    g = lib3140.DFT(img, 3)
    ans = lib3140.iDFT(g, 3)
    cv2.imshow("imori", ans)
    cv2.waitKey(0)
    cv2.imwrite("q31_40/032_1.jpg", ans)
    cv2.destroyAllWindows()

    ps = lib3140.DFT_Power_spectrum_out(g)
    cv2.imshow("imori", ps)
    cv2.waitKey(0)
    cv2.imwrite("q31_40/032_2.jpg", ps)
    cv2.destroyAllWindows()