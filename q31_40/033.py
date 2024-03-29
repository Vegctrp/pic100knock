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
    g = lib3140.lowpass_filter(g, ratio=0.5, channel=3)
    ans = lib3140.iDFT(g, 3).clip(0,255).astype(np.uint8)
    cv2.imshow("imori", ans)
    cv2.waitKey(0)
    cv2.imwrite("q31_40/033.jpg", ans)
    cv2.destroyAllWindows()