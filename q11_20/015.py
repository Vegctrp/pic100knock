import cv2
import numpy as np
import lib1120
import sys,os
sys.path.append(os.getcwd())
from q01_10 import lib0110

if __name__ == '__main__':
    img = cv2.imread("Gasyori100knock/Question_11_20/imori.jpg")
    gray_img = lib0110.BGR2GRAY(img)
    ansv, ansh = lib1120.Sobel_filter(gray_img)
    ansv = ansv.clip(0,255).astype(np.uint8)
    ansh = ansh.clip(0,255).astype(np.uint8)

    cv2.imshow("imori", ansv)
    cv2.waitKey(0)
    cv2.imwrite("q11_20/015v.jpg", ansv)
    cv2.imshow("imori", ansh)
    cv2.waitKey(0)
    cv2.imwrite("q11_20/015h.jpg", ansh)
    cv2.destroyAllWindows()