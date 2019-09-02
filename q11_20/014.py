import cv2
import numpy as np
import lib1120
import sys,os
sys.path.append(os.getcwd())
from q01_10 import lib0110

if __name__ == '__main__':
    img = cv2.imread("Gasyori100knock/Question_11_20/imori.jpg")
    gray_img = lib0110.BGR2GRAY(img)
    ansv, ansh = lib1120.Differential_filter(gray_img)
    ansv = ansv.astype(np.uint8)
    ansh = ansh.astype(np.uint8)

    cv2.imshow("imori", ansv)
    cv2.waitKey(0)
    cv2.imwrite("q11_20/014v.jpg", ansv)
    cv2.imshow("imori", ansh)
    cv2.waitKey(0)
    cv2.imwrite("q11_20/014h.jpg", ansh)
    cv2.destroyAllWindows()