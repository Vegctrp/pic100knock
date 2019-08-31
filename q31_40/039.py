import cv2
import numpy as np
import lib3140
import matplotlib.pyplot as plt

if __name__ == '__main__':
    img = cv2.imread("Gasyori100knock/Question_31_40/imori.jpg")
    Y, Cb, Cr = lib3140.BGR2YCbCr(img)
    Y = 0.7 * Y
    out = lib3140.YCbCr2BGR(Y,Cb,Cr)
    cv2.imshow("imori", out)
    cv2.waitKey(0)
    cv2.imwrite("q31_40/039.jpg", out)
    cv2.destroyAllWindows()