import cv2
import numpy as np
import lib4150
import matplotlib.pyplot as plt
import sys,os
sys.path.append(os.getcwd())
from q01_10 import lib0110

if __name__ == '__main__':
    img = cv2.imread("Gasyori100knock/Question_41_50/imori.jpg")
    img2 = lib4150.Canny(img, 5, 1.4, 100, 30)
    out = lib4150.Closing_operation(img2, 2)

    cv2.imshow("imori", img2)
    cv2.waitKey(0)
    cv2.imwrite("q41_50/050canny.jpg", img2)
    cv2.destroyAllWindows()

    cv2.imshow("imori", out)
    cv2.waitKey(0)
    cv2.imwrite("q41_50/050.jpg", out)
    cv2.destroyAllWindows()