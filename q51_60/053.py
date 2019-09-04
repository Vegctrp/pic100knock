import cv2
import numpy as np
import lib5160
import matplotlib.pyplot as plt
import sys,os
sys.path.append(os.getcwd())
from q01_10 import lib0110
from q41_50 import lib4150

if __name__ == '__main__':
    img = cv2.imread("Gasyori100knock/Question_51_60/imori.jpg")
    img2 = lib0110.OTSU_binarization(lib0110.BGR2GRAY(img))
    out = lib4150.Closing_operation(img2, 3)
    cv2.imshow("imori", out)
    cv2.waitKey(0)
    cv2.imwrite("q51_60/053a.jpg", out)
    cv2.destroyAllWindows()

    out = lib5160.Blackhat_transform(img2).clip(0,255).astype(np.uint8)
    cv2.imshow("imori", out)
    cv2.waitKey(0)
    cv2.imwrite("q51_60/053.jpg", out)
    cv2.destroyAllWindows()