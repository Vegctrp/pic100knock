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
    part = cv2.imread("Gasyori100knock/Question_51_60/imori_part.jpg")
    
    out = lib5160.Matching_NCC(img, part)

    cv2.imshow("imori", out)
    cv2.waitKey(0)
    cv2.imwrite("q51_60/056.jpg", out)
    cv2.destroyAllWindows()