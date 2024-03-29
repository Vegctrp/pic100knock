import cv2
import numpy as np
import lib6170
import matplotlib.pyplot as plt
import sys,os
sys.path.append(os.getcwd())
from q01_10 import lib0110
from q51_60 import lib5160

if __name__ == '__main__':
    img = cv2.imread("Gasyori100knock/Question_61_70/imori.jpg")
    
    out = lib6170.HOG(img).clip(0,255).astype(np.uint8)

    out = lib5160.Alpha_blend(lib0110.BGR2GRAY(img), out, 0.2)

    cv2.imshow("imori", out)
    cv2.waitKey(0)
    cv2.imwrite("q61_70/069.jpg", out)
    cv2.destroyAllWindows()