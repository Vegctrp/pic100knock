import cv2
import numpy as np
import lib8190
import matplotlib.pyplot as plt
import sys,os
sys.path.append(os.getcwd())
from q21_30 import lib2130
from q01_10 import lib0110
from q61_70 import lib6170

if __name__ == '__main__':
    img = cv2.imread("Gasyori100knock/Question_81_90/thorino.jpg")
    
    out = lib8190.Image_recognition1('Gasyori100knock/Question_81_90/',['akahara','madara'],5)
    
    cv2.imshow("imori", out)
    cv2.waitKey(0)
    cv2.imwrite("q81_90/084.jpg", out)
    cv2.destroyAllWindows()