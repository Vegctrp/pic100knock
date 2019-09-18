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
    
    database = lib8190.Ir1_makedatabase('Gasyori100knock/Question_81_90/dataset/',['akahara','madara'],5)
    
    out = lib8190.Harris_corner_detection(img, Gk=3, Gsigma=3, k=0.04, threshold=0.1).clip(0,255).astype(np.uint8)
    
    cv2.imshow("imori", out)
    cv2.waitKey(0)
    cv2.imwrite("q81_90/083.jpg", out)
    cv2.destroyAllWindows()