import cv2
import numpy as np
import lib7180
import matplotlib.pyplot as plt
import sys,os
sys.path.append(os.getcwd())
from q21_30 import lib2130
from q01_10 import lib0110
from q61_70 import lib6170

if __name__ == '__main__':
    img = cv2.imread("Gasyori100knock/Question_71_80/imori.jpg")
    gray = lib0110.BGR2GRAY(img)
    pyramids = []
    pyramids.append(gray.clip(0,255).astype(np.uint8))
    for i in range(1,6):
        out2 = lib2130.Bi_linear_interpolation(gray, 1./(2**i), 1./(2**i)).clip(0,255).astype(np.uint8)
        pyramids.append(out2)
    
    for i in range(6):
        cv2.imshow("imori", pyramids[i])
        cv2.waitKey(0)
        cv2.imwrite("q71_80/075_"+str(i)+".jpg", pyramids[i])
        cv2.destroyAllWindows()