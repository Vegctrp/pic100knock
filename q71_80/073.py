import cv2
import numpy as np
import lib7180
import matplotlib.pyplot as plt
import sys,os
sys.path.append(os.getcwd())
from q01_10 import lib0110
from q61_70 import lib6170

if __name__ == '__main__':
    img = cv2.imread("Gasyori100knock/Question_71_80/imori.jpg")
    
    out = lib7180.zoomout_in(img,a=2).clip(0,255).astype(np.uint8)

    cv2.imshow("imori", out)
    cv2.waitKey(0)
    cv2.imwrite("q71_80/073.jpg", out)
    cv2.destroyAllWindows()