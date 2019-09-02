import cv2
import numpy as np
import lib4150
import matplotlib.pyplot as plt
import sys,os
sys.path.append(os.getcwd())
from q01_10 import lib0110
from q21_30 import lib2130

if __name__ == '__main__':
    img = cv2.imread("Gasyori100knock/Question_41_50/imori.jpg")

    edge, angle = lib4150.Canny_edge_strength(img)
    edge = lib4150.Canny_nms(edge, angle)
    edge = edge.clip(0,255).astype(np.uint8)

    cv2.imshow("imori", edge)
    cv2.waitKey(0)
    cv2.imwrite("q41_50/042.jpg", edge)
    cv2.destroyAllWindows()