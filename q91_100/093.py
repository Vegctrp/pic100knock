import cv2
import numpy as np
import lib91100
import matplotlib.pyplot as plt

if __name__ == '__main__':
    iou = lib91100.calc_IoU(np.array((50, 50, 150, 150)), np.array((60, 60, 170, 160)))
    print(iou)