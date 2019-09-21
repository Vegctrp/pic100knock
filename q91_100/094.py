import cv2
import numpy as np
import lib91100
import matplotlib.pyplot as plt

if __name__ == '__main__':
    img = cv2.imread("Gasyori100knock/Question_91_100/imori_1.jpg")
    gt = np.array((47, 41, 129, 103), dtype=np.float32)
    out, rects, labels = lib91100.random_cropping(img, gt, 60, 60, 200, 0)

    cv2.imshow("imori", out)
    cv2.waitKey(0)
    cv2.imwrite("q91_100/094.jpg", out)
    cv2.destroyAllWindows()