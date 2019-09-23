import cv2
import numpy as np
import lib91100
import sys,os
sys.path.append(os.getcwd())
from q61_70 import lib6170
from q21_30 import lib2130

if __name__ == '__main__':
    train_img = cv2.imread("Gasyori100knock/Question_91_100/imori_1.jpg")
    gt = np.array((47, 41, 129, 103), dtype=np.float32)
    test_img = cv2.imread("Gasyori100knock/Question_91_100/imori_many.jpg")
    sizes = np.array(((42, 42), (56, 56), (70, 70)), dtype=np.float32)

    out, _, _ = lib91100.my_object_detection((train_img, gt, 60, 60, 200, 0, 32, 32), (test_img, sizes, 4, 4, 32, 32), (0.01, 2, 10000), 0.7, 8)
    cv2.imshow("imori", out)
    cv2.waitKey(0)
    cv2.imwrite("q91_100/098.jpg", out)
    cv2.destroyAllWindows()