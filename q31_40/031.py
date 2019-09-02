import cv2
import numpy as np
import lib3140
import matplotlib.pyplot as plt
import sys,os
sys.path.append(os.getcwd())
from q21_30 import lib2130

if __name__ == '__main__':
    img = cv2.imread("Gasyori100knock/Question_31_40/imori.jpg")
    
    ans = lib3140.Affine_skew(img, dx=30).astype(np.uint8)
    cv2.imshow("imori", ans)
    cv2.waitKey(0)
    cv2.imwrite("q31_40/031_1.jpg", ans)
    cv2.destroyAllWindows()

    ans = lib3140.Affine_skew(img, dy=30).astype(np.uint8)
    cv2.imshow("imori", ans)
    cv2.waitKey(0)
    cv2.imwrite("q31_40/031_2.jpg", ans)
    cv2.destroyAllWindows()

    ans = lib3140.Affine_skew(img, dx=30, dy=30).astype(np.uint8)
    cv2.imshow("imori", ans)
    cv2.waitKey(0)
    cv2.imwrite("q31_40/031_3.jpg", ans)
    cv2.destroyAllWindows()