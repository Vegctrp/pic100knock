import cv2
import numpy as np
import lib2130
import matplotlib.pyplot as plt

if __name__ == '__main__':
    img = cv2.imread("Gasyori100knock/Question_21_30/imori.jpg")
    
    ans = lib2130.Bi_cubic_interpolation(img, ax=1.5, ay=1.5)
    cv2.imshow("imori", ans)
    cv2.waitKey(0)
    cv2.imwrite("q21_30/027.jpg", ans)
    cv2.destroyAllWindows()