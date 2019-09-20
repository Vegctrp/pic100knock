import cv2
import numpy as np
import lib91100
import matplotlib.pyplot as plt

if __name__ == '__main__':
    img1 = cv2.imread("Gasyori100knock/Question_91_100/imori.jpg")
    out1 = lib91100.Kmeans_cr(img1, classes=5, seed=0).clip(0,255).astype(np.uint8)
    cv2.imshow("imori", out1)
    cv2.waitKey(0)
    cv2.imwrite("q91_100/092.jpg", out1)
    cv2.destroyAllWindows()
    
    img2 = cv2.imread("Gasyori100knock/Question_91_100/imori.jpg")
    out2 = lib91100.Kmeans_cr(img2, classes=10, seed=0).clip(0,255).astype(np.uint8)
    cv2.imshow("imori", out2)
    cv2.waitKey(0)
    cv2.imwrite("q91_100/092_k10.jpg", out2)
    cv2.destroyAllWindows()

    img3 = cv2.imread("Gasyori100knock/Question_91_100/madara.jpg")
    out3 = lib91100.Kmeans_cr(img3, classes=5, seed=0).clip(0,255).astype(np.uint8)
    cv2.imshow("imori", out3)
    cv2.waitKey(0)
    cv2.imwrite("q91_100/092_m.jpg", out3)
    cv2.destroyAllWindows()