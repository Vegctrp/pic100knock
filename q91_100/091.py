import cv2
import numpy as np
import lib91100
import matplotlib.pyplot as plt

if __name__ == '__main__':
    img = cv2.imread("Gasyori100knock/Question_91_100/imori.jpg")
    rimg, classmat, class_center = lib91100.Kmeans_cr1_init(img, classes=5, seed=0)

    print(class_center)
    out = (classmat*51).clip(0,255).astype(np.uint8).reshape(img.shape)
    
    cv2.imshow("imori", out)
    cv2.waitKey(0)
    cv2.imwrite("q91_100/091.jpg", out)
    cv2.destroyAllWindows()