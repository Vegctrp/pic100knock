import cv2
import numpy as np
import lib5160
import matplotlib.pyplot as plt

if __name__ == '__main__':
    img1 = cv2.imread("Gasyori100knock/Question_51_60/imori.jpg")
    img2 = cv2.imread("Gasyori100knock/Question_51_60/thorino.jpg")
    
    out = lib5160.Alpha_blend(img1, img2, alpha=0.6).clip(0,255).astype(np.uint8)

    cv2.imshow("imori", out)
    cv2.waitKey(0)
    cv2.imwrite("q51_60/060.jpg", out)
    cv2.destroyAllWindows()