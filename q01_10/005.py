import cv2
import numpy as np
import lib0110

if __name__ == '__main__':
    img = cv2.imread("Gasyori100knock/Question_01_10/imori.jpg")
    ans = lib0110.hue_inversion(img)

    cv2.imshow("imori", ans)
    cv2.waitKey(0)
    cv2.imwrite("q1-10/005.jpg", ans)
    cv2.destroyAllWindows()