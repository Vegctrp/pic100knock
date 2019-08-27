import cv2
import numpy as np
import lib0110

if __name__ == '__main__':
    img = cv2.imread("Gasyori100knock/Question_01_10/imori.jpg")
    ans = lib0110.BGR2GRAY(img).astype(np.uint8)

    cv2.imshow("imori", ans)
    cv2.waitKey(0)
    cv2.imwrite("q1-10/002.jpg", ans)
    cv2.destroyAllWindows()