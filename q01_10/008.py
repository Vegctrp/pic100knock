import cv2
import numpy as np
import lib0110

if __name__ == '__main__':
    img = cv2.imread("Gasyori100knock/Question_01_10/imori.jpg")
    ans = lib0110.max_pooling(img,8).astype(np.uint8)

    cv2.imshow("imori", ans)
    cv2.waitKey(0)
    cv2.imwrite("q01_10/008.jpg", ans)
    cv2.destroyAllWindows()