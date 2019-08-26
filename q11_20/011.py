import cv2
import numpy as np
import lib1120

if __name__ == '__main__':
    img = cv2.imread("Gasyori100knock/Question_11_20/imori.jpg")
    ans = lib1120.Smoothing_filter(img).astype(np.uint8)

    cv2.imshow("imori", ans)
    cv2.waitKey(0)
    cv2.imwrite("q11-20/011.jpg", ans)
    cv2.destroyAllWindows()