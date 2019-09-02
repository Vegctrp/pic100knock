import cv2
import numpy as np
import lib1120
import sys,os
sys.path.append(os.getcwd())
from q01_10 import lib0110

if __name__ == '__main__':
    img = cv2.imread("Gasyori100knock/Question_11_20/imori.jpg")
    gray_img = lib0110.BGR2GRAY(img)
    ans = lib1120.Laplacian_filter(gray_img).astype(np.uint8)

    cv2.imshow("imori", ans)
    cv2.waitKey(0)
    cv2.imwrite("q11_20/017.jpg", ans)
    cv2.destroyAllWindows()