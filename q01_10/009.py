import cv2
import numpy as np
import lib0110

if __name__ == '__main__':
    img = cv2.imread("Gasyori100knock/Question_01_10/imori_noise.jpg")
    ans = lib0110.Gaussian_filter(img, k=5, sigma=1.3).astype(np.uint8)

    cv2.imshow("imori", ans)
    cv2.waitKey(0)
    cv2.imwrite("q01_10/009.jpg", ans)
    cv2.destroyAllWindows()