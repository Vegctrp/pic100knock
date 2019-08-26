import cv2
import numpy as np

def color_reduction(img):
    img = img.astype(np.float64)
    height,width,C = img.shape

    out = (img // 64) *64 + 32
    return out


if __name__ == '__main__':
    img = cv2.imread("../Gasyori100knock/Question_01_10/imori.jpg")
    ans = color_reduction(img).astype(np.uint8)

    cv2.imshow("imori", ans)
    cv2.waitKey(0)
    cv2.imwrite("006.jpg", ans)
    cv2.destroyAllWindows()