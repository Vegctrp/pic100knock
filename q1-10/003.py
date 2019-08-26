import cv2
import numpy as np

def BGR2GRAY(img):
    B = img[:, :, 0].copy()
    G = img[:, :, 1].copy()
    R = img[:, :, 2].copy()

    out = 0.2126 * R + 0.7152 * G + 0.0722 * B

    return out

def binalization(img, threshold):
    img[img < threshold] = 0
    img[img >= threshold] = 255
    return img

img = cv2.imread("../Gasyori100knock/assets/imori.jpg")
grayimage = BGR2GRAY(img)
ans = binalization(grayimage,128).astype(np.uint8)

cv2.imshow("imori", ans)
cv2.waitKey(0)
cv2.imwrite("003.jpg", ans)
cv2.destroyAllWindows()