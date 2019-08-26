import cv2
import numpy as np

def BGR2GRAY(img):
    H,W,C = img.shape
    B = img[:, :, 0].copy()
    G = img[:, :, 1].copy()
    R = img[:, :, 2].copy()

    out = 0.2126 * R + 0.7152 * G + 0.0722 * B

    return out

def OTSU_binalization(img): # for gray-scale image
    max_t=0
    use_t=-1
    for t in range(0,256):
        v0 = img[np.where(img < t)]
        print(v0)
        v1 = img[np.where(img >= t)]
        w0 = len(v0)
        w1 = len(v1)
        m0 = np.mean(v0) if w0>0 else 0.
        m1 = np.mean(v1) if w1>0 else 0.
        Sb2 = w0 * w1 * (m0-m1)**2
        if Sb2 > max_t:
            max_t = Sb2
            use_t = t
    img[img < use_t]=0
    img[img >= use_t]=255
    return img

img = cv2.imread("../Gasyori100knock/assets/imori.jpg")
grayimage = BGR2GRAY(img)
ans = OTSU_binalization(grayimage).astype(np.uint8)

cv2.imshow("imori", ans)
cv2.waitKey(0)
cv2.imwrite("004.jpg", ans)
cv2.destroyAllWindows()