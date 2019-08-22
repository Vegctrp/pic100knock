import cv2
import numpy as np

def BGR2GRAY(img):
    H,W,C = img.shape
    B = img[:, :, 0].copy()
    G = img[:, :, 1].copy()
    R = img[:, :, 2].copy()

    out = 0.2126 * R + 0.7152 * G + 0.0722 * B
    print(out[3,3])

    return out

#img = cv2.imread("../Gasyori100knock/assets/imori.jpg")
#H,W,C = img.shape
#ans=img.copy()
#img2[:H,:W]=img2[:H,:W,(2,1,0)]

img = cv2.imread("../Gasyori100knock/assets/imori.jpg")
ans = BGR2GRAY(img).astype(np.uint8)

cv2.imshow("imori", ans)
cv2.waitKey(0)
cv2.imwrite("002.jpg", ans)
cv2.destroyAllWindows()