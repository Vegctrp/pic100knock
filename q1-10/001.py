import cv2
import numpy as np

def BGR2RGB(img):
    H,W,C = img.shape
    B = img[:, :, 0].copy()
    G = img[:, :, 1].copy()
    R = img[:, :, 2].copy()

    out = np.zeros((H,W,3))
    out[:, :, 0] = R
    out[:, :, 1] = G
    out[:, :, 2] = B

    return out

#img = cv2.imread("../Gasyori100knock/assets/imori.jpg")
#H,W,C = img.shape
#ans=img.copy()
#img2[:H,:W]=img2[:H,:W,(2,1,0)]

img = cv2.imread("../Gasyori100knock/assets/imori.jpg")
ans = BGR2RGB(img).astype(np.uint8)

cv2.imshow("imori", ans)
cv2.waitKey(0)
cv2.imwrite("001.jpg", ans)
cv2.destroyAllWindows()