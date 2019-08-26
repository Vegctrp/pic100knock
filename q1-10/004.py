import cv2
import numpy as np
import lib0110

img = cv2.imread("../Gasyori100knock/assets/imori.jpg")
grayimage = lib0110.BGR2GRAY(img)
ans = lib0110.OTSU_binalization(grayimage).astype(np.uint8)

cv2.imshow("imori", ans)
cv2.waitKey(0)
cv2.imwrite("004.jpg", ans)
cv2.destroyAllWindows()