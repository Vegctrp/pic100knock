import cv2
import numpy as np
import lib0110

#img = cv2.imread("../Gasyori100knock/assets/imori.jpg")
#H,W,C = img.shape
#ans=img.copy()
#img2[:H,:W]=img2[:H,:W,(2,1,0)]

img = cv2.imread("../Gasyori100knock/Question_01_10/imori.jpg")
ans = lib0110.BGR2RGB(img).astype(np.uint8)

cv2.imshow("imori", ans)
cv2.waitKey(0)
cv2.imwrite("001.jpg", ans)
cv2.destroyAllWindows()