import cv2

img = cv2.imread("../Gasyori100knock/assets/imori.jpg")
H,W,C = img.shape
img2=img.copy()
img2[:H//2,:W//2]=img2[:H//2,:W//2,(2,1,0)]
cv2.imshow("imori", img2)
cv2.waitKey(0)
cv2.imwrite("tutorial.jpg", img2)
cv2.destroyAllWindows()