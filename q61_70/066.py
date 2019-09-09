import cv2
import numpy as np
import lib6170
import matplotlib.pyplot as plt

if __name__ == '__main__':
    img = cv2.imread("Gasyori100knock/Question_61_70/imori.jpg")
    
    mag, ang = lib6170.HOG1_gradient(img)
    print(ang)

    mag = mag.clip(0,255).astype(np.uint8)
    ang = (ang * 31).astype(np.uint8)

    cv2.imshow("imori", mag)
    cv2.waitKey(0)
    cv2.imwrite("q61_70/066mag.jpg", mag)
    cv2.destroyAllWindows()

    cv2.imshow("imori", ang)
    cv2.waitKey(0)
    cv2.imwrite("q61_70/066ang.jpg", ang)
    cv2.destroyAllWindows()