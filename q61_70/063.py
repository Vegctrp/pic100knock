import cv2
import numpy as np
import lib6170
import matplotlib.pyplot as plt

if __name__ == '__main__':
    img = cv2.imread("Gasyori100knock/Question_61_70/gazo.png")
    
    out = lib6170.Thinning(img).clip(0,255).astype(np.uint8)

    cv2.imshow("imori", out)
    cv2.waitKey(0)
    cv2.imwrite("q61_70/063.jpg", out)
    cv2.destroyAllWindows()