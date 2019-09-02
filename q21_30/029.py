import cv2
import numpy as np
import lib2130
import matplotlib.pyplot as plt

if __name__ == '__main__':
    img = cv2.imread("Gasyori100knock/Question_21_30/imori.jpg")
    
    ans = lib2130.Affine_scale(img, ax=1.3, ay=0.8)
    ans1 = ans.clip(0,255).astype(np.uint8)
    cv2.imshow("imori", ans1)
    cv2.waitKey(0)
    cv2.imwrite("q21_30/029_1.jpg", ans1)
    cv2.destroyAllWindows()

    ans = lib2130.Affine_translation(ans, mx=30, my=-30).clip(0,255).astype(np.uint8)
    cv2.imshow("imori", ans)
    cv2.waitKey(0)
    cv2.imwrite("q21_30/029_2.jpg", ans)
    cv2.destroyAllWindows()