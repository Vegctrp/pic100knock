import cv2
import numpy as np
import lib2130
import matplotlib.pyplot as plt

if __name__ == '__main__':
    img = cv2.imread("Gasyori100knock/Question_21_30/imori.jpg")
    
    ans = lib2130.Histogram_equalization(img).clip(0,255).astype(np.uint8)
    cv2.imshow("imori", ans)
    cv2.waitKey(0)
    cv2.imwrite("q21_30/023.jpg", ans)
    cv2.destroyAllWindows()

    plt.hist(ans.ravel(), bins=255, range=(0,255), rwidth=0.8)
    plt.xlabel("pixel value")
    plt.ylabel("times")
    plt.savefig("q21_30/023result.png")
    plt.show()