import cv2
import numpy as np
import lib1120
import matplotlib.pyplot as plt

if __name__ == '__main__':
    img = cv2.imread("Gasyori100knock/Question_11_20/imori_dark.jpg")
    
    plt.hist(img.ravel(), bins=255, range=(0,255), rwidth=0.8)
    plt.xlabel("pixel value")
    plt.ylabel("times")
    plt.savefig("q11_20/020result.png")
    plt.show()