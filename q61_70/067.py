import cv2
import numpy as np
import lib6170
import matplotlib.pyplot as plt

if __name__ == '__main__':
    img = cv2.imread("Gasyori100knock/Question_61_70/imori.jpg")
    
    mag, ang = lib6170.HOG1_gradient(img)
    angs = lib6170.HOG2_histogram(mag, ang, N=8)
    for i in range(9):
        plt.subplot(3,3,i+1)
        plt.imshow(angs[i, ...])
        plt.axis('off')
        plt.xticks(color="None")
        plt.yticks(color="None")
    plt.savefig("./q61_70/067.png")
    plt.show()