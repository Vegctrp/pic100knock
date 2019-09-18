import cv2
import numpy as np
import lib8190
import matplotlib.pyplot as plt
import sys,os
sys.path.append(os.getcwd())
from q21_30 import lib2130
from q01_10 import lib0110
from q61_70 import lib6170

if __name__ == '__main__':
    img = cv2.imread("Gasyori100knock/Question_81_90/thorino.jpg")
    
    database = lib8190.Ir1_makedatabase('Gasyori100knock/Question_81_90/dataset/',['akahara','madara'],5)
    print(database)

    for i, path in enumerate(range(10)):
        plt.subplot(2, 5, i+1)
        plt.hist(database[i,:].ravel(), bins=12, rwidth=0.8)
        plt.title(path)
    plt.savefig('./q81_90/084.png')