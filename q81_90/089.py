import cv2
import numpy as np
import lib8190
import matplotlib.pyplot as plt

if __name__ == '__main__':
    filenames=[]
    for i,name in enumerate(['akahara','madara']):
        for num in range(1,6):
            filenames.append("train_"+name+"_"+str(num)+".jpg")
        for num in range(1,3):
            filenames.append("test_"+name+"_"+str(num)+".jpg")

    ans = lib8190.Kmeans2('Gasyori100knock/Question_81_90/dataset/',filenames, classes=2)
    
    for name,predict in zip(filenames,ans):
        print(name+" : " + str(predict))