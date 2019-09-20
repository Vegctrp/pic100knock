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

    database, classmean = lib8190.Kmeans1_makedatabase('Gasyori100knock/Question_81_90/dataset/',filenames,2)
    print(database)
    print(classmean)