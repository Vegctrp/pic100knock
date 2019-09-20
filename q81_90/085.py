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
    filenames=[]
    labels=[]
    for i,name in enumerate(['akahara','madara']):
        for num in range(1,6):
            filenames.append("train_"+name+"_"+str(num)+".jpg")
            labels.append(i)

    database = lib8190.Ir1_makedatabase('Gasyori100knock/Question_81_90/dataset/',filenames,labels)
    
    test_filenames=[]
    for i,name in enumerate(['akahara','madara']):
        for num in range(1,3):
            test_filenames.append("test_"+name+"_"+str(num)+".jpg")

    lib8190.Ir2_judge('Gasyori100knock/Question_81_90/dataset/',filenames,database,test_filenames)