import cv2
import numpy as np
import lib91100
import sys,os
sys.path.append(os.getcwd())
from q61_70 import lib6170
from q21_30 import lib2130

if __name__ == '__main__':
    img = cv2.imread("Gasyori100knock/Question_91_100/imori_1.jpg")
    gt = np.array((47, 41, 129, 103), dtype=np.float32)
    _, rects, labels = lib91100.random_cropping(img, gt, 60, 60, 200, 0)
    database = lib91100.nn_getHOGs(img, rects, 32, 32)

    train_x = np.array(database[:160])
    train_t = np.array(labels[:160])
    test_x = np.array(database[160:])
    test_t = np.array(labels[160:])

    train_t = np.expand_dims(train_t, axis=-1)
    test_t = np.expand_dims(test_t, axis=-1)

    nn = lib91100.NN(ind=train_x.shape[1], lr=0.01, seed=0)
    nn = lib91100.train_nn(nn, train_x, train_t, 10000)
    lib91100.test_nn(nn, train_x, train_t)
    lib91100.test_nn(nn, test_x, test_t)
