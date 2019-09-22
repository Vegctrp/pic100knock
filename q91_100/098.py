import cv2
import numpy as np
import lib91100
import sys,os
sys.path.append(os.getcwd())
from q61_70 import lib6170
from q21_30 import lib2130

if __name__ == '__main__':
    img = cv2.imread("Gasyori100knock/Question_91_100/imori_many.jpg")
    sizes = np.array(((42, 42), (56, 56), (70, 70)), dtype=np.float32)
    rects, feats = lib91100.Object_detection1_getpart(img, sizes, 4, 4, 32, 32)

    train_img = cv2.imread("Gasyori100knock/Question_91_100/imori_1.jpg")
    gt = np.array((47, 41, 129, 103), dtype=np.float32)
    _, tr_rects, tr_labels = lib91100.random_cropping(train_img, gt, 60, 60, 200, 0)
    tr_database = lib91100.nn_getHOGs(train_img, tr_rects, 32, 32)

    train_x = np.array(tr_database)
    train_t = np.array(tr_labels)
    test_x = np.array(tr_database[160:])
    test_t = np.array(tr_labels[160:])

    train_t = np.expand_dims(train_t, axis=-1).astype(np.float32)
    test_t = np.expand_dims(test_t, axis=-1).astype(np.float32)

    nn = lib91100.NN(ind=train_x.shape[1], lr=0.01, seed=2)
    nn = lib91100.train_nn(nn, train_x, train_t, 10000)
    lib91100.test_nn(nn, test_x, test_t)
    out = lib91100.Object_detection2_judge(img, nn, feats, rects, threshold=0.7)

    cv2.imshow("imori", out)
    cv2.waitKey(0)
    cv2.imwrite("q91_100/098.jpg", out)
    cv2.destroyAllWindows()