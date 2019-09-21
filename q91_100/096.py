import cv2
import numpy as np
import lib91100
import sys,os
sys.path.append(os.getcwd())
from q61_70 import lib6170
from q21_30 import lib2130

class NN:
    def __init__(self, ind=2, w1=64, w2=64, outd=1, lr=0.1, seed=-1):
        if seed != -1:
            np.random.seed(seed)
        self.w1 = np.random.normal(0, 1, [ind, w1])
        self.b1 = np.random.normal(0, 1, [w1])
        self.w2 = np.random.normal(0, 1, [w1, w2])
        self.b2 = np.random.normal(0, 1, [w2])
        self.wout = np.random.normal(0, 1, [w2, outd])
        self.bout = np.random.normal(0, 1, [outd])
        self.lr = lr

    def forward(self, x):
        self.z1 = x
        self.z2 = self.sigmoid(np.dot(self.z1, self.w1) + self.b1)
        self.z3 = self.sigmoid(np.dot(self.z2, self.w2) + self.b2)
        self.out = self.sigmoid(np.dot(self.z3, self.wout) + self.bout)
        return self.out

    def train(self, x, t):
        # backpropagation output layer
        #En = t * np.log(self.out) + (1-t) * np.log(1-self.out)
        En = (self.out - t) * self.out * (1 - self.out)
        grad_En = En #np.array([En for _ in range(t.shape[0])])
        grad_wout = np.dot(self.z3.T, En)
        grad_bout = np.dot(np.ones([En.shape[0]]), En)
        self.wout -= self.lr * grad_wout
        self.bout -= self.lr * grad_bout

        # backpropagation inter layer
        grad_u2 = np.dot(En, self.wout.T) * self.z3 * (1 - self.z3)
        grad_w2 = np.dot(self.z2.T, grad_u2)
        grad_b2 = np.dot(np.ones([grad_u2.shape[0]]), grad_u2)
        self.w2 -= self.lr * grad_w2
        self.b2 -= self.lr * grad_b2

        grad_u1 = np.dot(En, self.wout.T) * self.z2 * (1 - self.z2)
        grad_w1 = np.dot(self.z1.T, grad_u1)
        grad_b1 = np.dot(np.ones([grad_u1.shape[0]]), grad_u1)
        self.w1 -= self.lr * grad_w1
        self.b1 -= self.lr * grad_b1

    def sigmoid(self, x):
        return 1. / (1. + np.exp(-x))


def train_nn(nn, train_x, train_t, num):
    for _ in range(num):
        nn.forward(train_x)
        nn.train(train_x, train_t)
    return nn

def test_nn(nn, test_x, test_t):
    num = len(test_x)
    correct = 0
    for j in range(len(test_x)):
        x = test_x[j]
        t = test_t[j][0]
        #print("in:", x, "pred:", nn.forward(x), "ans:", t)
        ans = 0 if nn.forward(x)<=0.5 else 1
        if t==ans:
            correct += 1
    print("datasize : ", num, ", correct : ", correct)

if __name__ == '__main__':
    img = cv2.imread("Gasyori100knock/Question_91_100/imori_1.jpg")
    gt = np.array((47, 41, 129, 103), dtype=np.float32)
    _, rects, labels = lib91100.random_cropping(img, gt, 60, 60, 200, 0)
    database = []
    for rect in rects:
        rect = list(map(int, rect))
        part = img[rect[1]:rect[3], rect[0]:rect[2]]
        hei, wid, _ = part.shape
        p2 = lib2130.Bi_linear_interpolation(part, 32./hei, 32./wid)
        mag, ang = lib6170.HOG1_gradient(p2)
        angs = lib6170.HOG2_histogram(mag, ang, N=8)
        outs = lib6170.HOG3_normalization(angs, C=3, epsilon=1)
        feat = outs.ravel()
        database.append(feat.astype(np.float32))

    train_x = np.array(database[:160])
    train_t = np.array(labels[:160])
    test_x = np.array(database[160:])
    test_t = np.array(labels[160:])

    train_t = np.expand_dims(train_t, axis=-1)
    test_t = np.expand_dims(test_t, axis=-1)

    nn = NN(ind=train_x.shape[1], lr=0.01, seed=0)
    print("hoge")
    nn = train_nn(nn, train_x, train_t, 10000)
    print("hgoe1")
    test_nn(nn, train_x, train_t)
    test_nn(nn, test_x, test_t)
