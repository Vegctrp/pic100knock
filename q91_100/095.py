import numpy as np

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
        self.wout -= self.lr * grad_wout#np.expand_dims(grad_wout, axis=-1)
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
    for j in range(len(test_x)):
        x = test_x[j]
        t = test_t[j]
        print("in:", x, "pred:", nn.forward(x), "ans:", t)

if __name__ == '__main__':
    train_x = np.array([[0,0], [0,1], [1,0], [1,1]], dtype=np.float32)
    train_t = np.array([[0], [1], [1], [0]], dtype=np.float32)

    nn = NN(ind=train_x.shape[1], seed=0)
    nn = train_nn(nn, train_x, train_t, 1000)
    test_nn(nn, train_x, train_t)
