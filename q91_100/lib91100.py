import cv2
import numpy as np
from collections import Counter
import sys,os
sys.path.append(os.getcwd())
from q01_10 import lib0110
from q21_30 import lib2130
from q61_70 import lib6170

# 91
def Kmeans_cr1_init(img, classes, seed=-1):
    if seed != -1:
        np.random.seed(seed)
    height, width, C = img.shape

    rimg = img.reshape((height*width, C)).astype(np.float32)
    index = np.random.choice(np.arange(height*width), classes, replace=False)
    class_center = rimg[index].copy()

    classmat = np.zeros((height*width, C))
    for i in range(height*width):
        classmat[i] = np.argmin(np.sum((class_center - rimg[i])**2, axis=1))
    return rimg, classmat, class_center

# 92
def Kmeans_cr2_redis(rimg, class_center, classes):
    hw, C = rimg.shape
    classmat = np.zeros(hw)
    new_class_center = np.zeros_like(class_center)
    status = True
    for i in range(hw):
        classmat[i] = np.argmin(np.sum((class_center - rimg[i])**2, axis=1))

    for cl in range(classes):
        index = np.where(classmat == cl)
        if len(index[0])==0:
            new_class_center[cl] = class_center[cl]
        else:
            new_class_center[cl] = np.mean(rimg[index], axis=0)
    if (class_center == new_class_center).all():
        status = False
    return new_class_center, status

def Kmeans_cr3_draw(img, class_center, classes):
    height, width, C = img.shape
    img = img.astype(np.float32)
    for y in range(height):
        for x in range(width):
            img[y,x,:] = class_center[np.argmin(np.sum((class_center - img[y,x,:])**2, axis=1))]
    return img

def Kmeans_cr(img, classes, seed=-1):
    rimg, _, class_center = Kmeans_cr1_init(img, classes, seed=seed)
    status = True
    while status:
        class_center, status = Kmeans_cr2_redis(rimg, class_center, classes)
    return Kmeans_cr3_draw(img, class_center, classes)


# 93
def area(rect):
    return max(rect[2]-rect[0], 0) * max(rect[3]-rect[1], 0)

def calc_IoU(rect1, rect2):
    rect1 = rect1.astype(np.float32)
    rect2 = rect2.astype(np.float32)
    rol = np.array((max(rect1[0],rect2[0]), max(rect1[1],rect2[1]), min(rect1[2],rect2[2]), min(rect1[3],rect2[3])))
    iou = area(rol) / (area(rect1) + area(rect2) - area(rol))
    return iou


# 94
def random_cropping(img, gt, sizex, sizey, num, seed=-1):
    height, width, _ = img.shape
    if seed != -1:
        np.random.seed(seed)
    rects = []
    labels = []
    i2 = img.copy()
    for _ in range(num):
        x1 = np.random.randint(width-60)
        y1 = np.random.randint(height-60)
        rect = np.array((x1, y1, x1+sizex, y1+sizey), dtype=np.float32)
        rects.append(rect)
        iou = calc_IoU(gt, rect)
        label = 1 if iou>=0.5 else 0
        labels.append(label)
        if label==0:
            i2 = cv2.rectangle(i2, (rect[0],rect[1]), (rect[2],rect[3]), (255,0,0))
        elif label==1:
            i2 = cv2.rectangle(i2, (rect[0],rect[1]), (rect[2],rect[3]), (0,0,255))
    i2 = cv2.rectangle(i2, (gt[0],gt[1]), (gt[2],gt[3]), (0,255,0))
    return i2, rects, labels


# 96
def nn_getHOG(img, rect, rex, rey, HOG_N=8):
    rect = list(map(int, rect))
    part = img[rect[1]:rect[3], rect[0]:rect[2]]
    hei, wid, _ = part.shape
    p2 = lib2130.Bi_linear_interpolation(part, float(rex)/wid, float(rey)/hei)
    mag, ang = lib6170.HOG1_gradient(p2)
    angs = lib6170.HOG2_histogram(mag, ang, N=HOG_N)
    outs = lib6170.HOG3_normalization(angs, C=3, epsilon=1)
    feat = outs.ravel().astype(np.float32)
    return feat

def nn_getHOGs(img, rects, rex, rey, HOG_N=8):
    database = []
    for rect in rects:
        database.append(nn_getHOG(img, rect, rex, rey, HOG_N))
    return database

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
        score = nn.forward(x)
        ans = 0 if score<=0.5 else 1
        if t==ans:
            correct += 1
    print("datasize : ", num, ", correct : ", correct)

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

        grad_u1 = np.dot(grad_u2, self.w2.T) * self.z2 * (1 - self.z2)
        grad_w1 = np.dot(self.z1.T, grad_u1)
        grad_b1 = np.dot(np.ones([grad_u1.shape[0]]), grad_u1)
        self.w1 -= self.lr * grad_w1
        self.b1 -= self.lr * grad_b1

    def sigmoid(self, x):
        return 1. / (1. + np.exp(-x))


# 97
def Object_detection1_getpart(img, sizes, dx, dy, hogx, hogy, hogn=8):
    height, width, _ = img.shape
    rects = []
    feats = []
    for y in range(0,height,dy):
        for x in range(0,width, dx):
            for size in sizes:
                hx = size[0] // 2
                hy = size[1] // 2
                x1 = int(max(0, x-hx))
                x2 = int(min(width, x+hx))
                y1 = int(max(0, y-hy))
                y2 = int(min(height, y+hy))
                rect = [x1, y1, x2, y2]
                feat = nn_getHOG(img, rect, hogy, hogx, hogn)
                if (x1 != x2) and (y1 != y2):
                    rects.append(rect)
                    feats.append(feat)
    return rects, feats

# 98
def Object_detection2_judge(img, nn, feats, rects, threshold):
    i2 = img.copy()
    cor_rects = []
    cor_scores = []
    for feat, rect in zip(feats, rects):
        score = nn.forward(feat)
        if score >= threshold:
            print(rect, score)
            cor_rects.append(rect)
            cor_scores.append(score)
            i2 = cv2.rectangle(i2, (rect[0],rect[1]), (rect[2],rect[3]), (0,0,255), 1)
    return i2, cor_rects, cor_scores

def my_object_detection(trs, tes, nns, jus, HOG_N):
    tr_img, tr_gt, tr_sizex, tr_sizey, tr_num, tr_seed, tr_rex, tr_rey = trs
    te_img, te_sizes, te_dx, te_dy, te_hogx, te_hogy = tes
    nn_lr, nn_seed, nn_num = nns
    ju_threshold = jus
    #HOG_N

    _, tr_rects, tr_labels = random_cropping(tr_img, tr_gt, tr_sizex, tr_sizey, tr_num, tr_seed)
    tr_database = nn_getHOGs(tr_img, tr_rects, tr_rex, tr_rey, HOG_N)

    train_x = np.array(tr_database)
    train_t = np.array(tr_labels)
    train_t = np.expand_dims(train_t, axis=-1).astype(np.float32)

    nn = NN(ind=train_x.shape[1], lr=nn_lr, seed=nn_seed)
    nn = train_nn(nn, train_x, train_t, nn_num)

    test_x = np.array(tr_database[160:])
    test_t = np.array(tr_labels[160:])
    test_t = np.expand_dims(test_t, axis=-1).astype(np.float32)
    test_nn(nn, test_x, test_t)

    rects, feats = Object_detection1_getpart(te_img, te_sizes, te_dx, te_dy, te_hogx, te_hogy, HOG_N)
    out, cor_rects, cor_scores = Object_detection2_judge(te_img, nn, feats, rects, threshold=ju_threshold)
    return out, cor_rects, cor_scores

# 99
def Object_detection3_nms(img, rects, scores, threshold):
    flag = True
    selected_rects = []
    selected_scores = []
    #for rect,score in zip(rects, scores):
    #    print(rect,score)
    while flag:
        ind = np.argmax(scores)
        selected_rects.append(rects[ind])
        selected_scores.append(scores[ind])
        scores[ind] = 0
        for i,rect in enumerate(rects):
            iou = calc_IoU(np.array(rects[ind]), np.array(rect))
            if iou >= threshold:
                scores[i] = 0
        if np.sum(scores) == 0:
            flag = 0
    #print("rects : ",selected_rects, "scores : ",selected_scores)
    return selected_rects, selected_scores

def Object_detection4_draw(img, rects, scores):
    for rect,score in zip(rects, scores):
        img = cv2.rectangle(img, (rect[0],rect[1]), (rect[2],rect[3]), (0,0,255), 1)
        img = cv2.putText(img, str(score[0])[:4], (rect[0],rect[1]+9), cv2.FONT_HERSHEY_COMPLEX, 0.4, (0, 255, 255), 1)
    return img

def Object_detection5_evaluate(img, rects, scores, gts, threshold):
    #recall
    recall_num = gts.shape[0]
    recall_correct = 0
    for gt in gts:
        for rect in rects:
            iou = calc_IoU(gt, np.array(rect))
            #print(iou)
            if iou >= threshold:
                recall_correct += 1
                break
    Recall = float(recall_correct)/recall_num
    print("Recall : ", Recall, "(",recall_correct,":",recall_num,")")

    #precision
    precision_num = len(rects)
    precision_correct = 0
    mAP_judge = []
    for rect in rects:
        flag = False
        for gt in gts:
            iou = calc_IoU(gt, np.array(rect))
            if iou >= threshold:
                flag = True
                break
        if flag:
            precision_correct += 1
            mAP_judge.append(1)
        else:
            mAP_judge.append(0)
    Precision = float(precision_correct)/precision_num
    print("Precision : ", Precision, "(",precision_correct,":",precision_num,")")

    #fscore
    if Recall+Precision == 0:
        fscore = 0
    else:
        fscore = 2 * Recall * Precision / (Recall + Precision)
    print("F-score : ", fscore)

    #mAP
    mAP_value = 0
    mAP_num = 0
    for i,m in enumerate(mAP_judge):
        if m==1:
            mAP_num += 1
            mAP_value += mAP_num / (i+1)
    if not mAP_num==0:
        mAP_value /= mAP_num
    print("mAP : ", mAP_value)

    # drawing
    for rect,score,j in zip(rects, scores, mAP_judge):
        if j == 1:
            img = cv2.rectangle(img, (rect[0],rect[1]), (rect[2],rect[3]), (0,0,255), 1)
        else:
            img = cv2.rectangle(img, (rect[0],rect[1]), (rect[2],rect[3]), (255,0,0), 1)
        img = cv2.putText(img, str(score[0])[:4], (rect[0],rect[1]+9), cv2.FONT_HERSHEY_COMPLEX, 0.4, (0, 255, 255), 1)
    for gt in gts:
        img = cv2.rectangle(img, (gt[0],gt[1]), (gt[2],gt[3]), (0,255,0))
    return img