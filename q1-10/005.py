import cv2
import numpy as np

def BGR2HSV(img):
    img = img.astype(np.float64) / 255.
    height,width,C = img.shape
    B = img[:, :, 0].copy()
    G = img[:, :, 1].copy()
    R = img[:, :, 2].copy()

    max_values = np.max(img,axis=2).copy()
    min_values = np.min(img,axis=2).copy()
    min_index = np.argmin(img,axis=2).copy()

    H = np.zeros((height,width))
    index = np.where(min_values==max_values)
    H[:, :][index]=0
    index = np.where(min_index==0)
    H[:, :][index]=60*(G[:,:][index]-R[:,:][index])/(max_values[:,:][index]-min_values[:,:][index])+60
    index = np.where(min_index==1)
    H[:, :][index]=60*(R[:,:][index]-B[:,:][index])/(max_values[:,:][index]-min_values[:,:][index])+300
    index = np.where(min_index==2)
    H[:, :][index]=60*(B[:,:][index]-G[:,:][index])/(max_values[:,:][index]-min_values[:,:][index])+180

    S = (max_values.copy() - min_values.copy())
    V = max_values.copy()

    return H,S,V


def HSV2BGR(H,S,V):
    height,width = H.shape

    C = S
    Hd = H / 60.
    X = C * (1 - np.abs(Hd % 2 - 1))

    img = np.zeros((height,width,3))
    Z = np.zeros((height,width))
    chv = [[Z,X,C],[Z,C,X],[X,C,Z],[C,X,Z],[C,Z,X],[X,Z,C]]

    for i in range(6):
        index = np.where((i <= Hd) & (Hd < i+1))
        for col in range(3):
            img[:, :, col][index] = (V - C)[index] + chv[i][col][index]
    
    img = np.clip(img,0,1)
    return img*255


def hue_inversion(img):
    H,S,V = BGR2HSV(img)
    Hd = (H + 180) % 360
    out = HSV2BGR(Hd,S,V)
    return out


if __name__ == '__main__':
    img = cv2.imread("../Gasyori100knock/assets/imori.jpg")
    ans = hue_inversion(img).astype(np.uint8)

    cv2.imshow("imori", ans)
    cv2.waitKey(0)
    cv2.imwrite("005.jpg", ans)
    cv2.destroyAllWindows()