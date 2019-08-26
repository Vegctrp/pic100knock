import cv2
import numpy as np

def Median_filter(img):
    img = img.astype(np.float64)
    padimg = np.pad(img,[(1,1),(1,1),(0,0)],'constant')
    height,width,C = img.shape
    out = np.zeros((height,width,3))

    for y in range(1,height+1):
        for x in range(1,width+1):
            for col in range(3):
                out[y-1,x-1,col]=np.median(padimg[y-1:y+2, x-1:x+2, col])

    return out


if __name__ == '__main__':
    img = cv2.imread("../Gasyori100knock/Question_01_10/imori_noise.jpg")
    ans = Median_filter(img).astype(np.uint8)

    cv2.imshow("imori", ans)
    cv2.waitKey(0)
    cv2.imwrite("010.jpg", ans)
    cv2.destroyAllWindows()