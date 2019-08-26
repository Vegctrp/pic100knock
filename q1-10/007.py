import cv2
import numpy as np

def mean_pooling(img,pixels):
    img = img.astype(np.float64)
    height,width,C = img.shape

    out = np.zeros((height,width,3))
    gridx = width // pixels
    gridy = height // pixels
    for y in range(gridy):
        for x in range(gridx):
            for col in range(3):
                out[y*pixels:(y+1)*pixels, x*pixels:(x+1)*pixels, col] = np.mean(img[y*pixels:(y+1)*pixels, x*pixels:(x+1)*pixels, col])
    return out


if __name__ == '__main__':
    img = cv2.imread("../Gasyori100knock/Question_01_10/imori.jpg")
    ans = mean_pooling(img,8).astype(np.uint8)

    cv2.imshow("imori", ans)
    cv2.waitKey(0)
    cv2.imwrite("007.jpg", ans)
    cv2.destroyAllWindows()