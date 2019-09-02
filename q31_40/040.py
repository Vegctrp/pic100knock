import cv2
import numpy as np
import lib3140
import matplotlib.pyplot as plt

if __name__ == '__main__':
    img = cv2.imread("Gasyori100knock/Question_31_40/imori.jpg")
    Y, Cb, Cr = lib3140.BGR2YCbCr(img)
    fY = lib3140.DCTs(Y, t=8, channels=1)
    fCb = lib3140.DCTs(Cb, t=8, channels=1)
    fCr = lib3140.DCTs(Cr, t=8, channels=1)
    fY, fCb, fCr = lib3140.DCT_quantization8_YCbCr(fY,fCb,fCr)
    Y = lib3140.IDCTs(fY, t=8, k=8, channels=1)
    Cb = lib3140.IDCTs(fCb, t=8, k=8, channels=1)
    Cr = lib3140.IDCTs(fCr, t=8, k=8, channels=1)
    out = lib3140.YCbCr2BGR(Y,Cb,Cr).clip(0,255).astype(np.uint8)
    cv2.imshow("imori", out)
    cv2.waitKey(0)
    cv2.imwrite("q31_40/040.jpg", out)
    cv2.destroyAllWindows()