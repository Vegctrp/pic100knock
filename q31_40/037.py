import cv2
import numpy as np
import lib3140
import matplotlib.pyplot as plt

if __name__ == '__main__':
    img = cv2.imread("Gasyori100knock/Question_31_40/imori.jpg")
    
    k=4
    fs = lib3140.DCTs(img, t=8, channels=3)
    out = lib3140.IDCTs(fs, t=8, k=k, channels=3)
    out = out.clip(0,255).astype(np.uint8)
    print("psrn : "+str(lib3140.PSRN(img, out, channels=3, max=255)))
    print("bitrate : "+str(8*(k**2)/(8**2)))
    cv2.imshow("imori", out)
    cv2.waitKey(0)
    cv2.imwrite("q31_40/037.jpg", out)
    cv2.destroyAllWindows()