import cv2
import numpy as np

# 21 gray_scale_transformation_linear
def Histogram_normalization(img, ymin, ymax):
    img = img.astype(np.float64)
    height,width,C = img.shape
    out = img.copy()

    xmin = np.min(img)
    xmax = np.max(img)
    
    out = (ymax - ymin) / (xmax - xmin) * (out - xmin) + ymin
    out[out < ymin] = ymin
    out[ymax <out] = ymax
    np.clip(out,ymin,ymax)
    return out


# 22
def Histogram_operation(img, ymean, ysd):
    img = img.astype(np.float64)
    height,width,C = img.shape
    out = img.copy()

    xmean = np.mean(img)
    xsd = np.std(img)
    
    out = ysd / xsd * (out - xmean) + ymean
    return out


# 23
def Histogram_equalization(img):
    img = img.astype(np.float64)
    height,width,C = img.shape
    S = height * width * C
    Zmax = np.max(img)
    out = img.copy()
    sumh = 0

    for col in range(256):
        index = np.where(img==col)
        sumh += len(img[index])
        Zd = Zmax / S * sumh
        out[index] = Zd
    return out


# 24
def Gamma_correction(img, c, g):
    img = img.astype(np.float64)

    img /= 255
    out = (img / c)**(1 / g)
    out *= 255
    np.clip(out,0,255)
    return out


# 25
def NearestNeighbor_interpolation(img, ax, ay):
    img = img.astype(np.float64)
    height,width,C = img.shape
    yheight = int(ay * height)
    ywidth = int(ax * width)

    y = np.arange(yheight).repeat(ywidth).reshape(yheight, -1)
    x = np.tile(np.arange(ywidth), (yheight, 1))
    y = np.round(y / ay).astype(np.int)
    x = np.round(x / ax).astype(np.int)
    out = img[y,x]

    np.clip(out,0,255)
    return out


# 26
def Bi_linear_interpolation(img, ax, ay):
    img = img.astype(np.float64)
    if len(img.shape)==2:
        img = np.expand_dims(img, axis=-1)
    height,width,C = img.shape
    yheight = int(ay * height)
    ywidth = int(ax * width)

    y = np.arange(yheight).repeat(ywidth).reshape(yheight, -1)
    x = np.tile(np.arange(ywidth), (yheight, 1))
    y = y / ay
    x = x / ax
    ny = np.floor(y).astype(np.int)
    nx = np.floor(x).astype(np.int)
    ny = np.minimum(ny, width-2)
    nx = np.minimum(nx, height-2)
    dy = y - ny
    dx = x - nx
    dy = np.repeat(np.expand_dims(dy, axis=-1), C, axis=-1)
    dx = np.repeat(np.expand_dims(dx, axis=-1), C, axis=-1)

    out = (1-dx)*(1-dy)*img[ny, nx] + dx*(1-dy)*img[ny, nx+1] + dy*(1-dx)*img[ny+1, nx] + dx*dy*img[ny+1, nx+1]
    if C==1:
        out = out.reshape((yheight,ywidth))
    np.clip(out,0,255)
    return out


# 27
def Bi_cubic_interpolation(img, ax, ay):
    img = img.astype(np.float64)
    height,width,C = img.shape
    yheight = int(ay * height)
    ywidth = int(ax * width)
    out = np.zeros((yheight, ywidth, 3))

    y = np.arange(yheight).repeat(ywidth).reshape(yheight, -1)
    x = np.tile(np.arange(ywidth), (yheight, 1))
    y = y / ay
    x = x / ax
    ny2 = np.floor(y).astype(np.int)
    nx2 = np.floor(x).astype(np.int)
    dy2 = y - ny2
    dx2 = x - nx2
    dys = [dy2+1, dy2, 1-dy2, 2-dy2]
    dxs = [dx2+1, dx2, 1-dx2, 2-dx2]
    nys = [ny2-1, ny2, ny2+1, ny2+2]
    nxs = [nx2-1, nx2, nx2+1, nx2+2]
    nys = np.clip(nys, 0, height-1)
    nxs = np.clip(nxs, 0, width-1)

    def weight(t):
        a = -1
        at = np.abs(t)
        w = np.zeros_like(t)
        index = np.where(at <= 1)
        w[index] = ((a+2)*np.power(at,3) - (a+3)*np.power(at,2) + 1)[index]
        index = np.where((1 < at) & (at <= 2))
        w[index] = (a*np.power(at,3) - 5*a*np.power(at,2) + 8*a*at - 4*a)[index]
        return w.reshape(4,yheight,ywidth)

    hys = weight(dys)
    hxs = weight(dxs)

    hsum = np.zeros_like(hys[0])
    for yy in range(4):
        for xx in range(4):
            for col in range(3):
                out[:, :, col] += img[nys[yy], nxs[xx], col] * hys[yy] * hxs[xx]
            hsum += hys[yy] * hxs[xx]
    for col in range(3):
        out[:, :, col] /= hsum
    np.clip(out,0,255)
    return out


# 28
def Affine(ox,oy, a,b,c,d,tx,ty):
    revdiv = a*d-b*c
    iy = (a * oy - c * ox) / revdiv - ty
    ix = (d * ox - b * oy) / revdiv - tx
    return ix,iy

def Affine_out(img,rx,ry,fx,fy):
    img = img.astype(np.float64)
    height,width,C = img.shape
    yheight,ywidth = fx.shape
    padimg = np.pad(img,[(1,1),(1,1),(0,0)],'constant')
    out = np.zeros((yheight, ywidth, 3))
    iy = ry.clip(-1,height).astype(np.int)
    ix = rx.clip(-1,width).astype(np.int)
    out[fy,fx] = padimg[iy+1, ix+1]
    return out

def Affine_translation(img, mx, my):
    img = img.astype(np.float64)
    height,width,C = img.shape
    ysize = height
    xsize = width
    fy = np.arange(height).repeat(width).reshape(height, -1)
    fx = np.tile(np.arange(width), (height, 1))
    sx1,sy1 = Affine(fx,fy, 1,0,0,1,mx,my)
    out = Affine_out(img,sx1,sy1,fx,fy)
    return out


# 29
def Affine_scale(img, ax, ay):
    img = img.astype(np.float64)
    height,width,C = img.shape
    ysize = int(ay * height)
    xsize = int(ax * width)
    fy = np.arange(ysize).repeat(xsize).reshape(ysize, -1)
    fx = np.tile(np.arange(xsize), (ysize, 1))
    sx1,sy1 = Affine(fx,fy, ax,0,0,ay,0,0)
    out = Affine_out(img,sx1,sy1,fx,fy)
    return out


# 30
def Affine_rotation_corner(img, a):
    img = img.astype(np.float64)
    height,width,C = img.shape
    ysize = height
    xsize = width
    fy = np.arange(ysize).repeat(xsize).reshape(ysize, -1)
    fx = np.tile(np.arange(xsize), (ysize, 1))
    sx1,sy1 = Affine(fx,fy, np.cos(-np.pi*a/180),-np.sin(-np.pi*a/180),np.sin(-np.pi*a/180),np.cos(-np.pi*a/180),0,0)
    out = Affine_out(img,sx1,sy1,fx,fy)
    return out


def Affine_rotation_center(img, a):
    img = img.astype(np.float64)
    height,width,C = img.shape
    ysize = height
    xsize = width
    ycenter = ysize // 2
    xcenter = xsize // 2
    fy = np.arange(ysize).repeat(xsize).reshape(ysize, -1)
    fx = np.tile(np.arange(xsize), (ysize, 1))
    sx1,sy1 = Affine(fx-xcenter,fy-ycenter, np.cos(-np.pi*a/180),-np.sin(-np.pi*a/180),np.sin(-np.pi*a/180),np.cos(-np.pi*a/180),-xcenter,-ycenter)
    out = Affine_out(img,sx1,sy1,fx,fy)
    return out