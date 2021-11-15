import numpy as np
import polarTransform
import cv2
from skimage import transform, draw
import copy
import random


def normalization(x):
    lower = -1024.0
    upper = 2000.0

    x = (x - lower) / (upper - lower)
    x[x < 0.0] = 0.0
    x[x > 1.0] = 1.0
    return x


def GaussianHighFrequencyFilter(imarr, sigma):

    height, width = imarr.shape
    fft = np.fft.fft2(imarr)
    fft = np.fft.fftshift(fft)

    for i in range(height):
        for j in range(width):
            fft[i, j] *= (1 - np.exp(-((i - (height - 1) / 2) ** 2 + (j - (width - 1) / 2) ** 2) / 2 / sigma ** 2))
    fft = np.fft.ifftshift(fft)
    ifft = np.fft.ifft2(fft)
    ifft = np.real(ifft)
    max = np.max(ifft)
    min = np.min(ifft)
    res = np.zeros((height, width))
    for i in range(height):
        for j in range(width):
            # res[i, j] = 255 * (ifft[i, j] - min) / (max - min)
            res[i, j] = (ifft[i, j] - min) / (max - min)

    return res


def extract(image):
    w, h = np.shape(image)
    coff = 512/w
    gau = GaussianHighFrequencyFilter(image, 10)
    tran = polarTransform.convertToPolarImage(gau, initialRadius=0, finalRadius=w/2, initialAngle=0 * np.pi, finalAngle=2 * np.pi)
    img = transform.rescale(tran[0], (coff, coff))
    sob = cv2.Sobel(img, cv2.CV_64F, 1, 0)
    feature = GaussianHighFrequencyFilter(sob, 100)
    mid = MedianFilter(feature, 20)
    res = feature - mid
    print("finished extract")
    return res


def ring_artifact(image, radius, intensity, width=1):

    # width = random.randint(1,2)
    w, h = np.shape(image)
    r_e, c_e = draw.circle(w / 2, h / 2, radius)
    r_i, c_i = draw.circle(w / 2, h / 2, radius - width)
    # for i in r_e:
    #     for j in c_e:
    background = np.zeros([w, h])
    background[r_e, c_e] = intensity
    background[r_i, c_i] = 0
    # background = np.ma.masked_not_equal(background, intensity)
    # background = np.asarray(background)
    for i in range(w):
        for j in range(h):
            if background[i, j] == 0:
                background[i, j] = image[i, j]


    # print(background)
    # image[list(set(r_e)^set(r_i)),list(set(c_e)^set(c_i))] = intensity


    # image[r_e, c_e] = intensity
    #
    # # for i in r_i:
    # #     for j in c_i:
    # image[r_i, c_i] = intensity

    return background


def transverse_mean(image, n):
    inputs = copy.copy(image)
    h, w = np.shape(image)
    new_arr = np.zeros((h, w))
    print(h, w)
    for i in range(h):
        # print(np.shape(a[i,:]))
        new_arr[i, :] = np.convolve(image[i, :], np.ones([n]) / n, mode="same")
    return new_arr


def MedianFilter(img, k, padding=None):
    image = copy.copy(img)

    height, width = np.shape(img)
    edge = int((k - 1) / 2)
    if width - 1 - edge <= edge:
        print("The parameter k is too large.")
        return None
    # front = img[:, 0]
    # after = img[:, width - 1]
    #
    # for n in range(edge):
    #     image = np.insert(image, 0, front, axis=1)
    #     image = np.c_[image, after]

    new_arr = np.zeros((height, width))
    # for i in range(edge, height-edge):
    #     for j in range(edge, width-edge):
    #         new_arr[i, j] = np.median(image[i - edge:i + edge + 1, j - edge:j + edge + 1]) # 调用np.median求取中值
    for i in range(height):
        for j in range(width):
        # print(np.shape(a[i,:]))
            new_arr[i, j] = np.median(image[i , j - edge:j + edge + 1])

    where_are_nan = np.isnan(new_arr)
    new_arr[where_are_nan] = 0
    # print(np.shape(new_arr))
    # print(np.shape(new_arr[~np.isnan(new_arr)]))
    # print(type(new_arr))

    return new_arr


def circle(x0, y0, radius):
    f = 1 - radius
    ddf_x = 1
    ddf_y = -2 * radius
    x = 0
    y = radius

    clist=[]

    while x < y:
     if f >= 0:
      y -= 1
      ddf_y += 2
      f += ddf_y
     x += 1
     ddf_x += 2
     f += ddf_x
     clist.append([x0 + x, y0 + y])
     clist.append([x0 - x, y0 + y])
     clist.append([x0 + x, y0 - y])
     clist.append([x0 - x, y0 - y])
     clist.append([x0 + y, y0 + x])
     clist.append([x0 - y, y0 + x])
     clist.append([x0 + y, y0 - x])
     clist.append([x0 - y, y0 - x])
    return clist