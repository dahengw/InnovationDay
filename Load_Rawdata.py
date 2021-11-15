import numpy as np
import os
from PIL import Image
import pydicom
import matplotlib.pyplot as plt
import cv2
import SimpleITK as sitk

count = []
num = []

for_num = 0
for file in os.listdir("./Data/Ring/"):
    image = sitk.ReadImage('./Data/Ring/{}'.format(file))
    img = np.array(image)
    length = int(np.sqrt(len(img)))
    img = np.reshape(img, [length, length])
    # res = transform.resize(img, (256,256))
    for_num += 1
    print(for_num)
    print(np.shape(img), np.max(img), np.min(img), np.mean(img))
    # print(np.shape(res),np.max(res), np.min(res), np.mean(res))
    # img = img.astype("uint8")
    # plt.imshow(img, cmap=plt.cm.gray)
    # plt.savefig(str(for_num) + '.npy')
    np.save(str(for_num) + '.npy', img)
    # plt.show()
    # plt.show(block=False)
    # plt.close('all')
    # print(np.shape(count))


    # f = h5py.File("hdct.h5", 'w')
    # f.create_dataset('data', data=count)
    # f.close


# plt.imsave("01.pdf", img)