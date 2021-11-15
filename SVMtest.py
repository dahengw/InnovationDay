import cv2
import numpy as np
import SimpleITK as sitk
import os
from feature_extract import *
from skimage.feature import hog
import shutil
import sys
from sklearn.svm import SVC
import scipy.io as io
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import StratifiedKFold
import joblib

X = []
Y = []



    # 遍历文件夹，读取图片
# for f in os.listdir("./test data/HDCT/"):
#     image = sitk.ReadImage('./test data/HDCT/{}'.format(f))
#     # img = np.load('./test data/HDCT/{}'.format(f))
#     # img = np.load('./test data/npy/Axial/{}'.format(f))
#     img = np.array(image)
#     lenth = int(np.sqrt(len(img)))
#     img = np.reshape(img, [lenth, lenth])
#     feat = extract(img)
#     np.save('./test data/npy/HDCT/{}'.format(f), feat)
#     hist = np.squeeze(np.reshape(feat, [-1, 1]))
#     X.append(hist)
#     Y.append(1)

# for f in os.listdir("./test data/art_ring/"):
#     # image = sitk.ReadImage('./test data/art_ring/{}'.format(f))
#     # img = np.load('./test data/HDCT/{}'.format(f))
#     img = np.load('./test data/art_ring/{}'.format(f))
#     # img = np.array(image)
#     # lenth = int(np.sqrt(len(img)))
#     # img = np.reshape(img, [lenth, lenth])
#     feat = extract(img)
#     np.save('./test data/npy/art_ring/{}'.format(f), feat)
#     hist = np.squeeze(np.reshape(feat, [-1, 1]))
#     X.append(hist)
#     Y.append(0)
#
# for f in os.listdir("./test data/Axial/"):
#     image = sitk.ReadImage('./test data/Axial/{}'.format(f))
#     # img = np.load('./test data/HDCT/{}'.format(f))
#     # img = np.load('./test data/npy/Axial/{}'.format(f))
#     img = np.array(image)
#     lenth = int(np.sqrt(len(img)))
#     img = np.reshape(img, [lenth, lenth])
#     feat = extract(img)
#     np.save('./test data/npy/Axial/{}'.format(f), feat)
#     hist = np.squeeze(np.reshape(feat, [-1, 1]))
#     X.append(hist)
#     Y.append(1)

# for f in os.listdir("./test data/npy/abdomen/"):
#     img = np.load("./test data/npy/abdomen/{}".format(f))
#     # g = extract(img)
#     hist = np.squeeze(np.reshape(img, [-1,1]))
#     X.append(hist)
#     Y.append(1)
# #
# for f in os.listdir("./test data/npy/Axial/"):
#     img = np.load("./test data/npy/Axial/{}".format(f))
#     # g = extract(img)
#     hist = np.squeeze(np.reshape(img, [-1,1]))
#     X.append(hist)
#     Y.append(1)

for f in os.listdir("./test data/npy/art_ring/"):
    img = np.load("./test data/npy/art_ring/{}".format(f))
    # g = extract(img)
    hist = np.squeeze(np.reshape(img, [-1,1]))
    X.append(hist)
    Y.append(0)

for f in os.listdir("./test data/npy/HDCT/"):
    img = np.load("./test data/npy/HDCT/{}".format(f))
    # g = extract(img)
    hist = np.squeeze(np.reshape(img, [-1,1]))
    X.append(hist)
    Y.append(1)
#
# for f in os.listdir("./test data/npy/iDOSE_YD/"):
#     img = np.load("./test data/npy/iDOSE_YD/{}".format(f))
#     # g = extract(img)
#     hist = np.squeeze(np.reshape(img, [-1,1]))
#     X.append(hist)
#     Y.append(0)
# #
# for f in os.listdir("./test data/npy/iDOSE_YC/"):
#     img = np.load("./test data/npy/iDOSE_YC/{}".format(f))
#     # g = extract(img)
#     hist = np.squeeze(np.reshape(img, [-1,1]))
#     X.append(hist)
#     Y.append(1)
# #
# for f in os.listdir("./test data/npy/iDOSE_YD/"):
#     img = np.load("./test data/npy/iDOSE_YD/{}".format(f))
#     # g = extract(img)
#     hist = np.squeeze(np.reshape(img, [-1,1]))
#     X.append(hist)
#     Y.append(0)






# for f in os.listdir("./test data/iDOSE_YC/"):
#     image = sitk.ReadImage('./test data/iDOSE_YC/{}'.format(f))
#     # img = np.load('./test data/HDCT/{}'.format(f))
#     # img = np.load('./test data/npy/Axial/{}'.format(f))
#     img = np.array(image)
#     lenth = int(np.sqrt(len(img)))
#     img = np.reshape(img, [lenth, lenth])
#     feat = extract(img)
#     np.save('./test data/npy/iDOSE_YC/{}'.format(f), feat)
#     hist = np.squeeze(np.reshape(feat, [-1, 1]))
#     X.append(hist)
#     Y.append(1)
#
# for f in os.listdir("./test data/iDOSE_YD/"):
#     image = sitk.ReadImage('./test data/iDOSE_YD/{}'.format(f))
#     # img = np.load('./test data/HDCT/{}'.format(f))
#     # img = np.load('./test data/npy/Axial/{}'.format(f))
#     img = np.array(image)
#     lenth = int(np.sqrt(len(img)))
#     img = np.reshape(img, [lenth, lenth])
#     feat = extract(img)
#     np.save('./test data/npy/iDOSE_YD/{}'.format(f), feat)
#     hist = np.squeeze(np.reshape(feat, [-1, 1]))
#     X.append(hist)
#     Y.append(0)

# for f in os.listdir("./test data/npy/Axial/"):
#     g = np.load("./test data/npy/Axial/{}".format(f))
#     hist = np.squeeze(np.reshape(g, [-1,1]))
#     X.append(hist)
#     Y.append(0)

# for f in os.listdir("./HDCT/ring_00020001/"):
#     img = np.load("./HDCT/ring_00020001/{}".format(f))
#     g = extract(img)
#     hist = np.squeeze(np.reshape(g, [-1,1]))
#     X.append(hist)
#     Y.append(0)

# for f in os.listdir("./test data/IAC/"):
#     image = sitk.ReadImage('./test data/IAC/{}'.format(f))
#     img = np.array(image)
#     lenth = int(np.sqrt(len(img)))
#     img = np.reshape(img, [lenth, lenth])
#     hist = extract(img)
#     np.save('./test data/npy/IAC/{}'.format(f), hist)
#     hist = np.squeeze(np.reshape(hist, [-1, 1]))
#     X.append(hist)
#     x_t.append(0)

test_X = np.array(X)
test_Y = np.array(Y)



# test_Y = np.array(Y)
# test_y = np.array(y_t)
np.set_printoptions(suppress=True)


clf = joblib.load("model/model_3_11.2.m")
print(clf.predict_proba(test_X))
print(clf.score(test_X, test_Y))
# print(clf.recall_score(test_X, test_Y))

# print(clf.predict(test_Y))
# print(clf.score(test_Y, test_y))