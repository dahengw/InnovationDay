import cv2
import numpy as np
import os
from sklearn.svm import LinearSVC
from skimage.feature import hog
import SimpleITK as sitk
from sklearn.svm import SVC
import scipy.io as io
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import StratifiedKFold
import joblib
from feature_extract import *



X = []
Y = []
x_t = []
y_t = []
    # 遍历文件夹，读取图片


def norm(x):
    low = np.min(x)
    up = np.max(x)
    print(low, "-----", up)
    x = (x-low)/(up-low)
    x[x < 0.0] = 0.0
    x[x > 1.0] = 1.0
    return x


for f in os.listdir("./Data_npy/Ring/"):
    img = np.load("./Data_npy/Ring/{}".format(f))
    img = extract(img)
    # np.save("./Data/add_artifact_npy/{}.npy".format(f[0:3]), g)
    hist = np.reshape(img, [-1,1])
    X.append(hist)
    x_t.append(0)

for f in os.listdir("./Data_npy/Normal/"):
    g = np.load("./Data_npy/Normal/{}".format(f))
    img = extract(img)
    hist = np.reshape(g, [-1, 1])
    X.append(hist)
    x_t.append(1)

# for f in os.listdir("./Ring Artifacts/Brain Helical/npy/"):
#     g = np.load("./Ring Artifacts/Brain Helical/npy/{}".format(f))
#     hist = np.reshape(g, [-1,1])
#     X.append(hist)
#     x_t.append(0)


X = np.array(np.squeeze(X))
Y = np.array(x_t)

# for q in os.listdir("./test0/"):
#     # 打开一张图片并灰度化
#     Images = cv2.imread("./test0/{}".format(q))
#     Images = cv2.resize(Images,(256,256))
#     x_t.append(Images)
# x_t = np.array(x_t)
#
# for q in os.listdir("./test9/"):
#     # 打开一张图片并灰度化
#     Images = cv2.imread("./test9/{}".format(q))
#     Images = cv2.resize(Images,(256,256))
#     y_t.append(Images)
# y_t = np.array(y_t)

print(np.shape(X),np.shape(Y))


print("start training")
# 交叉验证
cv = StratifiedKFold(n_splits=5)  # 导入该模型，后面将数据划分多份
classifier = svm.SVC(kernel='rbf', gamma=0.0001, C=1000, probability=True, random_state=0)  # SVC模型 可以换作AdaBoost模型试试

# 画平均ROC曲线的两个参数
mean_tpr = 0.0  # 用来记录画平均ROC曲线的信息
mean_fpr = np.linspace(0, 1, 100)
cnt = 0
for i, (train, test) in enumerate(cv.split(X, Y)):  # 利用模型划分数据集和目标变量 为一一对应的下标
    cnt = cnt + 1
    probas_ = classifier.fit(X[train], Y[train]).predict_proba(X[test])  # 训练模型后预测每条样本得到两种结果的概率
    fpr, tpr, thresholds = roc_curve(Y[test], probas_[:, 1])  # 该函数得到伪正例、真正例、阈值，这里只使用前两个

    mean_tpr += np.interp(mean_fpr, fpr, tpr)  # 插值函数 interp(x坐标,每次x增加距离,y坐标)  累计每次循环的总值后面求平均值
    mean_tpr[0] = 0.0  # 将第一个真正例=0 以0为起点

    roc_auc = auc(fpr, tpr)  # 求auc面积
    plt.plot(fpr, tpr, lw=1, label='ROC fold {0:.2f} (area = {1:.2f})'.format(i, roc_auc))  # 画出当前分割数据的ROC曲线

plt.plot([0, 1], [0, 1], '--', color=(0.6, 0.6, 0.6), label='Luck')  # 画对角线

mean_tpr /= cnt  # 求数组的平均值
mean_tpr[-1] = 1.0  # 坐标最后一个点为（1,1）  以1为终点
mean_auc = auc(mean_fpr, mean_tpr)
print(mean_auc)
plt.plot(mean_fpr, mean_tpr, 'k--', label='Mean ROC (area = {0:.2f})'.format(mean_auc), lw=2)

plt.xlim([-0.05, 1.05])  # 设置x、y轴的上下限，设置宽一点，以免和边缘重合，可以更好的观察图像的整体
plt.ylim([-0.05, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')  # 可以使用中文，但需要导入一些库即字体
plt.title('Receiver operating characteristic example')
plt.legend(loc="lower right")
plt.show()

# 保存模型


os.chdir("model/")
joblib.dump(classifier, "train_model.m")
