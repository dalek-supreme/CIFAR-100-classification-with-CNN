#coding=gbk
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#加载文件
def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='latin1')
    return dict

testdataset=unpickle('C:/Users/Lenovo1/Desktop/cifar-100-python/test')
label = unpickle('C:/Users/Lenovo1/Desktop/cifar-100-python/meta')
traindataset = unpickle('C:/Users/Lenovo1/Desktop/cifar-100-python/train')

#数据重塑
x_train = traindataset['data'].reshape((len(traindataset['data']), 3, 32, 32)).transpose(0, 2, 3, 1)
y_train = traindataset['fine_labels']
z_train = traindataset['coarse_labels']
    
x_test = testdataset['data'].reshape((len(testdataset['data']), 3, 32, 32)).transpose(0, 2, 3, 1)
y_test = testdataset['fine_labels']
z_test = testdataset['coarse_labels']

x_train_rows = x_train.reshape(x_train.shape[0], 32 * 32 * 3)
x_test_rows = x_test.reshape(x_test.shape[0], 32 * 32 * 3)

# 对像素进行缩放
from sklearn.preprocessing import MinMaxScaler
minmax = MinMaxScaler()



x_train_rows = minmax.fit_transform(x_train_rows)
x_test_rows = minmax.fit_transform(x_test_rows)

from sklearn.neighbors import KNeighborsClassifier

# 执行KNN方法，同时通过迭代来探索k值多少才合适
k = [1]
for i in k:
    model = KNeighborsClassifier(n_neighbors=i, algorithm='ball_tree', n_jobs=6)
    model.fit(x_train_rows, z_train)
    preds = model.predict(x_test_rows)
    print('k = %s, Accuracy = %f' % (i, np.mean(z_test==preds)))
