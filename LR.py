#!/usr/bin/python
# -*- coding:utf-8 -*-

from time import time
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn import metrics
import numpy as np
import mnist
import roc

if __name__ == "__main__":
    # 读取Mnist数据集
    mnistSet = mnist.loadLecunMnistSet()
    train_X, train_Y, test_X, test_Y = mnistSet[0], mnistSet[1], mnistSet[2], mnistSet[3]

    t = time()

    model = LogisticRegression(C=0.000001, solver='lbfgs', multi_class='multinomial')
    model.fit(train_X, train_Y)
    train_Y_hat = model.predict(train_X)
    print '训练集精确度: ', metrics.accuracy_score(train_Y, train_Y_hat)
    test_Y_hat = model.predict(test_X)
    print '测试集精确度: ', metrics.accuracy_score(test_Y, test_Y_hat)

    # # 数据集总样本数
    # m, n = np.shape(train_X)
    # # 分批训练数据时每次拟合的样本数
    # num = 10000
    # idx = range(m)
    # model = LogisticRegressionCV(Cs=np.logspace(-7, -3, 5), cv=5, solver='lbfgs', multi_class='multinomial')
    # for i in range(int(np.ceil(1.0*m/num))):
    #     minEnd = min((i+1)*num, m)
    #     sub_idx = idx[i*num:minEnd]
    #     model.fit(train_X[sub_idx], train_Y[sub_idx])
    #     print '最优参数: ', model.C_

    print "总耗时:", time() - t, "秒"

    # 绘制ROC曲线
    n_class = len(np.unique(train_Y))
    roc.drawROC(n_class, test_Y, test_Y_hat)
