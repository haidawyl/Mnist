#!/usr/bin/python
# -*- coding:utf-8 -*-

from time import time
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
from sklearn import metrics
import numpy as np
import mnist
import roc

if __name__ == "__main__":
    # 读取Mnist数据集
    mnistSet = mnist.loadLecunMnistSet()
    train_X, train_Y, test_X, test_Y = mnistSet[0], mnistSet[1], mnistSet[2], mnistSet[3]

    # 数据集总样本数
    m = np.shape(train_X)[0]
    # 分批训练数据时每次拟合的样本数
    num = 10000

    print "**********测试先验为高斯分布的朴素贝叶斯**********"
    t = time()
    model = GaussianNB()
    # 整体拟合
    model.fit(train_X, train_Y)
    # 分批拟合
    # for i in range(int(np.ceil(m/num))):
    #     minEnd = min((i+1)*num, m)
    #     model.partial_fit(train_X[i*num:minEnd], train_Y[i*num:minEnd], classes=train_Y)
    idx = range(m)
    np.random.shuffle(idx)
    # 分批预测训练集
    all_train_Y_hat = []
    for i in range(int(np.ceil(m/num))):
        minEnd = min((i+1)*num, m)
        sub_idx = idx[i*num:minEnd]
        train_Y_hat = model.predict(train_X[sub_idx])
        print '训练集子集精确度: ', metrics.accuracy_score(train_Y[sub_idx], train_Y_hat)
        all_train_Y_hat[i*num:minEnd] = train_Y_hat[:]
    print '训练集精确度: ', metrics.accuracy_score(train_Y[idx], all_train_Y_hat)
    # 预测测试集
    test_Y_hat = model.predict(test_X)
    print '测试集精确度: ', metrics.accuracy_score(test_Y, test_Y_hat)
    print "总耗时:", time() - t, "秒"
    n_class = len(np.unique(train_Y))
    # 绘制ROC曲线
    roc.drawROC(n_class, test_Y, test_Y_hat)

    print "\n**********测试先验为多项式分布的朴素贝叶斯**********"
    t = time()
    model = MultinomialNB()
    # 拟合训练集数据
    model.fit(train_X, train_Y)
    idx = range(m)
    np.random.shuffle(idx)
    # 预测训练集
    train_Y_hat = model.predict(train_X[idx])
    print '训练集精确度: ', metrics.accuracy_score(train_Y[idx], train_Y_hat)
    # 预测测试集
    test_Y_hat = model.predict(test_X)
    print '测试集精确度: ', metrics.accuracy_score(test_Y, test_Y_hat)
    print "总耗时:", time() - t, "秒"
    # 绘制ROC曲线
    roc.drawROC(n_class, test_Y, test_Y_hat)

    print "\n**********测试先验为伯努利分布的朴素贝叶斯**********"
    t = time()
    model = BernoulliNB()
    # 拟合训练集数据
    model.fit(train_X, train_Y)
    idx = range(m)
    np.random.shuffle(idx)
    # 预测训练集
    train_Y_hat = model.predict(train_X[idx])
    print '训练集精确度: ', metrics.accuracy_score(train_Y[idx], train_Y_hat)
    # 预测测试集
    test_Y_hat = model.predict(test_X)
    print '测试集精确度: ', metrics.accuracy_score(test_Y, test_Y_hat)
    print "总耗时:", time() - t, "秒"
    # 绘制ROC曲线
    roc.drawROC(n_class, test_Y, test_Y_hat)
