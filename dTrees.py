#!/usr/bin/python
# -*- coding:utf-8 -*-

from sklearn.decomposition import PCA
from time import time
from sklearn import tree
from sklearn import metrics
import numpy as np
import mnist
import roc
import pydotplus

if __name__ == "__main__":
    # 读取Mnist数据集
    mnistSet = mnist.loadLecunMnistSet()
    train_X, train_Y, test_X, test_Y = mnistSet[0], mnistSet[1], mnistSet[2], mnistSet[3]

    t = time()
    m, n = np.shape(train_X)
    idx = range(m)
    # 分批训练数据时每次拟合的样本数
    num = 30000

    # 使用PCA降维, 看看占样本特征方差90%的特征数目, 可以根据这个数目指定DecisionTreeClassifier类的参数max_features
    # print np.shape(train_X)
    # pca = PCA(n_components=0.9, whiten=True, random_state=0)
    # for i in range(int(np.ceil(1.0*m/num))):
    #     minEnd = min((i+1)*num, m)
    #     sub_idx = idx[i*num:minEnd]
    #     train_pca_X = pca.fit_transform(train_X[sub_idx])
    #     print np.shape(train_pca_X)

    # pca = PCA(n_components=90, whiten=True, svd_solver='randomized', random_state=0)
    # train_X = pca.fit_transform(train_X)
    # test_X = pca.transform(test_X)

    model = tree.DecisionTreeClassifier(splitter='random', max_features=90, max_depth=30, min_samples_split=6, min_samples_leaf=2)
    # 拟合训练集数据
    model.fit(train_X, train_Y)

    dotData = tree.export_graphviz(model, out_file=None)
    graph = pydotplus.graph_from_dot_data(dotData)
    graph.write_pdf("mnist.pdf")

    np.random.shuffle(idx)
    # 预测训练集
    train_Y_hat = model.predict(train_X[idx])
    print '训练集精确度: ', metrics.accuracy_score(train_Y[idx], train_Y_hat)
    # 预测测试集
    test_Y_hat = model.predict(test_X)
    print '测试集精确度: ', metrics.accuracy_score(test_Y, test_Y_hat)
    print "总耗时:", time() - t, "秒"
    # 绘制ROC曲线
    n_class = len(np.unique(train_Y))
    roc.drawROC(n_class, test_Y, test_Y_hat)
