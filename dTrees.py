#!/usr/bin/python
# -*- coding:utf-8 -*-

from sklearn.decomposition import PCA
from time import time
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor, export_graphviz
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, mean_squared_error
from sklearn.model_selection import GridSearchCV, train_test_split
import numpy as np
import pandas as pd
import mnist
import roc
import pydotplus

if __name__ == "__main__":
    # 读取Mnist数据集
    mnistSet = mnist.loadLecunMnistSet()
    train_X, train_Y, test_X, test_Y = mnistSet[0], mnistSet[1], mnistSet[2], mnistSet[3]

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

    print "\n**********测试DecisionTreeClassifier类**********"
    t = time()
    # param_grid1 = {'max_depth': range(30, 71, 10), 'min_samples_split': range(4, 9, 2)}
    # param_grid2 = {'min_samples_split': range(4, 9, 2), 'min_samples_leaf': range(3, 12, 2)}
    # model = GridSearchCV(DecisionTreeClassifier(splitter='random', max_features=90, max_depth=50, min_samples_split=6),
    #                      param_grid=param_grid2, cv=3)
    # # 拟合训练数据集
    # model.fit(train_X, train_Y)
    # print "最好的参数是:%s, 此时的得分是:%0.2f" % (model.best_params_, model.best_score_)
    model = DecisionTreeClassifier(splitter='random', max_features=90, max_depth=50, min_samples_split=6,
                                   min_samples_leaf=2)
    # 拟合训练集数据
    model.fit(train_X, train_Y)

    # dotData = export_graphviz(model, out_file=None)
    # graph = pydotplus.graph_from_dot_data(dotData)
    # graph.write_pdf("mnist.pdf")

    np.random.shuffle(idx)
    # 预测训练集
    train_Y_hat = model.predict(train_X[idx])
    print '训练集精确度: ', accuracy_score(train_Y[idx], train_Y_hat)
    # 预测测试集
    test_Y_hat = model.predict(test_X)
    print '测试集精确度: ', accuracy_score(test_Y, test_Y_hat)
    print "总耗时:", time() - t, "秒"
    # 绘制ROC曲线
    n_class = len(np.unique(train_Y))
    roc.drawROC(n_class, test_Y, test_Y_hat)

    # 读取CCPP数据集, 测试AdaBoost的回归模型
    data = pd.read_excel("data/CCPP/Folds5x2_pp.xlsx")
    # AT:温度, V:压力, AP:湿度, RH:压强, PE:输出电力
    # 样本特征X
    X = data[['AT', 'V', 'AP', 'RH']]
    # 数据归一化
    X = StandardScaler().fit_transform(X)
    # 样本输出Y
    Y = data[['PE']]
    # 划分训练集和测试集，将数据集的70%划入训练集，30%划入测试集
    train_X, test_X, train_Y, test_Y = train_test_split(X, Y, test_size=0.3, random_state=1)

    m, n = np.shape(train_X)
    idx = range(m)
    np.random.shuffle(idx)

    print "\n**********测试DecisionTreeRegressor类**********"
    t = time()
    # param_grid1 = {'max_depth': range(10, 31, 5), 'min_samples_split': range(3, 12, 2)}
    # param_grid2 = {'min_samples_split': range(3, 12, 2), 'min_samples_leaf': range(2, 6, 1)}
    # model = GridSearchCV(DecisionTreeRegressor(splitter='random', max_depth=20, min_samples_split=5),
    #                      param_grid=param_grid2, cv=5)
    # # 拟合训练数据集
    # model.fit(train_X, train_Y.values.ravel())
    # print("最好的参数是:%s, 此时的得分是:%0.2f" % (model.best_params_, model.best_score_))
    model = DecisionTreeRegressor(splitter='random', max_depth=30, min_samples_split=5, min_samples_leaf=2)
    # 拟合训练数据集
    model.fit(train_X, train_Y.values.ravel())
    # 预测测试集
    test_Y_pred = model.predict(test_X)
    print "测试集得分:", model.score(test_X, test_Y)
    print "测试集MSE:", mean_squared_error(test_Y, test_Y_pred)
    print "测试集RMSE:", np.sqrt(mean_squared_error(test_Y, test_Y_pred))
    print "总耗时:", time() - t, "秒"
