#!/usr/bin/python
# -*- coding:utf-8 -*-

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score
from time import time
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import mnist
import roc

if __name__ == "__main__":
    # 读取Mnist数据集, 测试随机森林(Random Forest)的分类模型
    mnistSet = mnist.loadLecunMnistSet()
    train_X, train_Y, test_X, test_Y = mnistSet[0], mnistSet[1], mnistSet[2], mnistSet[3]

    m, n = np.shape(train_X)
    idx = range(m)
    np.random.shuffle(idx)

    # 使用PCA降维
    # num = 30000
    # pca = PCA(n_components=0.9, whiten=True, random_state=0)
    # for i in range(int(np.ceil(1.0 * m / num))):
    #     minEnd = min((i + 1) * num, m)
    #     sub_idx = idx[i * num:minEnd]
    #     train_pca_X = pca.fit_transform(train_X[sub_idx])
    #     print np.shape(train_pca_X)

    print "\n**********测试RandomForestClassifier类**********"
    t = time()
    # param_grid1 = {"n_estimators": range(1000, 2001, 100)}
    # param_grid2 = {'max_depth': range(30, 71, 10), 'min_samples_split': range(4, 9, 2)}
    # param_grid3 = {'min_samples_split': range(4, 9, 2), 'min_samples_leaf': range(3, 12, 2)}
    # model = GridSearchCV(
    #     estimator=RandomForestClassifier(max_features=90, n_estimators=1300, max_depth=30, min_samples_split=6),
    #     param_grid=param_grid3, cv=3)
    # # 拟合训练数据集
    # model.fit(train_X, train_Y)
    # print "最好的参数是:%s, 此时的得分是:%0.2f" % (model.best_params_, model.best_score_)
    model = RandomForestClassifier(max_features=90, n_estimators=1300, max_depth=30, min_samples_split=4,
                                   min_samples_leaf=1)
    # 拟合训练数据集
    model.fit(train_X, train_Y)
    # 预测训练集
    train_Y_hat = model.predict(train_X[idx])
    print "训练集精确度: ", accuracy_score(train_Y[idx], train_Y_hat)
    # 预测测试集
    test_Y_hat = model.predict(test_X)
    print "测试集精确度: ", accuracy_score(test_Y, test_Y_hat)
    print "总耗时:", time() - t, "秒"
    # 绘制ROC曲线
    n_class = len(np.unique(train_Y))
    roc.drawROC(n_class, test_Y, test_Y_hat)

    # 读取CCPP数据集, 测试随机森林(Random Forest)的回归模型
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

    print "\n**********测试RandomForestRegressor类**********"
    t = time()
    # param_grid1 = {"n_estimators": range(700, 1501, 100)}
    # param_grid2 = {'max_depth': range(3, 10, 2), 'min_samples_split': range(4, 9, 2)}
    # param_grid3 = {'min_samples_split': (4, 9, 2), 'min_samples_leaf': range(2, 11, 2)}
    # model = GridSearchCV(
    #     estimator=RandomForestRegressor(oob_score=True, n_estimators=700, max_depth=9, min_samples_split=4, min_samples_leaf=2),
    #     param_grid=param_grid3, cv=5)
    # # 拟合训练数据集
    # model.fit(train_X, train_Y.values.ravel())
    # print model.cv_results_
    # print("最好的参数是:%s, 此时的得分是:%0.2f" % (model.best_params_, model.best_score_))
    model = RandomForestRegressor(oob_score=True, n_estimators=700, max_depth=30, min_samples_split=4,
                                  min_samples_leaf=1)
    # 拟合训练数据集
    model.fit(train_X, train_Y.values.ravel())
    print "袋外分数:", model.oob_score_
    print "系数:", model.coef_
    print "截距:", model.intercept_
    print "训练集R2:", r2_score(train_Y, model.predict(train_X))
    # 预测测试集
    test_Y_pred = model.predict(test_X)
    print "测试集得分:", model.score(test_X, test_Y)
    print "测试集MSE:", mean_squared_error(test_Y, test_Y_pred)
    print "测试集RMSE:", np.sqrt(mean_squared_error(test_Y, test_Y_pred))
    print "测试集R2:", r2_score(test_Y, test_Y_pred)
    print "总耗时:", time() - t, "秒"
