#!/usr/bin/python
# -*- coding:utf-8 -*-

from sklearn.svm import SVC, NuSVC, LinearSVC, SVR, NuSVR, LinearSVR
from sklearn.model_selection import GridSearchCV, train_test_split, cross_val_predict
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, mean_squared_error
from time import time
import numpy as np
import pandas as pd
import operator
import mnist
import roc

if __name__ == "__main__":
    np.set_printoptions(linewidth=200, edgeitems=10)

    # 读取Mnist数据集, 测试SVM的分类模型
    mnistSet = mnist.loadLecunMnistSet()
    train_X, train_Y, test_X, test_Y = mnistSet[0], mnistSet[1], mnistSet[2], mnistSet[3]

    m, n = np.shape(train_X)
    idx = range(m)

    # 数据归一化
    # for i in range(int(np.ceil(1.0*m/num))):
    #     minEnd = min((i+1)*num, m)
    #     sub_idx = idx[i*num:minEnd]
    #     train_X[sub_idx] = StandardScaler().fit_transform(train_X[sub_idx])
    # test_X = StandardScaler().fit_transform(test_X)

    np.random.shuffle(idx)

    print "\n**********测试LinearSVC类**********"
    t = time()
    # model = GridSearchCV(LinearSVC(), param_grid={"C": np.logspace(-10, 0, 11)}, cv=5)
    # model.fit(train_X, train_Y)
    # print("最好的参数是:%s, 此时的得分是:%0.2f" % (model.best_params_, model.best_score_))

    model = LinearSVC(C=0.000001)
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
    # n_class = len(np.unique(train_Y))
    # roc.drawROC(n_class, test_Y, test_Y_hat)

    print "\n**********测试SVC类**********"
    t = time()
    # 分批处理时每次的样本数
    num = 10000
    # 分批拟合训练数据集
    paramsCount = {}
    model = GridSearchCV(SVC(cache_size=1000),
                         param_grid={"C": np.logspace(-3, 3, 7), "gamma": np.logspace(-10, 0, 11)}, cv=5)
    for i in range(int(np.ceil(1.0 * m / num))):
        minEnd = min((i + 1) * num, m)
        sub_idx = idx[i * num:minEnd]
        model.fit(train_X[sub_idx], train_Y[sub_idx])
        best_params_ = model.best_params_
        print("最好的参数是:%s, 此时的得分是:%0.2f" % (best_params_, model.best_score_))
        key = ""
        for param in best_params_.keys():
            key = key + param + ":" + str(best_params_.get(param)) + ";"
        key = key[:len(key) - 1]
        paramsCount[key] = paramsCount.get(key, 0) + 1
    sortedParamsCount = sorted(paramsCount.iteritems(), key=operator.itemgetter(1), reverse=True)
    bestParams = {}
    for params in sortedParamsCount:
        for item in params[0].split(";"):
            paramVal = item.split(":")
            if not bestParams.has_key(paramVal[0]):
                bestParams[paramVal[0]] = paramVal[1]
    print "最好的参数是:", bestParams

    # model = GridSearchCV(SVC(cache_size=1000),
    #                      param_grid={"C": np.logspace(-3, 3, 7), "gamma": np.logspace(-10, 0, 11)}, cv=5)
    # model.fit(train_X, train_Y)
    # print("最好的参数是:%s, 此时的得分是:%0.2f" % (model.best_params_, model.best_score_))

    model = SVC(C=10, gamma=0.0000001, cache_size=1000)
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
    # roc.drawROC(n_class, test_Y, test_Y_hat)

    print "\n**********测试NuSVC类**********"
    t = time()
    model = GridSearchCV(NuSVC(cache_size=1000), param_grid={"nu": np.linspace(0.1, 1, 10),
                                                             "gamma": np.logspace(-3, 3, 7)}, cv=5)
    model.fit(train_X, train_Y)
    print("最好的参数是:%s, 此时的得分是:%0.2f" % (model.best_params_, model.best_score_))

    model = NuSVC(gamma=0.0000001, cache_size=1000)
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
    # roc.drawROC(n_class, test_Y, test_Y_hat)


    # 读取CCPP数据集, 测试SVM的回归模型
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

    print "\n**********测试LinearSVR类**********"
    t = time()
    model = GridSearchCV(LinearSVR(), param_grid={"C": np.logspace(-3, 3, 7)}, cv=5)
    model.fit(train_X, train_Y.values.ravel())
    print("最好的参数是:%s, 此时的得分是:%0.2f" % (model.best_params_, model.best_score_))

    model = LinearSVR(C=10.0)
    # 拟合训练集
    model.fit(train_X, train_Y.values.ravel())
    # 打印模型的系数
    # print model.intercept_
    # print model.coef_
    # 预测测试集
    test_Y_pred = model.predict(test_X)
    print "测试集MSE:", mean_squared_error(test_Y, test_Y_pred)
    print "测试集RMSE:", np.sqrt(mean_squared_error(test_Y, test_Y_pred))
    print "总耗时:", time() - t, "秒"

    print "\n**********测试SVR类**********"
    t = time()
    model = GridSearchCV(SVR(cache_size=1000), param_grid={"C": np.logspace(-3, 3, 7), "gamma": np.logspace(-3, 3, 7),
                                                           "epsilon": np.logspace(-3, 3, 7)}, cv=5)
    model.fit(train_X, train_Y.values.ravel())
    print("最好的参数是:%s, 此时的得分是:%0.2f" % (model.best_params_, model.best_score_))

    model = SVR(C=100.0, gamma=0.1, cache_size=500)
    # 拟合训练集
    model.fit(train_X, train_Y.values.ravel())
    # 打印模型的系数
    # print model.intercept_
    # print model.dual_coef_
    # 预测测试集
    test_Y_pred = model.predict(test_X)
    print "测试集MSE:", mean_squared_error(test_Y, test_Y_pred)
    print "测试集RMSE:", np.sqrt(mean_squared_error(test_Y, test_Y_pred))
    print "总耗时:", time() - t, "秒"

    print "\n**********测试NuSVR类**********"
    t = time()
    model = GridSearchCV(NuSVR(cache_size=1000), param_grid={"C": np.logspace(-3, 3, 7), "nu": np.linspace(0.1, 1, 10),
                                                             "gamma": np.logspace(-3, 3, 7)}, cv=5)
    model.fit(train_X, train_Y.values.ravel())
    print("最好的参数是:%s, 此时的得分是:%0.2f" % (model.best_params_, model.best_score_))

    model = NuSVR(C=100.0, nu=0.3, gamma=0.1, cache_size=500)
    # 拟合训练集
    model.fit(train_X, train_Y.values.ravel())
    # 打印模型的系数
    # print model.intercept_
    # print model.dual_coef_
    # 预测测试集
    test_Y_pred = model.predict(test_X)
    print "测试集MSE:", mean_squared_error(test_Y, test_Y_pred)
    print "测试集RMSE:", np.sqrt(mean_squared_error(test_Y, test_Y_pred))
    print "总耗时:", time() - t, "秒"
