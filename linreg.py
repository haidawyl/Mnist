#!/usr/bin/python
# -*- coding:utf-8 -*-

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_predict
from sklearn.linear_model import LinearRegression, Ridge, RidgeCV, Lasso, LassoCV, LassoLars, LassoLarsCV, LassoLarsIC, \
    ElasticNet, ElasticNetCV, OrthogonalMatchingPursuit, OrthogonalMatchingPursuitCV, MultiTaskLasso, MultiTaskLassoCV, \
    MultiTaskElasticNet, MultiTaskElasticNetCV, BayesianRidge, ARDRegression
from sklearn.metrics import mean_squared_error

if __name__ == "__main__":
    # 读取数据集
    data = pd.read_excel("data/CCPP/Folds5x2_pp.xlsx")
    # AT:温度, V:压力, AP:湿度, RH:压强, PE:输出电力
    # 打印前五行数据
    print "data.head()=\n", data.head()
    # 打印后五行数据
    print "data.tail()=\n", data.tail()
    # 打印数据的维度信息
    print data.shape

    # 样本特征X
    X = data[['AT', 'V', 'AP', 'RH']]
    print "X.head()=\n", X.head()
    print "X.tail()=\n", X.tail()
    print X.shape

    # 样本输出Y
    Y = data[['PE']]
    print "Y.head()=\n", Y.head()
    print "Y.tail()=\n", Y.tail()
    print Y.shape

    # 划分训练集和测试集，将数据集的70%划入训练集，30%划入测试集
    train_X, test_X, train_Y, test_Y = train_test_split(X, Y, test_size=0.3, random_state=1)
    print "train_X.shape =", train_X.shape
    print "test_X.shape =", test_X.shape
    print "train_Y.shape =", train_Y.shape
    print "test_Y.shape =", test_Y.shape

    print "\n**********测试LinearRegression类**********"
    linreg = LinearRegression()
    # 拟合训练集
    linreg.fit(train_X, train_Y)
    # 打印模型的系数
    print linreg.intercept_
    print linreg.coef_

    # 其中的一组输出为
    # [ 458.39877507]
    # [[-1.97137593 -0.23772975  0.05834485 -0.15731748]]
    # 线性模型 Y = theta0 + theta1 * x1 + theta2 * x2 + theta3 * x3 + theta4 * x4
    # PE = 458.39877507 - 1.97137593 * AT - 0.23772975 * V + 0.05834485 * AP - 0.15731748 * RH

    # 对于线性回归模型, 一般使用均方误差(Mean Squared Error,MSE)或者
    # 均方根误差(Root Mean Squared Error,RMSE)在测试集上的表现来评该价模型的好坏.
    test_Y_pred = linreg.predict(test_X)
    print "测试集MSE:", mean_squared_error(test_Y, test_Y_pred)
    print "测试集RMSE:", np.sqrt(mean_squared_error(test_Y, test_Y_pred))

    # 可以通过交叉验证来持续优化模型, 下面采用的是10折交叉验证
    Y_pred = cross_val_predict(linreg, X, Y, cv=10)
    print "10折交叉验证MSE:", mean_squared_error(Y, Y_pred)
    print "10折交叉验证RMSE:", np.sqrt(mean_squared_error(Y, Y_pred))

    mpl.rcParams['font.sans-serif'] = u'SimHei'  # 指定显示字体
    mpl.rcParams['axes.unicode_minus'] = False  # 解决保存图像中负号'-'显示为方块的问题
    plt.figure(figsize=(8, 6), facecolor='w')
    plt.subplot(111)
    plt.scatter(Y, Y_pred)
    plt.plot([Y.min(), Y.max()], [Y.min(), Y.max()], 'k--', lw=4)
    plt.xlabel(u'实际值', fontsize=14)
    plt.ylabel(u'预测值', fontsize=14)
    plt.xlim((410, 500))
    plt.ylim((410, 500))
    plt.grid(b=True, ls=":")
    plt.show()

    print "\n**********测试Ridge类**********"
    # 在初始化Ridge类时, 指定超参数α, 默认值是1.0.
    ridge = Ridge(alpha=1.0)
    # 拟合训练集
    ridge.fit(train_X, train_Y)
    # 打印模型的系数
    print ridge.intercept_
    print ridge.coef_

    # 其中的一组输出为
    # [ 458.39070895]
    # [[-1.97134533 -0.23774135  0.05835247 -0.15731236]]
    # 线性模型 Y = theta0 + theta1 * x1 + theta2 * x2 + theta3 * x3 + theta4 * x4
    # PE = 458.39070895 - 1.97134533 * AT - 0.23774135 * V + 0.05835247 * AP - 0.15731236 * RH

    # 对于线性回归模型, 一般使用均方误差(Mean Squared Error,MSE)或者
    # 均方根误差(Root Mean Squared Error,RMSE)在测试集上的表现来评该价模型的好坏.
    test_Y_pred = ridge.predict(test_X)
    print "测试集MSE:", mean_squared_error(test_Y, test_Y_pred)
    print "测试集RMSE:", np.sqrt(mean_squared_error(test_Y, test_Y_pred))

    print "\n**********测试RidgeCV类**********"
    # 在初始化RidgeCV类时, 提供一组备选的α值, RidgeCV类会帮我们选择一个合适的α值.
    ridgecv = RidgeCV(alphas=[0.01, 0.1, 0.5, 1, 3, 5, 7, 10, 20, 100], cv=5)
    # 拟合训练集
    ridgecv.fit(train_X, train_Y)
    # 打印最优的α值
    print "最优的alpha值: ", ridgecv.alpha_
    # 打印模型的系数
    print ridgecv.intercept_
    print ridgecv.coef_

    # 对于线性回归模型, 一般使用均方误差(Mean Squared Error,MSE)或者
    # 均方根误差(Root Mean Squared Error,RMSE)在测试集上的表现来评该价模型的好坏.
    test_Y_pred = ridgecv.predict(test_X)
    print "测试集MSE:", mean_squared_error(test_Y, test_Y_pred)
    print "测试集RMSE:", np.sqrt(mean_squared_error(test_Y, test_Y_pred))

    # 10x10的矩阵_X, 表示一组有10个样本, 每个样本有10个特征的数据.
    _X = 1. / (np.arange(1, 11) + np.arange(0, 10)[:, np.newaxis])
    # 10x1的向量_Y代表样本的输出.
    _Y = np.ones(10)
    # 超参数α的数目
    n_alphas = 200
    # 所有的超参数都在10的-10次方和10的-2次方之间
    alphas = np.logspace(-10, -2, n_alphas)
    _ridge = Ridge(fit_intercept=False)
    coefs = []
    # 遍历所有的超参数
    for a in alphas:
        # 设置本次循环的超参数alpha
        _ridge.set_params(alpha=a)
        # 针对每一个超参数alpha做ridge回归
        _ridge.fit(_X, _Y)
        # 把每一个超参数alpha对应的theta存下来
        coefs.append(_ridge.coef_)
    # 绘图, 以α为x轴, θ的10个维度为y轴
    ax = plt.gca()
    ax.plot(alphas, coefs)
    # 将alpha的值取对数便于画图
    ax.set_xscale('log')
    # 翻转x轴的大小方向, 让alpha从大到小显示
    ax.set_xlim(ax.get_xlim()[::-1])
    plt.xlabel('alpha')
    plt.ylabel('weights')
    plt.title('Ridge coefficients as a function of the regularization')
    plt.rc('font', **{'sans-serif': 'Arial', 'family': 'sans-serif'})  # 指定plt的显示字体
    plt.axis('tight')
    plt.show()

    print "\n**********测试Lasso类**********"
    # 在初始化Lasso类时, 指定超参数α, 默认值是1.0.
    lasso = Lasso(alpha=1.0)
    # 拟合训练集
    lasso.fit(train_X, train_Y)
    # 打印模型的系数
    print lasso.intercept_
    print lasso.coef_

    # 对于线性回归模型, 一般使用均方误差(Mean Squared Error,MSE)或者
    # 均方根误差(Root Mean Squared Error,RMSE)在测试集上的表现来评该价模型的好坏.
    test_Y_pred = lasso.predict(test_X)
    print "测试集MSE:", mean_squared_error(test_Y, test_Y_pred)
    print "测试集RMSE:", np.sqrt(mean_squared_error(test_Y, test_Y_pred))

    print "\n**********测试LassoCV类**********"
    # 在初始化LassoCV类时, 提供一组备选的α值, LassoCV类会帮我们选择一个合适的α值.
    lassocv = LassoCV(alphas=[0.01, 0.1, 0.5, 1, 3, 5, 7, 10, 20, 100], cv=5)
    # 拟合训练集
    lassocv.fit(train_X, train_Y.values.ravel())
    # 打印最优的α值
    print "最优的alpha值: ", lassocv.alpha_
    # 打印模型的系数
    print lassocv.intercept_
    print lassocv.coef_

    # 对于线性回归模型, 一般使用均方误差(Mean Squared Error,MSE)或者
    # 均方根误差(Root Mean Squared Error,RMSE)在测试集上的表现来评该价模型的好坏.
    test_Y_pred = lassocv.predict(test_X)
    print "测试集MSE:", mean_squared_error(test_Y, test_Y_pred)
    print "测试集RMSE:", np.sqrt(mean_squared_error(test_Y, test_Y_pred))

    print "\n**********测试LassoLars类**********"
    # 在初始化LassoLars类时, 指定超参数α, 默认值是1.0.
    lassoLars = LassoLars(alpha=0.005)
    # 拟合训练集
    lassoLars.fit(train_X, train_Y)
    # 打印模型的系数
    print lassoLars.intercept_
    print lassoLars.coef_

    # 对于线性回归模型, 一般使用均方误差(Mean Squared Error,MSE)或者
    # 均方根误差(Root Mean Squared Error,RMSE)在测试集上的表现来评该价模型的好坏.
    test_Y_pred = lassoLars.predict(test_X)
    print "测试集MSE:", mean_squared_error(test_Y, test_Y_pred)
    print "测试集RMSE:", np.sqrt(mean_squared_error(test_Y, test_Y_pred))

    print "\n**********测试LassoLarsCV类**********"
    lassoLarscv = LassoLarsCV(cv=5)
    # 拟合训练集
    lassoLarscv.fit(train_X, train_Y.values.ravel())
    # 打印模型的系数
    print lassoLarscv.intercept_
    print lassoLarscv.coef_

    # 对于线性回归模型, 一般使用均方误差(Mean Squared Error,MSE)或者
    # 均方根误差(Root Mean Squared Error,RMSE)在测试集上的表现来评该价模型的好坏.
    test_Y_pred = lassoLarscv.predict(test_X)
    print "测试集MSE:", mean_squared_error(test_Y, test_Y_pred)
    print "测试集RMSE:", np.sqrt(mean_squared_error(test_Y, test_Y_pred))

    print "\n**********测试LassoLarsIC类**********"
    lassoLarsIC = LassoLarsIC()
    # lassoLarsIC = LassoLarsIC(criterion='bic')
    # 拟合训练集
    lassoLarsIC.fit(train_X, train_Y.values.ravel())
    # 打印模型的系数
    print lassoLarsIC.intercept_
    print lassoLarsIC.coef_

    # 对于线性回归模型, 一般使用均方误差(Mean Squared Error,MSE)或者
    # 均方根误差(Root Mean Squared Error,RMSE)在测试集上的表现来评该价模型的好坏.
    test_Y_pred = lassoLarsIC.predict(test_X)
    print "测试集MSE:", mean_squared_error(test_Y, test_Y_pred)
    print "测试集RMSE:", np.sqrt(mean_squared_error(test_Y, test_Y_pred))

    print "\n**********测试ElasticNet类**********"
    # 在初始化ElasticNet类时, 指定超参数α和ρ, 默认值分别是1.0和0.5.
    elasticNet = ElasticNet(alpha=1.0, l1_ratio=0.5)
    # 拟合训练集
    elasticNet.fit(train_X, train_Y)
    # 打印模型的系数
    print elasticNet.intercept_
    print elasticNet.coef_

    # 对于线性回归模型, 一般使用均方误差(Mean Squared Error,MSE)或者
    # 均方根误差(Root Mean Squared Error,RMSE)在测试集上的表现来评该价模型的好坏.
    test_Y_pred = elasticNet.predict(test_X)
    print "测试集MSE:", mean_squared_error(test_Y, test_Y_pred)
    print "测试集RMSE:", np.sqrt(mean_squared_error(test_Y, test_Y_pred))

    print "\n**********测试ElasticNetCV类**********"
    # 在初始化ElasticNetCV类时, 提供一组备选的α值, ElasticNetCV类会帮我们选择一个合适的α值.
    elasticNetCV = ElasticNetCV(l1_ratio=0.7, alphas=[0.01, 0.1, 0.5, 1, 3, 5, 7, 10, 20, 100], cv=5)
    # 拟合训练集
    elasticNetCV.fit(train_X, train_Y.values.ravel())
    # 打印最优的α值
    print "最优的alpha值: ", elasticNetCV.alpha_
    # 打印模型的系数
    print elasticNetCV.intercept_
    print elasticNetCV.coef_

    # 对于线性回归模型, 一般使用均方误差(Mean Squared Error,MSE)或者
    # 均方根误差(Root Mean Squared Error,RMSE)在测试集上的表现来评该价模型的好坏.
    test_Y_pred = elasticNetCV.predict(test_X)
    print "测试集MSE:", mean_squared_error(test_Y, test_Y_pred)
    print "测试集RMSE:", np.sqrt(mean_squared_error(test_Y, test_Y_pred))

    print "\n**********测试OrthogonalMatchingPursuit类**********"
    # 在初始化OrthogonalMatchingPursuit类时, 指定参数n_nonzero_coefs, 默认值是None.
    omp = OrthogonalMatchingPursuit(n_nonzero_coefs=3)
    # 拟合训练集
    omp.fit(train_X, train_Y)
    # 打印模型的系数
    print omp.intercept_
    print omp.coef_

    # 对于线性回归模型, 一般使用均方误差(Mean Squared Error,MSE)或者
    # 均方根误差(Root Mean Squared Error,RMSE)在测试集上的表现来评该价模型的好坏.
    test_Y_pred = omp.predict(test_X)
    print "测试集MSE:", mean_squared_error(test_Y, test_Y_pred)
    print "测试集RMSE:", np.sqrt(mean_squared_error(test_Y, test_Y_pred))

    print "\n**********测试OrthogonalMatchingPursuitCV类**********"
    ompCV = OrthogonalMatchingPursuitCV(cv=5)
    # 拟合训练集
    ompCV.fit(train_X, train_Y.values.ravel())
    # 打印最好的n_nonzero_coefs值
    print "最好的n_nonzero_coefs值: ", ompCV.n_nonzero_coefs_
    # 打印模型的系数
    print ompCV.intercept_
    print ompCV.coef_

    # 对于线性回归模型, 一般使用均方误差(Mean Squared Error,MSE)或者
    # 均方根误差(Root Mean Squared Error,RMSE)在测试集上的表现来评该价模型的好坏.
    test_Y_pred = ompCV.predict(test_X)
    print "测试集MSE:", mean_squared_error(test_Y, test_Y_pred)
    print "测试集RMSE:", np.sqrt(mean_squared_error(test_Y, test_Y_pred))

    print "\n**********测试MultiTaskLasso类**********"
    # 在初始化MultiTaskLasso类时, 指定参数alpha, 默认值是1.0.
    multiTaskLasso = MultiTaskLasso(alpha=1.0)
    # 拟合训练集
    multiTaskLasso.fit(train_X, train_Y)
    # 打印模型的系数
    print multiTaskLasso.intercept_
    print multiTaskLasso.coef_

    # 对于线性回归模型, 一般使用均方误差(Mean Squared Error,MSE)或者
    # 均方根误差(Root Mean Squared Error,RMSE)在测试集上的表现来评该价模型的好坏.
    test_Y_pred = multiTaskLasso.predict(test_X)
    print "测试集MSE:", mean_squared_error(test_Y, test_Y_pred)
    print "测试集RMSE:", np.sqrt(mean_squared_error(test_Y, test_Y_pred))

    print "\n**********测试MultiTaskLassoCV类**********"
    # 在初始化MultiTaskLassoCV类时, 提供一组备选的α值, MultiTaskLassoCV类会帮我们选择一个合适的α值.
    multiTaskLassoCV = MultiTaskLassoCV(alphas=[0.01, 0.1, 0.5, 1, 3, 5, 7, 10, 20, 100], cv=5)
    # 拟合训练集
    multiTaskLassoCV.fit(train_X, train_Y)
    # 打印最优的α值
    print "最优的alpha值: ", multiTaskLassoCV.alpha_
    # 打印模型的系数
    print multiTaskLassoCV.intercept_
    print multiTaskLassoCV.coef_

    # 对于线性回归模型, 一般使用均方误差(Mean Squared Error,MSE)或者
    # 均方根误差(Root Mean Squared Error,RMSE)在测试集上的表现来评该价模型的好坏.
    test_Y_pred = multiTaskLassoCV.predict(test_X)
    print "测试集MSE:", mean_squared_error(test_Y, test_Y_pred)
    print "测试集RMSE:", np.sqrt(mean_squared_error(test_Y, test_Y_pred))

    print "\n**********测试MultiTaskElasticNet类**********"
    # 在初始化MultiTaskElasticNet类时, 指定超参数α和ρ, 默认值分别是1.0和0.5.
    multiTaskElasticNet = MultiTaskElasticNet(alpha=0.01, l1_ratio=0.7)
    # 拟合训练集
    multiTaskElasticNet.fit(train_X, train_Y)
    # 打印模型的系数
    print multiTaskElasticNet.intercept_
    print multiTaskElasticNet.coef_

    # 对于线性回归模型, 一般使用均方误差(Mean Squared Error,MSE)或者
    # 均方根误差(Root Mean Squared Error,RMSE)在测试集上的表现来评该价模型的好坏.
    test_Y_pred = multiTaskElasticNet.predict(test_X)
    print "测试集MSE:", mean_squared_error(test_Y, test_Y_pred)
    print "测试集RMSE:", np.sqrt(mean_squared_error(test_Y, test_Y_pred))

    print "\n**********测试MultiTaskElasticNetCV类**********"
    # 在初始化MultiTaskElasticNetCV类时, 提供一组备选的α值, MultiTaskElasticNetCV类会帮我们选择一个合适的α值.
    multiTaskElasticNetCV = MultiTaskElasticNetCV(alphas=[0.01, 0.1, 0.5, 1, 3, 5, 7, 10, 20, 100], cv=5)
    # 拟合训练集
    multiTaskElasticNetCV.fit(train_X, train_Y)
    # 打印最优的α值
    print "最优的alpha值: ", multiTaskElasticNetCV.alpha_
    # 打印模型的系数
    print multiTaskElasticNetCV.intercept_
    print multiTaskElasticNetCV.coef_

    # 对于线性回归模型, 一般使用均方误差(Mean Squared Error,MSE)或者
    # 均方根误差(Root Mean Squared Error,RMSE)在测试集上的表现来评该价模型的好坏.
    test_Y_pred = multiTaskElasticNetCV.predict(test_X)
    print "测试集MSE:", mean_squared_error(test_Y, test_Y_pred)
    print "测试集RMSE:", np.sqrt(mean_squared_error(test_Y, test_Y_pred))

    print "\n**********测试BayesianRidge类**********"
    bayesianRidge = BayesianRidge()
    # 拟合训练集
    bayesianRidge.fit(train_X, train_Y.values.ravel())
    # 打印模型的系数
    print bayesianRidge.intercept_
    print bayesianRidge.coef_

    # 对于线性回归模型, 一般使用均方误差(Mean Squared Error,MSE)或者
    # 均方根误差(Root Mean Squared Error,RMSE)在测试集上的表现来评该价模型的好坏.
    test_Y_pred = bayesianRidge.predict(test_X)
    print "测试集MSE:", mean_squared_error(test_Y, test_Y_pred)
    print "测试集RMSE:", np.sqrt(mean_squared_error(test_Y, test_Y_pred))

    print "\n**********测试ARDRegression类**********"
    ardRegression = ARDRegression()
    # 拟合训练集
    ardRegression.fit(train_X, train_Y.values.ravel())
    # 打印模型的系数
    print ardRegression.intercept_
    print ardRegression.coef_

    # 对于线性回归模型, 一般使用均方误差(Mean Squared Error,MSE)或者
    # 均方根误差(Root Mean Squared Error,RMSE)在测试集上的表现来评该价模型的好坏.
    test_Y_pred = ardRegression.predict(test_X)
    print "测试集MSE:", mean_squared_error(test_Y, test_Y_pred)
    print "测试集RMSE:", np.sqrt(mean_squared_error(test_Y, test_Y_pred))
