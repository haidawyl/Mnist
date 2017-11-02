#!/usr/bin/python
# -*- coding:utf-8 -*-

from sklearn.decomposition import PCA, IncrementalPCA
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd

if __name__ == "__main__":
    # 读取CCPP数据集, PCA降维
    data = pd.read_excel("data/CCPP/Folds5x2_pp.xlsx")
    # AT:温度, V:压力, AP:湿度, RH:压强, PE:输出电力
    # 样本特征X
    X = data[['AT', 'V', 'AP', 'RH']]
    # 样本输出Y
    Y = data[['PE']]
    # 划分训练集和测试集，将数据集的70%划入训练集，30%划入测试集
    train_X, test_X, train_Y, test_Y = train_test_split(X, Y, test_size=0.3, random_state=1)

    m, n = np.shape(train_X)
    # 先不降维, 只对数据进行投影, 看看投影后各个维度的方差分布
    pca = PCA(n_components=n)
    pca.fit(train_X)
    print '各维度的方差: ', pca.explained_variance_
    print '各维度的方差值占总方差值的比例: ', pca.explained_variance_ratio_, '\n'

    # 降维, 指定降维后的维度数目
    pca = PCA(n_components=2)
    pca.fit(train_X)
    print '各维度的方差: ', pca.explained_variance_
    print '各维度的方差值占总方差值的比例: ', pca.explained_variance_ratio_
    print '降维后的维度数量: ', pca.n_components_, '\n'

    # 降维, 指定主成分的方差和所占的最小比例阈值
    pca = PCA(n_components=0.9)
    pca.fit(train_X)
    print '各维度的方差: ', pca.explained_variance_
    print '各维度的方差值占总方差值的比例: ', pca.explained_variance_ratio_
    print '占总方差值90%的维度数量: ', pca.n_components_, '\n'

    # 降维, 使用MLE算法计算降维后维度数量
    pca = PCA(n_components='mle')
    pca.fit(train_X)
    print '各维度的方差: ', pca.explained_variance_
    print '各维度的方差值占总方差值的比例: ', pca.explained_variance_ratio_
    print '降维后的维度数量: ', pca.n_components_, '\n'

    ipca = IncrementalPCA(n_components=2, batch_size=2)
    ipca.partial_fit(train_X)
    print '各维度的方差: ', ipca.explained_variance_
    print '各维度的方差值占总方差值的比例: ', ipca.explained_variance_ratio_
