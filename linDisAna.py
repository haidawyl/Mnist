#!/use/bin/python
# -*- coding:utf-8 -*-

from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import numpy as np
import mnist
'''
线性判别分析实例
'''

if __name__ == "__main__":
    np.set_printoptions(linewidth=200)
    # 读取Mnist数据集
    mnistSet = mnist.loadLecunMnistSet()
    train_X, train_Y, test_X, test_Y = mnistSet[0], mnistSet[1], mnistSet[2], mnistSet[3]

    m, n = np.shape(train_X)
    idx = range(m)
    np.random.shuffle(idx)

    num = 30000
    # 使用PCA降维
    pca = PCA(n_components=0.9, whiten=True, random_state=0)
    for i in range(int(np.ceil(1.0 * m / num))):
        minEnd = min((i + 1) * num, m)
        sub_idx = idx[i * num:minEnd]
        train_pca_X = pca.fit_transform(train_X[sub_idx])
        print np.shape(train_pca_X)

    num = 10000
    # 类别数量
    n_class = len(np.unique(train_Y))
    for n in range(2, n_class - 1):
        print "n = ", n
        # 使用LDA降维
        lda = LinearDiscriminantAnalysis(n_components=n)
        for i in range(int(np.ceil(1.0 * m / num))):
            minEnd = min((i + 1) * num, m)
            sub_idx = idx[i * num:minEnd]
            lda.fit(train_X[sub_idx], train_Y[sub_idx])
            train_lda_X = lda.transform(train_X[sub_idx])
            # print "截距: ", lda.intercept_
            # print "系数: ", lda.coef_
            print "各维度的方差值占总方差值的比例: ", lda.explained_variance_ratio_, "各维度的方差值之和占总方差值的比例: ", np.sum(
                lda.explained_variance_ratio_)
            print "类别权重: ", lda.priors_, "类别权重之和: ", np.sum(lda.priors_)
