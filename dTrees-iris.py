#!/usr/bin/python
# -*- coding:utf-8 -*-

# 载入类库
from sklearn.datasets import load_iris
from sklearn import tree
import pydotplus
from IPython.display import Image
import os
os.environ["PATH"] += os.pathsep + 'C:/Program Files (x86)/Graphviz2.38/bin/'

# 载入sciki-learn自带的鸢尾花数据, 进行拟合, 得到决策树模型:
iris = load_iris()
clf = tree.DecisionTreeClassifier()
clf = clf.fit(iris.data, iris.target)

# 将模型存入dot文件iris.dot
with open("iris.dot", 'w') as f:
    f = tree.export_graphviz(clf, out_file=f)
# 第一种方法是用graphviz的dot命令生成决策树的可视化文件, 敲完这个命令后当前目录就可以看到决策树的可视化文件iris.pdf了. 打开就能够看到决策树的模型图.
# 在命令行中执行下面的命令
# dot -Tpdf iris.dot -o iris.pdf

# 第二种方法是用pydotplus直接生成iris.pdf, 这样就不用再命令行去专门生成pdf文件了.
dot_data = tree.export_graphviz(clf, out_file=None)
graph = pydotplus.graph_from_dot_data(dot_data)
graph.write_pdf("iris.pdf")

# 第三种办法是直接把图产生在ipython的notebook中, 这是最常用的做法.
dot_data = tree.export_graphviz(clf, out_file=None,
                                feature_names=iris.feature_names,
                                class_names=iris.target_names,
                                filled=True, rounded=True,
                                special_characters=True)
graph = pydotplus.graph_from_dot_data(dot_data)
Image(graph.create_png())
