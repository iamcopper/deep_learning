#!/usr/bin/env python

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
import graphviz
from sklearn.metrics import accuracy_score

###########################################
# 加载数据集(玩具数据集，用来练手的helloworld数据集)
iris = load_iris()
# 查看label
print(list(iris.target_names))
# 查看feature
print(iris.feature_names)

###########################################
# 按属性和标签载入数据
X, y = load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=104, train_size=0.7, shuffle=True)
# 初始化模型，可以调整 max_depth 来观察模型的表现，
# 也可以调整 criterion 为 gini 来使用 gini 指数构建决策树
clf = tree.DecisionTreeClassifier()
# 训练决策树模型
clf = clf.fit(X_train, y_train)

###########################################
# 使用graphviz包来展示构建好的决策树
feature_names = ['萼片长度', '萼片宽度', '花瓣长度', '花瓣宽']
target_names = ['山鸢尾', '杂色鸢尾', '维吉尼亚鸢尾']
# 可视化生成的决策树
dot_data = tree.export_graphviz(clf, out_file=None,
                                feature_names=feature_names,
                                class_names=target_names,
                                filled=True, rounded=True,
                                special_characters=True)
graph = graphviz.Source(dot_data)
graph.view()

###########################################
y_test_predict = clf.predict(X_test)
print(accuracy_score(y_test, y_test_predict))