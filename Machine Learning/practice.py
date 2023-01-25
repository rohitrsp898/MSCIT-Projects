# import csv
#
# datasets = []
# num_f = 6
# with open("data_find_s_1b.csv", "r") as file:
#     data = csv.reader(file)
#
#     for d in data:
#         datasets.append(d)
#
# hypothesis = ["0"] * num_f
#
# for i in range(0, num_f):
#     hypothesis[i] = datasets[1][i]
#
# for i in range(1, len(datasets)):
#     if datasets[i][-1] == "Yes":
#         for j in range(0, num_f):
#             if hypothesis[j] == datasets[i][j]:
#                 hypothesis[j] = datasets[i][j]
#             else:
#                 hypothesis[j] = "?"
#     print(hypothesis)

# PCA And SC
#
# import pandas as pd
# from sklearn.datasets import load_iris
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import StandardScaler
# from sklearn.decomposition import PCA
#
# data = load_iris()
# print(data.feature_names)
#
# df = pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data",
#                  names=['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)', "class"])
# print(df.head())
#
# # features = df.values[:, :-1]
# # target = df.values[:, -1]
#
# features = df.drop("class", 1)
# target = df["class"]
#
# x_train, x_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=0)
#
# # print(x_train, y_train, x_test, y_test)
# sc = StandardScaler()
# x_train = sc.fit_transform(x_train)
# x_test = sc.transform(x_test)
#
# print(x_train, x_test)
#
# pca = PCA()
# # pca.fit(x_train)
# x_train = pca.fit_transform(x_train)
# x_test = pca.transform(x_test)
#
# print(x_train, x_test)


# import pandas as pd
# from sklearn.linear_model import LogisticRegression
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import accuracy_score
#
# dataset = pd.read_csv("https://raw.githubusercontent.com/plotly/datasets/master/diabetes.csv")
#
# X = dataset.values[:, :8]
# Y = dataset.values[:, -1]
#
# x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=0)
#
# model = LogisticRegression()
# model.fit(x_train, y_train)
# predict = model.predict(x_test)
# print(predict)
# print(accuracy_score(y_test, predict) * 100)


# from sklearn import datasets
# from sklearn.neighbors import KNeighborsClassifier
# from sklearn.model_selection import train_test_split
# import pandas as pd
#
# # data = datasets.load_iris()
# #
# # X = data.data
# # Y = data.target
# url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'
# names = ["sepal-length", "sepal-width", "petal-length", "petal-width", "class"]
# dataset = pd.read_csv(url, names=names)
#
# X = dataset.values[:, :4]
# Y = dataset.values[:, -1]
#
# print(X, Y)
#
# x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=0)
#
# model = KNeighborsClassifier(n_neighbors=5, p=2)
# model.fit(x_train, y_train)
#
# print("accuracy :", model.score(x_train, y_train))
#
# predict = model.predict(x_test)
#
# print(predict)

#
# from math import sqrt
# from sklearn.metrics import confusion_matrix, classification_report
#
#
# def ec(a, b):
#     return sqrt(sum((e1 - e2) for e1, e2 in zip(a, b)))
#
#
# def man(a, b):
#     return sum(abs(e1 - e2) for e1, e2 in zip(a, b))
#
#
# def min(a, b, p):
#     return (sum((e1 - e2) ** p for e1, e2 in zip(a, b))) ** (1 / p)
#
#
# a = [1, 0, 0, 1, 0, 0, 1, 0, 0, 1]
#
# b = [1, 0, 0, 1, 0, 0, 0, 1, 0, 0]
#
# print(ec(a,b))
#
# print(man(a,b))
#
# print(min(a,b,2))
#
# print(confusion_matrix(a,b))
#
# print(classification_report(a,b))

from sklearn.cluster import KMeans
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score
import pandas as pd

# url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'
# names = ["sepal-length", "sepal-width", "petal-length", "petal-width", "class"]
# dataset = pd.read_csv(url, names=names)
datasets = datasets.load_iris()
X = datasets.data
Y = datasets.target

# print(X,Y)

x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=0)
# print(x_train, x_test, y_train, y_test)

for i in range(1,11):
    model = KMeans(n_clusters=4)
    model.fit(x_train, y_train)

p = model.predict(x_test)
print(p)

print(confusion_matrix(y_test, p))
print(accuracy_score(y_test, p) * 100)
