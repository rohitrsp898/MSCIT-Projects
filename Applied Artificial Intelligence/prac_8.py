"""
Practical : 8
Aim : Write an application to implement Clustering
algorithm.
"""

from numpy import where
from sklearn.datasets import make_classification
from matplotlib import pyplot

x, y = make_classification(n_samples=1000, n_features=2, n_redundant=0,
                           n_clusters_per_class=1, random_state=4)

for class_value in range(2):
    row_ix = where(y == class_value)
    pyplot.scatter(x[row_ix, 0], x[row_ix, 1])

pyplot.show()