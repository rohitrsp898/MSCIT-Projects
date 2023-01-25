# Least Square

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams["figure.figsize"] = (12.0, 9.0)

data = pd.read_csv("4a_salary_data.csv")
X = data.iloc[:, 0]
Y = data.iloc[:, 1]
plt.scatter(X, Y)
plt.show()

x_mean = np.mean(X)
y_mean = np.mean(Y)
num = 0
den = 0

for i in range(len(X)):
    num += (X[i] - x_mean) * (Y[i] - y_mean)
    den += (X[i] - x_mean) ** 2

m = num / den
c = y_mean - m * x_mean

print(m, c)

y_pred = m * X + c

plt.scatter(X, Y)
plt.plot([min(X), max(X)], [min(y_pred), max(y_pred)], color="red")

plt.show()
