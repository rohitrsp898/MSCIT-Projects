# Scatter plot

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
import seaborn as sns
import pandas as pd

# importing the dataset
dataset = pd.read_csv(
    "https://gist.githubusercontent.com/netj/8836201/raw/6f9306ad21398ea43cba4f7d537619d0e07d5ae3/iris.csv")
dataset.describe()

# Splitting the dataset into the Training set and Test set

X = dataset.iloc[:, [0, 1, 2, 3]].values
Y = dataset.iloc[:, 4].values

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.25, random_state=0)

sc = StandardScaler()

X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Fitting Logistic Regression to the Training set
classifier = LogisticRegression(random_state=0, solver='lbfgs', multi_class="auto")
classifier.fit(X_train, y_train)

# Predicting the Test set results

y_pred = classifier.predict(X_test)  # Predict probabilities

probs_y = classifier.predict_proba(X_test)
cm = confusion_matrix(y_test, y_pred)

print(cm)

# Plot confusion matrix

# confusion matrix sns heatmap

ax = plt.axes()

df_cm = cm

sns.heatmap(df_cm, annot=True, annot_kws={"size": 30}, fmt="d", cmap="Blues", ax=ax)
ax.set_title('Confusion Matrix')

plt.show()

# CLustering

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import scipy.cluster.hierarchy as shc
from sklearn.cluster import AgglomerativeClustering

customer_data = pd.read_csv(
    "https://raw.githubusercontent.com/tirthajyoti/Machine-Learning-with-Python/master/Datasets/Mall_Customers.csv")
customer_data.shape
customer_data.head()

data = customer_data.iloc[:, 3:5].values

plt.figure(figsize=(10, 7))
plt.title("Customers Dendograms")

dend = shc.dendrogram(shc.linkage(data, method='ward'))

cluster = AgglomerativeClustering(n_clusters=5, affinity='euclidean', linkage='ward')
cluster.fit_predict(data)

plt.figure(figsize=(10, 7))
plt.scatter(data[:, 0], data[:, 1], c=cluster.labels_, cmap="rainbow")
plt.show()
