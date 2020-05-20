
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv('Social_Network_Ads.csv')
X = dataset.iloc[:, [2,3]].values
y = dataset.iloc[:, 4].values

from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state= 0)

from sklearn.preprocessing import StandardScaler
sc_X  = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)

from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors = 5)
classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

from matplotlib.colors import ListedColormap
X_set, y_set = X_train, y_train
X1, X2 = np.meshgrid( np.arange(X_set[:, 0].min()-1, X_set[:, 0].max()+1, 0.01 ),
                     np.arange(X_set[:, 1].min()-1, X_set[:, 1].max()+1, 0.01))
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.65, cmap =ListedColormap(('red', 'green')) )
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i,j in enumerate(np.unique(y_set)):
    plt.scatter( X_set[y_set == j, 0], X_set[y_set == j, 1], s = 10,
                c = ListedColormap(('red', 'green'))(i), label=j )
plt.title("KNN (Training Set)")
plt.xlabel("Age")
plt.ylabel("Estimated Salaries")
plt.legend()
plt.show()

from matplotlib.colors import ListedColormap
X_set, y_set = X_test, y_test
X1, X2 = np.meshgrid( np.arange(X_set[:, 0].min()-1, X_set[:, 0].max()+1, 0.01 ),
                     np.arange(X_set[:, 1].min()-1, X_set[:, 1].max()+1, 0.01))
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.65, cmap =ListedColormap(('red', 'green')) )
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i,j in enumerate(np.unique(y_set)):
    plt.scatter( X_set[y_set == j, 0], X_set[y_set == j, 1], s = 10,
                c = ListedColormap(('red', 'green'))(i), label=j )
plt.title("KNN (Test Set)")
plt.xlabel("Age")
plt.ylabel("Estimated Salaries")
plt.legend()
plt.show()



