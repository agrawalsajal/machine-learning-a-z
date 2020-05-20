
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd 

dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, 2].values

from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
sc_y = StandardScaler()
X = sc_X.fit_transform(X)
y = sc_y.fit_transform(y.reshape((len(y), 1)))

from sklearn.svm import SVR
svr = SVR(kernel='rbf')
svr.fit(X, y)

y_pred = svr.predict(sc_X.transform(6.5))
y_pred = sc_y.inverse_transform(y_pred)

plt.scatter(X, y)
plt.plot(X, svr.predict(X))
plt.title('Truth or Bluff (SVR)')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()

X_grid = np.arange(min(X), max(X), 0.1)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X, y)
plt.plot(X, svr.predict(X))
plt.title('Truth or Bluff (SVR)')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()
