
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:, 1:2].values
y = dataset.iloc[:,2].values

from sklearn.linear_model import LinearRegression
lin_reg1 = LinearRegression()
lin_reg1.fit(X, y)

from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree = 4)
X_poly = poly_reg.fit_transform(X)
poly_reg.fit(X_poly, y)
lin_reg2 = LinearRegression()
lin_reg2.fit(X_poly , y)


plt.scatter(X, y, color='red')
plt.plot(X, lin_reg1.predict(X), color='blue')
plt.title('Truth or Bluff (linear reg)')
plt.xlabel('Level')
plt.ylabel('Salary')
plt.show()

plt.scatter(X, y, color='red')
plt.plot(X, lin_reg2.predict(poly_reg.fit_transform(X)), color='blue')
plt.title('Truth or Bluff (polynomial reg)')
plt.xlabel('Level')
plt.ylabel('Salary')
plt.show()

lin_reg2.predict(poly_reg.fit_transform(6.5))

X_grid = np.arange(min(X), max(X), 0.1)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X, y, color='red')
plt.plot(X_grid, lin_reg2.predict(poly_reg.fit_transform(X_grid)), color='blue')
plt.title('Truth or Bluff (polynomial reg)')
plt.xlabel('Level')
plt.ylabel('Salary')
plt.show()