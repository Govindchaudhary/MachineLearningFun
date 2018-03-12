import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#importing the datasets

dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:,1:2].values  #taking columns between 1:2 excluding 2 and make a matrix
Y = dataset.iloc[:,2].values  # taking only column at index 2 



# splitting the dataset into training set and test set

"""from sklearn.cross_validation import train_test_split
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=.2,random_state=0)"""

#feaature scaling(already done by most libraries)

"""from sklearn.preprocessing import StandardScaler
scale_X = StandardScaler()
X_train = scale_X.fit_transform(X_train)
X_test = scale_X.transform(X_test) """

#fitting linear regression model to the dataset

from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(X,Y)

#fitting the polynomial regression model to the dataset

from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree=4)
X_poly = poly_reg.fit_transform(X)
lin_reg2 = LinearRegression()
lin_reg2.fit(X_poly,Y)

#visuliazing the linear

plt.scatter(X,Y,color='red')
plt.plot(X,lin_reg.predict(X),color='blue')
plt.title('Truth or bluff(linear)')
plt.xlabel('salary')
plt.ylabel('level')
plt.show()

#visuliazing the polynomial

plt.scatter(X,Y,color='red')
plt.plot(X,lin_reg2.predict(poly_reg.fit_transform(X)),color='blue')
plt.title('Truth or bluff(polynomail)')
plt.xlabel('salary')
plt.ylabel('level')
plt.show()

#predicting the salary at 6.5 level using linear regression

lin_reg.predict(6.5)

#predicting the salery at 6.5 using polynomial
lin_reg2.predict(poly_reg.fit_transform(6.5))


