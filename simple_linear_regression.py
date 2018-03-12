import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#importing the datasets

dataset = pd.read_csv('Salary_Data.csv')
X = dataset.iloc[:,:-1].values  #taking all the columns except last one and make a matrix
Y = dataset.iloc[:,1].values  # taking only column at index 3 



# splitting the dataset into training set and test set

from sklearn.cross_validation import train_test_split
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=1/3,random_state=0)

#feaature scaling(already done by most libraries)

"""from sklearn.preprocessing import StandardScaler
scale_X = StandardScaler()
X_train = scale_X.fit_transform(X_train)
X_test = scale_X.transform(X_test) """


#fitting linear regression to the training set

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()  # making regressor as our linear model
regressor.fit(X_train,Y_train) #we are making our model to learn the co-relation between X_train and Y_train

#predicting the test test results

Y_pred = regressor.predict(X_test)

#visulaizing the Training set results

plt.scatter(X_train,Y_train,color='red')
plt.plot(X_train,regressor.predict(X_train),color='blue') #plotting our model
plt.title('Salary vs Experience(Training set)')
plt.xlabel('Experience')
plt.ylabel('Salary')
plt.show()

#visulaizing the test set results

plt.scatter(X_test,Y_test,color='red')
plt.plot(X_train,regressor.predict(X_train),color='blue') #plotting our model
plt.title('Salary vs Experience(Testing set)')
plt.xlabel('Experience')
plt.ylabel('Salary')
plt.show()


