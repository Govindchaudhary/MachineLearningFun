import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#importing the datasets

dataset = pd.read_csv('50_Startups.csv')
X = dataset.iloc[:,:-1].values  #taking all the columns except last one and make a matrix
Y = dataset.iloc[:,4].values  # taking only column at index 3 

#encoding the ategorical data
#encoding the independent variable

from sklearn.preprocessing import LabelEncoder,OneHotEncoder
labelencoder_X = LabelEncoder()
X[:,3] = labelencoder_X.fit_transform(X[:,3]) #return an array of encoded values starting from 0
onehotencoder = OneHotEncoder(categorical_features=[3])
X = onehotencoder.fit_transform(X).toarray()

#avoiding the dummy variable trap
X = X[:,1:]

# splitting the dataset into training set and test set

from sklearn.cross_validation import train_test_split
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=.2,random_state=0)

#feaature scaling(already done by most libraries)

"""from sklearn.preprocessing import StandardScaler
scale_X = StandardScaler()
X_train = scale_X.fit_transform(X_train)
X_test = scale_X.transform(X_test) """

#fitting multiple linear regressions to the training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train,Y_train) # we are making the model to learn the corelation between X_train and Y_train

#predicting the test set results
Y_pred = regressor.predict(X_test)


#building the optimal model using backward elimination

import statsmodels.formula.api as sm
# this doesn't consider const (b0 + b1*x1+ b2*x2+ b3*x3 +...) ie b0 so we have
#to make it aware about this by(b0*x0 + b1*x1 + b2*x2 +...) x0=1
#for this we add a column of all ones in our dataset.and we want this our ist column


X = np.append(arr = np.ones((50,1)).astype(int),values=X,axis=1)
X_opt = X[:,[0,1,2,3,4,5]]
regressor_OLS = sm.OLS(endog=Y, exog=X_opt).fit()
regressor_OLS.summary()

X_opt = X[:,[0,1,3,4,5]]
regressor_OLS = sm.OLS(endog=Y, exog=X_opt).fit()
regressor_OLS.summary()

X_opt = X[:,[0,3,4,5]]
regressor_OLS = sm.OLS(endog=Y, exog=X_opt).fit()
regressor_OLS.summary()

X_opt = X[:,[0,3,5]]
regressor_OLS = sm.OLS(endog=Y, exog=X_opt).fit()
regressor_OLS.summary()

X_opt = X[:,[0,3]]
regressor_OLS = sm.OLS(endog=Y, exog=X_opt).fit()
regressor_OLS.summary()

