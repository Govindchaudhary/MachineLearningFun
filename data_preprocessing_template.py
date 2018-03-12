import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#importing the datasets

dataset = pd.read_csv('Data.csv')
X = dataset.iloc[:,:-1].values  #taking all the columns except last one and make a matrix
Y = dataset.iloc[:,3].values  # taking only column at index 3 



# splitting the dataset into training set and test set

from sklearn.cross_validation import train_test_split
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=.2,random_state=0)

#feaature scaling(already done by most libraries)

"""from sklearn.preprocessing import StandardScaler
scale_X = StandardScaler()
X_train = scale_X.fit_transform(X_train)
X_test = scale_X.transform(X_test) """


