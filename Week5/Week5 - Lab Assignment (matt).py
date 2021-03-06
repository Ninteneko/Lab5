# Matt Brierley
# Yarelit Mendoza

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.feature_extraction import DictVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
pd.options.mode.chained_assignment = None

'''
Lab 5
'''

######### Part 1 ###########


'''
    1-1) Download the iris-data-3 from Canvas, use pandas.read_csv to load it. This dataset has 5 input features: [sepal_length, sepal_width, petal_length, petal_width, color]
    1-2) Find and drop duplicate samples (use "ID" to find duplicates)
    1-3) Split your data into train(70% of data) and test(30% of data) via random selection
'''
# YOUR CODE GOES HERE  

iris = pd.read_csv("iris-data-3.csv")
iris = iris.drop_duplicates(subset="ID")
X = iris.drop(columns="species")
y = iris["species"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

'''
    2-1) Train a KNN classifier for the (k = 5, metric='manhattan'). Did you encounter an error? what is the error? how can we fix it?
    Fix the error and train your model. (Hint: pandas.get_dummies())
    2-2) Test your classifier with the test set and report the results.
    2-3) Print the confusion matrix for the results on the test set. 
'''
# YOUR CODE GOES HERE  

knn = KNeighborsClassifier(n_neighbors=5, metric='manhattan')
X_train = pd.get_dummies(X_train)
knn.fit(X_train, y_train)

X_test = pd.get_dummies(X_test)
X_test = X_test.reindex(columns=X_train.columns, fill_value=0)

y_pred = knn.predict(X_test)

print(f"Fix: get_dummies\nConfusion Matrix:\n{confusion_matrix(y_test, y_pred)}\n")


'''
    3) Test your trained model with the given test set below and report the performance.
'''

X_test = np.asarray([[5 , 1, 0.2 , 5,'red'],[0.9 , 7, 6.2 , 2.1,'red'], [0.9 , 7, 6.2 , 2.1,'pink'] , [1.9 , 4, 5 , 0.1,'purple'], [5.9 , 3.3, 0.2 , 2.7,'blue']])
Y_test = np.asarray(['virginica', 'virginica','virginica', 'versicolor' ,'setosa'])
# YOUR CODE GOES HERE  

X_test = pd.DataFrame(data=X_test)
Y_test = pd.DataFrame(data=Y_test)
X_test = pd.get_dummies(X_test, columns=[4])
X_test = X_test.reindex(columns=X_train.columns, fill_value=0)
Y_pred = knn.predict(X_test)
print(f"Accuracy: {accuracy_score(Y_test, Y_pred)}\n")

'''
    4)  Use DictVectorizer from sklearn.feature_extraction to solve Q2
'''
# YOUR CODE GOES HERE  

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

X_train_dict = X_train.to_dict(orient='records')
dv = DictVectorizer(sparse=False)
X_train_encoded = dv.fit_transform(X_train_dict)

knn.fit(X_train_encoded, y_train)

X_test_dict = X_test.to_dict(orient='records')
X_test_encoded = dv.transform(X_test_dict)

y_pred = knn.predict(X_test_encoded)

print(f"Fix: DictVectorizer\nConfusion Matrix:\n{confusion_matrix(y_test, y_pred)}\n")

'''
    5)  Use OneHotEncoder and LabelEncoder from sklearn.preprocessing to solve Q2
'''
# YOUR CODE GOES HERE  

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

le = LabelEncoder()
X_train['color'] = le.fit_transform(X_train['color'])
X_test['color'] = le.transform(X_test['color'])

ohe = OneHotEncoder(sparse=False)
transformed = ohe.fit_transform(X_train['color'].to_numpy().reshape(-1,1))
ohe_df = pd.DataFrame(transformed, columns=ohe.get_feature_names())
X_train = pd.concat([X_train.reset_index(drop=True), ohe_df.reset_index(drop=True)], axis=1, join='outer').drop(['color'], axis=1)

transformed = ohe.transform(X_test['color'].to_numpy().reshape(-1,1))
ohe_df = pd.DataFrame(transformed, columns=ohe.get_feature_names())
X_test = pd.concat([X_test.reset_index(drop=True), ohe_df.reset_index(drop=True)], axis=1, join='outer').drop(['color'], axis=1)


knn.fit(X_train, y_train)

y_pred = knn.predict(X_test)

print(f"Fix: LabelEncoder and OneHotEncoder\nConfusion Matrix:\n{confusion_matrix(y_test, y_pred)}\n")