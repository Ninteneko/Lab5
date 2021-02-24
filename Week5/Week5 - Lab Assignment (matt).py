import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

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
X = pd.get_dummies(X, dummy_na=True)
y = iris["species"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

'''
    2-1) Train a KNN classifier for the (k = 5, metric='manhattan'). Did you encounter an error? what is the error? how can we fix it?
    Fix the error and train your model. (Hint: pandas.get_dummies())
    2-2) Test your classifier with the test set and report the results.
    2-3) Print the confusion matrix for the results on the test set. 
'''
# YOUR CODE GOES HERE  

knn = KNeighborsClassifier(n_neighbors=5, metric='manhattan')
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)
print(confusion_matrix(y_test, y_pred))


'''
    3) Test your trained model with the given test set below and report the performance.
'''

X_test = np.asarray([[5 , 1, 0.2 , 5,'red'],[0.9 , 7, 6.2 , 2.1,'red'], [0.9 , 7, 6.2 , 2.1,'pink'] , [1.9 , 4, 5 , 0.1,'purple'], [5.9 , 3.3, 0.2 , 2.7,'blue']])
Y_test = np.asarray(['virginica', 'virginica','virginica', 'versicolor' ,'setosa'])
# YOUR CODE GOES HERE  

#X_test = pd.DataFrame(data=X_test)
#Y_test = pd.DataFrame(data=Y_test)
#X_test = pd.get_dummies(X_test, dummy_na=True)
#print(X_train, X_test)
#Y_pred = knn.predict(X_test)
#print(accuracy_score(Y_test, Y_pred))

'''
    4)  Use DictVectorizer from sklearn.feature_extraction to solve Q2
'''
# YOUR CODE GOES HERE  


'''
    5)  Use OneHotEncoder and LabelEncoder from sklearn.preprocessing to solve Q2
'''
# YOUR CODE GOES HERE  