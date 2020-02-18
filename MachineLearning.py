import pandas
import os
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

from sklearn.tree import DecisionTreeClassifier

import numpy

from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

data = pandas.read_csv('data.csv')

#used to examine our data and see column names
data_top = data.head()

features = data[['Time','Source','Destination','Length']]

label = data[['Protocol']]


#the head function is used to make sure that the categorical data has been encoded correctly
features.head()


label.head()




features = pandas.get_dummies(features, columns=['Source', 'Destination'])

label = pandas.get_dummies(label)

#Note x denotes features and y denotes labels as per standard machine learning notation.

# the test size means that 20% of our data will be used as a testing category

# and the other 80% will ne used to train the machine learning algorithm

# the random state integer value is a value used ot generate a random function, 42 is the most
#used value for this area.

x_train, x_test, y_train, y_test = train_test_split(features,label,test_size=0.2, random_state=42)


# the first model will be the tree classifier

model1 = DecisionTreeClassifier()

# the second model is the K-Nearest-Neighbors classifier
model2 = KNeighborsClassifier()


#fitting the tree model with our x_training set and our y_training set
model1.fit(x_train, y_train)

#fitting the K-Nearest Neighbors with our x training set and our y training set
model2.fit(x_train, y_train)


# now predicting both models
predictionTree = model1.predict(x_test)
predictionKNN = model2.predict(x_test)

y_test = y_test.to_numpy()

y_test1 = numpy.argmax(y_test, axis=-1)
predictionTree1 = numpy.argmax(predictionTree, axis=-1)
predictionKNN1 = numpy.argmax(predictionKNN, axis=-1)

print(accuracy_score(predictionTree, y_test))
print(accuracy_score(predictionKNN, y_test))

cmTree = confusion_matrix(y_test1, predictionTree1, labels=[0,1,2,3,4,5,6,7])
cmKNN = confusion_matrix(y_test1, predictionKNN1, labels=[0,1,2,3,4,5,6,7])

print(cmTree)
print(cmKNN)
