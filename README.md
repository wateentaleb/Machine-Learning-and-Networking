# Machine Learning and Networking
In this project, I will create a network topology consisting of three hosts and one switch in Mininet, as follows:

<img width="635" alt="architecture" src="https://user-images.githubusercontent.com/16707828/74704249-a371fc00-51dd-11ea-938c-17d55ca95849.png">

All links in the network will have a bandwidth of 1 Mbps and a propagation delay of 50 ms.

## Objective 
The objective of this project is to generate a network trace dataset and apply machine learning algorithms to perform protocol classification. Through this project I will learn how machine learning and data science can be leveraged and applied to networking problems, and I will be introduced to basic machine learning workflows including data acquisition, pre-processing, model building, and evaluation.


## Protocol Classification Based on Captured Traces

I will implement two machine learning models to attempt and classify the various packets captured by Wireshark during Part 1 of this lab. Machine Learning is a branch of data science which has received praise in industry and academia alike for its ability to extract meaningful information and relationships from data. With the imminent implementation of 5G networks, machine learning will be leveraged to provide intelligence and automation to networking systems. The following section will be completed in Python and will require the use of the pandas, os, sklearn, and numpy libraries.

The first section of this implementation requires the importing of all the necessary libraries. The code begins with the following declerations: 

`````````````
import pandas
import os
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

import numpy
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
`````````````
Since we are in the same working directory as the .csv file, we can simply import it using the Pandas library. Pandas is a very efficient and useful library used by data scientists around the world. Import the data contained in the .csv folder and save it to a variable called data.

`````````````
data = pandas.read_csv('data.csv')
`````````````

In machine learning classification we use a set of features to predict an output label. To do this, we will split the data accordingly. The features we will use will be time, source IP, destination IP, and packet length. The label we are trying to predict is the protocol used. To split the data we will isolate the columns containing the information we are looking for and we will save them to new variables.

<img width="1092" alt="Screen Shot 2020-02-17 at 11 41 13 PM" src="https://user-images.githubusercontent.com/16707828/74704609-00ba7d00-51df-11ea-88d3-8cf78da00bbf.png">

`````````````
features = data[['Time','Source','Destination','Length']]

label = data[['Protocol']]
`````````````

As seen above, we have various different data types (int, string, etc.) Additionally, some of the data is categorical meaning that there are specific categories present (ie. Source IP, destination IP, Protocol). In order to create data which can be interpreted by our models we will convert the categorical data into One Hot Encoded data. Pandas allows us to do this very effectively using the get_dummies() function.

`````````````
features = data[['Time','Source','Destination','Length']]

label = data[['Protocol']]
`````````````

In order to train a machine learning model and evaluate its performance, we will need to split data into a training and testing set. The model will be trained using the features and labels of the training set, whereby it will attempt to establish relationships between the features and labels. Once trained we will use the features of the test set and ask the model to predict the associated labels. By comparing the labels predicted by the model to those of the test set, we will see how accurate our model is. Sklearn offers an excellent function for splitting our data into training and test sets. Note x denotes features and y denotes labels as per standard machine learning notation

`````````````
x_train, x_test, y_train, y_test = train_test_split(features,label,test_size=0.2, random_state=42)
`````````````

## Machine Learning Models Used 

### Decision Tree Classifier 

The decision tree algorithm has a training time faster compared to neural network algorithms, and its time complexity is related to the number of records and number of attributes given in the data. This implementation is chosen due to its capability of dealing with high dimensional data and provide good accuracy. The way that this specific algorithm works is as follows, the first step is to select the best attribute using an ASM known as Attribute Selection Measures to split the records. Secondly, the selected attribute is made as a decision node and then consequently breaks the data set into smaller subsets. This process is repeated recursively for each child until one of the following conditions match:

**1. All tuples belong to the same attribute value**

**2. There are no more remaining attributes**

**3. There are no more instances.**

![decision-tree](https://user-images.githubusercontent.com/16707828/74704967-36ac3100-51e0-11ea-848d-ba783f302b39.png)


### K-Nearest Neighbors 
The k-Nearest Neighbors (kNN) algorithm assumes that instances that in close proximity are similar. In order to find which instances are near each other in proximity, the distance between those instances needs to be calculated. Furthermore, the most widely used method for calculating distance whilst using this algorithm is known as the Euclidean distance, which is also known as the straight-line distance. The k-Nearest Neighbors algorithm works as follows:

1. Load the data of instances

2. Initialze the k variable to the chosen number of neighbors

3. For each instance in the data 
+ Calculate the distance between the query example and the current example from the data.
+ Add the distances and the index of the example to an ordered collection

4. Sorting the ordered collection is required for the distances and indices. The order of the set
is in ascending order, meaning smallest to largest by distances.

Image showing how similar data points typically exist close to each other, 


![knn](https://user-images.githubusercontent.com/16707828/74705151-da95dc80-51e0-11ea-8aae-291125010d09.png)

## Confusion Matrix 

In order to explain a confusion matrix, a simple 2x2 model will be used to explain it's behaviour and function. 

In the 2x2 confusion matrix, they are only two possible predicted cases, “yes” and “no”. If the instance that was examined for instance was that a student would receive a mark of 90% on the networking final, yes would be that the student indeed attains that mark and no otherwise.
In terms of the most basic terms, there are four that need to be analyzed which will be listed and explained below:

+ **True positives (TP):** True positives mean that the instance which was predicted did in fact meet the conclusion. In other words, as known in the area of discrete mathematics, this can be related to implications which are denoted by p -> q, meaning that p (the hypothesis) implies q. In this case, our hypothesis was true, and so was the conclusion. The truth table for implications is shown below

+ **True negatives (TN):** True positives are the case when the hypothesis predicted the result was going to be no, and the conclusion was also no, moreover; this can also be related that the hypothesis implies the conclusion, the truth table will be shown below.

+ **False positives (FP):** false positives are the instances where the prediction was “yes” however the conclusion concluded that the result was “no”.

+ **False negatives (FN):** false negatives are the instance where the prediction was “no” however the conclusion concluded that the result was “yes”

Furthermore, some of the rates that can be calculated will be shown below:

<img width="188" alt="Screen Shot 2020-02-17 at 11 57 45 PM" src="https://user-images.githubusercontent.com/16707828/74705267-509a4380-51e1-11ea-8e1e-fae6f834e424.png">

Accuracy denotes how often the classifier was correct. It’s a useful rate to find if your model is useful.


Miss classification is used to find how many times the classifier was wrong with its predictions.

<img width="227" alt="Screen Shot 2020-02-17 at 11 58 32 PM" src="https://user-images.githubusercontent.com/16707828/74705312-6c054e80-51e1-11ea-8be6-d3ca34cdc4c1.png">




