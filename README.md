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



