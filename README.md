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
