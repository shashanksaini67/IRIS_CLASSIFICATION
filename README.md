import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.neighbors import KNeighborsClassifier

#LOADING DATASET
iris= datasets.load_iris()

features=iris.data
labels=iris.target

print(features[10])
print(labels[10])

#PRINTING DESCRIPTIONS:

print(iris.keys())
print(iris.DESCR)
#here 0=Iris-Setosa, 1=Iris-Versicolour, 2=Iris-Virginica

#TRANING THE CLASSIFIRE:
#USING KNEARESTNEIGHBOR CLASSIFIRE : from sklearn.neighbors import KNeighborsClassifier
clf= KNeighborsClassifier() # created model names as clf 

clf.fit(features,labels)

predict=clf.predict([[2,3,4,5]])

predict

