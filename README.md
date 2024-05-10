# Implementation-of-SVM-For-Spam-Mail-Detection

## AIM:
To write a program to implement the SVM For Spam Mail Detection.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Moodle-Code Runner

## Algorithm
```
1. Import the required packages.
2. Import the dataset to operate on.
3. Split the dataset.
4. Predict the required output.
5. End the program.
```

## Program:
```
Program to implement the SVM For Spam Mail Detection..
Developed by: subikshan.p
RegisterNumber: 212223240161

import pandas as pd
data=pd.read_csv("spam.csv",encoding='latin-1')
data.head()
data.info()
data.isnull().sum()
x=data["v1"].values
y=data["v2"].values
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)
from sklearn.feature_extractiaon.text import CountVectorizer
cv=CountVectorizer()
x_train=cv.fit_transform(x_train)
x_test=cv.transform(x_test)
from sklearn.svm import SVC
svc=SVC()
svc.fit(x_train,y_train)
y_pred=svc.predict(x_test)
y_pred
from sklearn import metrics
accuracy=metrics.accuracy_score(y_test,y_pred)
accuracy
```

## Output:
### Data Head:
![head](https://user-images.githubusercontent.com/93427208/173846588-8b6564a7-5154-4768-8392-d7e73101f989.png)

### Data Info:
![info](https://user-images.githubusercontent.com/93427208/173846683-02ba0f71-d91f-4b25-bfd0-567c99c1a53f.png)

### Data isnull():
![isnull](https://user-images.githubusercontent.com/93427208/173846738-9177ccfc-1f83-41e6-829f-45970c7ad578.png)

### y_pred:
![ypred](https://user-images.githubusercontent.com/93427208/173846832-fe73c991-28a1-44e1-9397-fc7bdff02b88.png)

### Accuracy:
![accracy](https://user-images.githubusercontent.com/93427208/173846885-378f6ef8-4913-4bd1-a146-7a72d0c72d19.png)


## Result:
Thus the program to implement the SVM For Spam Mail Detection is written and verified using python programming.
