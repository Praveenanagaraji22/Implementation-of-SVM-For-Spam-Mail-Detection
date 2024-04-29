# Implementation-of-SVM-For-Spam-Mail-Detection

## AIM:
To write a program to implement the SVM For Spam Mail Detection.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
Step 1 - Start 
Step 2 - Import the required packages.
Step 3 - Import the dataset to operate on.
Step 4 - Split the dataset.
Step 5 - Predict the required output.
Step 6 - Stop

## Program:
```
/*
Program to implement the SVM For Spam Mail Detection..
Developed by: PRAVEENA N
RegisterNumber: 212222040122
*/

import chardet
file='/content/spam.csv'
with open(file,'rb') as rawdata:
  result = chardet.detect(rawdata.read(100000))
result
import pandas as pd
data=pd.read_csv('/content/spam.csv',encoding='Windows-1252')
data.head()
data.info()
data.isnull().sum()
x=data["v1"].values
y=data["v2"].values
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)
from sklearn.feature_extraction.text import CountVectorizer
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
![image](https://github.com/Praveenanagaraji22/Implementation-of-SVM-For-Spam-Mail-Detection/assets/119393514/cb437bcd-ce4b-4129-93c9-b49a28996ec5)

### data.head():
![image](https://github.com/Praveenanagaraji22/Implementation-of-SVM-For-Spam-Mail-Detection/assets/119393514/7e0730a8-fbc6-4e5e-8c4d-92650538d429)

### data.info():
![image](https://github.com/Praveenanagaraji22/Implementation-of-SVM-For-Spam-Mail-Detection/assets/119393514/bb78c3ba-5528-434b-af73-036fef989126)

### data.isnull.sum():
![image](https://github.com/Praveenanagaraji22/Implementation-of-SVM-For-Spam-Mail-Detection/assets/119393514/daee9789-feba-4a8f-92d9-05ebf503fd5b)

### Y_prediction value:
![image](https://github.com/Praveenanagaraji22/Implementation-of-SVM-For-Spam-Mail-Detection/assets/119393514/14110d77-80cb-4484-b1ce-eb9aab780d46)

### Accuracy value:
![image](https://github.com/Praveenanagaraji22/Implementation-of-SVM-For-Spam-Mail-Detection/assets/119393514/94b8566d-bb61-4010-9b6b-d3716815c1a7)

## Result:
Thus the program to implement the SVM For Spam Mail Detection is written and verified using python programming.
