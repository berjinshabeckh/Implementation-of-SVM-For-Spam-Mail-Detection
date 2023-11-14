# Implementation-of-SVM-For-Spam-Mail-Detection

## AIM:
To write a program to implement the SVM For Spam Mail Detection.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the necessary packages.
2. Read the given csv file and display the few contents of the data.
3. Assign the features for x and y respectively.
4. Split the x and y sets into train and test sets.
5. Convert the Alphabetical data to numeric using CountVectorizer.
6. Predict the number of spam in the data using SVC (C-Support Vector Classification) method of SVM (Support vector machine) in sklearn library.
7. Find the accuracy of the model.

## Program:
```
/*
Program to implement the SVM For Spam Mail Detection..
Developed by:H.Berjin Shabeck
RegisterNumber:  212222240018
*/
```
```
import chardet
file = '/content/spam.csv'
with open(file,'rb') as rawdata:
  result = chardet.detect(rawdata.read(100000))
result

import pandas as pd
data= pd.read_csv("/content/spam.csv",encoding='Windows-1252')

data.head()

data.info()

data.isnull().sum()

x=data["v1"].values

y=data["v2"].values

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)

from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer()

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
![image](https://github.com/R-Guruprasad/Implementation-of-SVM-For-Spam-Mail-Detection/assets/119390308/c1cbcff6-7c3e-49af-b0ce-3bd17ec6b50f)
## data.head()
![image](https://github.com/R-Guruprasad/Implementation-of-SVM-For-Spam-Mail-Detection/assets/119390308/b3f4cbf1-6b18-498d-8774-8aa3afd66e1b)
## data.info()
![image](https://github.com/R-Guruprasad/Implementation-of-SVM-For-Spam-Mail-Detection/assets/119390308/17890907-8243-42cc-9d92-35043106f1e9)
## data.isnull().sum()
![image](https://github.com/R-Guruprasad/Implementation-of-SVM-For-Spam-Mail-Detection/assets/119390308/63ef50fa-b7a9-41d8-bd23-aa3ad0875cf0)
## Y_prediction value
![image](https://github.com/R-Guruprasad/Implementation-of-SVM-For-Spam-Mail-Detection/assets/119390308/328dfa13-c1c6-45b8-9d39-f9cde7e2873e)
## Accuracy value
![image](https://github.com/R-Guruprasad/Implementation-of-SVM-For-Spam-Mail-Detection/assets/119390308/2b1de9ee-131b-4f08-84ed-2eda7f79a693)

## Result:
Thus the program to implement the SVM For Spam Mail Detection is written and verified using python programming.
