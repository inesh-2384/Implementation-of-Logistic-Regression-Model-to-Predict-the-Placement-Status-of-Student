# Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student

## AIM:
To write a program to implement the the Logistic Regression Model to Predict the Placement Status of Student.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the required packages and print the present data.

2.Print the placement data and salary data.

3.Find the null and duplicate values.

4.Using logistic regression find the predicted values of accuracy , confusion matrices.

5.Display the results. 

## Program:
```
/*
Program to implement the the Logistic Regression Model to Predict the Placement Status of Student.
Developed by: inesh.n
RegisterNumber:  2122232220036
## Developed by: CHARUMATHI R
## RegisterNumber: 212222240021

import pandas as pd
data=pd.read_csv("Placement_Data.csv")
data.head()

data1=data.copy()
data1=data1.drop(["sl_no","salary"],axis=1)#Browses the specified row or column
data1.head()

data1.isnull().sum()

data1.duplicated().sum()

from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
data1["gender"]=le.fit_transform(data1["gender"])
data1["ssc_b"]=le.fit_transform(data1["ssc_b"])
data1["hsc_b"]=le.fit_transform(data1["hsc_b"])
data1["hsc_s"]=le.fit_transform(data1["hsc_s"])
data1["degree_t"]=le.fit_transform(data1["degree_t"])
data1["workex"]=le.fit_transform(data1["workex"])
data1["specialisation"]=le.fit_transform(data1["specialisation"] )     
data1["status"]=le.fit_transform(data1["status"])       
data1 

x=data1.iloc[:,:-1]
x

y=data1["status"]
y

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)

from sklearn.linear_model import LogisticRegression
lr=LogisticRegression(solver="liblinear")
lr.fit(x_train,y_train)
y_pred=lr.predict(x_test)
y_pred

from sklearn.metrics import accuracy_score
accuracy=accuracy_score(y_test,y_pred)
accuracy

from sklearn.metrics import confusion_matrix
confusion=confusion_matrix(y_test,y_pred)
confusion

from sklearn.metrics import classification_report
classification_report1 = classification_report(y_test,y_pred)
print(classification_report1)

lr.predict([[1,80,1,90,1,1,90,1,0,85,1,85]])


```

## Output:
##Placement Data:
![image](https://github.com/inesh-2384/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/146412203/b558225b-03f1-4eb2-a1d7-dc84aace4c50)
##Salary Data:
![image](https://github.com/inesh-2384/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/146412203/b85a2e85-dda0-408c-b11c-29760faad93b)
##Checking the null() function:
![image](https://github.com/inesh-2384/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/146412203/99731719-574f-4a93-8c51-41c46aeafe11)
##Data Duplicate:
![image](https://github.com/inesh-2384/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/146412203/80acb648-b5a1-4467-ad1d-215a47e88e0c)
##Print Data:
![image](https://github.com/inesh-2384/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/146412203/1f628a39-227c-4806-a7dd-5e242e67894c)
##Data-Status:
![image](https://github.com/inesh-2384/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/146412203/f1726097-9913-48d2-a79c-89844af2f7c6)
##Y_prediction array:
![image](https://github.com/inesh-2384/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/146412203/694fdfb2-920a-4f1e-9ffa-724af4e7b160)
##Accuracy value:
![image](https://github.com/inesh-2384/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/146412203/cd3171f6-ac2c-4a79-8b74-b94579e71bbf)
##Confusion array:
![image](https://github.com/inesh-2384/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/146412203/66773064-d0c9-424d-acc3-1b8807ff23f6)
##Classification Report:
##Prediction of LR:
![image](https://github.com/inesh-2384/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/146412203/ddea14c9-cdb6-4336-a97e-b9528a444a5f)

## Result:
Thus the program to implement the the Logistic Regression Model to Predict the Placement Status of Student is written and verified using python programming.
