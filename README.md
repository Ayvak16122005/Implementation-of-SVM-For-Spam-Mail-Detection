#  Implementation-of-SVM-For-Spam-Mail-Detection


## AIM:
To write a program to implement the SVM For Spam Mail Detection.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Character Encoding Detection:

* You’ve used the chardet library to detect the character encoding of the file “spam.csv”.

* The detected encoding is likely to be Windows-1252 (also known as cp1252).
Data Loading and Exploration:

* You’ve loaded the dataset from “spam.csv” using pandas and specified the encoding as Windows-1252.

* You’ve printed the first five rows of the dataset using data.head().

* Additionally, you’ve displayed information about the dataset using data.info().

2. Data Preprocessing:

* You’ve split the data into training and testing sets using train_test_split.

* You’ve used CountVectorizer to convert text data (in column “v2”) into numerical features for SVM training.

3. Model Training and Prediction:

* You’ve initialized an SVM classifier (svc) and trained it on the training data.

* You’ve predicted the labels for the test data using y_pred.

4. Model Evaluation:

* You’ve calculated the accuracy of the model using metrics.accuracy_score.

## Program:
```python

Program to implement the SVM For Spam Mail Detection..
Developed by: KAVYA T
RegisterNumber:  2305003004

import chardet
file='spam.csv'
with open(file,'rb') as rawdata:
    result=chardet.detect(rawdata.read(100000))
print(result)
import pandas as pd
data=pd.read_csv("spam.csv",encoding='Windows-1252')
print("The First five Data:\n")
print(data.head())
print("\nThe Information:\n")
print(data.info())
print("\nTo count the Number of null values in dataset:\n")
print(data.isnull().sum())
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
print("\nThe Y_prediction\n")
print(y_pred)
from sklearn import metrics
accuracy=metrics.accuracy_score(y_test,y_pred)
print("\nAccuracy:\n")
print(accuracy)

```

## Output:

## Data.head()

![image](https://github.com/22009150/Implementation-of-SVM-For-Spam-Mail-Detection/assets/118708624/fe0b7f7f-83fe-4a2f-92cf-6ff2d9fc10dc)

## Data.info()

![image](https://github.com/22009150/Implementation-of-SVM-For-Spam-Mail-Detection/assets/118708624/12d98c81-9143-4867-bfa8-a21824dd384c)

## Data.innull().sum()

![image](https://github.com/22009150/Implementation-of-SVM-For-Spam-Mail-Detection/assets/118708624/189a779a-8a86-4162-9097-190a187dedd5)

## y_pred:

![image](https://github.com/22009150/Implementation-of-SVM-For-Spam-Mail-Detection/assets/118708624/15ee7319-a90a-419a-925a-630f98ad29cb)

## Accuracy:

![image](https://github.com/22009150/Implementation-of-SVM-For-Spam-Mail-Detection/assets/118708624/b5ab93d7-ca77-4de7-9bf1-5194b0ab4666)



## Result:
Thus the program to implement the SVM For Spam Mail Detection is written and verified using python programming.

