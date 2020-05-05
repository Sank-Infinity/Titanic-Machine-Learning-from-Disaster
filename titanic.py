# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
      
train = pd.read_csv('train.csv', usecols=["Survived","Age","Pclass","Sex"])

def impute_na(data, variable):
    # function to fill na with a random sample
    df = data.copy()
    
    # random sampling
    df[variable+'_random'] = df[variable]
    
    # extract the random sample to fill the na
    random_sample = df[variable].dropna().sample(df[variable].isnull().sum(), random_state=0)
    
    # pandas needs to have the same index in order to merge datasets
    random_sample.index = df[df[variable].isnull()].index
    df.loc[df[variable].isnull(), variable+'_random'] = random_sample
    
    return df[variable+'_random']


train['Age']=impute_na(train,'Age')

train = pd.get_dummies(train)
df = pd.get_dummies(train["Pclass"])
train = pd.concat([train,df], axis=1)
train = train.drop(["Pclass"], axis=1)

X_train = train.iloc[:, 1:6].values
Y_train = train.iloc[:, [0]].values


from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators= 300, criterion='entropy', random_state=0)
classifier.fit(X_train, Y_train)

y_pred = classifier.predict(X_train)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(Y_train, y_pred)


      
test = pd.read_csv('test.csv', usecols=["Age","Pclass","Sex"])

test['Age']=impute_na(test,'Age')

test = pd.get_dummies(test)
df_1 = pd.get_dummies(test["Pclass"])
test = pd.concat([test,df_1], axis=1)
test = test.drop(["Pclass"], axis=1)

X_test = test.iloc[:, :5].values

"""
Survived = classifier.predict(X_test)

submission = pd.to_csv()
"""










