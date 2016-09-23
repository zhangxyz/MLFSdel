#!usr/bin/env python
#-*- coding: utf-8 -*-
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import load_svmlight_file
import numpy as np
train_X,train_y=load_svmlight_file('train.scale')
train_x=train_X.toarray()
def infogain_result(x,y): 
    infogain_coef=[]
    n=len(x[0])-1
    clf = DecisionTreeClassifier(criterion='entropy')
    clf = clf.fit(train_x, train_y)
    for i in range(0,n):
        infogain_coef.append(clf.feature_importances_[i])
    return infogain_coef

