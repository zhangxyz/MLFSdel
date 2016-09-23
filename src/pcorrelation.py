#!usr/bin/env python
#-*- coding: utf-8 -*-
import numpy as np
from sklearn import feature_selection
from sklearn.datasets import load_svmlight_file
from math import sqrt

train_X,train_y=load_svmlight_file('train.scale')
train_x=train_X.toarray() 

   
def corrcoef(x,y):
    num=np.cov(x,y)
    den=sqrt((np.cov(x))*(np.cov(y)))
    return num/den    
def corrcoef_result(x,y):
    corr=[]
    number_i=len(x[0])  
    for i in range(number_i):
        p=corrcoef([x[i] for x in train_x],train_y)
        corr.append(abs(p[0][1]))
    return corr
    

