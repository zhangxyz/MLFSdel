#!usr/bin/env python
#-*- coding: utf-8 -*-
from sklearn import metrics 
from sklearn import cross_validation
import numpy as np
from sklearn import preprocessing
import math
from sklearn.cross_validation import cross_val_score
from sklearn.datasets import load_svmlight_file
from sklearn.feature_selection import SelectPercentile, f_classif, SelectKBest
from sklearn.pipeline import Pipeline
import sys
import os
from subprocess import *
from sklearn.ensemble import RandomForestClassifier

def calculate_result(actual,pred):  
    m_precision = metrics.precision_score(actual,pred);  
    m_recall = metrics.recall_score(actual,pred);  
    print 'predict info:'  
    print 'accuracy:{0:.3f}'.format(metrics.accuracy_score(actual,pred))  
    print 'precision:{0:.3f}'.format(m_precision)  
    print 'recall:{0:0.3f}'.format(m_recall);  
    print 'f1-score:{0:.3f}'.format(metrics.f1_score(actual,pred));  

def calculate_n(x,y):
    clf = RandomForestClassifier(n_estimators=3)
    max_metric=cross_val_score(clf,x,y,cv=5,scoring='f1').mean()
    n=3
    for i in range(4,100):
      clf = RandomForestClassifier(n_estimators=i)
      metric = cross_val_score(clf,x,y,cv=5,scoring='f1').mean()   
      if max_metric < metric :
        max_metric=metric
        n=i
    return n
def randomforest_result(train_x,train_y,test_x,test_y,out_file): 
    print '********************\nRamdomForest\n******************'   
    #c= calculate_n(train_x,train_y)   
    clf = RandomForestClassifier(n_estimators=99)
    clf.fit(train_x, train_y)
    pred=clf.predict(test_x)
    calculate_result(test_y,pred)
    np.savetxt(out_file,pred,fmt='%d')
    #print(c)
    
def randomforest_select(train_x,train_y,test_x,test_y):  
    #c= calculate_n(train_x,train_y)   
    clf = RandomForestClassifier(n_estimators=99)
    clf.fit(train_x, train_y)
    pred=clf.predict(test_x)
    return metrics.precision_score(test_y,pred)
    #print(c)


