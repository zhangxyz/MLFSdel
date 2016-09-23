#!usr/bin/env python
#-*- coding: utf-8 -*-
from sklearn import preprocessing
from sklearn import metrics 
from sklearn import cross_validation
from sklearn.datasets import load_svmlight_file
from sklearn.cross_validation import cross_val_score
import numpy as np
import math
from sklearn.datasets import load_svmlight_file
import sys
import os
from subprocess import *
from sklearn.feature_selection import SelectPercentile, f_classif, SelectKBest
from sklearn.pipeline import Pipeline
from sklearn.ensemble import GradientBoostingClassifier

def calculate_result(actual,pred):  
    m_precision = metrics.precision_score(actual,pred);  
    m_recall = metrics.recall_score(actual,pred);  
    print 'predict info:' 
    print 'accuracy:{0:.3f}'.format(metrics.accuracy_score(actual,pred))   
    print 'precision:{0:.3f}'.format(m_precision)  
    print 'recall:{0:0.3f}'.format(m_recall);  
    print 'f1-score:{0:.3f}'.format(metrics.f1_score(actual,pred));
       
def calculate_n(x,y):
    clf = GradientBoostingClassifier(n_estimators=3)
    max_metric=cross_val_score(clf,x,y,cv=10,scoring='f1').mean()
    n=3
    for i in [100,200,300,400,500,600,700,800,900,1000]:
      clf = GradientBoostingClassifier(n_estimators=i)
      metric = cross_val_score(clf,x,y,cv=10,scoring='f1').mean()   
      if max_metric < metric :
        max_metric=metric
        n=i
    return n
    
def gbdt_result(train_x,train_y,test_x,test_y,out_file):
    print '********************\nGBDT\n******************************'
    #c=calculate_n(train_x,train_y)
    clf =  GradientBoostingClassifier(n_estimators=1000)    
    #clf=GradientBoostingClassifier(n_estimators=c)#调整参数n_estimators
    clf.fit(train_x,train_y)
    pred=clf.predict(test_x)
    calculate_result(test_y,pred)
    np.savetxt(out_file,pred,fmt='%d')
    #print(c)
    
    
def gbdt_select(train_x,train_y,test_x,test_y):
    #c=calculate_n(train_x,train_y)
    clf =  GradientBoostingClassifier(n_estimators=1000)    
    #clf=GradientBoostingClassifier(n_estimators=c)#调整参数n_estimators
    clf.fit(train_x,train_y)
    pred=clf.predict(test_x)
    calculate_result(test_y,pred)
    return metrics.precision_score(test_y,pred)
    #print(c)




 
