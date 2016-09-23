#!usr/bin/env python
#-*- coding: utf-8 -*-
from sklearn import preprocessing
from sklearn import metrics 
from sklearn import cross_validation
from sklearn.datasets import load_svmlight_file
from sklearn.cross_validation import cross_val_score
from sklearn.cross_validation import train_test_split
import numpy as np
import math
from sklearn.datasets import load_svmlight_file
import sys
import os
from subprocess import *
from sklearn.feature_selection import SelectPercentile, f_classif, SelectKBest
from sklearn.pipeline import Pipeline
from sklearn import tree

def calculate_result(actual,pred):  
    m_precision = metrics.precision_score(actual,pred);  
    m_recall = metrics.recall_score(actual,pred);  
    print 'predict info:'  
    print 'accuracy:{0:.3f}'.format(metrics.accuracy_score(actual,pred))  
    print 'precision:{0:.3f}'.format(m_precision)  
    print 'recall:{0:0.3f}'.format(m_recall);  
    print 'f1-score:{0:.3f}'.format(metrics.f1_score(actual,pred));
       
def cart_result(train_x,train_y,test_x,test_y,out_file):
    print '********************\nCART\n************'
    clf = tree.DecisionTreeClassifier()
    clf.fit(train_x,train_y)
    pred=clf.predict(test_x)
    calculate_result(test_y,pred)
    np.savetxt(out_file,pred,fmt='%d')
    
def cart_select(train_x,train_y,test_x,test_y):
    clf = tree.DecisionTreeClassifier()
    clf.fit(train_x,train_y)
    pred=clf.predict(test_x)
    return metrics.precision_score(test_y,pred)






 
