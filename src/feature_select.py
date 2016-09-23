#!usr/bin/env python
#-*- coding: utf-8 -*-
import sys
import os
import getopt
from sklearn import cross_validation
import numpy as np
from sklearn import metrics
from sklearn.datasets import load_svmlight_file
from sklearn import metrics 
import math
from sklearn.datasets import load_svmlight_file
from subprocess import *
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier 
from sklearn.ensemble import GradientBoostingClassifier
from sklearn import tree
import randomforest as rf
import GBDT as gb
import CART as ca
import knn as k
import pcorrelation as pcorr
import infortancegain as info

def del_feature_select(info,number):   
    del_n=[]
    for i in range(number):
        n=info.index(min(info))
        del_n.append(n)
        info.remove(min(info))
    return del_n


def del_data(x,del_n):
    n_i=len(x[0])
    n_j=len(x)
    n_d=len(del_n)
    n=len(del_n)
    x_array=[[0]*(n_i) for i in range(n_j)] 
    for i in range(n_i):
            for j in range(n_j):
                x_array[j][i]=x[j][i]
    for i in range(n):
        for j in range(n_j):
            del x_array[j][del_n[i]]
    return x_array

def feature_select(train_x,train_y,test_x,test_y,infogain):
    n_i=len(train_x[0])
    max_precision=0
    max_number=0 
    for i in range(1,n_i):
        info=infogain[:]
        x=train_x[:]
        y=test_x[:]
        del_n=del_feature_select(info,i)
        print del_n
        x_train_array=del_data(x,del_n)
        x_test_array=del_data(y,del_n)
        precision=k.knn_select(x_train_array,train_y,x_test_array,test_y)
        print precision
        if max_precision<precision:
            max_precision=precision
            max_number=i
    return max_number



      
           
           
   
