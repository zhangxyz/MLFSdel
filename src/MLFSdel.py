#!/usr/bin/env python
#coding=utf-8

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
import feature_select as fs

if len(sys.argv) <= 3:
    print('Usage: [options] training_file testing_file ')
    print('options:')
    print('-m model_type: set type of machine learning model')
    print('\t0 -- randomforest')
    print('\t1 -- GBDT')
    print('\t2 -- CART')
    print('\t3 -- knn')
    print('-n feature_number: set number of feature select')
    print('\t(Arbitrary Integer) -- consider feature_number features at each select ')
    print('\tbest -- choose the best select')
    print('\tnull -- no feature select,direct detection')
    print('-o out_file: output file')
    raise SystemExit

try:
    opts, args = getopt.getopt(sys.argv[1:], "hm:n:o:")
except getopt.GetoptError:
    sys.exit()
    
if len(opts) == 0:
    usage()
    sys.exit()
    
for opt, arg in opts:
    if opt == '-m':
        model_type=arg
    elif opt == '-n':
        feature_number=arg
    elif opt == '-o':
        out_file=arg
    elif opt == '-h':
        usage()
        sys.exit()
train_file=args[0]
test_file=args[1]

train_X,train_y=load_svmlight_file(train_file)
train_x=train_X.toarray()
test_X,test_y=load_svmlight_file(test_file)
test_x=test_X.toarray()
number=len(train_x[0])
	
corrcoef=pcorr.corrcoef_result(train_x,train_y)
infogain=info.infogain_result(train_x,train_y)


n_i=len(train_x[0])   
if feature_number=='best':
    del_feature=fs.feature_select(train_x,train_y,test_x,test_y,infogain)
    print '\n特征选择后的特征数量:\n'
    remain_feature=n_i-del_feature
    print (remain_feature)
    info_s=infogain[:]
    x_s=train_x[:]
    y_s=test_x[:]
    del_n=fs.del_feature_select(info_s,del_feature)
    x_train_array=fs.del_data(x_s,del_n)
    x_test_array=fs.del_data(y_s,del_n)
    np.savetxt('train.txt',x_train_array,fmt=('%f\t'*remain_feature),newline='\n')
    np.savetxt('test.txt',x_test_array,fmt=('%f\t'*remain_feature),newline='\n')
    if model_type == '0':
        rf.randomforest_result(x_train_array,train_y,x_test_array,test_y,out_file)
    elif model_type == '1':
        gb.gbdt_result(x_train_array,train_y,x_test_array,test_y,out_file)
    elif model_type == '2':
        ca.cart_result(x_train_array,train_y,x_test_array,test_y,out_file)
    elif model_type == '3':
        k.knn_result(x_train_array,train_y,x_test_array,test_y,out_file)
    else:
	    print("Wrong options")  
elif feature_number.isdigit():
    print '\n自定义的特征选择过程:\n'
    del_feature=n_i-int(feature_number)  
    info_s=infogain[:]
    x_s=train_x[:]
    y_s=test_x[:]
    del_n=fs.del_feature_select(info_s,del_feature)
    x_train_array=fs.del_data(x_s,del_n)
    x_test_array=fs.del_data(y_s,del_n)
    if model_type == '0':
        rf.randomforest_result(x_train_array,train_y,x_test_array,test_y,out_file)
    elif model_type == '1':
        gb.gbdt_result(x_train_array,train_y,x_test_array,test_y,out_file)
    elif model_type == '2':
        ca.cart_result(x_train_array,train_y,x_test_array,test_y,out_file)
    elif model_type == '3':
        k.knn_result(x_train_array,train_y,x_test_array,test_y,out_file)
    else:
	    print("Wrong options")          
else:
    print '\n无特征选择:\n'
    if model_type == '0':
        rf.randomforest_result(train_x,train_y,test_x,test_y,out_file)
    elif model_type == '1':
        gb.gbdt_result(train_x,train_y,test_x,test_y,out_file)
    elif model_type == '2':
        ca.cart_result(train_x,train_y,test_x,test_y,out_file)
    elif model_type == '3':
        k.knn_result(train_x,train_y,test_x,test_y,out_file)
    else:
	    print("Wrong options")
 
        
        
        

    
            

    
    
    







