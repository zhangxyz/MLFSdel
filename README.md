download from https://github.com/zhangxyz 

system requirements
============
Python 2.7
scikit-learn 
NumPy >= 1.6.1
SciPy >= 0.9 and a working C/C++ compiler. For the development version,
you will also require Cython >=0.23.

Installing MLFSdel
============
download MLFSdel-0.1.1.tar.gz from https://github.com/zhangxyz 

Running MLFSdel
============
'python MLFSdel.py -h' for help

Usage: [options] training_file testing_file 
options:
-m model_type: set type of machine learning model
	0 -- randomforest
	1 -- GBDT
	2 -- CART
	3 -- knn
-n feature_number: set number of feature select
	(Arbitrary Integer) -- consider feature_number features at each select 
	best -- choose the best select
	null -- no feature select,direct detection
-o out_file: output file

Example
============
example 1: 
python MLFSdel.py -m 0 -n 20 -o out.txt training_file testing_file
example 2: 
python MLFSdel.py -m 1 -n best -o out.txt training_file testing_file
example 3:
python MLFSdel.py -m 1 -n null -o out.txt training_file testing_file

output	
============
'train.txt':training feature datasets after feature selections
'test.txt':testing feature datasets after feature selections



