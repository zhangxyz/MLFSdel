download from https://github.com/zhangxyz 

system requirements
============
Python 2.7<br>
scikit-learn <br>
NumPy >= 1.6.1<br>
SciPy >= 0.9 and a working C/C++ compiler. For the development version,
you will also require Cython >=0.23.<br>

Installing MLFSdel
============
download MLFSdel-0.1.1.tar.gz from https://github.com/zhangxyz <br>

Running MLFSdel
============
command 'python MLFSdel.py -h' for help <br>

Usage: [options] training_file testing_file <br>
options:<br>
-m model_type: set type of machine learning model<br>
	0 -- randomforest<br>
	1 -- GBDT<br>
	2 -- CART<br>
	3 -- knn<br>
-n feature_number: set number of feature select<br>
	(Arbitrary Integer) -- consider feature_number features at each select <br>
	best -- choose the best select<br>
	null -- no feature select,direct detection<br>
-o out_file: output file<br>

Example
============
example 1: <br>
python MLFSdel.py -m 0 -n 20 -o out.txt training_file testing_file<br>
example 2: <br>
python MLFSdel.py -m 1 -n best -o out.txt training_file testing_file<br>
example 3:<br>
python MLFSdel.py -m 1 -n null -o out.txt training_file testing_file<br>

output	
============
'train.txt':training feature datasets after feature selections<br>
'test.txt':testing feature datasets after feature selections<br>



