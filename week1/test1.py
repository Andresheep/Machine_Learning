import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append('/path/to/parent/dir/')

iris = np.genfromtxt("data/iris.txt",delimiter=None) # load the data 
Y = iris[:,-1]
X = iris[:,0:2]
# Note: indexing with ":" indicates all values (in this case, all rows);
# indexing with a value ("0", "1", "-1", etc.) extracts only that value (here, columns); # indexing rows/columns with a range ("1:-1") extracts any row/column in that range.
import mltools as ml
#We'll use some data manipulation routines in the provided class code
#Make sure the "mltools" directory is in a directory on your Python path, e.g.,
#export PYTHONPATH=$\$${PYTHONPATH}:/path/to/parent/dir
# or add it to your path inside Python:

X,Y = ml.shuffleData(X,Y); # shuffle data randomly
# (This is a good idea in case your data are ordered in some pathological way, 
# as the Iris data are)
Xtr,Xva,Ytr,Yva = ml.splitData(X,Y, 0.75); # split data into 75/25 train/validation

knn = ml.knn.knnClassify() # create the object and train it
knn.train(Xtr, Ytr, 1) # where K is an integer, e.g. 1 for nearest neighbor prediction
YvaHat = knn.predict(Xva) # get estimates of y for each data point in Xva
ml.plotClassify2D( knn, Xtr, Ytr ); # make 2D classification plot with data (Xtr,Ytr)


