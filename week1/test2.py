import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append('/path/to/parent/dir/')

iris = np.genfromtxt("data/iris.txt",delimiter=None) # load the data 
Y = iris[:,-1]
X = iris[:,0:-1]
# Note: indexing with ":" indicates all values (in this case, all rows);
# indexing with a value ("0", "1", "-1", etc.) extracts only that value (here, columns); 
# indexing rows/columns with a range ("1:-1") extracts any row/column in that range.
import mltools as ml
#We'll use some data manipulation routines in the provided class code
numpy.random.seed(0)
X,Y = ml.shuffleData(X,Y); # shuffle data randomly
# (This is a good idea in case your data are ordered in some pathological way, 
# as the Iris data are)
Xtr,Xva,Ytr,Yva = ml.splitData(X,Y, 0.75); # split data into 75/25 train/validation

K=[1,2,5,10,50,100,200] 
errTrain = []
errTest = []
for i, k in enumerate(K) :
    
    counttr = 0
    countte = 0
    learner = ml.knn.knnClassify(Xtr,Ytr,k) # train model
    Yhattr = learner.predict(Xtr) #  predict results for training Y on training data
    for t in range(len(Ytr)):
        if Yhattr[t]!= Ytr[t] : counttr += 1 # count what fraction of training data predictions are wrong 
    errTrain.append(float(counttr)/len(Ytr))
    Yhatte = learner.predict(Xva) #  predict results for validation Y
    for m in range(len(Yva)):
        if Yhatte[m]!=Yva[m] : countte += 1
    errTest.append(float(countte)/len(Yva))
plt.semilogx(K,errTrain,color = 'green',label = 'train error')
plt.legend()
plt.semilogx(K,errTest,color = 'red',label = 'val error')
plt.legend()
plt.show()