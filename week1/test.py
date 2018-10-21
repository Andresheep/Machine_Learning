import numpy as np
import matplotlib.pyplot as plt

iris = np.genfromtxt("data/iris.txt",delimiter=None) # load the text file 
Y = iris[:,-1] # target value is the last column
X = iris[:,0:-1] # features are the other columns
plt.xlabel("Feature 1 Value")
plt.ylabel("Feature 2 Value")
class0x = []
class0y = []
class1x = []
class1y = []
class2x = []
class2y = []
for i in range(148):
    if Y[i] == 0:  # y = 0
        class0x.append(X[i,0])
        class0y.append(X[i,1])

    if Y[i] == 1:  # y = 1
        class1x.append(X[i,0])
        class1y.append(X[i,1])

    if Y[i] == 2:  # y = 2
        class2x.append(X[i,0])
        class2y.append(X[i,1])
plt.scatter(class0x,class0y,c = 'b', label = 'y = 0')
plt.legend()
plt.scatter(class1x,class1y,c = 'g', label = 'y = 1')
plt.legend()
plt.scatter(class2x,class2y,c = 'r', label = 'y = 2')
plt.legend()
plt.show()