import numpy as np
import matplotlib.pyplot as plt
import mltools as ml

data = np.genfromtxt("data/curve80.txt",delimiter=None) # load the data

X = data[:,0]
X = np.atleast_2d(X).T # code expects shape (M,N) so make sure it's 2-dimensional 
Y = data[:,1] # doesn't matter for Y
Xtr,Xte,Ytr,Yte = ml.splitData(X,Y,0.75) # split data set 75/25
lr = ml.linear.linearRegress( Xtr, Ytr ) # create and train model
xs = np.linspace(0,10,200)
xs = xs[:,np.newaxis]
ys = lr.predict( xs )
Xtr2 = np.zeros( (Xtr.shape[0],2) ) # create Mx2 array to store features 
Xtr2[:,0] = Xtr[:,0] # place original "x" feature as X1
Xtr2[:,1] = Xtr[:,0]**2 # place "x^2" feature as X2
# Now, Xtr2 has two features about each data point: "x" and "x^2"


degrees =[1, 3, 5, 7, 10, 18]
mseTr=[]
mseTe=[]
for degree in degrees:
    XtrP = ml.transforms.fpoly(Xtr, degree, bias=False)
    # Rescale the data matrix so that the features have similar ranges / variance
    XtrP,params = ml.transforms.rescale(XtrP)
    # "params" returns the transformation parameters (shift & scale)
    # Then we can train the model on the scaled feature matrix:
    lr = ml.linear.linearRegress( XtrP, Ytr ) # create and train model
    # Now, apply the same polynomial expansion & scaling transformation to Xtest:
    XteP,_ = ml.transforms.rescale( ml.transforms.fpoly(Xte,degree,False), params)
    XsP,_= ml.transforms.rescale( ml.transforms.fpoly(xs,degree,False), params)
    ys = lr.predict( XsP )
    plt.scatter(Xtr,Ytr,c='r')
    ax = plt.axis()
    plt.title("Degree: " + str(degree))
    plt.axis(ax)
    plt.plot(xs,ys)
    plt.show()
    YTrainPred = lr.predict(XtrP)
    YTestPred= lr.predict(XteP)
    mseTrain = np.mean((YTrainPred - Ytr) ** 2)
    mseTest = np.mean((YTestPred - Yte) ** 2)
    mseTr.append(mseTrain)
    mseTe.append(mseTest)
plt.semilogy(degrees, mseTr, c = 'red')
plt.semilogy(degrees, mseTe, c = 'green')
plt.title("Error-Degree")
plt.show()