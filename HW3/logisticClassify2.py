import numpy as np
import mltools as ml
import matplotlib.pyplot as plt

# Fix the required "not implemented" functions for the homework ("TODO")

################################################################################
## LOGISTIC REGRESSION BINARY CLASSIFIER #######################################
################################################################################


class logisticClassify2(ml.classifier):
    """A binary (2-class) logistic regression classifier

    Attributes:
        classes : a list of the possible class labels
        theta   : linear parameters of the classifier
    """

    def __init__(self, *args, **kwargs):
        """
        Constructor for logisticClassify2 object.

        Parameters: Same as "train" function; calls "train" if available

        Properties:
           classes : list of identifiers for each class
           theta   : linear coefficients of the classifier; numpy array
        """
        self.classes = [0,1]              # (default to 0/1; replace during training)
        self.theta = np.array([])         # placeholder value before training

        if len(args) or len(kwargs):      # if we were given optional arguments,
            self.train(*args,**kwargs)    #  just pass them through to "train"


## METHODS ################################################################

    def plotBoundary(self,X,Y):
        if len(self.theta) != 3: raise ValueError('Data & model must be 2D');
        K = np.unique(Y)
        x1b = np.linspace(min(X[:,0])-1, max(X[:,0])+1, 200)
        x2b = -(self.theta[1]/self.theta[2])*x1b-self.theta[0]/self.theta[2]
        plt.plot(X[Y==K[0],0],X[Y==K[0],1],'.',color='g')
        plt.plot(X[Y==K[1],0],X[Y==K[1],1],'.',color='r')
        plt.plot(x1b, x2b)
    
    def predictSoft(self, X):
        """ Return the probability of each class under logistic regression """
        raise NotImplementedError
        ## You do not need to implement this function.
        ## If you *want* to, it should return an Mx2 numpy array "P", with
        ## P[:,1] = probability of class 1 = sigma( theta*X )
        ## P[:,0] = 1 - P[:,1] = probability of class 0
        return P

    def predict(self, X):
        """ Return the predictied class of each data point in X"""
        Yhat = np.zeros(X.shape[0]);
        for i in range(X.shape[0]):
            r = self.theta[0] + self.theta[1]*X[i,0]+ self.theta[2]*X[i,1]
            if r > 0:
                Yhat[i] = self.classes[1]
            else:
                Yhat[i] = self.classes[0]
        return Yhat

    def train(self, X, Y, initStep=1.0, stopTol=1e-4, stopEpochs=5000, plot=None):
        """ Train the logistic regression using stochastic gradient descent """
        M,N = X.shape;                     # initialize the model if necessary:
        self.classes = np.unique(Y);       # Y may have two classes, any values
        XX = np.hstack((np.ones((M,1)),X)) # XX is X, but with an extra column of ones
        YY = ml.toIndex(Y,self.classes);   # YY is Y, but with canonical values 0 or 1
        if len(self.theta)!=N+1: self.theta=np.random.rand(N+1);
        # init loop variables:
        def sigma(r):
            return 1/(1+np.exp(-r))
        epoch=0; done=False; Jnll=[]; J01=[];
        while not done:
            stepsize, epoch = initStep*2.0/(2.0+epoch), epoch+1; # update stepsize
            # Do an SGD pass through the entire data set:
            for i in np.random.permutation(M):
                ri = np.dot(self.theta,XX[i,:]);     # TODO: compute linear response r(x)
                gradi = (-YY[i]+sigma(ri))*XX[i,:];     # TODO: compute gradient of NLL loss
                self.theta -= stepsize * gradi;  # take a gradient step

            J01.append( self.err(X,Y) )  # evaluate the current error rate
            j = 0
            for i in np.random.permutation(M):
                j += -YY[i]*np.log(sigma(np.dot(self.theta, XX[i,:])))-\
                 (1-YY[i])*np.log(1-sigma(np.dot(self.theta, XX[i,:])))
            ## TODO: compute surrogate loss (logistic negative log-likelihood)
            Jsur = j/M
            Jnll.append( Jsur ) # TODO evaluate the current NLL loss

            ## For debugging: you may want to print current parameters & losses
            # print self.theta, ' => ', Jnll[-1], ' / ', J01[-1]
            # raw_input()   # pause for keystroke
            if epoch >stopEpochs or ((epoch>1) and np.abs(Jnll[-2] - Jnll[-1]) < stopTol):
                  done = True
            # TODO check stopping criteria: exit if exceeded # of epochs ( > stopEpochs)
            # or if Jnll not changing between epochs ( < stopTol )
        plt.figure(1)
        plt.plot(Jnll,'b-',label='loss')
        plt.legend()
        plt.plot(J01,'r-',label='error')
        plt.legend()
        plt.figure(2)
        self.plotBoundary(X,Y)
        plt.show()


################################################################################
################################################################################
################################################################################
