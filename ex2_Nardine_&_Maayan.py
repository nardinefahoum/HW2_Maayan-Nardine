# * write a model `Ols` which has a propoery $w$ and 3 methods: `fit`, `predict` and `score`.? hint: use [numpy.linalg.pinv](https://docs.scipy.org/doc/numpy-1.15.1/reference/generated/numpy.linalg.pinv.html) to be more efficient.
from sklearn import datasets
from sklearn.model_selection import train_test_split
import numpy as np
from matplotlib import pyplot as plt
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression, Ridge, Lasso


####################################### Exercise 1 - Ordinary Least Square ###########################
boston = datasets.load_boston()
X = boston.data
Y = boston.target


class Ols(object):
    def __init__(self):
        self.w = None
    
    @staticmethod
    def pad(X):
        X=np.hstack((np.ones((X.shape[0],1)), X))
        return X
  
    def fit(self, X, Y):
        #remeber pad with 1 before fitting
        self.Y=Y
        self.X=Ols.pad(X)
        self.w=(np.linalg.pinv(self.X)@self.Y).reshape(self.X.shape[1],1)
        return self

    def predict(self, X):
        #return wx
        self.ypred=Ols.pad(X)@self.w
        return self.ypred
    
    def score(self, X, Y):
        #return MSE
        return mean_squared_error(Y,self.ypred)


print("𝑝 is:", X.shape[1], "and 𝑛 is:",X.shape[0] )
ols=Ols()
ols.fit(X, Y)
ols.predict(X)

print("The MSE is:", ols.score( X, Y))
plt.scatter(Y, ols.predict(X))
plt.xlabel('True_y')
plt.ylabel('predicted_y')

mse_train=np.array([])
mse_test=np.array([])

for i in range(20):
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, train_size=0.75)
    ols=Ols()
    ols.fit(X_train, Y_train)
    ols.predict(X_train)
    mse_train=np.append(mse_train,ols.score( X_train, Y_train))
    ols.predict( X_test)
    mse_test=np.append(mse_test,ols.score(X_test, Y_test))

print("The average MSE for train is:",mse_train.mean(), "and the average MSE for test is:",mse_test.mean())

# t-test
t, p = stats.ttest_rel(mse_train,mse_test)


print("The p-value for the hypothesis that the MSE of the train is smaller than the test is: ", p/2)
if p/2 <0.1:
  print("Thus, the MSE for training, in 90% confident level, is significant smaller than the MSE testing (P<0.1) ")
else:
  print("Thus, the MSE for training, in 90% confident level, is not significant smaller than the MSE testing (P>0.1)")




# Write a new class OlsGd which solves the problem using gradinet descent. 
# The class should get as a parameter the learning rate and number of iteration. 
# Plot the loss convergance. for each alpha, learning rate plot the MSE with respect to number of iterations.
# What is the effect of learning rate? 
# How would you find number of iteration automatically? 
# Note: Gradient Descent does not work well when features are not scaled evenly (why?!). Be sure to normalize your feature first.
class Normalizer(): 
    def __init__(self):
        self.Z=None
    def fit(self, X):
        self.X=X
        self.sd=np.std(self.X, axis=0)
        self.mean=np.mean(self.X, axis=0)
        return self

    def predict(self, X):
        #apply normalization
        self.Z=(X-self.mean)/self.sd
        return self.Z
    
class OlsGd(Ols):
    def __init__(self, learning_rate=.05, 
               num_iteration=1000, 
               normalize=True,
               early_stop=True,
               verbose=True):
        super(OlsGd, self).__init__()
        self.learning_rate = learning_rate
        self.num_iteration = num_iteration
        self.early_stop = early_stop
        self.normalize = normalize
        self.normalizer = Normalizer()    
        self.verbose = verbose
    
    def _fit(self, X, Y, reset=True):
        #remeber to normalize the data before starting
        self.Y=Y
        self.X=X
        if self.normalize:
            self.normalizer.fit(self.X)
            self.X=self.normalizer.predict(self.X)
        self.X=Ols.pad(self.X)
        if reset:
            self.w=np.random.normal(size=(self.X.shape[1],1))
        else:
            self.fit(self.X,self.Y)
        return self
        
    def _predict(self, X):
        #remeber to normalize the data before starting
        if self.normalize:
            X=self.normalizer.predict(X)
        self.ypred=Ols.pad(X)@self.w
        return self.ypred
    
    def _step(self, X, Y):
        # use w update for gradient descent
        self.loss = np.array([])
        self.mse = np.array([])
        for self.i in range(self.num_iteration):
            dloss = (self.X.T@(self.X@self.w-self.Y.reshape(self.Y.shape[0],1)))/self.X.shape[0]
            D = self.X@self.w-(self.Y.reshape(self.Y.shape[0],1))
            self.loss = np.append(self.loss, 0.5*(D.T@D))
            self.mse=np.append(self.mse, np.mean((self.Y-self.X@self.w)**2))
            self.w = self.w-self.learning_rate*dloss
            
            if self.verbose:
                print("num_iteration:", self.i+1, "loss:", (0.5*(D.T@D)).item())
            if self.early_stop and self.i!=0 and np.abs(self.loss[self.i]-self.loss[self.i-1])<1:
                break

        return self
#Plot the loss convergance.        
gd=OlsGd(verbose=False) 
gd._fit(X, Y,reset=True)
gd._step(X, Y)

plt.plot(np.arange(gd.i+1),gd.loss)
plt.xlabel("num_iteration")
plt.ylabel("loss")


###################################### Exercise 2 - Ridge Linear Regression #######################

class RidgeLs(Ols):
    def __init__(self, ridge_lambda, *wargs, **kwargs):
        super(RidgeLs,self).__init__(*wargs, **kwargs)
        self.ridge_lambda = ridge_lambda
    def _fit(self, X, Y):
        #Closed form of ridge regression
        self.Y=Y
        self.X=Ols.pad(X)
        self.w=(np.linalg.pinv(self.X.T@self.X+self.ridge_lambda*np.eye(self.X.shape[1]))@self.X.T@self.Y).reshape(self.X.shape[1],1)
        return self
rd=RidgeLs(ridge_lambda=0.7)
rd._fit(X, Y)


####################################### scikitlearn implementation for OLS, Ridge and Lasso ######################################################
ols = LinearRegression()
ols.fit(X, Y)
ols.predict(X)

ridge = Ridge(alpha=0.7)
ridge.fit(X, Y)
ridge.predict(X)

lasso = Lasso(alpha=1.0)
lasso.fit(X,Y)
lasso.predict(X)
