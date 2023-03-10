import numpy as np
from scipy.optimize import minimize

class KernelLogisticRegression:
    
    # Initialize class instance variables
    def __init__(self, kernel, **kernel_kwargs):
        self.kernel = kernel
        self.kernel_kwargs = kernel_kwargs
        self.fit_called = False
        
    def pad(self, X):
        return np.append(X, np.ones((X.shape[0], 1)), 1)
    
    def fit(self, X, y):
        # Condition if fit has been called
        self.fit_called = True

        # First transform our feature matrix X
        X_ = self.pad(X)
        
        # Get some random variable initial v in R^n
        v_0 = np.random.rand(X_.shape[0]) - 0.5

        self.v = v_0
        
        # Save X_ as an instance variable
        self.X_train = X_
        
        # Compute kernel matrix of X
        km = self.kernel(X_, X_, **self.kernel_kwargs)
        
        # Perform minimization
        result = minimize(lambda w: self.empirical_risk(y, km, w), x0 = v_0) 
        
        self.v = result.x # Save our optimized weight vector to self.v
        
    def predict(self, X):
        '''
        Accepts feature matrix X and returns binary labels {0,1}
        '''
        if (not self.fit_called):
            raise Exception("You must call KLR.fit first!")
        else: 
            X_ = self.pad(X)
            
            # Generate the kernel between self.X_train and X
            km = self.kernel(self.X_train, X_, **self.kernel_kwargs)

            # Compute inner product between v and column of km
            inner_product = self.v @ km

            # Return y_hat
            y_hat = 1*(inner_product > 0)
            return y_hat
        
    def score(self, X, y):
        '''
        Computes the accuracy of the model predictions on the feature matrix X with label vector y
        '''
        if (not self.fit_called):
            raise Exception("You must call KLR.fit first!")
        else:
            # Get our predicted label vector
            y_hat = self.predict(X)
            
            # Take the predicted y_hat and convert the vector to 1s and 0s
            y_hat_comp = 1*(y_hat > 0)
    
            return (y == y_hat_comp).mean() # Returns the accuracy of predictions as the mean of the comparison between y and y_
    
    # From Convexity Lecture Notes
    def sigmoid(self, z):
        '''
        Input: Some vector z 
        Output: Returns the sigmoid of z 
        '''
        return 1 / (1 + np.exp(-z))
    
    # From Convexity lecture notes
    def loss(self, y_hat, y):
        '''
        Input: Feature matrix X and true label vector y 
        Output: Returns overall loss of the current weights on X and y  
        '''
        return -y * np.log(self.sigmoid(y_hat)) - (1 - y)*np.log(1 - self.sigmoid(y_hat))
    
    def empirical_risk(self, y, km, w):
        '''
        Input: Feature matrix X and true label vector y
        Output: Returns the average of the loss between X and y
        '''
        inner_product = w @ km # This is our y_hat
        return self.loss(inner_product, y).mean()