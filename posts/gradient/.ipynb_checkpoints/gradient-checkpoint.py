import numpy as np
import random

class LogisticRegression:
    
    # From Convexity Lecture Notes
    def sigmoid(self, z):
        '''
        Input:
        Output:
        '''
        return 1 / (1 + np.exp(-z))

    # From Convexity Lecture Notes
    def empirical_risk(self, X, y):
        '''
        Input:
        Output:
        '''
        return self.loss(X, y).mean()
    
    def gradient_empirical(self, X, y):
        '''
        Returns the gradient of the empirical risk loss
        '''
        y_hat = self.predict(X)
        loss_deriv = self.loss_deriv(X, y)
        matrix_loss_deriv = loss_deriv[:, np.newaxis]
    
        # print(matrix_loss_deriv.shape)
        # print(X.shape)
        return (matrix_loss_deriv * X).mean(axis = 0)
    
    def pad(self, X):
        '''
        Input: Feature matrix X
        Output: Transformed feature matrix X with extra column of 1 
        '''
        return np.append(X, np.ones((X.shape[0], 1)), 1)
        
    def fit(self, X, y, alpha, max_epochs):
        '''
        Input: Feature matrix X and true label vector y
        Output: None
        '''
        # Transform feature matrix X with extra column
        X_ = self.pad(X)
        
        # Initalize class instance variables: weights vector, loss and score history 
        self.w = np.random.rand(X_.shape[1])
        self.loss_history = []
        self.score_history = []
        
        prev_loss = np.inf # Set initial loss to infinity
        
        # From Gradient notes
        for _ in range(max_epochs):
            # print("Running time: " + str(_))
            # For each step, compute the gradient of the current weight vector
            gradient = self.gradient_empirical(X_, y)
            # Update weight vector 
            self.w -= gradient * alpha
            
            # Calculate our loss after updated weight vector
            new_loss = self.empirical_risk(X_, y)
            # Append to our loss history
            self.loss_history.append(new_loss)
            
            # Check if new_loss is the same as prev_loss
            if np.allclose(new_loss, prev_loss):
                print("Loss are the same")
                break
            else:
                prev_loss = new_loss
        print("Max epochs reached")
        
    def predict(self, X):
        '''
        Input: Feature matrix X 
        Output: Vector y_hat in {0,1}^n of predicted labels
        '''
        return (X@self.w) # No longer using indicator since 0-1 loss is not convex
    
    def score(self, X, y):
        ''' 
        Input: Feature matrix X and true label vector y
        Output: Returns accuracy of predictions as a number between 0 and 1, with 1 being a perfect classification
        '''
        # Generate our y_hat value using predict function
        y_hat = self.predict(X)
        return (y == y_hat).mean() # Returns the accuracy of predictions as the mean of the comparison between y and y_
    
    # From Convexity lecture notes
    def loss(self, X, y):
        '''
        Input: Feature matrix X and true label vector y 
        Output: Returns overall loss of the current weights on X and y  
        '''
        y_hat = self.predict(X)
        return -y * np.log(self.sigmoid(y_hat)) - (1 - y)*np.log(1 - self.sigmoid(y_hat))
    
    def loss_deriv(self, X, y):
        '''
        Takes the derivative of our little loss function with respect to y_hat
        '''
        y_hat = self.predict(X)
        return self.sigmoid(y_hat) - y
        
        