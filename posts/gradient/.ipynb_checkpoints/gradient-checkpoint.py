import numpy as np
import random

class LogisticRegression:
    
    # From Convexity Lecture Notes
    def sigmoid(self, z):
        '''
        Input: Some vector z 
        Output: Returns the sigmoid of z 
        '''
        return 1 / (1 + np.exp(-z))

    # From Convexity Lecture Notes
    def empirical_risk(self, X, y):
        '''
        Input: Feature matrix X and true label vector y
        Output: Returns the average of the loss between X and y
        '''
        return self.loss(X, y).mean()
    
    def gradient_empirical(self, X, y):
        '''
        Input: Feature matrix X and true label vector y
        Output: Returns the gradient of the empirical risk loss
        '''
        y_hat = self.predict(X)
        loss_deriv = self.loss_deriv(X, y)
        matrix_loss_deriv = loss_deriv[:, np.newaxis]
    
        return (matrix_loss_deriv * X).mean(axis = 0)
    
    def pad(self, X):
        '''
        Input: Feature matrix X
        Output: Transformed feature matrix X with extra column of 1 
        '''
        return np.append(X, np.ones((X.shape[0], 1)), 1)
        
    def fit(self, X, y, alpha, max_epochs):
        '''
        Input: Feature matrix X, true label vector y, learning rate alpha, and max epochs
        Output: Normal gradient descent model trained on data set
        '''
        # Transform feature matrix X with extra column
        X_ = self.pad(X)
        
        # Initalize class instance variables: weights vector, loss and score history 
        self.w = np.random.rand(X_.shape[1])
        self.loss_history = []
        self.score_history = []
        
        prev_loss = np.inf # Set initial loss to infinity
        
        # From Gradient notes
        for j in range(max_epochs):
            # For each step, compute the gradient of the current weight vector
            gradient = self.gradient_empirical(X_, y)
            # Update weight vector 
            self.w -= gradient * alpha
            
            # Calculate our score and append to score history
            score = self.score(X_, y)
            self.score_history.append(score)
            
            # Calculate our loss after updated weight vector and append to loss history
            new_loss = self.empirical_risk(X_, y)
            
            # Check if new_loss is the same as prev_loss
            if np.allclose(new_loss, prev_loss):
                self.loss_history.append(new_loss)
                break
            else:
                self.loss_history.append(new_loss)
                prev_loss = new_loss
        
    def fit_stochastic(self, X, y, alpha, max_epochs, batch_size, momentum = False):
        '''
        Input: Feature matrix X, true label vector y, learning rate alpha, max epochs, and momentum
        Output: Stochastic gradient model trained on data set with momentum
        '''
         # If momentum is set true, set beta value to 0.8
        beta = 0.8 if momentum else 0

        # From assignment post
        # Transform feature matrix X with extra column
        X_ = self.pad(X)
        
        n = X_.shape[0]
            
        # Initalize class instance variables: weights vector, loss and score history 
        initial_w = np.random.rand(X_.shape[1])
        self.w = initial_w
        self.loss_history = []
        prev_loss = np.inf
        
        for j in np.arange(max_epochs):
            # Reshuffling our subset of points
            order = np.arange(n)
            np.random.shuffle(order)
            
            # Create batch size to generate a subset of our feature matrix
            for batch in np.array_split(order, n // batch_size + 1):
                x_batch = X_[batch,:]
                y_batch = y[batch]
                gradient = self.gradient_empirical(x_batch, y_batch) 
                
                # Update weight vector using momentum
                curr_w = self.w
                next_w = curr_w - gradient * alpha + (beta * (self.w - initial_w))
                self.w = next_w
                initial_w = curr_w
                
            # At the end of epoch, we calculate and update loss history
            new_loss = self.empirical_risk(X_, y)
                
            # Terminate if found no difference between prev and new loss
            if np.allclose(new_loss, prev_loss):
                self.loss_history.append(new_loss)
                break
            else:
                self.loss_history.append(new_loss)
                prev_loss = new_loss
        
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
        # Take the predicted y_hat and convert the vector to 1s and 0s
        y_hat_comp = 1*(y_hat > 0)
        return (y == y_hat_comp).mean() # Returns the accuracy of predictions as the mean of the comparison between y and y_
    
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
        Input: Feature matrix X and true label vector y
        Output: Returns the derivative of little loss function with respect to predicted y_hat
        '''
        y_hat = self.predict(X)
        return self.sigmoid(y_hat) - y
        
        