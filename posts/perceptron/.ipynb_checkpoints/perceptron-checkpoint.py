import numpy as np

class Perceptron:
        
    def fit(self, X, y, max_steps):
        '''
        No return value, but trains the perceptron to fit a best linear classifier on data set
        '''
        # Produce modified feature matrix X
        X_ = np.append(X, np.ones((X.shape[0], 1)), 1)
        # Initialize random weight vector with same dimensions as X_
        w_ = np.random.rand(X_.shape[1])
        
        # Create variable instances weights and history
        self.w = w_
        self.history = []
        
        for _ in range(max_steps):
            
            i_ = np.random.randint(0,X_.shape[0]-1) # Choose random index i from 1-(n-1)
            y_ = 2*y[i_] - 1 # Change y labels to -1 and 1
            
            # Take the accuracy of perceptron using score function
            accuracy = self.score(X, y)
            
            # Base case to terminate loop
            if (accuracy == 1):
                # Append the last known accuracy before terminating loop 
                self.history.append(accuracy)
                break
            
            # For each iteration, append current accuracy to Perceptron's evolution history
            self.history.append(accuracy)
                
            # Perceptron update happens here
            perceptron_update = self.update(X_, i_, y_)
            # Update our weight vector 
            self.w += perceptron_update

    def update(self, X, index, y):
        '''
        Returns a new weight vector to update the perceptron
        '''
        dot_product = X[index]@self.w
        indicator_cond = 1*((y * dot_product) < 0)
        perceptron_update = y * X[index]
        return indicator_cond * perceptron_update
    
    def predict(self, X):
        '''
        Returns a vector y in (0,1)^n
        '''
        X_ = np.append(X, np.ones((X.shape[0], 1)), 1)
        # Generates a vector of truths/false based off X_ and the weighted vector associated with the perceptron
        return 1*((X_@self.w) >= 0)
    
    def score(self, X, y):
        '''
        Returns accuracy of the perceptron as a number between 0 and 1 corresponding to perfect classification
        '''
        # Generate our y_hat to find the accuracy through its averaged sums of the comparison between y 
        y_hat = self.predict(X) 
        compare = (y_hat == y)
        return compare.mean()

                
        