import numpy as np
from matplotlib import pyplot as plt

class LinearRegression:
    
    def pad(self, X):
        return np.append(X, np.ones((X.shape[0], 1)), 1)

    def fit_analytic(self, X, y):
        # Pad our X feature matrix
        X = self.pad(X)
        
        transpose_X = np.transpose(X)
        inverse_matr = np.linalg.inv((transpose_X @ X))

        # Calculate our optimized weight vector
        w_hat = inverse_matr @ transpose_X @ y
        self.w = w_hat
    
    def fit_gradient(self, X, y, max_iter, alpha):
        # Pad our X feature matrix
        X = self.pad(X)
        
        # Initialize some random guess for w
        self.w = np.random.rand(X.shape[1])
        # Initialize our score history
        self.score_history = []
        
        # Compute our P and q once
        transpose_X = np.transpose(X)
        P = transpose_X @ X
        q = transpose_X @ y

        for _ in range(max_iter):
            # Compute our gradient 
            gradient = np.subtract((P @ self.w), q)
            self.w -= gradient * alpha
            
            # Compute our score and append to score_history
            score = self.score(X, y, self.w, gradient = True)
            self.score_history.append(score)
            
    # Predict function to compute y_hat
    def predict(self, X, w):
        return X@w

    # Score function
    def score(self, X, y, w, gradient = False):

        if gradient is True:
            X = X
        else:
            X = self.pad(X)
                        
        # Get y_hat
        y_hat = self.predict(X,w)

        # Top sum
        top_sum = np.sum((y_hat - y)**2)

        # Get mean of y
        y_mean = np.mean(y)

        # Bottom sum
        bottom_sum = np.sum((y_mean - y)**2)

        return 1 - (top_sum / bottom_sum) # Coefficient of determination
    
    # Function to generate linear regression data
    def LR_data(self, n_train = 100, n_val = 100, p_features = 1, noise = .1, w = None):
        if w is None: 
            w = np.random.rand(p_features + 1) + .2

        X_train = np.random.rand(n_train, p_features)
        y_train = self.pad(X_train)@w + noise*np.random.randn(n_train)

        X_val = np.random.rand(n_val, p_features)
        y_val = self.pad(X_val)@w + noise*np.random.randn(n_val)

        return X_train, y_train, X_val, y_val