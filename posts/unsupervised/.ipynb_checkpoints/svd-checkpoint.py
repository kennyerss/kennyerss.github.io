import numpy as np
import math
from matplotlib import pyplot as plt

class SVD:
    
    def __init__(self, img, k):
        self.img = img
        self.k = k
        
    def svd_reconstruct(self, img, k):
        '''
        Input: Image to reconstruct and number of k singular values to use
        Output: Reconstructs image from its singular value decomposition and storage amount
        '''
        # From lecture notes
        U, sigma, V = np.linalg.svd(img)
        
        # Create the D matrix in the SVD
        D = np.zeros_like(img,dtype=float) # matrix of zeros of same shape as img
        # Singular values on the main diagonal
        D[:min(img.shape),:min(img.shape)] = np.diag(sigma)      
        
        # Approximate using the first k columns of U, D, and V
        U_ = U[:, :k]
        D_ = D[:k, :k]
        V_ = V[:k, :]
        
        # Reconstruct our image 
        img_ = U_ @ D_ @ V_

        # Get the dimensions of our img
        m, n = img_.shape

        # Calculate the number of pixels to store for reconstructed image
        storage = ((k*m) + k + (k*n)) / (m*n) * 100

        return img_, round(storage, 1)
  
    def svd_experiment(self):
        '''
        Output: Plots out varied k singular values and its reconstructed images using SVD
        '''
        # Initialize how many rows and columns we want the subplot axes to have
        plt_row = 3
        plt_col = 3
        
        fig, axarr = plt.subplots(plt_row, plt_col, figsize = (12,6))
        
        # Plotting each new reconstructed image onto subplot
        for i in range(plt_row):
            for j in range(plt_col):
                img_, storage = self.svd_reconstruct(self.img, self.k)
                
                axarr[i, j].imshow(img_, cmap = "Greys")
                axarr[i, j].axis("off")
                axarr[i, j].set(title = f"{self.k} components, % storage = {storage}")
                
                # Update k value
                self.k += 5

        # Adjust spacing for each subplot
        plt.tight_layout()
        plt.show()