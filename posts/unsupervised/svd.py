import numpy as np
from matplotlib import pyplot as plt

class SVD:
    
    def svd_reconstruct(img, k):
        '''
        Input: Image to reconstruct and number of k singular values to use
        Output: Reconstructs image from its singular value decomposition
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
        return img_
    
    
    def svd_experiment(img, k):
        '''
        Input: Image to 
        Output:
        '''