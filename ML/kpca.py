import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
import warnings

class KPCA:
    def __init__(self, X, kernel, d):
        """
        KPCA object
        Parameters
        ----------
        X: dxn matrix
        kernel: kernel function from kernel class
        d: number of principal components to be chosen
        """
        self.X = X
        self.kernel = kernel 
        self.d = d
        self.K = None
        self.sigma = None
        self.v = None
        self.scores = None
        self.tuplas_eig = None
    
    def _is_pos_semidef(self, x):
        return np.all(x >= 0)

    def _kernel_matrix(self):
        """
        Compute kernel matrix
        Output:
        K: nxn matrix
        """
        c = self.X.shape[1]
        K = np.array([[self.kernel(self.X[:, i], self.X[:, j]) for j in range(c)] for i in range(c)])
        # Centering K
        ones = np.ones((c, c)) / c
        K = K - ones @ K - K @ ones + ones @ K @ ones
        return K
    
    def _decompose(self):
        """
        Decomposition of K
        Output:
        tuplas_eig: List of ordered tuples by singular values; (singular_value, eigenvector)
        """
        self.K = self._kernel_matrix()
        eigval, eigvec = np.linalg.eig(self.K)
        if not self._is_pos_semidef(eigval):
            warnings.warn("The matrix K is not positive semi-definite")
        # Normalize eigenvectors and compute singular values of K
        tuplas_eig = sorted(
            [(np.sqrt(eigval[i]), eigvec[:, i] / np.sqrt(eigval[i])) for i in range(len(eigval))],
            key=lambda x: x[0], reverse=True
        )
        return tuplas_eig
    
    def project(self):
        """
        Compute scores
        Output:
        scores: T = sigma * V_d^t
        """
        self.tuplas_eig = self._decompose()
        tuplas_eig_dim = self.tuplas_eig[:self.d]
        self.sigma = np.diag([i[0] for i in tuplas_eig_dim])
        self.v = np.array([j[1] for j in tuplas_eig_dim]).T
        self.sigma = np.real_if_close(self.sigma, tol=1)
        self.v = np.real_if_close(self.v, tol=1)
        self.scores = self.sigma @ self.v.T
        return self.scores
    
    def plot_singular_values(self, grid=True):
        eig_plot = [np.real_if_close(e, tol=1) for (e, _) in self.tuplas_eig if e > 0.01]
        plt.style.use('seaborn-whitegrid')
        plt.figure(figsize=(15, 7.5))
        plt.plot(range(1, len(eig_plot) + 1), eig_plot)
        plt.grid(grid)
        plt.title('Singular values of the matrix $K$ greater than 0')
        plt.ylabel('$\sigma^2$')
        plt.show()
        
    def plot_scores_2d(self, colors, grid=True, dim_1=1, dim_2=2):
        if self.d < 2:
            warnings.warn("Insufficient principal components")
            return
        
        plt.style.use('seaborn-whitegrid')
        plt.figure(figsize=(15, 10))
        plt.axhline(color='black', alpha=0.2)
        plt.axvline(color='black', alpha=0.2)
        plt.scatter(self.scores[dim_1 - 1, :], self.scores[dim_2 - 1, :], c=colors)
        plt.grid(grid)
        plt.title('KPCA Space')
        plt.xlabel('${}^a$ principal component in space $\phi(X)$'.format(dim_1))
        plt.ylabel('${}^a$ principal component in space $\phi(X)$'.format(dim_2))
        plt.show()
        
    def plot_scores_3d(self, colors, grid=True, dim_1=1, dim_2=2, dim_3=3):
        if self.d < 3:
            warnings.warn("Insufficient principal components")
            return
        
        plt.style.use('seaborn-whitegrid')
        fig = plt.figure(figsize=(15, 10))
        ax = fig.add_subplot(111, projection="3d")
        ax.scatter(self.scores[dim_1 - 1, :], self.scores[dim_2 - 1, :], self.scores[dim_3 - 1, :], c=colors)
        plt.grid(grid)
        plt.title('KPCA Space')
        ax.set_xlabel('${}^a$ principal component in space $\phi(X)$'.format(dim_1))
        ax.set_ylabel('${}^a$ principal component in space $\phi(X)$'.format(dim_2))
        ax.set_zlabel('${}^a$ principal component in space $\phi(X)$'.format(dim_3))
        plt.show()
        
    def plot_density(self, labels, dim=1, grid=False):
        plt.style.use('seaborn-whitegrid')
        plt.figure(figsize=(15, 5))
        for label in np.unique(labels):
            sns.kdeplot(self.scores[dim - 1, labels == label], linewidth=3, label=label)
        plt.grid(grid)
        plt.legend()
        plt.title('Distributions in the ${}^a$ principal component'.format(dim))
        plt.show()
