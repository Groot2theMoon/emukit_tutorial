import math
import random

class GaussianProcess:
    def __init__(self, kernel, noise_variance=1e-6):
        self.kernel = kernel
        self.noise_variance = noise_variance
        self.X_train = []
        self.y_train = []
        self.L = []
        self.alpha = []

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y
        
        # Compute kernel matrix
        K = self.kernel_matrix(X, X)
        
        # Add noise to diagonal
        n = len(K)
        for i in range(n):
            K[i][i] += self.noise_variance
            
        # Cholesky decomposition
        self.L = self.cholesky(K)
        
        # Solve L@L.T@alpha = y
        self.alpha = self.solve_cholesky(self.L, y)

    def predict(self, X_test):
        K_s = self.kernel_matrix(self.X_train, X_test)
        K_ss = self.kernel_matrix(X_test, X_test)
        
        # Compute predictive mean: K_s.T @ alpha
        mean = self.matmul(self.transpose(K_s), self.alpha)  # Now handles vector
        
        # Compute predictive variance
        v = self.solve_triangular(self.L, K_s)
        v_sq = [[x**2 for x in row] for row in v]
        variance = [self.get_diag(K_ss, i) - sum(col[i] for col in v_sq) 
                   for i in range(len(X_test))]
        
        return mean, [math.sqrt(max(x, 0)) for x in variance]

    # Helper functions for linear algebra
    def kernel_matrix(self, X1, X2):
        return [[self.kernel(x1, x2) for x2 in X2] for x1 in X1]

    def cholesky(self, A):
        n = len(A)
        L = [[0.0]*n for _ in range(n)]
        for i in range(n):
            for j in range(i+1):
                s = sum(L[i][k] * L[j][k] for k in range(j))
                if i == j:
                    L[i][j] = math.sqrt(A[i][i] - s)
                else:
                    L[i][j] = (A[i][j] - s) / L[j][j]
        return L

    def solve_triangular(self, L, b):
        n = len(L)
        m = len(b[0]) if isinstance(b[0], list) else 1
        x = [[0.0]*m for _ in range(n)]
        
        for col in range(m):
            for i in range(n):
                if isinstance(b[i], list):
                    val = b[i][col]
                else:
                    val = b[i]
                s = sum(L[i][j] * x[j][col] for j in range(i))
                x[i][col] = (val - s) / L[i][i]
        return x

    def solve_cholesky(self, L, y):
        # Solve L@L.T@x = y
        n = len(L)
        # Forward substitution: L@z = y
        z = [0.0]*n
        for i in range(n):
            s = sum(L[i][j] * z[j] for j in range(i))
            z[i] = (y[i] - s) / L[i][i]
        
        # Backward substitution: L.T@x = z
        x = [0.0]*n
        for i in reversed(range(n)):
            s = sum(L[j][i] * x[j] for j in range(i+1, n))
            x[i] = (z[i] - s) / L[i][i]
        return x

    def matmul(self, A, B):
        """Handle both matrix-matrix and matrix-vector multiplication"""
        if isinstance(B[0], (int, float)):
            # Matrix-vector multiplication
            return [sum(a*b for a,b in zip(row, B)) for row in A]
        else:
            # Matrix-matrix multiplication
            return [[sum(a*b for a,b in zip(row, col)) 
                    for col in zip(*B)] for row in A]

    def transpose(self, A):
        return list(map(list, zip(*A)))

    def get_diag(self, A, i):
        return A[i][i] if i < len(A) else 0.0

# RBF kernel implementation
def rbf_kernel(l=1.0, sigma_f=1.0):
    def kernel(x1, x2):
        sq_dist = sum((a - b)**2 for a,b in zip(x1, x2))
        return sigma_f**2 * math.exp(-0.5 * sq_dist / l**2)
    return kernel

# Example usage
if __name__ == "__main__":
    # Generate synthetic data
    random.seed(42)
    X = [[x] for x in [-5 + 10*i/19 for i in range(20)]]
    y = [math.sin(x[0]) + random.gauss(0, 0.1) for x in X]

    # Initialize GP
    gp = GaussianProcess(rbf_kernel(l=1.0, sigma_f=1.0), 0.1**2)
    gp.fit(X, y)

    # Make predictions
    X_test = [[x] for x in [-7 + 14*i/99 for i in range(100)]]
    mean_pred, std_pred = gp.predict(X_test)

    # Simple ASCII plot
    print("Approximate GP Regression Result:")
    for x, m, s in zip(X_test, mean_pred, std_pred):
        bar = '#' * int(10*(m + 1.96*s - (m - 1.96*s)))
        print(f"X={x[0]:5.1f} | Mean: {m:6.2f} | Uncertainty: {bar}")