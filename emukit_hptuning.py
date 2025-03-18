import numpy as np
import pandas as pd
from emukit.core import ParameterSpace, ContinuousParameter
from emukit.core.initial_designs import RandomDesign
from emukit.bayesian_optimization.loops import BayesianOptimizationLoop
from emukit.bayesian_optimization.acquisitions import ExpectedImprovement
from emukit.core.optimization import GradientAcquisitionOptimizer
from GPy.models import GPRegression
from GPy.kern import RBF, White
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist

# Load data and prepare features/labels (same as before)
url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'
columns = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'class']
data = pd.read_csv(url, header=None, names=columns)
class_mapping = {'Iris-setosa': 0, 'Iris-versicolor': 1, 'Iris-virginica': 2}
data['class'] = data['class'].map(class_mapping)
X = data.iloc[:, :-1].values
y = data['class'].values

# k-NN functions (same as before)
def predict_knn(X_train, y_train, X_test, k):
    distances = cdist(X_test, X_train, 'euclidean')
    nearest_indices = np.argsort(distances, axis=1)[:, :k]
    nearest_labels = y_train[nearest_indices]
    return np.array([np.bincount(labels).argmax() for labels in nearest_labels])

def cross_val_score_knn(X, y, k, cv=5):
    fold_size = len(X) // cv
    accuracies = []
    indices = np.random.permutation(len(X))
    for i in range(cv):
        test_indices = indices[i*fold_size : (i+1)*fold_size]
        train_indices = np.setdiff1d(indices, test_indices)
        X_train, y_train = X[train_indices], y[train_indices]
        X_test, y_test = X[test_indices], y[test_indices]
        y_pred = predict_knn(X_train, y_train, X_test, k)
        accuracies.append(np.mean(y_pred == y_test))
    return np.mean(accuracies)

# Enhanced GP Wrapper with numerical stability
class StableGPRegression(GPRegression):
    def set_data(self, X, Y):
        # Ensure 2D array format
        self.X = np.atleast_2d(X)
        self.Y = np.atleast_2d(Y)
        self.update_model(True)
        self.optimize_restarts(5)  # Multiple restarts for better optimization

# Modified objective function
def objective(params):
    return np.array([[-cross_val_score_knn(X, y, max(1, min(20, int(np.round(p[0])))] for p in params])

# Create parameter space
space = ParameterSpace([ContinuousParameter('k', 1, 20)])

# Initialize with more points for better coverage
design = RandomDesign(space)
initial_k = design.get_samples(5)
initial_y = objective(initial_k)

# Create kernel with white noise for numerical stability
kernel = RBF(1) + White(1, variance=1e-4)
model = StableGPRegression(initial_k, initial_y, kernel)
model.likelihood.variance = 0.01  # Initial noise estimate
model.Gaussian_noise.constrain_bounded(1e-5, 1e-2)  # Constrain noise parameter

# Configure Bayesian optimization
acquisition = ExpectedImprovement(model, jitter=0.1)  # Increased jitter
optimizer = GradientAcquisitionOptimizer(space)

# Run optimization loop
bo_loop = BayesianOptimizationLoop(
    model=model,
    space=space,
    acquisition=acquisition,
    acquisition_optimizer=optimizer
)
bo_loop.run_loop(objective, 15)  # Increased iterations

# Results visualization
best_idx = np.argmin(bo_loop.loop_state.Y)
best_k = int(np.round(bo_loop.loop_state.X[best_idx][0]))
best_acc = -bo_loop.loop_state.Y[best_idx][0]
print(f"Optimal k: {best_k}, Accuracy: {best_acc:.4f}")

# Plot convergence
plt.plot(np.minimum.accumulate(bo_loop.loop_state.Y * -1))
plt.xlabel('Iterations')
plt.ylabel('Best Accuracy')
plt.title('Bayesian Optimization Convergence')
plt.show()