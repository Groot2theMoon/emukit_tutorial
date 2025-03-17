import numpy as np
import pandas as pd
from emukit.core import ParameterSpace, ContinuousParameter
from emukit.core.initial_designs import RandomDesign
from emukit.bayesian_optimization.loops import BayesianOptimizationLoop
from emukit.bayesian_optimization.acquisitions import ExpectedImprovement, NegativeLowerConfidenceBound
from emukit.core.optimization import GradientAcquisitionOptimizer
from GPy.models import GPRegression
from GPy.kern import RBF
import matplotlib.pyplot as plt

# **Step 1: Load Iris dataset and define functions**
# Load the Iris dataset from a URL.
url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'
columns = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'class']
data = pd.read_csv(url, header=None, names=columns)
# Map class labels to integers.
class_mapping = {'Iris-setosa': 0, 'Iris-versicolor': 1, 'Iris-virginica': 2}
data['class'] = data['class'].map(class_mapping)
X = data.iloc[:, :-1].values  # Features.
y = data['class'].values      # Labels.

# Implement k-NN classifier.
from scipy.spatial.distance import cdist

def predict_knn(X_train, y_train, X_test, k):
    """
    Predict labels for test points using k-NN.
    Inputs: X_train, y_train (training data), X_test (test data), k (neighbors).
    Output: Predicted labels.
    """
    distances = cdist(X_test, X_train, 'euclidean')
    nearest_indices = np.argsort(distances, axis=1)[:, :k]
    nearest_labels = y_train[nearest_indices]
    predictions = np.array([np.bincount(labels).argmax() for labels in nearest_labels])
    return predictions

# Implement cross-validation.
def cross_val_score_knn(X, y, k, cv=5):
    """
    Compute mean accuracy over 5 folds for k-NN.
    Inputs: X (features), y (labels), k (neighbors), cv (folds).
    Output: Mean accuracy.
    """
    fold_size = len(X) // cv
    accuracies = []
    indices = np.arange(len(X))
    for i in range(cv):
        test_indices = indices[i * fold_size:(i + 1) * fold_size]
        train_indices = np.setdiff1d(indices, test_indices)
        X_train, y_train = X[train_indices], y[train_indices]
        X_test, y_test = X[test_indices], y[test_indices]
        y_pred = predict_knn(X_train, y_train, X_test, k)
        accuracy = np.mean(y_pred == y_test)
        accuracies.append(accuracy)
    return np.mean(accuracies)

# Define objective function.
def objective(params):
    """
    Objective function returning negative accuracy.
    Input: params (2D array of k values).
    Output: 2D array of negative accuracies.
    """
    results = []
    for param in params:
        k = max(1, min(20, int(np.round(param[0]))))
        accuracy = cross_val_score_knn(X, y, k)
        results.append(-accuracy)
    return np.array(results)[:, np.newaxis]

# **Step 2: Set up parameter space and initial points**
# Define the search space for k.
space = ParameterSpace([ContinuousParameter('k', 1, 20)])
# Generate consistent initial points for fair comparison across runs.
design = RandomDesign(space)
initial_k = design.get_samples(3)  # 3 random k values.
initial_y = objective(initial_k)   # Evaluate objective at these points.

# **Step 3: Set up GP kernel**
# Use an RBF kernel for the GP model (1D input: k).
kernel = RBF(input_dim=1)

# **Step 4: Set up acquisition optimizer**
# Use a gradient-based optimizer to maximize acquisition functions.
optimizer = GradientAcquisitionOptimizer(space)

# **Step 5: Function to run Bayesian optimization**
def run_bo(acquisition):
    """
    Run Bayesian optimization with a given acquisition function.
    Input: acquisition (acquisition function object).
    Output: Convergence history (best accuracy per iteration).
    """
    # Initialize a fresh GP model for each run to avoid interference.
    model = GPRegression(initial_k, initial_y, kernel)
    # Set up the Bayesian optimization loop.
    bo_loop = BayesianOptimizationLoop(
        model=model,
        space=space,
        acquisition=acquisition,
        acquisition_optimizer=optimizer
    )
    max_iter = 10  # Number of iterations.
    bo_loop.run_loop(objective, max_iter)  # Run the optimization.
    # Extract observed objective values.
    Y = bo_loop.loop_state.Y
    # Compute best accuracy so far at each iteration.
    convergence = -np.minimum.accumulate(Y[:, 0])
    return convergence

# **Step 6: Define different acquisition functions**
# Create a dummy model to initialize acquisition functions.
dummy_model = GPRegression(initial_k, initial_y, kernel)
acquisitions = [
    ExpectedImprovement(dummy_model, jitter=0.01),  # EI with low jitter.
    ExpectedImprovement(dummy_model, jitter=0.1),   # EI with higher jitter.
    NegativeLowerConfidenceBound(dummy_model, beta=1.96),  # LCB with moderate exploration.
    NegativeLowerConfidenceBound(dummy_model, beta=3.0)    # LCB with more exploration.
]
labels = [
    'EI jitter=0.01',
    'EI jitter=0.1',
    'LCB beta=1.96',
    'LCB beta=3'
]

# **Step 7: Run optimization for each acquisition function**
# Collect convergence curves for each acquisition function.
convergence_curves = [run_bo(acq) for acq in acquisitions]

# **Step 8: Plot convergence curves**
# Plot the best accuracy over iterations for each acquisition function.
for curve, label in zip(convergence_curves, labels):
    plt.plot(range(1, len(curve) + 1), curve, label=label, marker='o')
plt.xlabel('Iteration')
plt.ylabel('Best Cross-Validation Accuracy So Far')
plt.title('Convergence Plot for Different Acquisition Functions')
plt.legend()
plt.grid(True)
plt.show()