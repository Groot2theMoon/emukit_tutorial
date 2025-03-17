import numpy as np
import pandas as pd
from emukit.core import ParameterSpace, ContinuousParameter
from emukit.core.initial_designs import RandomDesign
from emukit.bayesian_optimization.loops import BayesianOptimizationLoop
from emukit.bayesian_optimization.acquisitions import ExpectedImprovement
from emukit.core.optimization import GradientAcquisitionOptimizer
from GPy.models import GPRegression
from GPy.kern import RBF
import matplotlib.pyplot as plt

# **Step 1: Load the Iris dataset**
# Load the dataset from a public URL using Pandas to avoid dependency on Scikit-Learn.
url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'
columns = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'class']
data = pd.read_csv(url, header=None, names=columns)
# Convert class labels to integers for classification tasks.
class_mapping = {'Iris-setosa': 0, 'Iris-versicolor': 1, 'Iris-virginica': 2}
data['class'] = data['class'].map(class_mapping)
X = data.iloc[:, :-1].values  # Extract feature matrix (all columns except the last).
y = data['class'].values      # Extract target vector (class labels).

# **Step 2: Implement k-NN classifier from scratch**
# Define a custom k-NN classifier using Euclidean distance and majority voting.
from scipy.spatial.distance import cdist

def predict_knn(X_train, y_train, X_test, k):
    """
    Predict labels for test points using k-NN.
    Inputs:
        X_train: Training feature matrix.
        y_train: Training labels.
        X_test: Test feature matrix.
        k: Number of neighbors.
    Output: Predicted labels for X_test.
    """
    # Compute Euclidean distances between each test point and all training points.
    distances = cdist(X_test, X_train, 'euclidean')
    # Sort distances and get indices of the k nearest neighbors for each test point.
    nearest_indices = np.argsort(distances, axis=1)[:, :k]
    # Extract labels of the k nearest neighbors.
    nearest_labels = y_train[nearest_indices]
    # Predict the label by majority vote among the k neighbors.
    predictions = np.array([np.bincount(labels).argmax() for labels in nearest_labels])
    return predictions

# **Step 3: Implement 5-fold cross-validation from scratch**
# Evaluate the k-NN classifier's performance using 5-fold cross-validation.
def cross_val_score_knn(X, y, k, cv=5):
    """
    Compute mean accuracy over 5 folds for k-NN with a given k.
    Inputs:
        X: Feature matrix.
        y: Labels.
        k: Number of neighbors.
        cv: Number of folds (default=5).
    Output: Mean accuracy across folds.
    """
    fold_size = len(X) // cv  # Calculate size of each fold.
    accuracies = []           # Store accuracy for each fold.
    indices = np.arange(len(X))  # Array of indices for splitting data.
    # Iterate over each fold.
    for i in range(cv):
        # Define test indices for the current fold.
        test_indices = indices[i * fold_size:(i + 1) * fold_size]
        # Define training indices as all indices not in the test set.
        train_indices = np.setdiff1d(indices, test_indices)
        # Split data into training and test sets.
        X_train, y_train = X[train_indices], y[train_indices]
        X_test, y_test = X[test_indices], y[test_indices]
        # Make predictions on the test set.
        y_pred = predict_knn(X_train, y_train, X_test, k)
        # Compute accuracy for this fold (fraction of correct predictions).
        accuracy = np.mean(y_pred == y_test)
        accuracies.append(accuracy)
    # Return the mean accuracy across all folds.
    return np.mean(accuracies)

# **Step 4: Define the objective function for Bayesian optimization**
# Emukit minimizes the objective, so return negative accuracy to maximize accuracy.
def objective(params):
    """
    Objective function for optimization.
    Input:
        params: 2D array (n_points, 1) where each row is a value of k.
    Output: 2D array (n_points, 1) of negative accuracies.
    """
    results = []
    for param in params:
        # Convert parameter to integer k, constrained between 1 and 20.
        k = max(1, min(20, int(np.round(param[0]))))
        # Compute cross-validation accuracy for this k.
        accuracy = cross_val_score_knn(X, y, k)
        # Append negative accuracy since Emukit minimizes the objective.
        results.append(-accuracy)
    # Return as a 2D array as required by Emukit.
    return np.array(results)[:, np.newaxis]

# **Step 5: Set up the parameter space**
# Define the search space for k as a continuous range from 1 to 20.
space = ParameterSpace([ContinuousParameter('k', 1, 20)])

# **Step 6: Generate initial points**
# Use a random design to select initial points for the Gaussian Process (GP) model.
design = RandomDesign(space)
initial_k = design.get_samples(3)  # Generate 3 random k values.
initial_y = objective(initial_k)   # Evaluate the objective at these points.

# **Step 7: Set up the Gaussian Process model**
# Model the objective function using GP regression with an RBF kernel.
kernel = RBF(input_dim=1)  # Radial Basis Function kernel for 1D input (k).
model = GPRegression(initial_k, initial_y, kernel)  # GP model with initial data.

# **Step 8: Define the acquisition function**
# Use Expected Improvement (EI) to decide the next point to evaluate.
acquisition = ExpectedImprovement(model, jitter=0.01)  # Small jitter for numerical stability.

# **Step 9: Set up the acquisition optimizer**
# Use a gradient-based optimizer to maximize the acquisition function.
optimizer = GradientAcquisitionOptimizer(space)

# **Step 10: Set up and run the Bayesian optimization loop**
# Configure the Bayesian optimization loop with the model, space, acquisition, and optimizer.
max_iter = 10  # Number of iterations to run the optimization.
bo_loop = BayesianOptimizationLoop(
    model=model,
    space=space,
    acquisition=acquisition,
    acquisition_optimizer=optimizer
)
# Run the loop, evaluating the objective at new points suggested by the acquisition function.
bo_loop.run_loop(objective, max_iter)

# **Step 11: Extract and display results**
# Get all observed objective values (negative accuracies) from the loop.
Y = bo_loop.loop_state.Y
# Best accuracy is the negative of the minimum objective value (since we minimized -accuracy).
best_accuracy = -np.min(Y)
# Find the k value corresponding to the best accuracy.
best_k = bo_loop.loop_state.X[np.argmin(Y)]
print(f"Best k: {int(np.round(best_k[0]))}, Best cross-validation accuracy: {best_accuracy:.4f}")

# **Step 12: Plot convergence**
# Compute the best accuracy found so far at each iteration.
convergence = -np.minimum.accumulate(Y[:, 0])  # Convert negative accuracies to positive and track minimum.
plt.plot(range(1, len(convergence) + 1), convergence, marker='o')
plt.xlabel('Iteration')
plt.ylabel('Best Cross-Validation Accuracy So Far')
plt.title('Convergence Plot for Hyperparameter Tuning')
plt.grid(True)
plt.show()