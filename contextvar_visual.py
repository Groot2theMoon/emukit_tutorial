import numpy as np
import matplotlib.pyplot as plt
from emukit.test_functions import branin_function
from emukit.core import ParameterSpace, ContinuousParameter
from emukit.core.initial_designs import RandomDesign
from GPy.models import GPRegression
from emukit.model_wrappers import GPyModelWrapper
from emukit.bayesian_optimization.acquisitions import ExpectedImprovement
from emukit.bayesian_optimization.loops import BayesianOptimizationLoop

# load the Branin function and define the parameter space
f, parameter_space = branin_function()

def run_optimization(context=None, max_iter=10):
    # Collect initial random design points
    design = RandomDesign(parameter_space)
    X_init = design.get_samples(10)
    Y_init = f(X_init)
    
    # Create a GP model and wrap it in Emukit
    model_gpy = GPRegression(X_init, Y_init)
    model_emukit = GPyModelWrapper(model_gpy)
    
    # Define the acquisition function and create the Bayesian optimization loop
    expected_improvement = ExpectedImprovement(model=model_emukit)
    bayesopt_loop = BayesianOptimizationLoop(model=model_emukit,
                                              space=parameter_space,
                                              acquisition=expected_improvement,
                                              batch_size=1)
    # Run the optimization loop with (optional) context variable
    bayesopt_loop.run_loop(f, max_iter, context=context)
    return bayesopt_loop.loop_state.X, bayesopt_loop.loop_state.Y

# Run three optimization sequences:
# 1. Fixing 'x1' at 0.3
X1, Y1 = run_optimization(context={'x1': 0.3})
# 2. Fixing 'x2' at 0.1
X2, Y2 = run_optimization(context={'x2': 0.1})
# 3. No context (full exploration)
X3, Y3 = run_optimization(context=None)

# Prepare a grid to plot the Branin function contour
x1_vals = np.linspace(-5, 10, 200)
x2_vals = np.linspace(0, 15, 200)
X1_grid, X2_grid = np.meshgrid(x1_vals, x2_vals)
grid = np.vstack([X1_grid.ravel(), X2_grid.ravel()]).T
Z = f(grid).reshape(X1_grid.shape)

# Plot the Branin function and the evaluated points from each optimization run
plt.figure(figsize=(10, 8))
contour = plt.contourf(X1_grid, X2_grid, Z, levels=50, cmap='viridis')
plt.colorbar(contour)
plt.scatter(X1[:, 0], X1[:, 1], color='red', label="Run 1: context {'x1': 0.3}")
plt.scatter(X2[:, 0], X2[:, 1], color='blue', label="Run 2: context {'x2': 0.1}")
plt.scatter(X3[:, 0], X3[:, 1], color='orange', label="Run 3: no context")
plt.xlabel('x1')
plt.ylabel('x2')
plt.title('Bayesian Optimization on Branin Function with Context Variables')
plt.legend()
plt.show()
