"""
<External objective function evaluation in Bayesian optimization with Emukit

the Bayesian optimization component of Emukit aloows for objective functions to be evaluated externally.
If users opt for this approach, they can use Emukit to suggest the next point for evaulation, and then evaluate the objective function themselves as well as decide on
the stopping criteria of the evaluation loop.
this notebook shall demonstrate how to carry out this procedure. The main benefit of using Emukit in this manner is that you can externally 
mange issues such as parallelizing the computation of the objective function, which is convenient in many scenarios. 
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors as mcolors

### --- Figure config
colors = dict(mcolors.BASE_COLORS, **mcolors.CSS4_COLORS)
LEGEND_SIZE = 15
TITLE_SIZE = 25
AXIS_SIZE = 15

"""
<Handling the loop yourself>
for the purposes of this notebook we are going to use one of the predefined objective fs that come with GPyOpt.
However, the key thing to realize is that the function could be anything. 
As long as users are able to externally evaluate the suggested points and provide GPyOpt with the results, the library has options for setting the objective f's origin.
"""

from emukit.test_functions import forrester_function
from emukit.core.loop import UserFunctionWrapper

target_function, space = forrester_function()

"""
First, we are going to run the optimization loop outside of Emukit, and only use the library to get the next point at which to evaluate our function.
there are 2 things to pay attention whe creating the main optimization object:

    1) Since we recreate the object anew for each iteration, we need to pass data about all previous iterations to it.
    2) The model gets optimized from the scratch in every iteration but the parameters of the model could be saved and used to update the state (TODO)

we start with three initial points at which we evaluate the objective funtion. 
"""

X = np.array([[0.1], [0.6], [0.9]])
Y = target_function(X)

from emukit.examples.gp_bayesian_optimization.single_objective_bayesian_optimization import GPBayesianOptimization
from emukit.core.loop import UserFunctionResult

num_iterations = 10

bo = GPBayesianOptimization(variables_list=space.parameters, X=X, Y=Y)
results = None

for _ in range(num_iterations):
    X_new = bo.get_next_points(results)
    Y_new = target_function(X_new)
    results = [UserFunctionResult(X_new[0], Y_new[0])]

X = bo.loop_state.X
Y = bo.loop_state.Y

"""
Let's visualize the results. The size of the marker denotes the order in which the point was evaluated. - the bigger the marker, the later was the evaluation.
"""

x = np.arange(0.0, 1.0, 0.01)
y = target_function(x)

plt.figure()
plt.plot(x,y)
for i, (xs, ys) in enumerate(zip(X, Y)):
    plt.plot(xs, ys, 'ro', markersize=10 + 10*(i+1)/len(X))

plt.show()

"""
<Comparing with the high leve API>
to compare the results, let's now execute the whole loop with Emukit.
"""

X = np.array([[0.1], [0.6], [0.9]])
Y = target_function(X)

bo_loop = GPBayesianOptimization(variables_list=space.parameters, X=X, Y=Y)
bo_loop.run_optimization(target_function, num_iterations)

"""
now let's print the results of this optimization and compare it to the previous external evaluation run.
A before, the size of the marker corresponds to its evaluation order.
"""

x = np.arange(0.0, 1.0, 0.01)
y = target_function(x)

plt.figure()
plt.plot(x, y)
for i, (xs, ys) in enumerate(zip(bo_loop.model.model.X, bo_loop.model.model.Y)):
    plt.plot(xs, ys, 'ro', markersize=10+10*(i+1)/len(bo_loop.model.model.X))

plt.show()