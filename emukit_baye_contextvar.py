"""
<Bayesian optimization with context variables>
In this notebook we are going to see how to use Emukit to solve optimization problems in which certain variables are fixed during the optimization phase.
Thesea re called context variables.
This is useful when some of the variable s in the optimization are controllable/known factors. 
example is the optimization of the movement of a robot under conditions of the "known" environment change.
"""

from emukit.test_functions import branin_function
from emukit.core import ParameterSpace, ContinuousParameter, DiscreteParameter
from emukit.core.initial_designs import RandomDesign
from GPy.models import GPRegression
from emukit.model_wrappers import GPyModelWrapper
from emukit.bayesian_optimization.acquisitions import ExpectedImprovement
from emukit.bayesian_optimization.loops import BayesianOptimizationLoop
from emukit.core.loop import FixedIterationsStoppingCondition

# loading the problem and the loop

f, parameter_space = branin_function()

# define the domain of the function to optimize. 
# we build the model. 

design = RandomDesign(parameter_space) # collect random points.
X = design.get_samples(10)
Y = f(X)
model_gpy = GPRegression(X,Y) # train and wrapt the model in Emukit
model_emukit = GPyModelWrapper(model_gpy)

# prepare the optimization object to run the loop

expected_improvement = ExpectedImprovement(model = model_emukit)
bayesopt_loop = BayesianOptimizationLoop(model = model_emukit,
                                         space = parameter_space,
                                         acquisition = expected_improvement,
                                         batch_size=1)

# set the number of iterations to run to 10

max_iter = 10

"""
<Running the optimization by setting a context variabel>
set a context, we just need to create a dictionary with the variables to fix and pass it to the Bayesian optimization object when running the optimization. 
Note that, everytime we run new iterations we can set other variables to be the context. 
We run 3 sequences of 10 iterations each with different values of the context. 
"""

bayesopt_loop.run_loop(f, max_iter, context={'x1':0.3}) # we set x1 as the context variable
bayesopt_loop.run_loop(f, max_iter, context={'x2':0.1}) # we set x2 as the context variable
bayesopt_loop.run_loop(f, max_iter) # no context

# inspect the collected points 

bayesopt_loop.loop_state.X

