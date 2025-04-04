"""
<Bayesian Optimization integrating model hyper_parameters>,
in this notebook we are going to see how to use Emukit to solve optimization problems 
when the acquisition function is integrated with respect to the hyper-parameteres of the model.

To show this with an example, use the Six-hump camel function.

f(x1,x2)=(4−2.1x21=x413)x21+x1x2+(−4+4x22)x22,
in [−3,3]×[−2,2], This functions has two global minima, at (0.0898,−0.7126) and (−0.0898,0.7126)
"""

import numpy as np

# loading the problem and generate initial data

from emukit.test_functions import sixhumpcamel_function
f, parameter_space = sixhumpcamel_function()

# now we define the domain of the function to optimize.

### --- Generate data
from emukit.core.initial_designs import RandomDesign

design = RandomDesign(parameter_space) # collect random points
num_data_points = 5
X = design.get_samples(num_data_points)
Y = f(X)

# Train the model on the initial data

import GPy

model_gpy_mcmc = GPy.models.GPRegression(X, Y)
model_gpy_mcmc.kern.set_prior(GPy.priors.Uniform(0,5))
model_gpy_mcmc.likelihood.variance.constrain_fixed(0.001)

from emukit.model_wrappers import GPyModelWrapper
model_emukit = GPyModelWrapper(model_gpy_mcmc)
model_emukit.model.plot()
model_emukit.model




"""
<create acquitision function>
We use a combination of [IntegratedHyperParameterAcquisition] and [ExpectedImprovement] classes to create the integrated expected imporvement acquisition object.
the [IntegratedHyperParameterAcquisition] can convert any acquisition function into one that is integrated over model hyper-parameters.

we need to pass a function that will return an acqisition object to [IntegratedHyperParameterAcquisiton] this function takes in the model as an input only.
"""

from emukit.core.acquisition import IntegratedHyperParameterAcquisition
from emukit.bayesian_optimization.acquisitions import ExpectedImprovement

acquisition_generator = lambda model: ExpectedImprovement(model, jitter=0)
expected_improvement_integrated = IntegratedHyperParameterAcquisition(model_emukit, acquisition_generator)

from emukit.bayesian_optimization.loops import BayesianOptimizationLoop

bayesopt_loop = BayesianOptimizationLoop(model = model_emukit,
                                         space = parameter_space,
                                         acquisition = expected_improvement_integrated,
                                         batch_size=1)

max_iter = 10
bayesopt_loop.run_loop(f, max_iter)

# now, once the loop is completed we can visualize the distribution of the hyperparameters given the data.
import matplotlib.pyplot as plt

labels = ['rbf variance', 'rbf lengthscale']

plt.figure(figsize=(14,5))
samples = bayesopt_loop.candidate_point_calculator.acquisition.samples

plt.subplot(1,2,1)
plt.plot(samples, label = labels)
plt.title('Hyperparameters samples',size=25)
plt.xlabel('Sample index',size=15)
plt.ylabel('Value',size=15)

plt.subplot(1,2,2)
from scipy import stats
xmin = samples.min()
xmax = samples.max()
xs = np.linspace(xmin,xmax,100)
for i in range(samples.shape[1]):
    kernel = stats.gaussian_kde(samples[:,i])
    plt.plot(xs,kernel(xs),label=labels[i])
_ = plt.legend()
plt.title('Hyperparameters densities',size=25)
plt.xlabel('Value',size=15)
plt.ylabel('Frequency',size=15)


plt.plot(np.minimum.accumulate(bayesopt_loop.loop_state.Y))
plt.ylabel('Current best')
plt.xlabel('Iteration');
