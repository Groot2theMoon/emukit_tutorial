###General imports
#%matplotlib inline
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors as mcolors

###Figure config
LEGEND_SIZE = 15

"""
define example 1-D forrester function (6x-2)^2 * sin(12x-4), which is defined over interval x=[0,1]
"""
from emukit.test_functions import forrester_function
from emukit.core.loop.user_function import UserFunctionWrapper
from emukit.core import ContinuousParameter, ParameterSpace

target_function, space = forrester_function()

"""
the [space] object defines the input space X=[0,1] which in this case is purely continuous and only one dimensional.
In a later section we will see how we can also apply Bayesian optimization in other domains that contain discrete or categorical parameters.

of course in reality, evaluating f on a grid wouldn't be possible.
"""

x_plot = np.linspace(space.parameters[0].min, space.parameters[0].max, 200)[:, None]
y_plot = target_function(x_plot)

plt.figure(figsize=(12,8))
plt.plot(x_plot, y_plot, "k", label="Objective Function")
plt.legend(loc=2, prop={'size': LEGEND_SIZE})
plt.xlabel(r"$x$")
plt.ylabel(r"$f(x)$")
plt.grid(True)
plt.xlim(0,1)
plt.show()

"""
<The Initial Design>
Useually, before we star the actual BO loop we need to gather a few observations such that we can fit the model.
This is called the initial design and common strategies are either a predefined grid or sampling points uniformly at random.
"""

X_init = np.array([[0.2], [0.6], [0.9]])
Y_init = target_function(X_init)

plt.figure(figsize=(12,8))
plt.plot(X_init, Y_init, "ro", markersize=10, label="Observations")
plt.plot(x_plot, y_plot, "k", label="Objective Function")
plt.legend(loc=2, prop={'size': LEGEND_SIZE})
plt.xlabel(r"$x$")
plt.ylabel(r"$f(x)$")
plt.grid(True)
plt.xlim(0,1)
plt.show()

"""
<The Model>
Now we can start with the BO loop by first fitting a model on the collected data. 
The arguably most popular model for BO is a Gaussian process (GP) which defines a probability distribution across classes of functions, typically smooth, 
such that each linear finite-dimensional restriction is multivariate Gaussian. GPs are fully parametrized by a mean μ(x) and a covariance function k(x,x′). 
Without loss of generality μ(x) is assumed to be zero. The covariance function k(x,x′) characterizes the smoothness and other properties of f. 

It is known that the kernel of the process and has to be continuous, symmetric and positive definite. A widely used kernel is the squared exponential or
 RBF kernel: k(x,x′)=θ0⋅exp(−∥x−x′∥2θ1)
where θ0 and and θ1 are hyperparameters. 

To denote that f is a sample from a GP with mean μ and covariance k we write f(x)∼GP(μ(x),k(x,x′)).

For regression tasks, the most important feature of GPs is that process priors are conjugate to the likelihood from finitely many observations y=(y1,…,yn)T
and X={x1,...,xn} , xi∈X  of the form yi=f(xi)+ϵi  where ϵi∼N(0,σnoise) and we estimate σnoise by an additional hyperparameter θ2. 
 
We obtain the Gaussian posterior f(x∗)|X,y,θ∼N(μ(x∗),σ2(x∗)), where μ(x∗) and σ2(x∗) have a close form.

Note that Gaussian process are also characterized by hyperparameters θ={θ0,...θk} such as for instance the kernel lengthscales. 

For simplicitly we keep these hyperparameters fixed here. 
However, we usually either optimize or sample these hyperparameters using the marginal loglikelihood of the GP. 

Of course we could also use any other model that returns a mean μ(x) and variance σ2(x) on an arbitrary input points x such as Bayesian neural networks or random forests.

"""

import GPy
from emukit.model_wrappers.gpy_model_wrappers import GPyModelWrapper

gpy_model = GPy.models.GPRegression(X_init, Y_init, GPy.kern.RBF(1, lengthscale=0.08, variance=20), noise_var=1e-10)
emukit_model = GPyModelWrapper(gpy_model)

mu_plot, var_plot = emukit_model.predict(x_plot)

plt.figure(figsize=(12,8))
plt.plot(X_init, Y_init, "ro", markersize=10, label="Observations")
plt.plot(x_plot, y_plot, "k", label="Objective Function")

plt.plot(x_plot, mu_plot, "C0", label="Model")

plt.fill_between(x_plot[:, 0],
                 mu_plot[:, 0] + np.sqrt(var_plot)[:, 0],
                 mu_plot[:, 0] - np.sqrt(var_plot)[:, 0], color="C0", alpha=0.6) #fill_between alpha : Transparaency (0-1)
plt.fill_between(x_plot[:, 0],
                 mu_plot[:, 0] + 2 * np.sqrt(var_plot)[:, 0],
                 mu_plot[:, 0] - 2 * np.sqrt(var_plot)[:, 0], color="C0", alpha=0.4)
plt.fill_between(x_plot[:, 0],
                 mu_plot[:, 0] + 3 * np.sqrt(var_plot)[:, 0],
                 mu_plot[:, 0] - 3 * np.sqrt(var_plot)[:, 0], color="C0", alpha=0.2)

plt.legend(loc=2, prop={'size': LEGEND_SIZE})
plt.xlabel(r"$x$")
plt.ylabel(r"$f(x)$")
plt.grid(True)
plt.xlim(0,1)
plt.show()

"""
<The Acquisition Function>
the second step of our BO loop we use our model to compute the acquisition function.
Various different acquisition functions exist such as :

1) Probability of Improvement (PI) : Given the currently best observed value y⋆∈argmin{y0,…,yn}
, PI simply maximizes
aPI(x)=Φ(γ(x))

where γ(x)=y⋆−μ(x)σ(x) and Φ is the CDF of a standard normal distribution

2) Negative Lower Confidence Bound (NLCB) : This acuisition function is based on the famous upper confidence bound bandit strategy.
It maximized the function:
aLCB=−(μ(x)−βσ(x))

where β is a user-defined hyperparameter that controls exploitation / exploration.

3) Expected Improvement (EI) : probably the most often used acquisition function
computes : Ep(f|D)[max(y⋆−f(x),0)]

where y⋆∈argmin{y0,…,yn}. Assuming p(f|D) to be a Gaussian, we can compute EI in closed form by:

σ(x)(γ(x)Φ(γ(x)))+ϕ(γ(x))

here γ(x)=y⋆−μ(x)σ(x) and Φ is the CDF and ϕ is the PDF of a standard normal distribution.

All of these acquisition function ony rely on the model and hence are cheap to evaluate. 
Furthermore we can easily compute the gradients and use a simple gradient optimization method to find xn+1∈argmaxx∈Xa(x)
"""

from emukit.bayesian_optimization.acquisitions import ExpectedImprovement, NegativeLowerConfidenceBound, ProbabilityOfImprovement

ei_acquisition = ExpectedImprovement(emukit_model)
nlcb_acquisition = NegativeLowerConfidenceBound(emukit_model)
pi_acquisition = ProbabilityOfImprovement(emukit_model)

ei_plot = ei_acquisition.evaluate(x_plot)
nlcb_plot = nlcb_acquisition.evaluate(x_plot)
pi_plot = pi_acquisition.evaluate(x_plot)

plt.figure(figsize=(12, 8))
plt.plot(x_plot, (ei_plot - np.min(ei_plot)) / (np.max(ei_plot) - np.min(ei_plot)), "green", label="EI")
plt.plot(x_plot, (nlcb_plot - np.min(nlcb_plot)) / (np.max(nlcb_plot) - np.min(nlcb_plot)), "purple", label="NLCB")
plt.plot(x_plot, (pi_plot - np.min(pi_plot)) / (np.max(pi_plot) - np.min(pi_plot)), "orange", label="PI")

plt.legend(loc=1, prop={'size': LEGEND_SIZE})
plt.xlabel(r"$x$")
plt.ylabel(r"$f(x)$")
plt.grid(True)
plt.xlim(0,1)
plt.show()

"""
<Evaluating the objective function>
To find the next point to evaluate we optimize the acquisition function using a standard gradient descent optimizer.
"""

from emukit.core.optimization import GradientAcquisitionOptimizer

optimizer = GradientAcquisitionOptimizer(space)
x_new, _ = optimizer.optimize(ei_acquisition)
x_new_nlcb, _ = optimizer.optimize(nlcb_acquisition)
x_new_pi, _ = optimizer.optimize(pi_acquisition)

plt.figure(figsize=(12,8))
plt.plot(x_plot, (ei_plot - np.min(ei_plot)) / (np.max(ei_plot) - np.min(ei_plot)), "red", label="EI")
plt.axvline(x_new, color="darkgreen", label="x_next", linestyle="--")
"""
plt.plot(x_plot, (nlcb_plot - np.min(nlcb_plot)) / (np.max(nlcb_plot) - np.min(nlcb_plot)), "purple", label="NLCB")
plt.axvline(x_new_nlcb, color="purple", label="x_next_nlcb", linestyle="--")

plt.plot(x_plot, (pi_plot - np.min(pi_plot)) / (np.max(pi_plot) - np.min(pi_plot)), "orange", label="PI")
plt.axvline(x_new_pi, color="darkorange", label="x_next_pi", linestyle="--")
"""
plt.legend(loc=1, prop={'size': LEGEND_SIZE})
plt.xlabel(r"$x$")
plt.ylabel(r"$f(x)$")
plt.grid(True)
plt.xlim(0,1)
plt.show()

"""
Afterwards we evaluate the true objective function and append it to our initial observations.
"""
y_new = target_function(x_new)

X = np.append(X_init, x_new, axis=0)
Y = np.append(Y_init, y_new, axis=0)

"""
After updating the model, you can see that the uncertainty about the true objective function in this region decreases and our model becomes more certain.
"""
emukit_model.set_data(X, Y)

mu_plot, var_plot = emukit_model.predict(x_plot)

plt.figure(figsize=(12,8))
plt.plot(emukit_model.X, emukit_model.Y, "ro", markersize=10, label="Observations")
plt.plot(x_plot, y_plot, "k", label="Objective Function")
plt.plot(x_plot, mu_plot, "C0", label="Model")

plt.fill_between(x_plot[:, 0],
                 mu_plot[:, 0] + np.sqrt(var_plot)[:, 0],
                 mu_plot[:, 0] - np.sqrt(var_plot)[:, 0], color="C0", alpha=0.6) #fill_between alpha : Transparaency (0-1)
plt.fill_between(x_plot[:, 0],
                 mu_plot[:, 0] + 2 * np.sqrt(var_plot)[:, 0],
                 mu_plot[:, 0] - 2 * np.sqrt(var_plot)[:, 0], color="C0", alpha=0.4)
plt.fill_between(x_plot[:, 0],
                 mu_plot[:, 0] + 3 * np.sqrt(var_plot)[:, 0],
                 mu_plot[:, 0] - 3 * np.sqrt(var_plot)[:, 0], color="C0", alpha=0.2)

plt.legend(loc=2, prop={'size': LEGEND_SIZE})
plt.xlabel(r"$x$")
plt.ylabel(r"$f(x)$")
plt.grid(True)
plt.xlim(0,1)
plt.show()

"""
<Emukit's Bayesian optimization interface>
of course inpractice we don't want to implement all of these steps our self. Emukit provides a convenient and flexible interface to apply Bayesian optimaztion.
Below we can see how to run Bayesian optimization of the exact same function for 10 iterations. 
"""

from emukit.examples.gp_bayesian_optimization.single_objective_bayesian_optimization import GPBayesianOptimization

bo = GPBayesianOptimization(variables_list=[ContinuousParameter('x1', 0, 1)],
                            X=X_init, Y=Y_init)
bo.run_optimization(target_function, 10)

mu_plot, var_plot = bo.model.predict(x_plot)

plt.figure(figsize=(12,8))

plt.plot(bo.loop_state.X, bo.loop_state.Y, "ro", markersize=10, label="Observations")
plt.plot(x_plot, y_plot, "k", label="Objective Function")
plt.plot(x_plot, mu_plot, "C0", label="Model")
plt.fill_between(x_plot[:, 0],
                 mu_plot[:, 0] + np.sqrt(var_plot)[:, 0],
                 mu_plot[:, 0] - np.sqrt(var_plot)[:, 0], color="C0", alpha=0.6)

plt.fill_between(x_plot[:, 0],
                 mu_plot[:, 0] + 2 * np.sqrt(var_plot)[:, 0],
                 mu_plot[:, 0] - 2 * np.sqrt(var_plot)[:, 0], color="C0", alpha=0.4)

plt.fill_between(x_plot[:, 0],
                 mu_plot[:, 0] + 3 * np.sqrt(var_plot)[:, 0],
                 mu_plot[:, 0] - 3 * np.sqrt(var_plot)[:, 0], color="C0", alpha=0.2)
plt.legend(loc=2, prop={'size': LEGEND_SIZE})
plt.xlabel(r"$x$")
plt.ylabel(r"$f(x)$")
plt.grid(True)
plt.xlim(0, 1)

plt.show()