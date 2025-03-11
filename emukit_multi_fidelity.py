"""
<An Introduction to Multi-fidelity Modeling in Emukit>

a common issue encoutered when appliying machine learning to environmental sciences and engineering problems is the difficulty or cost required to obtain sufficient 
data for building robust models. possible examples include aerospace and nautical engineering, where it is both infeasible and prohibitively expensive to runan vast number
of experiments using the actual vehicle. Even when there is no physical rtifact involved, sich as in climate modeling, data my still be hard to obtain when these can only
be collected by running an expensive computer experiment, where the time required an individual data sample restricts the volume of data that can later be 
used for modeling. 

Constructing a reliable model when only few observations are available is challenging, which is why it is common practice to devolop simulators of the actual system,
from which data points can be more easily obtained. in engineering applications, such simulators often take the form of CFD tools which approximate the behaviour of the true artifact
for a given design or configuration. However, although it is now possible to obtain more data samples, it is highly unlikely that these simulators model the 
true system exactly. instead, these are expected to contain some degree of bias and noise.

From the above, one can deduce that naivelgy combining observations from multiple information sources could result int he model giving biased predictions which do not
accurately reflect the treu problem. To this end, nultifidelity models are designed to augment the limited true observations available with cheaply-obtained
approximations in a principled manner. in such models, observations obtained from the true source are referred to as high-fidelity observations, whereas approximations
are denoted as being low-fidelity. these low-fidelity observations are then systemically combined with the more accurate (but limited) observations in order to predict the 
high-fidelity output more effectively. Note than we can generally combine information from multiple lower fidelity sources, which can all be seen ase auxillary tasks in 
support of a single primary task.

In this notebook, we shall investigate a selection of multi-fidelty models based on Gaussian processes which are readily available in Emukit. We start by investigaing the
traditional linear multi-fidelity model. Subsequently, we shall illustrate why this model can be unsuitable when the mapping from low to high-fidelity observations is nonlinear,
and demonstrate how an alternate model can alleviate theis issue. 
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors as mcolors
colors = dict(mcolors.BASE_COLORS, **mcolors.CSS4_COLORS)

np.random.seed(20)

"""
<Linear multi-fidelity model>
the linear multi-fidelity model propoesed is widely viewed as a reference point for all such models. In this model, the high-fidelity function is modeled 
as a scaled sum of the low-fidelity function plus ann error term.

fhigh(x)=ferr(x)+ρflow(x)

in this equation, f_low is taken to be a Gaussina process modeling the outputs of the lower fidelity function, while rho is a scaling factor indicating the magnitude of the 
correlation to the high-fidelity data. Setting this to 0 implies that there is no correlation between observations at different fidelities. 
f_error denotes yet another Gaussian process which models the bias term for the high=fidelity data. 
Note that f_error and f_low are assumed to be a independent processes which are only related by the equation given above. 

Note : while we shall limit our explanation to the case of two fidelities, this set-up can easily be generalized to cater for T fidelities as follows : 

ft(x)=ft(x)+ρt−1ft−1(x),t=1,…,T)

If the training points are sorted such that the low and high-fidelity points are grouped together : (X_low X_high)^T

we can express the model as a single Gaussian process having the following prior: 

[f_low(h) f_high(h)]^T ~ GP(0, [[k_low rho*k_low]^T [rho*k_low rho^2*k_low + k_err]^T])
"""

"""
<linear multi-fidelity modeling in emukit>
as a first example of how the linear multi-fidelity model implemented in Emukit [emukit.multi_fidelity.models.GPyLinearMultiFidelityModel] can be used,
we shall consider the two-fidelity forrester function. 
This benchmark is frequently used to illustrate the capapbilities of multi-fidelity models.
"""

import GPy
import emukit.multi_fidelity
import emukit.test_functions
from emukit.model_wrappers.gpy_model_wrappers import GPyMultiOutputWrapper
from emukit.multi_fidelity.models import GPyLinearMultiFidelityModel

# Generate samples from the Forrester fucntion

high_fidelity = emukit.test_functions.forrester.forrester
low_fidelity = emukit.test_functions.forrester.forrester_low

x_plot = np.linspace(0, 1, 200)[: , None]
y_plot_l = low_fidelity(x_plot)
y_plot_h = high_fidelity(x_plot)

x_train_l = np.atleast_2d(np.random.rand(12)).T
x_train_h = np.atleast_2d(np.random.permutation(x_train_l)[:6])
y_train_l = low_fidelity(x_train_l)
y_train_h = high_fidelity(x_train_h)

"""
the inputs to the models ar expected to take the form of ndarrays where the last column where the last column indicates the fidelity of the observed points. 
Although only the input points, X, are augmented with the fidelity level, the observed outputs Y must also be converted to array form.

For example, a dataset consisting of 3 low-fidelity points and 2 high-fidelity points would be represented as follows,
where the input is three-dimensional while the output is one-dimensional.


A similar procedure must be carried out for obtaining predictions at new test oints, whereby the fidelity indicated in the column then indicates the fidelity
at which the function must be predicted for a designated point. 

for convenience of use, we provide helper methods for eaisly converting between list of arrays (ordered from the lowest to the highest fidelity) and the required ndarray
representation. this is found in emukit.multi_fidelity.convert_lists_to_array.
"""

# convert lists of arrays to ndarrays augmented with fidelity indicators

from emukit.multi_fidelity.convert_lists_to_array import convert_x_list_to_array, convert_xy_lists_to_arrays

X_train, Y_train = convert_xy_lists_to_arrays([x_train_l, x_train_h], [y_train_l, y_train_h])

# plot the original functions.

plt.figure(figsize=(12,8))
plt.plot(x_plot, y_plot_l, 'b')
plt.plot(x_plot, y_plot_h, 'r')
plt.scatter(x_train_l, y_train_l, color='b', s=40)
plt.scatter(x_train_h, y_train_h, color='r', s=40)
plt.ylabel('f(x)')
plt.xlabel('x')
plt.legend(['Low fidelity', 'High fidelity'])
plt.title('High and low fidelity Forrester functions');

plt.show()