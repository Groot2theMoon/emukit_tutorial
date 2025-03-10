###General imports
%matplotlib inline
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

x_plot = np.linspace(space.parameters[0].min, space.parameters[0].max, 200)[:, None]
y_plot = target_function(x_plot)

plt.figure(figsize=(12,8))
plt.plot(x_plot, )