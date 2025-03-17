# example data
import matplotlib.pyplot as plt
import numpy as np
rng = np.random.RandomState(1)

#example data visualization
X = np.linspace(0,10,1000).reshape(-1,1)
y= np.squeeze(X*np.sin(X))

fig, ax = plt.subplots(figsize=(12,8))
ax.plot(X,y)

plt.show()

"""
임의의 x좌표 6개를 골라 예제 데이터를 뽑는다. (evidence)
1) 측정을 하면 정확한 값을 찾아내는 경우
2) 측정이 불확실성을 안고 있는 경우. x 하나당 10회 측정, 표준편차=1 이라고 가정.
"""

training_indices = rng.choice(np.arange(y.size), size=6, replace=False)

# @ exact observation
X_train = X[training_indices]
y_train = y[training_indices]

# @ noisy situation
noise_std = 1
X_train_noisy = np.array(X[training_indices].tolist()*10)
y_train_noisy = np.array(list(y[training_indices])*10)+rng.normal(0, noise_std, size=y_train.shape[0]*10)

#visualization
fig, ax = plt.subplots(figsize=(12,8))
ax.plot(X,y, c="lightgray", label="true")
ax.scatter(X_train, y_train, label="sample without noise")
ax.scatter(X_train_noisy, y_train_noisy, label="sample with noise", s=5, alpha=0.5)
ax.legend

plt.show()

"""
<Gaussian Process>
1) without noise
evidence가 참값인 경우의 Gaussian process를 실행. 
커널은 RBF 를 사용. 
length_scale=1, 범위 0.01 ~ 100 으로 fitting 
"""

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF

kernel = 1*RBF(length_scale=1.0, length_scale_bounds=(1e-2, 1e2))
gpr = GaussianProcessRegressor(kernel, n_restarts_optimizer=9)
gpr.fit(X_train, y_train)
print(gpr.kernel_)
print(gpr.kernel_.theta)