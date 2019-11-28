import numpy as np 
from itertools import product
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from matplotlib import pyplot as plt

def f(x):
    """The function to predict."""
    return x * np.sin(x)

# Truth data
x_true = np.linspace(0, 10, num=100)
y_true = f(x_true).ravel()

# Get data
x_train = np.linspace(0, 10, num=20).reshape(-1, 1)
y_train = f(x_train).ravel()
print(y_train)
# Set the kernel
kernel = C(1.0, (1e-3, 1e3)) * RBF(5, (1e-2, 1e2))

# Instantiate the GP
gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=15)

# Train the GP
gp.fit(x_train, y_train)

# Predict using the GP
x_pred = np.linspace(1, 9, num=30).reshape(-1, 1)
y_pred, sigma = gp.predict(x_pred, return_std=True)

# Plot
plt.figure()
plt.plot(x_true, y_true, 'r:', label=r'$f(x) = x\,\sin(x)$')
plt.plot(x_train, y_train, 'r.', markersize=10, label='Observations')
plt.plot(x_pred, y_pred, 'b-', label='Prediction')
plt.show()