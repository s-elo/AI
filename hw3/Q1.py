import numpy as np
import math
import matplotlib.pyplot as plt
from rbf import Rbfn
import random

np.random.seed(22)

train_x = np.arange(-1.6, 1.6, 0.08)
train_x = np.reshape(train_x, (train_x.shape[0], 1))
# nosie
normal_gaussian = 0.3 * \
    np.random.normal(loc=0, scale=1, size=train_x.shape[0]).reshape(
        (train_x.shape[0], 1))
train_y = 1.2 * np.sin(math.pi * train_x) - \
    np.cos(2.4 * math.pi * train_x) + normal_gaussian
print(
    f'training input shape: {train_x.shape}, training label shape: {train_y.shape}')

test_x = np.arange(-1.6, 1.6, 0.01)
test_x = np.reshape(test_x, (test_x.shape[0], 1))
test_y = 1.2 * np.sin(math.pi * test_x) - np.cos(2.4 * math.pi * test_x)
print(f'test input shape: {test_x.shape}, test label shape: {test_y.shape}')
print('\n')

rbf = Rbfn()

print(f'=======Exact Interpolation======')
approximator = rbf.fit(train_x, train_y, strategy='interpolation', std=0.1)

outputs = approximator(test_x)

plt.figure()
plt.plot(test_x, test_y)
plt.plot(test_x, outputs)
plt.legend(['testset', 'approximated'])
plt.title(f'Exact Interpolation')

print(f'=======Fixed Centers Selected at Random======')
# use the appropriate size std
approximator = rbf.fit(train_x, train_y, strategy='fix', std=0)

outputs = approximator(test_x)

plt.figure()
plt.plot(test_x, test_y)
plt.plot(test_x, outputs)
plt.legend(['testset', 'approximated'])
plt.title(f'Fixed Centers Selected at Random')

print(f'=======Exact Interpolation With Regularization======')
regs = [0.01, 0.1, 1, 10]

fig, ax = plt.subplots(1, 4, figsize=(16,5))

for idx, reg in enumerate(regs):
    approximator = rbf.fit(
        train_x, train_y, strategy='interpolation', regularization=reg, std=0.1)

    outputs = approximator(test_x)

    ax[idx].plot(test_x, outputs, label=f'reg: {reg}')
    ax[idx].plot(test_x, test_y, label=f'testset')
    ax[idx].legend()

fig.suptitle(f'Exact Interpolation With Regularization')

plt.show()
