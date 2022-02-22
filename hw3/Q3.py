import numpy as np
import numpy.matlib as mb
import scipy.special as sc
import math
import matplotlib.pyplot as plt
from som import Som


def a():
    print(f'=====one dimensional SOM for hat=====')
    train_data = np.linspace(-math.pi, math.pi, 400)
    # (sample, feature)
    train_data = np.array([train_data, np.sinc(train_data)]).T

    # print(train_data.shape)
    plt.figure()
    plt.plot(train_data[:, 0], train_data[:, 1], marker='+', c='red')

    som = Som()

    neuron_weights = som.one_dim_som(train_data, map_size=(1, 40), max_iter=8000)
    # print(neuron_weights.shape)
    plt.plot(neuron_weights[:, 0], neuron_weights[:, 1], marker='o', c='blue')
    plt.legend(['train data', 'neuron weights'])
    plt.title('one dimensional SOM')


def b():
    x = np.random.randn(800, 2)
    s = np.sum(np.multiply(x, x), axis=1).reshape(800, 1)
    train_data = np.multiply(x, mb.repmat(
        np.divide(1 * np.sqrt(sc.gammainc(s/2, 1)), np.sqrt(s)), 1, 2))

    print(train_data.shape)
    plt.figure()
    plt.scatter(train_data[:, 0], train_data[:, 1], marker='+', c='red')


# a()
b()

plt.show()
