import numpy as np
import math
import matplotlib.pyplot as plt
from som import Som


def a():
    train_data = np.linspace(-math.pi, math.pi, 400)
    # (sample, feature)
    train_data = np.array([train_data, np.sinc(train_data)]).T

    print(train_data.shape)
    plt.figure()
    plt.plot(train_data[:, 0], train_data[:, 1], marker='+', c='red')

    som = Som()

    neuron_weights = som.one_dim_som(train_data, map_size=(1, 40), max_iter=20)
    print(neuron_weights.shape)
    plt.plot(neuron_weights[:, 0], neuron_weights[:, 1], marker='o', c='blue')

a()

plt.show()
