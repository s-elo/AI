import numpy as np
import matplotlib.pyplot as plt
from perceptron import Perceptron


def draw_line(weights, input_data, input_labels, regression=False):
    if regression:
        bias = weights[1]
        slop = weights[0]
    else:
        bias = -weights[1] / weights[2]
        slop = -weights[0] / weights[2]

    plt.figure()
    plt.plot([-1, 5], [-1*slop + bias, 5*slop + bias])

    if regression:
        plt.scatter(input_data, input_labels[0:], c='red', marker='x')
    else:
        plt.scatter(input_data[0:1], input_data[1:], c='red', marker='x')


def Q3():
    # (feature, sample)
    X = np.array([[0, 0], [1, 0], [1, 1], [0, 1]]).T
    Y = np.array([[0, 1, 1, 1]])

    classifier = Perceptron(input_shape=X.shape)

    # (input_feature, neuron_num)
    classifier.add_layer(neuron_num=1, activation='step')

    weights = classifier.train(X, Y, learning_rate=0.001, epochs=100)
    print(weights)

    draw_line(weights.T[0], X, Y)

    plt.show()


def Q4():
    regression_data = np.array(
        [[0], [0.8], [1.6], [3], [4], [5]]).T
    regression_labels = np.array([[0.5, 1, 4, 5, 6, 8]])

    regression = Perceptron(input_shape=regression_data.shape)

    LLS_weights = regression.regression_LLS(regression_data, regression_labels)
    print(LLS_weights)

    LMS_weights, w_cache, b_cache = regression.regression_LMS(
        regression_data, regression_labels, learning_rate=0.01)
    print(LMS_weights)

    draw_line(LLS_weights, regression_data,
              regression_labels, regression=True)

    draw_line(LMS_weights, regression_data,
              regression_labels, regression=True)

    plt.figure()
    plt.plot(np.arange(0, 100, 1), w_cache)
    plt.plot(np.arange(0, 100, 1), b_cache)
    plt.legend(['w', 'b'])

    regression.add_layer(neuron_num=1, activation='linear')
    LMS_w = regression.train(
        regression_data, regression_labels, epochs=100, learning_rate=0.01)
    print(LMS_w)

    plt.show()


# Q3()
Q4()
