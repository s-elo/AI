import numpy as np
import matplotlib.pyplot as plt
from perceptron import Perceptron
import sys


def draw_line(weights, input_data, input_labels, regression=False, title=''):
    if regression:
        bias = weights[1]
        slop = weights[0]
    else:
        bias = -weights[2] / weights[1]
        slop = -weights[0] / weights[1]

    plt.figure()
    plt.plot([-1, 5], [-1*slop + bias, 5*slop + bias])

    if regression:
        plt.scatter(input_data, input_labels[0:], c='red', marker='x')
    else:
        plt.scatter(input_data[0:1], input_data[1:], c='red', marker='x')

    plt.title(title)


def Q3():
    data = {
        'OR': (np.array([[0, 0], [1, 0], [1, 1], [0, 1]]).T, np.array([[0, 1, 1, 1]])),
        'AND': (np.array([[0, 0], [1, 0], [1, 1], [0, 1]]).T, np.array([[0, 0, 1, 0]])),
        'NAND': (np.array([[0, 0], [1, 0], [1, 1], [0, 1]]).T, np.array([[1, 1, 0, 1]])),
        'COMPLEMENT': (np.array([[0], [1]]).T, np.array([[1, 0]])),
        'XOR': (np.array([[0, 0], [1, 0], [1, 1], [0, 1]]).T, np.array([[0, 1, 0, 1]]))
    }

    learning_rate = [1, 0.01, 0.001]

    for logic in data:
        print(f'{logic}:')
        # (feature, sample)
        x = data[logic][0]
        # (1, sample)
        y = data[logic][1]

        for lr in learning_rate:
            print(
                f'==================learning_rate: {lr}=====================')
            classifier = Perceptron(input_shape=x.shape)

            # (input_feature, neuron_num)
            classifier.add_layer(neuron_num=1, activation='step')

            weights = classifier.train(x, y, learning_rate=lr, epochs=100)
            print(f'{logic} weights (w1, w2, ..., wn, b): {weights[0].T[0]}')
        print('\n')

        # except complement
        if x.shape != data['COMPLEMENT'][0].shape:
            draw_line(weights[0].T[0], x, y,
                      title=f'{logic} (learning_rate: 0.001)')


def Q4():
    regression_data = np.array(
        [[0], [0.8], [1.6], [3], [4], [5]]).T
    regression_labels = np.array([[0.5, 1, 4, 5, 6, 8]])

    regression = Perceptron(input_shape=regression_data.shape)

    LLS_weights = regression.regression_LLS(regression_data, regression_labels)
    print(f'LLS weights (w, b): {LLS_weights.T[0]}')

    LMS_weights, w_cache, b_cache = regression.regression_LMS(
        regression_data, regression_labels, learning_rate=0.01)
    print(f'LMS weights (w, b): {LMS_weights.T[0]}')

    draw_line(LLS_weights, regression_data,
              regression_labels, regression=True, title='LLS regression')

    draw_line(LMS_weights, regression_data,
              regression_labels, regression=True, title='LMS regression')

    plt.figure()
    plt.plot(np.arange(0, 100, 1), w_cache)
    plt.plot(np.arange(0, 100, 1), b_cache)
    plt.legend(['w', 'b'])
    plt.title('Trajectories of the weights versus learning steps')

    # regression.add_layer(neuron_num=1, activation='linear')
    # LMS_w = regression.train(
    #     regression_data, regression_labels, epochs=100, learning_rate=0.01)
    # print(LMS_w)


args = sys.argv[1]

if args == 'Q3':
    Q3()
elif args == 'Q4':
    Q4()
else:
    Q3()

plt.show()
