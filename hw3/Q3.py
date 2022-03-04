import numpy as np
import numpy.matlib as mb
import scipy.special as sc
import math
import matplotlib.pyplot as plt
from scipy.io import loadmat
from som import Som

som = Som()


def a():
    print(f'=====a. one dimensional SOM for hat=====')
    train_data = np.linspace(-math.pi, math.pi, 400)
    # (sample, feature)
    train_data = np.array([train_data, np.sinc(train_data)]).T
    # print(train_data.shape)

    plt.figure()
    plt.plot(train_data[:, 0], train_data[:, 1], marker='+', c='red')

    # (neuron_num, feature_num)
    neuron_weights = som.one_dim_som(
        train_data, map_size=(1, 40), max_iter=8000)
    # print(neuron_weights.shape)
    plt.plot(neuron_weights[:, 0], neuron_weights[:, 1], marker='o', c='blue')
    plt.legend(['train data', 'neuron weights'])
    plt.title('one dimensional SOM')


def b():
    print(f'=====b. two dimensional SOM for circle=====')
    x = np.random.randn(800, 2)
    s = np.sum(np.multiply(x, x), axis=1).reshape(800, 1)
    train_data = np.multiply(x, mb.repmat(
        np.divide(1 * np.sqrt(sc.gammainc(s/2, 1)), np.sqrt(s)), 1, 2))
    # print(train_data.shape)

    plt.figure()
    plt.scatter(train_data[:, 0], train_data[:, 1],
                marker='+', c='red', label='train data')

    # (neuron_row_num, neuron_col_num, feature_num)
    neuron_weights = som.two_dim_som(
        train_data, map_size=(8, 8), max_iter=8000)
    # print(neuron_weights.shape)

    som.draw_neurons(neuron_weights, row_num=8, col_num=8, set_label=True)
    plt.legend(loc='upper left')

    plt.title('two dimensional SOM')


def c():
    print(f'=====c. som for classification=====')
    dataset = loadmat('./dataset/Digits.mat')

    # (784, 1000).T (1, 1000).T (784, 100).T (1, 100).T
    train_data = dataset['train_data'].T
    train_label = dataset['train_classlabel'].T
    test_data = dataset['test_data'].T
    test_label = dataset['test_classlabel'].T

    # omit the class 4 and class 0
    train_x = []
    train_y = []
    test_x = []
    test_y = []
    for idx in range(0, train_label.shape[0]):
        label = train_label[idx]

        if label != 4 and label != 0:
            train_x.append(train_data[idx])
            train_y.append(label)

    for idx in range(0, test_label.shape[0]):
        label = test_label[idx]

        if label != 4 and label != 0:
            test_x.append(test_data[idx])
            test_y.append(label)

    train_x = np.array(train_x)
    train_y = np.array(train_y).reshape((len(train_y), 1))
    test_x = np.array(test_x)
    test_y = np.array(test_y).reshape((len(test_y), 1))
    # print(train_x.shape, train_y.shape,
    #       test_x.shape, test_y.shape)

    iters = [1000, 2000, 5000, 10000, 12000, 18000, 22000]
    train_acc = []
    test_acc = []
    for n in iters:
        print(f'==={n} iterations===')
        classifier, neuron_weights = som.fit(train_x, train_y, max_iter=n)
        # print(neuron_weights.shape)

        if n == 18000:
            som.draw_weight_map(neuron_weights)

        test_outputs = classifier(test_x)
        train_outputs = classifier(train_x)
        # print(outputs.shape)

        train_accuracy = np.sum(train_outputs == train_y) / train_y.shape[0]
        test_accuracy = np.sum(test_outputs == test_y) / test_y.shape[0]
        print(f'The train accuracy: {train_accuracy * 100}%')
        print(f'The test accuracy: {test_accuracy * 100}%')

        train_acc.append(train_accuracy)
        test_acc.append(test_accuracy)

    plt.figure()
    plt.plot(iters, train_acc, label='train set')
    plt.plot(iters, test_acc, label='test set')
    plt.xlabel('iterations')
    plt.ylabel('accruacy')
    plt.legend()
    plt.title('accuracy with different iterations')


a()
b()
c()

plt.show()
