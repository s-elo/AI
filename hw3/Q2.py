from scipy.io import loadmat
import numpy as np


def loadData():
    dataset = loadmat('./dataset/mnist_m.mat')

    train_data = dataset['train_data']
    train_label = dataset['train_classlabel']
    test_data = dataset['test_data']
    test_label = dataset['test_classlabel']

    train_num = train_data.shape[1]
    test_num = test_data.shape[1]

    # 9, 4 -> 1, remainging -> 0
    new_train_label = []
    new_test_label = []
    for idx in range(0, train_num):
        label = train_label[0][idx]

        if label == 9 or label == 4:
            new_train_label.append(1)
        else:
            new_train_label.append(0)

    for idx in range(0, test_num):
        label = test_label[0][idx]

        if label == 9 or label == 4:
            new_test_label.append(1)
        else:
            new_test_label.append(0)

    return (train_data, np.array(new_train_label).reshape((1, train_num)), test_data, np.array(new_test_label).reshape((1, test_num)))


# (784, 1000) (1, 1000) (784, 250) (1, 250)
train_data, train_label, test_data, test_label = loadData()
