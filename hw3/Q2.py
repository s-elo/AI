from scipy.io import loadmat
import numpy as np
from rbf import Rbfn


def loadData():
    dataset = loadmat('./dataset/mnist_m.mat')

    # (784, 1000) (1, 1000) (784, 250) (1, 250)
    train_data = dataset['train_data']
    train_label = dataset['train_classlabel']
    test_data = dataset['test_data']
    test_label = dataset['test_classlabel']

    train_num = train_data.shape[1]
    test_num = test_data.shape[1]

    # 9, 4 -> 1, remainging -> 0
    new_train_label = []
    new_test_label = []
    # 9, 4 -> [0,1], remainging -> [1,0] (one-hot)
    one_hot_train_label = []
    one_hot_test_label = []
    for idx in range(0, train_num):
        label = train_label[0][idx]

        if label == 9 or label == 4:
            new_train_label.append(1)
            one_hot_train_label.append([0, 1])
        else:
            new_train_label.append(0)
            one_hot_train_label.append([1, 0])

    for idx in range(0, test_num):
        label = test_label[0][idx]

        if label == 9 or label == 4:
            new_test_label.append(1)
            one_hot_test_label.append([0, 1])
        else:
            new_test_label.append(0)
            one_hot_test_label.append([1, 0])

    return (train_data.T, np.array(one_hot_train_label), test_data.T, np.array(one_hot_test_label), np.array(new_train_label).reshape((train_num, 1)), np.array(new_test_label).reshape((test_num, 1)))


# (1000, 784) (1000, 2) (250, 784) (250, 2) (1000, 1) (250, 1)
train_data, train_label, test_data, test_label, nor_train_label, nor_test_label = loadData()
# print(train_data.shape, train_label.shape, test_data.shape,
#       test_label.shape, nor_train_label.shape, nor_test_label.shape)


rbf = Rbfn()

approximator = rbf.fit(train_data, train_label,
                       strategy='interpolation', regularization=0, std=100)
train_outputs = approximator(train_data)
test_outputs = approximator(test_data)

train_acc = rbf.get_classification_score(train_outputs, nor_train_label)
test_acc = rbf.get_classification_score(test_outputs, nor_test_label)
print(f'train accuracy: {train_acc * 100}%')
print(f'test accuracy: {test_acc * 100}%')
