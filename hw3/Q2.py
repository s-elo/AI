from scipy.io import loadmat
import numpy as np
from rbf import Rbfn
import matplotlib.pyplot as plt


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


def a():
    print(f'========a. Exact Interpolation==========')
    regs = [0, 0.1, 1, 10]
    for reg in regs:
        if reg == 0:
            print(f'===exact interpolation without regularization===')
        else:
            print(f'===exact interpolation with regularization {reg}===')
        approximator = rbf.fit(train_data, train_label,
                               strategy='interpolation', regularization=reg, std=100)
        train_outputs = approximator(train_data)
        test_outputs = approximator(test_data)
        train_acc = rbf.get_classification_score(
            train_outputs, nor_train_label)
        test_acc = rbf.get_classification_score(test_outputs, nor_test_label)
        print(f'train accuracy: {train_acc * 100}%')
        print(f'test accuracy: {test_acc * 100}%')


print('\n')


def b():
    print(f'========b. Fixed Centers Selected at Random==========')
    stds = [0, 0.1, 1, 10, 100, 1000, 10000]
    for std in stds:
        if std == 0:
            print(f'===widths fixed at an appropriate size===')
        else:
            print(f'===widths is {std}===')

        approximator = rbf.fit(train_data, train_label,
                               strategy='fix', std=std, center_num=100)

        train_outputs = approximator(train_data)
        test_outputs = approximator(test_data)
        train_acc = rbf.get_classification_score(
            train_outputs, nor_train_label)
        test_acc = rbf.get_classification_score(test_outputs, nor_test_label)
        print(f'train accuracy: {train_acc * 100}%')
        print(f'test accuracy: {test_acc * 100}%')


def c():
    print(f'========c. K-Mean Clustering==========')
    approximator, k_mean_centers = rbf.fit(train_data, train_label,
                                           strategy='k_mean', std=0.1, center_num=2)

    train_outputs = approximator(train_data)
    test_outputs = approximator(test_data)
    train_acc = rbf.get_classification_score(
        train_outputs, nor_train_label)
    test_acc = rbf.get_classification_score(test_outputs, nor_test_label)
    print(f'train accuracy: {train_acc * 100}%')
    print(f'test accuracy: {test_acc * 100}%')

    # seperate the training data for each class
    c0 = []
    c1 = []
    for idx in range(0, nor_train_label.shape[0]):
        c = nor_train_label[idx]
        if c == 0:
            c0.append(train_data[idx])
        elif c == 1:
            c1.append(train_data[idx])
    plt.figure()
    plt.subplot(221)
    plt.imshow(np.mean(c0, axis=0).reshape((28, 28)), cmap='gray')
    plt.title(f'mean of training data')
    plt.subplot(222)
    plt.imshow(np.mean(c1, axis=0).reshape((28, 28)), cmap='gray')
    plt.title(f'mean of training data')
    plt.subplot(223)
    plt.imshow(k_mean_centers[0].reshape((28, 28)), cmap='gray')
    plt.title(f'center of k mean')
    plt.subplot(224)
    plt.imshow(k_mean_centers[1].reshape((28, 28)), cmap='gray')
    plt.title(f'center of k mean')

    # means = np.array([np.mean(c0, axis=0), np.mean(c1, axis=0)])
    # # c0_mean = np.mean(c0, axis=0)
    # # c1_mean = np.mean(c1, axis=0)
    # print(means[0].shape, k_mean_centers[0].shape)

    # dims = [24, 29]

    # print(means[:, dims[1]])
    # plt.figure()
    # plt.scatter(np.array(c0)[:, dims[0]], np.array(c0)
    #             [:, dims[1]], s=5, c='blue', label='class0')
    # plt.scatter(np.array(c1)[:, dims[0]], np.array(c1)
    #             [:, dims[1]], s=5, c='green', label='class1')
    # plt.scatter(means[:, dims[0]], means[:, dims[1]],
    #             s=200, c='red', label='mean of training data')
    # plt.scatter(k_mean_centers[:, dims[0]],
    #             k_mean_centers[:, dims[1]], s=200, c='black', label='center of k mean')
    # plt.legend()

    # print(np.abs(means - k_mean_centers).sum() / (2 * 784))


# a()
# b()
c()

plt.show()
