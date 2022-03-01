from scipy.io import loadmat
import numpy as np
from rbf import Rbfn
import matplotlib.pyplot as plt
import math


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
    regs = [0, 0.01, 0.1, 1, 10, 20]

    row_num = 2
    col_num = 3
    fig, ax = plt.subplots(2, 3, figsize=(16, 10))

    for idx, reg in enumerate(regs):
        if reg == 0:
            print(f'===exact interpolation without regularization===')
            title = f'without regularization'
        else:
            print(f'===exact interpolation with regularization {reg}===')
            title = f'regularization {reg}'
        approximator = rbf.fit(train_data, nor_train_label,
                               strategy='interpolation', regularization=reg, std=100)
        train_outputs = approximator(train_data)
        test_outputs = approximator(test_data)

        tr_acc, te_acc, thrs = rbf.get_classification_score(
            train_outputs, test_outputs, nor_train_label, nor_test_label)

        print(f'train final accuracy: {tr_acc[len(tr_acc) - 1] * 100}%')
        print(f'train maximum accuracy: {max(tr_acc) * 100}%')
        print(f'test final accuracy: {te_acc[len(te_acc) - 1] * 100}%')
        print(f'test maximum accuracy: {max(te_acc) * 100}%')

        col = idx % col_num
        row = math.floor(idx / col_num)

        ax[row, col].plot(thrs, tr_acc, label='train accuracy')
        ax[row, col].plot(thrs, te_acc, label='test accuracy')
        ax[row, col].legend()
        ax[row, col].set_xlabel('treshold')
        ax[row, col].set_ylabel('accuracy')
        ax[row, col].set_title(title)

        # one hot format
        # train_acc = rbf.get_classification_score(
        #     train_outputs, nor_train_label)
        # test_acc = rbf.get_classification_score(test_outputs, nor_test_label)
        # print(f'train accuracy: {train_acc * 100}%')
        # print(f'test accuracy: {test_acc * 100}%')


print('\n')


def b():
    print(f'========b. Fixed Centers Selected at Random==========')
    stds = [0, 0.1, 1, 10, 50, 100, 1000, 10000, 20, 70]

    col_num = 4
    fig, ax = plt.subplots(2, 4, figsize=(16, 10))

    max_te_acc = []

    for idx, std in enumerate(stds):
        if std == 0:
            print(f'===widths fixed at an appropriate size===')
            title = f'appropriate width'
        else:
            print(f'===widths is {std}===')
            title = f'widths is {std}'

        approximator = rbf.fit(train_data, nor_train_label,
                               strategy='fix', std=std, center_num=100)

        train_outputs = approximator(train_data)
        test_outputs = approximator(test_data)

        tr_acc, te_acc, thrs = rbf.get_classification_score(
            train_outputs, test_outputs, nor_train_label, nor_test_label)

        print(f'train final accuracy: {tr_acc[len(tr_acc) - 1] * 100}%')
        print(f'train maximum accuracy: {max(tr_acc) * 100}%')
        print(f'test final accuracy: {te_acc[len(te_acc) - 1] * 100}%')
        print(f'test maximum accuracy: {max(te_acc) * 100}%')

        max_te_acc.append(max(te_acc))

        if idx >= 8:
            continue

        col = idx % col_num
        row = math.floor(idx / col_num)

        ax[row, col].plot(thrs, tr_acc, label='train accuracy')
        ax[row, col].plot(thrs, te_acc, label='test accuracy')
        ax[row, col].legend()
        ax[row, col].set_xlabel('treshold')
        ax[row, col].set_ylabel('accuracy')
        ax[row, col].set_title(title)

        # one hot format
        # train_acc = rbf.get_classification_score(
        #     train_outputs, nor_train_label)
        # test_acc = rbf.get_classification_score(test_outputs, nor_test_label)
        # print(f'train accuracy: {train_acc * 100}%')
        # print(f'test accuracy: {test_acc * 100}%')

    plt.figure()
    sorted_idx = np.argsort(stds)
    plt.plot(np.array(stds)[sorted_idx], np.array(max_te_acc)[sorted_idx])
    plt.xlabel('width')
    plt.ylabel('maximum test accuracy')
    plt.title('maximum test accuracy with different width')


def c():
    print(f'========c. K-Mean Clustering==========')
    approximator, k_mean_centers = rbf.fit(train_data, nor_train_label,
                                           strategy='k_mean', std=0.1, center_num=2)

    train_outputs = approximator(train_data)
    test_outputs = approximator(test_data)

    tr_acc, te_acc, thrs = rbf.get_classification_score(
        train_outputs, test_outputs, nor_train_label, nor_test_label)

    print(f'train final accuracy: {tr_acc[len(tr_acc) - 1] * 100}%')
    print(f'train maximum accuracy: {max(tr_acc) * 100}%')
    print(f'test final accuracy: {te_acc[len(te_acc) - 1] * 100}%')
    print(f'test maximum accuracy: {max(te_acc) * 100}%')

    # one hot
    # train_acc = rbf.get_classification_score(
    #     train_outputs, nor_train_label)
    # test_acc = rbf.get_classification_score(test_outputs, nor_test_label)
    # print(f'train accuracy: {train_acc * 100}%')
    # print(f'test accuracy: {test_acc * 100}%')

    plt.plot(thrs, tr_acc, label='train accuracy')
    plt.plot(thrs, te_acc, label='test accuracy')
    plt.legend()
    plt.xlabel('treshold')
    plt.ylabel('accuracy')
    plt.title('k mean accuracy')

    # seperate the training data for each class
    c0 = []
    c1 = []
    for idx in range(0, nor_train_label.shape[0]):
        c = nor_train_label[idx]
        if c == 0:
            c0.append(train_data[idx])
        elif c == 1:
            c1.append(train_data[idx])
    plt.figure(figsize=(5, 7))
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


a()
b()
c()

plt.show()
