import math
import time
import numpy as np
from scipy.spatial import cKDTree
import matplotlib.pyplot as plt

class KNearest:
    def __init__(self, train, test, K, kd_tree):
        self.train = train
        self.test = test

        self.K = K
        self.kd_tree = kd_tree

    def get_K_nearest(self, sample):
        # p = 2 means using Euclidean distance
        distances, index = self.kd_tree.query(sample, k=self.K, p=2)

        return index

    def predict(self, sample, isPrint = False):
        K_nearest_index = self.get_K_nearest(sample)

        # get the number of classifications within the K-nearest
        c1 = 0
        c2 = 0

        for i in range(self.K):
            # when K = 1, K_nearest_index is just a number
            feature_index = K_nearest_index[i] if self.K != 1 else K_nearest_index
            # get the corresponding label
            label = self.train.getOneLabel(feature_index)

            if (label == 1):
                c1 = c1 + 1
            else:
                c2 = c2 + 1

        if (isPrint):
            print('c1:', c1, 'c2:', c2)

        if (c1 >= c2):
            return 1
        else:
            return 0

    def getErrorRate(self, data='test'):
        error_num = 0

        if (data == 'test'):
            data_set = self.test
        else:
            data_set = self.train

        for i in range(data_set.sampleNum):
            sample = data_set.getOneSample(i, 'log')
            label = data_set.getOneLabel(i)

            # if (i == 10):
            #     ret = self.predict(sample, True)
            # else:
            #     ret = self.predict(sample)

            ret = self.predict(sample)

            if (ret != label):
                error_num = error_num + 1

        return error_num / data_set.sampleNum


def k_nearest_simul(train, test, K=None):
    start = time.time()

    # construct a global kd-tree
    kd_tree = cKDTree(train.getData('log'), copy_data=True)

    # just test one K
    if (K != None):
        K_model = KNearest(train, test, K, kd_tree)
        print(K_model.getErrorRate())

        return

    step = 1
    K = 1

    # error rate for different K
    train_ret = []
    test_ret = []

    # to store the K value as the X axis
    K_label = []

    while (K <= 100):
        K_model = KNearest(train, test, K, kd_tree)

        train_error_rate = K_model.getErrorRate('train')
        test_error_rate = K_model.getErrorRate('test')

        train_ret.append(train_error_rate)
        test_ret.append(test_error_rate)

        K_label.append(K)

        if (K == 1 or K == 10 or K == 100):
            print('===when K = ' + str(K) + '===')
            print('train_error_rate:', train_error_rate)
            print('test_error_rate:', test_error_rate)
            print('\n')
            
        if (K == 10):
            step = 5

        K = K + step

    end = time.time()

    print('run time:', end - start)
    
    plt.plot(K_label, train_ret, 'r')
    plt.plot(K_label, test_ret, 'b')
    plt.ylabel('Error Rate')
    plt.xlabel('K')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()

    return (train_ret, test_ret, K_label)
