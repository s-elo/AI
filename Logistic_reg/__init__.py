import numpy as np
import math
import time
import matplotlib.pyplot as plt

class LogisticReg:
    def __init__(self, train, test, reg='true', lamada = 1):
        self.train = train
        self.test = test

        # using regularization if it is true
        self.reg = reg

        # if no regularization
        if (self.reg != True):
            self.lamada = 0
        else:
            self.lamada = lamada

        self.W = np.zeros(shape=(train.featureNum + 1, 1))

    # one iteration for compute the changes of weights
    def iteration(self):
        lamada = self.lamada

        # (feature_number + 1, sample_number)
        sample_number = self.train.sampleNum

        # add one more feature as the bias with value 1
        samples = np.append(self.train.getData('log').T,
                            np.ones(shape=(1, sample_number)), 0)

        # (1, sample_number)
        labels = self.train.labels.T

        # (feature_number + 1, 1)
        W = self.W

        # get the regularization value
        W_row_num = np.size(W, 0)

        # remove the last row to get rid of the bias
        # no_bias = W[0: W_row_num - 1]
        # reg_value = (1 / 2) * lamada * np.dot(no_bias.T, no_bias)

        assert(samples.shape == (self.train.featureNum + 1, sample_number))
        assert(labels.shape == (1, sample_number))
        assert(W.shape == (self.train.featureNum + 1, 1))

        # get the activated value
        # (1, sample_number)
        A = self.sigmoid(np.dot(W.T, samples))

        # calculate the NNL result
        # nnl = -np.sum(labels * np.log(A) + (1 - labels)
        #               * np.log(1 - A)) + reg_value

        # cost = (1 / sample_number) * nnl

        # get the derivative of nnl to W and bias
        # (D + 1, N)*(N, 1) + lamada * (D + 1, 1)= (D + 1, 1)
        W_bias_zero = W.copy()
        # make the bias as 0
        W_bias_zero[W_row_num - 1] = [0]

        # regularize the g
        g = np.dot(samples, (A.T - labels.T)) + lamada * W_bias_zero

        # get the second derivative h of nnl to W
        # (D + 1, N) * (N, N) * (N, D + 1) = (D + 1, D + 1)
        f = lambda x: x * (1 - x)
        # s = np.diag(list(map(lambda x: x * (1 - x), A[0])))
        s = np.diag(f(A[0]))
        h = np.dot(np.dot(samples, s), samples.T)

        # for regularization
        I = np.eye(W.shape[0])
        I[0][0] = 0

        h = h + lamada * I

        return (h, g)

    def train_weights(self, max_iteration=50, isPrint=False):
        h, g = self.iteration()
        diff = np.dot(np.linalg.inv(h), g)

        ret = []

        n = 0

        # stop when every diff is less than 0.000001 
        # or n is greater than the max_iteration
        while (((np.abs(diff) > 0.000001).all() and n <= max_iteration)):
            h, g = self.iteration()

            diff = np.dot(np.linalg.inv(h), g)

            # for draw the diff changes along the way
            ret.append(np.sum(np.abs(diff)))    

            self.W = self.W - diff

            n = n + 1

        if (isPrint):
            print('\n')
            print('iteration number:', n)

        return ret

    def predict(self, data='test'):
        samples = self.train.getData(
            'log').T if data == 'train' else self.test.getData('log').T
        sample_number = samples.shape[1]

        # add the bias
        samples = np.append(samples, np.ones(shape=(1, sample_number)), 0)

        Y_predict = np.zeros((1, sample_number))

        # get the activated value
        # (1, sample_number)
        A = self.sigmoid(np.dot(self.W.T, samples))

        for i in range(A.shape[1]):
            Y_predict[0, i] = 1 if A[0, i] > 0.5 else 0

        assert(Y_predict.shape == (1, sample_number))

        return Y_predict

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))


def logistic_reg_simul(train, test, reg=True):
    start = time.time()

    error_rate_test = []
    error_rate_train = []

    step = 1
    lamada = 1

    # store the lamada as X axis
    l = []

    while (lamada <= 100):
        lg = LogisticReg(train, test, reg, lamada)

        isPrint = lamada == 1 or lamada == 10 or lamada == 100

        lg.train_weights(isPrint=isPrint)

        predict_train = lg.predict('train')
        predict_test = lg.predict('test')

        # compare with the labels
        train_error = 1 - np.sum((predict_train[0] == train.labels.T[0]) * 1) / predict_train.size
        test_error = 1 - np.sum((predict_test[0] == test.labels.T[0]) * 1) / predict_test.size

        error_rate_train.append(train_error)
        error_rate_test.append(test_error)

        if (lamada == 1 or lamada == 10 or lamada == 100):
            print('===when λ = ' + str(lamada) + '===')
            print('train_error_rate:', train_error)
            print('test_error_rate:', test_error)

        if (lamada >= 10):
            step = 5
        
        l.append(lamada)

        lamada += step
    
    end = time.time()

    print('run time:', end - start)

    plt.plot(l, error_rate_train, 'r')
    plt.plot(l, error_rate_test, 'b')
    plt.ylabel('Error Rate')
    plt.xlabel('λ')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()

    return (error_rate_test, error_rate_train)
