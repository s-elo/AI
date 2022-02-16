from random import random
import numpy as np


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def predict(w, b, X):
    sample_number = X.shape[1]

    Y_predict = np.zeros((1, sample_number))

    # the output
    A = sigmoid(np.dot(w.T, X) + b)  # (1, sample_number)

    for i in range(A.shape[1]):
        Y_predict[0, i] = 1 if A[0, i] > 0.5 else 0

    assert(Y_predict.shape == (1, sample_number))

    return Y_predict


def get_acc(prediction, label):
    return 100 - np.mean(np.abs(prediction - label)) * 100


def init_weights(dimensions, random_state=1):
    np.random.seed(random_state)
    w = np.random.randn(dimensions, 1) * 0.01

    b = 0

    assert(w.shape == (dimensions, 1))
    assert(isinstance(b, float) or isinstance(b, int))

    return (w, b)


def iteration(w, b, X, Y):
    """
    w - weights: (num_px * num_px * 3, 1)

    b - bias

    X - training input: (num_px * num_px * 3, sample_number)

    Y - labels: (1, sample_number)
    """

    # forward propagation
    sample_number = X.shape[1]

    A = sigmoid(np.dot(w.T, X) + b)
    cost = (-1 / sample_number) * np.sum(Y *
                                         np.log(A) + (1 - Y) * np.log(1 - A))

    # back propagation
    dw = (1 / sample_number) * np.dot(X, (A - Y).T)
    db = (1 / sample_number) * np.sum(A - Y)

    assert(w.shape == dw.shape)
    assert(db.dtype == float)
    cost = np.squeeze(cost)
    assert(cost.shape == ())

    grads = {
        'dw': dw,
        'db': db
    }

    return (grads, cost)


def optimize(w, b, X, Y, tx, ty, iteration_number, learning_rate, print_cost=False):
    costs = []
    train_acc = []
    test_acc = []

    for i in range(iteration_number):
        grads, cost = iteration(w, b, X, Y)

        dw = grads['dw']
        db = grads['db']

        w = w - learning_rate * dw
        b = b - learning_rate * db

        train_acc.append(get_acc(predict(w, b, X), Y) / 100)
        test_acc.append(get_acc(predict(w, b, tx), ty) / 100)

        # record the cost every 100 iteration
        if (i % 100 == 0):
            costs.append(cost)

        if ((print_cost) and (i % 100 == 0)):
            print('iteration number: %i, wrong rate: %f' % (i, cost))

    params = {
        'w': w,
        'b': b
    }

    grads = {
        'dw': dw,
        'db': db
    }

    return (params, grads, costs, train_acc, test_acc)


def slp_model(train_data, train_label, test_data, test_label, iteration_number, learning_rate, random_state=1, print_cost=False):
    w, b = init_weights(train_data.shape[0], random_state=random_state)

    params, grads, costs, train_acc, test_acc = optimize(
        w, b, train_data, train_label, test_data, test_label, iteration_number, learning_rate, print_cost)

    # optimized params
    w = params['w']
    b = params['b']

    train_a = get_acc(predict(w, b, train_data), train_label)

    test_a = get_acc(predict(w, b, test_data), test_label)

    # get the accuracy
    print(f"training data accuracy: {train_a}%")
    print(f"testing data accuracy at the end of the trianing steps: {test_a}%")

    return (train_acc, test_acc)
