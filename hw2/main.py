from sklearn.neural_network import MLPRegressor
import matplotlib.pyplot as plt
import numpy as np
import random
import math

def Q1(lr=0.001):
    def fn(x, y):
        return (1 - x)**2 + 100 * (y - x**2)**2

    def dfn_x(x, y):
        return -2 * (1 - x) - 400 * (y - x**2) * x

    def dfn_y(x, y):
        return 200 * (y - x**2)

    def dfn_x_x(x, y):
        return 2 - 400 * y + 1200 * x**2

    def dfn_x_y(x, y):
        return -400 * x

    def dfn_y_y(x, y):
        return 200

    def dfn_y_x(x, y):
        return -400 * x

    th = 1e-6

    # starting point
    x = random.uniform(-1, 1)
    y = random.uniform(-1, 1)

    def gradient(x, y):
        x_data = []
        y_data = []
        fn_data = []

        k = 0
        while (abs(fn(x, y) - 0) > th):
            k = k + 1

            # get the gradient
            dx = lr * dfn_x(x, y)
            dy = lr * dfn_y(x, y)

            # cache the current point
            x_data.append(x)
            y_data.append(y)
            fn_data.append(fn(x, y))

            x = x - dx
            y = y - dy

        print(f'Gradient methods: x: {x}, y: {y}, iteration_num: {k}')
        steps = np.arange(0, len(x_data), 1)

        plt.figure()
        plt.scatter(x_data, y_data, c='red', marker='x')
        plt.plot(x_data, y_data)
        plt.xlabel("x")
        plt.ylabel("y")
        plt.title('trajectory of (x, y) using gradient method')

        plt.figure()
        plt.plot(steps, fn_data)
        plt.xlabel("iteration")
        plt.ylabel("function value")
        plt.title('function value versus iteration of gradient method')

    def hessian(x, y):
        x_data = []
        y_data = []
        fn_data = []

        w = np.array([[x, y]]).T

        k = 0
        while (abs(fn(w[0][0], w[1][0]) - 0) > th):
            k = k + 1

            x = w[0][0]
            y = w[1][0]

            hess = np.array([[dfn_x_x(x, y), dfn_x_y(x, y)],
                            [dfn_y_x(x, y), dfn_y_y(x, y)]])
            grad = np.array([[dfn_x(x, y)], [dfn_y(x, y)]])

            x_data.append(x)
            y_data.append(y)
            fn_data.append(fn(x, y))

            w = w - np.dot(np.linalg.inv(hess), grad)

        print(f'Newton\'s method: x: {x}, y: {y}, iteration_num: {k}')
        steps = np.arange(0, len(x_data), 1)

        plt.figure()
        plt.scatter(x_data, y_data, c='red', marker='x')
        plt.plot(x_data, y_data)
        plt.xlabel("x")
        plt.ylabel("y")
        plt.title('trajectory of (x, y) using Newton\'s method')

        plt.figure()
        plt.plot(steps, fn_data)
        plt.xlabel("iteration")
        plt.ylabel("function value")
        plt.title('function value versus iteration of Newton\'s method')

    gradient(x, y)
    hessian(x, y)


def Q2():
    x = np.arange(-1.6, 1.6, 0.05)
    y1 = np.sin(x)
    y2 = np.cos(x)
    plt.plot(x, y1, label="sin")
    plt.plot(x, y2, label="cos", linestyle="--")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title('sin & cos')
    plt.legend()

    x1 = np.reshape(x, (1, 60))
    y1 = np.reshape(y1, (1, 60))

    regr = MLPRegressor(hidden_layer_sizes=(
        3,), random_state=1, max_iter=500, activation='relu').fit(x1, y1)
    output = regr.predict(x1)

    plt.figure()
    plt.plot(x1[0], output[0])


Q1(lr=0.001)
# Q2()


plt.show()
