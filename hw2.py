from sklearn.neural_network import MLPRegressor
import matplotlib.pyplot as plt
import numpy as np

x = np.arange(0, 6, 0.1)
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

plt.show()
