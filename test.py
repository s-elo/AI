from data_handler import *
from Beta_NB import BetaNB
import math
import time
from scipy.spatial import cKDTree

train, test = loadData()

# print('training data:', train.getData().shape)
# print('training label:', train.labels.shape)
# print('test data:', test.getData().shape)
# print('test label:', test.labels.shape)

# cn = 0

# for i in range(train.sampleNum):
#     if (train.getOneLabel(i) == 1):
#         cn = cn + 1

# for i in range(test.sampleNum):
#     if (test.getOneLabel(i) == 1):
#         cn = cn + 1

# print(cn)

# print(train.getOneSample(3000, 'binary'))
# print(train.getOneSample(3000, 'log'))
# print(train.getOneSample(3000))

# print(train.getData().shape[1])

# def gaussian(x, mean, variance):
#     return (1 / math.sqrt(2 * math.pi * variance)) * math.exp(-0.5 * (x - mean)**2 / variance)

# print(gaussian(1, 2, 4))

# print(-math.inf > -1000)

# def t(x):
#     return x * (1 - x)

# print(list(map(t, np.zeros(shape=(1, 3)))))
# print(np.zeros(shape=(1, 3)))
# print(np.abs([1, -2, -3]))

# o = np.ones(shape = (1, 5))
# oo = np.append(o, np.ones(shape = (1, 5)), 0)
# print(oo)

# w = np.ones(shape = (5, 2))
# print(w)

# w[np.size(w, 0) - 1] = [2, 2]
# print(w)

# w1 = w[0:np.size(w, 0) - 1]

# print(w1)

# print(2 * w)

# d = np.eye(3)
# d[0][0] = 0
# print(d)

tt = [[1, 2], [3, 5], [3, 6, 2, 1, 8]]

g, p, j = tt
print(g, p)
print(tt.pop(0))

print(np.square(np.array(g) - np.array(p)))
print(np.sort(np.array(j), kind = 'heapsort')[0:3])

arr = [{'name': 'leo', 'age': 22},{'name': 'pit', 'age': 21},{'name': 'git', 'age': 12}]

print(sorted(arr, key=lambda x: x['age']))

print(sum(list(map(lambda x: x['age'], arr))))

test_data = np.array([
    [1, 2, 3],
    [2, 5, 6],
    [5, 6, 9],
    [5, 4, 8],
    [3, 5, 9],
    [5, 6, 3]
])

start = time.time()
kd_tree = cKDTree(test_data, copy_data=True)
# print(kd_tree.data)

test_point = np.array([9, 2, 2])

dis, index = kd_tree.query(test_point, k=1, p=2)

print(dis, index)
end = time.time()

print(end - start)