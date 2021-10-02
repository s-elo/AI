import numpy as np
from PIL import Image
import random
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from loadData import loadData
from sklearn.metrics import accuracy_score

img = Image.open('./dataset/PIE/2/3.jpg')
print(img.size[0]*img.size[1])
arr = np.array(img).reshape(1, 1024).reshape(32, 32)
print(arr.shape)
plt.imshow(arr)
plt.show()
# print(arr)
# print(arr.reshape(1024))

# for i in range(1, 5):
#     print(i)

# print(np.zeros(3))

# arr1 = [1, 2, 3, 4]
# arr2 = [4, 5, [2]]

# print(arr1 + arr2)

# train, test = train_test_split(arr1, test_size=0.5, random_state=1)

# print(train, test)

# random_state = random.seed(1)

# train_data, train_labels, test_data, test_labels = loadData(random_state=random_state)

# print(len(train_data), len(test_data))
# print(len(train_labels))

arr = [
    [2, 7, 3],
    [3, 1, 3],
    [1, 1, 3]
]

# print(np.mean(arr, axis=0).reshape(1, 3).T)
# print(arr - np.mean(arr, axis=0))

# # print(np.cov(arr, rowvar=False))
eigenVals, eigenVecs = np.linalg.eig(arr)
print(eigenVals[1], eigenVecs[:, 0:2])
# print(np.dot(arr, eigenVecs[:, 1].T), eigenVals[1]*eigenVecs[:, 1])

# arr1 = [1, 2, 3, 4]
# arr2 = [2, 1, 3, 6]

# print(accuracy_score(arr1, arr2))