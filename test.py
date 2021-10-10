import numpy as np
from PIL import Image
import random
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from loadData import loadData
from sklearn.metrics import accuracy_score

# img = Image.open('./dataset/selfies/3.jpg')
# print(img.size[0]*img.size[1])

# plt.imshow(img.resize((32,32)).convert('L'), cmap ='gray')
# plt.show()
# def selfieProcess():
#     for i in range(1, 11):
#         img = Image.open('./dataset/selfies/' + str(i) + '.jpg').resize((32,32)).convert('L')
#         img.save('./dataset/selfies/' + str(i) + '.jpg')

# selfieProcess()


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
# eigenVals, eigenVecs = np.linalg.eig(arr)
# idx = eigenVals.argsort()[::-1]   
# eigenVals = eigenVals[idx]
# eigenVecs = eigenVecs[:,idx]
# print(eigenVals)
# print(eigenVecs[:, 0:2])
# print('\n')
# print(np.dot(arr, eigenVecs[:, 1].T), eigenVals[1]*eigenVecs[:, 1])

# u, s, vh = np.linalg.svd(arr, full_matrices=True)
# print(u)
# print(s)
# print(vh)

# arr1 = [1, 2, 3, 4]
arr2 = [2, 1, 3, 6]
print(arr2[-1])
# print(accuracy_score(arr1, arr2))

# print(arr2[0:-3])