import numpy as np
from PIL import Image
from sklearn.utils import shuffle

datasetPath = './dataset/'


def loadData(train_num=450, test_num=50):
    train_data = []
    test_data = []

    train_labels = []
    test_labels = []

    subjects = ['automobile', 'dog']

    for subject in subjects:
        # load the training set
        for img_idx in range(0, train_num + test_num):
            # 0 -> 000
            fileName = str(img_idx).rjust(3, '0')

            # get one image and convert into grayscale
            img = Image.open(
                f'{datasetPath}/{subject}/{fileName}.jpg').resize((32, 32)).convert('L')

            # convert into a vector
            img_vector = np.array(img).reshape(img.size[0]*img.size[1])

            if img_idx < train_num:
                train_data.append(img_vector)
                train_labels.append(0 if subject == 'automobile' else 1)
            else:
                test_data.append(img_vector)
                test_labels.append(0 if subject == 'automobile' else 1)

    train_data, train_labels = shuffle(
        np.array(train_data), np.array(train_labels), random_state=0)
    test_data, test_labels = shuffle(
        np.array(test_data), np.array(test_labels), random_state=0)

    return train_data, train_labels.reshape(1, train_labels.shape[0]), test_data, test_labels.reshape(1, test_labels.shape[0])


train_x, train_y, test_x, test_y = loadData()
print(train_x.shape, test_y.shape)
