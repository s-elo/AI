import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split


def loadData(sub_range=(1, 26), img_num=170, random_state=1):
    train_data = []
    test_data = []

    train_labels = []
    test_labels = []

    for subject in range(sub_range[0], sub_range[1]):
        # store the images in each subject
        imgs = []
        labels = []

        for img_idx in range(1, img_num + 1):
            # get one image
            img = Image.open('./dataset/PIE/' +
                             str(subject) + '/' + str(img_idx) + '.jpg')

            # convert into a vector
            img_vector = np.array(img).reshape(img.size[0]*img.size[1])

            # add to the corresponding subject
            imgs.append(img_vector)

            labels.append(subject)

        # sperate the training and testing dataset
        train_part, test_part = train_test_split(
            imgs, test_size=0.3, random_state=random_state)

        train_data += train_part
        test_data += test_part

        train_labels += labels[0:len(train_part)]
        test_labels += labels[0:len(test_part)]

    imgs = []
    labels = []

    # get the selfies
    for i in range(1, 11):
        img = Image.open('./dataset/selfies/' + str(i) + '.jpg')

        img_vector = np.array(img).reshape(img.size[0]*img.size[1])

        imgs.append(img_vector)
        # 0 represents selfie subject
        labels.append(0)

    # sperate the training and testing dataset
    train_part, test_part = train_test_split(
        imgs, test_size=0.3, random_state=random_state)

    train_data += train_part
    test_data += test_part

    train_labels += labels[0:len(train_part)]
    test_labels += labels[0:len(test_part)]

    return (np.array(train_data), np.array(train_labels), np.array(test_data), np.array(test_labels))
