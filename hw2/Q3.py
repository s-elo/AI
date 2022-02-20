import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from slp import slp_model

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
        np.array(train_data), np.array(train_labels), random_state=1)
    test_data, test_labels = shuffle(
        np.array(test_data), np.array(test_labels), random_state=0)

    # make them in [0, 1]
    return train_data / 255, train_labels, test_data / 255, test_labels


train_x, train_y, test_x, test_y = loadData()
print(train_x.shape, test_y.shape)


def mlp(train_x, train_y, test_x, test_y, opts={}, title='', normalization=False):
    '''
    Default setting:
        learning rate: 0.001
        activation: relu
        solver: adam
        batch size: 200
        hidden_size: (100,)
        max_iter: 1000
    x: (sample, feature)
    y: (sample, 1)
    '''
    if normalization:
        train_x = (train_x - train_x.mean(axis=0)) / train_x.std(axis=0)
        test_x = (test_x - test_x.mean(axis=0)) / test_x.std(axis=0)

    mlp = MLPClassifier(random_state=1, max_iter=1, **opts)
    params = mlp.get_params()

    epochs = 300

    train_acc = []
    test_acc = []

    for i in range(0, epochs):
        mlp.set_params(**params)

        # Update the model with a single iteration over the given data
        mlp.partial_fit(train_x, train_y, [1, 0])

        # set the params for next iteration
        params = mlp.get_params()

        # get training set accuracy
        prediction = mlp.predict(train_x)
        train_acc.append(accuracy_score(train_y, prediction))

        # get test set accuracy
        prediction = mlp.predict(test_x)
        test_acc.append(accuracy_score(test_y, prediction))

    # display the final accuracy
    print(f'training set accuracy: {train_acc[len(train_acc) - 1] * 100}%')
    print(
        f'test set accuracy at the end of the training step: {test_acc[len(test_acc) - 1]*100}%')

    max_test_acc = max(test_acc)
    print(
        f'maximum test accuracy during the training step: {max_test_acc * 100}%')

    plt.figure()
    plt.plot(range(0, epochs), train_acc)
    plt.plot(range(0, epochs), test_acc)
    plt.xlabel('epochs')
    plt.ylabel('accuracy')
    plt.legend(['train_acc', 'test_acc'])
    plt.title(f'MLP-{title}')

    # plt.plot(mlp.loss_curve_)


def slp(train_x, train_y, test_x, test_y, normalization=False):
    if (normalization):
        # zero mean and unit variance
        # scaler = preprocessing.StandardScaler().fit(train_x)
        # train_x = scaler.transform(train_x)
        # test_x = scaler.transform(test_x)

        train_x = (train_x - train_x.mean(axis=0)) / train_x.std(axis=0)
        test_x = (test_x - train_x.mean(axis=0)) / train_x.std(axis=0)

    # make it suitable for the slp
    train_x = train_x.T
    train_y = train_y.reshape(1, train_y.shape[0])
    test_x = test_x.T
    test_y = test_y.reshape(1, test_y.shape[0])

    iter_num = 1000
    train_acc, test_acc = slp_model(train_x, train_y, test_x, test_y,
                                    iteration_number=iter_num, learning_rate=0.05, random_state=1, print_cost=False)

    max_test_acc = max(test_acc)
    print(
        f'maximum test accuracy during the training step: {max_test_acc*100}%')

    plt.figure()
    plt.plot(range(0, iter_num), train_acc)
    plt.plot(range(0, iter_num), test_acc)
    plt.xlabel('epochs')
    plt.ylabel('accuracy')
    plt.legend(['train_acc', 'test_acc'])
    title = 'Normalized data' if normalization else 'Non-normalized data'
    plt.title(f'SLP-{title}')


print(f'==================single layer perceptron=====================')
print(f'===Without data normalization===')
slp(train_x, train_y, test_x, test_y, normalization=False)
print(f'===With data normalization===')
slp(train_x, train_y, test_x, test_y, normalization=True)
print('\n')

print(f'==================multiple layers perceptron=====================')
print(f'=======batch mode======')
mlp_opts = {
    'hidden_layer_sizes': (28, 5),
    'batch_size': 900,
    # l2 penalty
    # 'alpha': 10
}
mlp(train_x, train_y, test_x, test_y, mlp_opts,
    title='batch mode', normalization=True)
print(f'=======mini batch mode with regularization======')
mlp_opts = {
    'hidden_layer_sizes': (28, 5),
    'batch_size': 200,
    # l2 penalty
    'alpha': 10
}
mlp(train_x, train_y, test_x, test_y, mlp_opts,
    title='Mini batch mode with regularization', normalization=True)
print(f'=======sequential mode======')
mlp_opts = {
    'hidden_layer_sizes': (28, 5),
    'batch_size': 1,
    # l2 penalty
    # 'alpha': 10
}
mlp(train_x, train_y, test_x, test_y, mlp_opts,
    title='sequential mode', normalization=True)
plt.show()
