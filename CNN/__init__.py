import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt
from loadData import loadData


class Cnn:
    def __init__(self, train_data, train_labels, test_data, test_labels):
        # (N, 32, 32, 1) w=32 h=32 c=1
        self.train_data = train_data.reshape(train_data.shape[0], 32, 32, 1)
        self.test_data = test_data.reshape(test_data.shape[0], 32, 32, 1)
        # (N)
        self.train_labels = train_labels
        self.test_labels = test_labels

        self.build_model()

    def build_model(self):
        model = models.Sequential()
        # first conv layer with 20 filters by 5*5 => output: 28*28*20
        model.add(layers.Conv2D(20, (5, 5), activation='relu',
                  input_shape=self.train_data.shape[1:]))
        # first max pooling filter with size 2*2 and stride of 2
        # => output: 14*14*20
        model.add(layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

        # second round => output: 10*10*50
        model.add(layers.Conv2D(50, (5, 5), activation='relu',
                  input_shape=self.train_data.shape[1:]))
        # output: 4*4*50
        model.add(layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

        # flatten as one dimension
        model.add(layers.Flatten())

        # fully connected layer 500 neurons
        model.add(layers.Dense(500, activation='relu'))

        # final fully connected layer 26 neurons with respect to 26 subjects
        model.add(layers.Dense(26, activation='softmax'))

        self.model = model


def cnn_simul():
    random_state = 20

    train_data, train_labels, test_data, test_labels = loadData(
        random_state=random_state)

    cnn = Cnn(train_data, train_labels, test_data, test_labels)

    model = cnn.model

    model.summary()

    model.compile(optimizer='adam',
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(
                      from_logits=True),
                  metrics=['accuracy'])

    history = model.fit(cnn.train_data, cnn.train_labels, epochs=28)

    plt.plot(history.history['accuracy'], label='accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.ylim([0.5, 1])
    plt.legend(loc='lower right')

    test_loss, test_acc = model.evaluate(
        cnn.test_data,  cnn.test_labels, verbose=2)

    print('test set loss:', test_loss)
    print('test set accuracy:', test_acc)

    plt.show()
