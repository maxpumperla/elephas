import pytest
from tensorflow.keras import Sequential, Input, Model
from tensorflow.keras.datasets import mnist, boston_housing
from tensorflow.keras.layers import Dense, Activation, Dropout
from tensorflow.keras.utils import to_categorical


@pytest.fixture
def classification_model():
    model = Sequential()
    model.add(Dense(128, input_dim=784))
    model.add(Activation('relu'))
    model.add(Dropout(0.2))
    model.add(Dense(128))
    model.add(Activation('relu'))
    model.add(Dropout(0.2))
    model.add(Dense(10))
    model.add(Activation('softmax'))
    return model


@pytest.fixture
def regression_model():
    model = Sequential()
    model.add(Dense(64, activation='relu', input_shape=(13,)))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(1, activation='linear'))
    return model


@pytest.fixture
def classification_model_functional():
    input_layer = Input(shape=(784, ))
    hidden = Dense(128, activation='relu')(input_layer)
    dropout = Dropout(0.2)(hidden)
    hidden2 = Dense(128, activation='relu')(dropout)
    dropout2 = Dropout(0.2)(hidden2)
    output = Dense(10, activation='softmax')(dropout2)
    model = Model(inputs=input_layer, outputs=output)
    return model


@pytest.fixture(scope='session')
def mnist_data():
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    x_train = x_train.reshape(60000, 784)
    x_test = x_test.reshape(10000, 784)
    x_train = x_train.astype("float32")
    x_test = x_test.astype("float32")
    x_train /= 255
    x_test /= 255

    # Convert class vectors to binary class matrices
    y_train = to_categorical(y_train, 10)
    y_test = to_categorical(y_test, 10)
    return x_train, y_train, x_test, y_test


@pytest.fixture(scope='session')
def boston_housing_dataset():
    (x_train, y_train), (x_test, y_test) = boston_housing.load_data()
    return x_train, y_train, x_test, y_test

