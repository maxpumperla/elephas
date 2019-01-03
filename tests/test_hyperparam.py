import pytest
from hyperopt import STATUS_OK
from hyperas.distributions import choice, uniform
import six.moves.cPickle as pickle

from elephas.hyperparam import HyperParamModel

pytest.mark.usefixtures("spark_context")


def data():
    from keras.datasets import mnist
    from keras.utils import np_utils
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = x_train.reshape(60000, 784)
    x_test = x_test.reshape(10000, 784)
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255
    x_test /= 255
    nb_classes = 10
    y_train = np_utils.to_categorical(y_train, nb_classes)
    y_test = np_utils.to_categorical(y_test, nb_classes)
    return x_train, y_train, x_test, y_test


def model(x_train, y_train, x_test, y_test):
    from keras.models import Sequential
    from keras.layers.core import Dense, Dropout, Activation
    from keras.optimizers import RMSprop

    keras_model = Sequential()
    keras_model.add(Dense(512, input_shape=(784,)))
    keras_model.add(Activation('relu'))
    keras_model.add(Dropout({{uniform(0, 1)}}))
    keras_model.add(Dense({{choice([256, 512, 1024])}}))
    keras_model.add(Activation('relu'))
    keras_model.add(Dropout({{uniform(0, 1)}}))
    keras_model.add(Dense(10))
    keras_model.add(Activation('softmax'))

    rms = RMSprop()
    keras_model.compile(loss='categorical_crossentropy',
                        optimizer=rms, metrics=['acc'])

    keras_model.fit(x_train, y_train,
                    batch_size={{choice([64, 128])}},
                    epochs=1,
                    verbose=2,
                    validation_data=(x_test, y_test))
    score, acc = keras_model.evaluate(x_test, y_test, verbose=0)
    print('Test accuracy:', acc)
    return {'loss': -acc, 'status': STATUS_OK, 'model': keras_model.to_yaml(),
            'weights': pickle.dumps(keras_model.get_weights())}


def test_hyper_param_model(spark_context):
    hyperparam_model = HyperParamModel(spark_context)
    hyperparam_model.minimize(model=model, data=data, max_evals=1)
