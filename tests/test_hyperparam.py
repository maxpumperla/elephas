import numpy as np
import pytest
from hyperopt import STATUS_OK
from hyperas.distributions import choice, uniform
import six.moves.cPickle as pickle

from elephas.hyperparam import HyperParamModel

pytest.mark.usefixtures("spark_context")


def test_that_requires_sc(spark_context):
    assert spark_context.parallelize(np.zeros((10, 10))).count() == 10


def test_hyper_param_model(spark_context):
    def data():
        from keras.datasets import mnist
        from keras.utils import np_utils
        (X_train, y_train), (X_test, y_test) = mnist.load_data()
        X_train = X_train.reshape(60000, 784)
        X_test = X_test.reshape(10000, 784)
        X_train = X_train.astype('float32')
        X_test = X_test.astype('float32')
        X_train /= 255
        X_test /= 255
        nb_classes = 10
        Y_train = np_utils.to_categorical(y_train, nb_classes)
        Y_test = np_utils.to_categorical(y_test, nb_classes)
        return X_train, Y_train, X_test, Y_test

    def model(X_train, Y_train, X_test, Y_test):
        from keras.models import Sequential
        from keras.layers.core import Dense, Dropout, Activation
        from keras.optimizers import RMSprop

        model = Sequential()
        model.add(Dense(512, input_shape=(784,)))
        model.add(Activation('relu'))
        model.add(Dropout({{uniform(0, 1)}}))
        model.add(Dense({{choice([256, 512, 1024])}}))
        model.add(Activation('relu'))
        model.add(Dropout({{uniform(0, 1)}}))
        model.add(Dense(10))
        model.add(Activation('softmax'))

        rms = RMSprop()
        model.compile(loss='categorical_crossentropy', optimizer=rms)

        model.fit(X_train, Y_train,
                  batch_size={{choice([64, 128])}},
                  nb_epoch=1,
                  show_accuracy=True,
                  verbose=2,
                  validation_data=(X_test, Y_test))
        score, acc = model.evaluate(X_test, Y_test, verbose=0)
        print('Test accuracy:', acc)
        return {'loss': -acc, 'status': STATUS_OK, 'model': model.to_yaml(),
                'weights': pickle.dumps(model.get_weights())}

    hyperparam_model = HyperParamModel(spark_context)
    hyperparam_model.minimize(model=model, data=data, max_evals=5)
