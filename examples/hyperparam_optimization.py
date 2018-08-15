from pyspark import SparkContext, SparkConf

from hyperopt import STATUS_OK
from hyperas.distributions import choice, uniform
import six.moves.cPickle as pickle

from elephas.hyperparam import HyperParamModel


def data():
    """Data providing function:

    Make sure to have every relevant import statement included here and return data as
    used in model function below. This function is separated from model() so that hyperopt
    won't reload data for each evaluation run.
    """
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
    """Model providing function:

    Create Keras model with double curly brackets dropped-in as needed.
    Return value has to be a valid python dictionary with two customary keys:
        - loss: Specify a numeric evaluation metric to be minimized
        - status: Just use STATUS_OK and see hyperopt documentation if not feasible
    The last one is optional, though recommended, namely:
        - model: specify the model just created so that we can later use it again.
    """
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
    keras_model.compile(loss='categorical_crossentropy', optimizer=rms, metrics=['acc'])

    keras_model.fit(x_train, y_train,
                    batch_size={{choice([64, 128])}},
                    epochs=1,
                    verbose=2,
                    validation_data=(x_test, y_test))
    score, acc = keras_model.evaluate(x_test, y_test, verbose=0)
    print('Test accuracy:', acc)
    return {'loss': -acc, 'status': STATUS_OK, 'model': keras_model.to_yaml(),
            'weights': pickle.dumps(keras_model.get_weights())}


# Create Spark context
conf = SparkConf().setAppName('Elephas_Hyperparameter_Optimization').setMaster('local[8]')
sc = SparkContext(conf=conf)

# Define hyper-parameter model and run optimization.
hyperparam_model = HyperParamModel(sc)
hyperparam_model.minimize(model=model, data=data, max_evals=5)
