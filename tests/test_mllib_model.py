from keras.datasets import mnist
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import RMSprop
from keras.utils import np_utils

from elephas.spark_model import SparkMLlibModel, load_spark_model
from elephas.utils.rdd_utils import to_labeled_point

import pytest
pytest.mark.usefixtures("spark_context")

# Define basic parameters
batch_size = 64
nb_classes = 10
epochs = 3

# Load data
(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train.reshape(60000, 784)[:1000]
x_test = x_test.reshape(10000, 784)
x_train = x_train.astype("float32")
x_test = x_test.astype("float32")
x_train /= 255
x_test /= 255
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

# Convert class vectors to binary class matrices
y_train = np_utils.to_categorical(y_train, nb_classes)
y_test = np_utils.to_categorical(y_test, nb_classes)

model = Sequential()
model.add(Dense(128, input_dim=784))
model.add(Activation('relu'))
model.add(Dropout(0.2))
model.add(Dense(128))
model.add(Activation('relu'))
model.add(Dropout(0.2))
model.add(Dense(10))
model.add(Activation('softmax'))

# Compile model
rms = RMSprop()
model.compile(rms, 'categorical_crossentropy', ['acc'])


def test_serialization():
    spark_model = SparkMLlibModel(
        model, frequency='epoch', mode='synchronous', num_workers=2)
    spark_model.save("test.h5")
    load_spark_model("test.h5")


def test_mllib_model(spark_context):
    # Build RDD from numpy features and labels
    lp_rdd = to_labeled_point(spark_context, x_train,
                              y_train, categorical=True)

    # Initialize SparkModel from Keras model and Spark context
    spark_model = SparkMLlibModel(
        model=model, frequency='epoch', mode='synchronous')

    # Train Spark model
    spark_model.fit(lp_rdd, epochs=5, batch_size=32, verbose=0,
                    validation_split=0.1, categorical=True, nb_classes=nb_classes)

    # Evaluate Spark model by evaluating the underlying model
    score = spark_model.master_network.evaluate(x_test, y_test, verbose=2)
    print('Test accuracy:', score[1])
