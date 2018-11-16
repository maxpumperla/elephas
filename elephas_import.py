from elephas.java import java_classes
from elephas.dl4j import ParameterAveragingModel
from elephas.utils import rdd_utils
import keras
from keras.utils import np_utils


def main():
    # Set Java Spark context
    conf = java_classes.SparkConf().setMaster('local[*]').setAppName("elephas_dl4j")
    jsc = java_classes.JavaSparkContext(conf)

    # Define Keras model
    model = keras.models.Sequential()
    model.add(keras.layers.Dense(128, input_dim=784))
    model.add(keras.layers.Dense(units=10, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])

    # Define DL4J Elephas model
    spark_model = ParameterAveragingModel(java_spark_context=jsc, model=model, num_workers=4, batch_size=32)

    # Load data and build DL4J DataSet RDD under the hood
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
    x_train = x_train.reshape(60000, 784)
    x_test = x_test.reshape(10000, 784)
    x_train = x_train.astype("float64")
    x_test = x_test.astype("float64")

    # Convert class vectors to binary class matrices
    y_train = np_utils.to_categorical(y_train, 10)
    y_test = np_utils.to_categorical(y_test, 10)
    y_train = y_train.astype("float64")
    y_test = y_test.astype("float64")
    x_train /= 255
    x_test /= 255
    java_rdd = rdd_utils.to_java_rdd(jsc, x_train, y_train, 32)

    import timeit

    start = timeit.default_timer()
    # Fit model
    spark_model.fit_rdd(java_rdd, 2)
    stop = timeit.default_timer()
    print('Time: ', stop - start)

    # Retrieve resulting weights from training, set to original Keras model, evaluate.
    keras_model = spark_model.get_keras_model()
    score = keras_model.evaluate(x_test, y_test, verbose=0)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])

    import os
    if os.path.exists("temp.h5"):
        os.remove("temp.h5")


if __name__ == '__main__':
    main()
