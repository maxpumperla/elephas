
from tensorflow.keras.optimizers import RMSprop

from elephas.spark_model import SparkMLlibModel, load_spark_model
from elephas.utils.rdd_utils import to_labeled_point

# Define basic parameters
batch_size = 64
nb_classes = 10
epochs = 3

# Compile model


def test_serialization(classification_model):
    rms = RMSprop()
    classification_model.compile(rms, 'categorical_crossentropy', ['acc'])
    spark_model = SparkMLlibModel(
        classification_model, frequency='epoch', mode='synchronous', num_workers=2)
    spark_model.save("test.h5")
    loaded_model = load_spark_model("test.h5")
    assert loaded_model.master_network.to_json()


def test_mllib_model(spark_context, classification_model, mnist_data):
    rms = RMSprop()
    classification_model.compile(rms, 'categorical_crossentropy', ['acc'])
    x_train, y_train, x_test, y_test = mnist_data
    x_train = x_train[:1000]
    y_train = y_train[:1000]
    # Build RDD from numpy features and labels
    lp_rdd = to_labeled_point(spark_context, x_train,
                              y_train, categorical=True)

    # Initialize SparkModel from tensorflow.keras model and Spark context
    spark_model = SparkMLlibModel(
        model=classification_model, frequency='epoch', mode='synchronous')

    # Train Spark model
    spark_model.fit(lp_rdd, epochs=5, batch_size=32, verbose=0,
                    validation_split=0.1, categorical=True, nb_classes=nb_classes)

    # Evaluate Spark model by evaluating the underlying model
    score = spark_model.master_network.evaluate(x_test, y_test, verbose=2)
    assert score