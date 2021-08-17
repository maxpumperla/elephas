import numpy as np
import pytest
from pyspark.ml import Pipeline
from pyspark.mllib.evaluation import MulticlassMetrics, RegressionMetrics
from pyspark.sql.types import DoubleType
from tensorflow.keras import optimizers
from tensorflow.keras.activations import relu
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential

from elephas.ml.adapter import to_data_frame
from elephas.ml_model import ElephasEstimator, load_ml_estimator, ElephasTransformer, load_ml_transformer
from elephas.utils.model_utils import ModelType, argmax


def test_serialization_transformer(classification_model):
    transformer = ElephasTransformer()
    transformer.set_keras_model_config(classification_model.to_json())
    transformer.save("test.h5")
    loaded_model = load_ml_transformer("test.h5")
    assert loaded_model.get_model().to_json() == classification_model.to_json()


def test_serialization_estimator(classification_model):
    estimator = ElephasEstimator()
    estimator.set_keras_model_config(classification_model.to_json())
    estimator.set_loss("categorical_crossentropy")

    estimator.save("test.h5")
    loaded_model = load_ml_estimator("test.h5")
    assert loaded_model.get_model().to_json() == classification_model.to_json()


def test_serialization_transformer_and_predict(spark_context, classification_model, mnist_data):
    _, _, x_test, y_test = mnist_data
    df = to_data_frame(spark_context, x_test, y_test, categorical=True)
    transformer = ElephasTransformer(weights=classification_model.get_weights(), model_type=ModelType.CLASSIFICATION)
    transformer.set_keras_model_config(classification_model.to_json())
    transformer.save("test.h5")
    loaded_transformer = load_ml_transformer("test.h5")
    loaded_transformer.transform(df)


def test_spark_ml_model_classification(spark_context, classification_model, mnist_data):
    batch_size = 64
    nb_classes = 10
    epochs = 1

    x_train, y_train, x_test, y_test = mnist_data
    x_train = x_train[:1000]
    y_train = y_train[:1000]
    df = to_data_frame(spark_context, x_train, y_train, categorical=True)
    test_df = to_data_frame(spark_context, x_test, y_test, categorical=True)

    sgd = optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    sgd_conf = optimizers.serialize(sgd)

    # Initialize Spark ML Estimator
    estimator = ElephasEstimator()
    estimator.set_keras_model_config(classification_model.to_json())
    estimator.set_optimizer_config(sgd_conf)
    estimator.set_mode("synchronous")
    estimator.set_loss("categorical_crossentropy")
    estimator.set_metrics(['acc'])
    estimator.set_epochs(epochs)
    estimator.set_batch_size(batch_size)
    estimator.set_validation_split(0.1)
    estimator.set_categorical_labels(True)
    estimator.set_nb_classes(nb_classes)

    # Fitting a model returns a Transformer
    pipeline = Pipeline(stages=[estimator])
    fitted_pipeline = pipeline.fit(df)

    # Evaluate Spark model by evaluating the underlying model
    prediction = fitted_pipeline.transform(test_df)
    pnl = prediction.select("label", "prediction")
    pnl.show(100)

    # since prediction in a multiclass classification problem is a vector, we need to compute argmax
    # the casting to a double is just necessary for using MulticlassMetrics
    pnl = pnl.select('label', argmax('prediction').astype(DoubleType()).alias('prediction'))
    prediction_and_label = pnl.rdd.map(lambda row: (row.label, row.prediction))
    metrics = MulticlassMetrics(prediction_and_label)
    print(metrics.accuracy)


def test_functional_model(spark_context, classification_model_functional, mnist_data):
    batch_size = 64
    epochs = 1

    x_train, y_train, x_test, y_test = mnist_data
    x_train = x_train[:1000]
    y_train = y_train[:1000]
    df = to_data_frame(spark_context, x_train, y_train, categorical=True)
    test_df = to_data_frame(spark_context, x_test, y_test, categorical=True)

    sgd = optimizers.SGD()
    sgd_conf = optimizers.serialize(sgd)
    estimator = ElephasEstimator()
    estimator.set_keras_model_config(classification_model_functional.to_json())
    estimator.set_optimizer_config(sgd_conf)
    estimator.set_mode("synchronous")
    estimator.set_loss("categorical_crossentropy")
    estimator.set_metrics(['acc'])
    estimator.set_epochs(epochs)
    estimator.set_batch_size(batch_size)
    estimator.set_validation_split(0.1)
    estimator.set_categorical_labels(True)
    estimator.set_nb_classes(10)
    pipeline = Pipeline(stages=[estimator])
    fitted_pipeline = pipeline.fit(df)
    prediction = fitted_pipeline.transform(test_df)
    pnl = prediction.select("label", "prediction")
    pnl = pnl.select('label', argmax('prediction').astype(DoubleType()).alias('prediction'))
    pnl.show(100)

    prediction_and_label = pnl.rdd.map(lambda row: (row.label, row.prediction))
    metrics = MulticlassMetrics(prediction_and_label)
    print(metrics.accuracy)


def test_regression_model(spark_context, regression_model, boston_housing_dataset):
    batch_size = 64
    epochs = 10

    x_train, y_train, x_test, y_test = boston_housing_dataset
    df = to_data_frame(spark_context, x_train, y_train)
    test_df = to_data_frame(spark_context, x_test, y_test)

    sgd = optimizers.SGD(lr=0.00001)
    sgd_conf = optimizers.serialize(sgd)
    estimator = ElephasEstimator()
    estimator.set_keras_model_config(regression_model.to_json())
    estimator.set_optimizer_config(sgd_conf)
    estimator.set_mode("synchronous")
    estimator.set_loss("mae")
    estimator.set_metrics(['mae'])
    estimator.set_epochs(epochs)
    estimator.set_batch_size(batch_size)
    estimator.set_validation_split(0.01)
    estimator.set_categorical_labels(False)

    pipeline = Pipeline(stages=[estimator])
    fitted_pipeline = pipeline.fit(df)
    prediction = fitted_pipeline.transform(test_df)
    pnl = prediction.select("label", "prediction")
    pnl.show(100)

    prediction_and_observations = pnl.rdd.map(lambda row: (row.label, row.prediction))
    metrics = RegressionMetrics(prediction_and_observations)
    print(metrics.r2)


def test_set_cols_deprecated(spark_context, regression_model, boston_housing_dataset):
    with pytest.deprecated_call():
        batch_size = 64
        epochs = 10

        x_train, y_train, x_test, y_test = boston_housing_dataset
        df = to_data_frame(spark_context, x_train, y_train)
        df = df.withColumnRenamed('features', 'scaled_features')
        df = df.withColumnRenamed('label', 'ground_truth')
        test_df = to_data_frame(spark_context, x_test, y_test)
        test_df = test_df.withColumnRenamed('features', 'scaled_features')
        test_df = test_df.withColumnRenamed('label', 'ground_truth')

        sgd = optimizers.SGD(lr=0.00001)
        sgd_conf = optimizers.serialize(sgd)
        estimator = ElephasEstimator()
        estimator.set_keras_model_config(regression_model.to_json())
        estimator.set_optimizer_config(sgd_conf)
        estimator.setFeaturesCol('scaled_features')
        estimator.setOutputCol('output')
        estimator.setLabelCol('ground_truth')
        estimator.set_mode("synchronous")
        estimator.set_loss("mae")
        estimator.set_metrics(['mae'])
        estimator.set_epochs(epochs)
        estimator.set_batch_size(batch_size)
        estimator.set_validation_split(0.01)
        estimator.set_categorical_labels(False)

        pipeline = Pipeline(stages=[estimator])
        fitted_pipeline = pipeline.fit(df)
        prediction = fitted_pipeline.transform(test_df)
        pnl = prediction.select("ground_truth", "output")
        pnl.show(100)

        prediction_and_observations = pnl.rdd.map(lambda row: (row['ground_truth'], row['output']))
        metrics = RegressionMetrics(prediction_and_observations)
        print(metrics.r2)


def test_set_cols(spark_context, regression_model, boston_housing_dataset):
    batch_size = 64
    epochs = 10

    x_train, y_train, x_test, y_test = boston_housing_dataset
    df = to_data_frame(spark_context, x_train, y_train)
    df = df.withColumnRenamed('features', 'scaled_features')
    df = df.withColumnRenamed('label', 'ground_truth')
    test_df = to_data_frame(spark_context, x_test, y_test)
    test_df = test_df.withColumnRenamed('features', 'scaled_features')
    test_df = test_df.withColumnRenamed('label', 'ground_truth')

    sgd = optimizers.SGD(lr=0.00001)
    sgd_conf = optimizers.serialize(sgd)
    estimator = ElephasEstimator(labelCol='ground_truth', outputCol='output', featuresCol='scaled_features')
    estimator.set_keras_model_config(regression_model.to_json())
    estimator.set_optimizer_config(sgd_conf)
    estimator.set_mode("synchronous")
    estimator.set_loss("mae")
    estimator.set_metrics(['mae'])
    estimator.set_epochs(epochs)
    estimator.set_batch_size(batch_size)
    estimator.set_validation_split(0.01)
    estimator.set_categorical_labels(False)

    pipeline = Pipeline(stages=[estimator])
    fitted_pipeline = pipeline.fit(df)
    prediction = fitted_pipeline.transform(test_df)
    pnl = prediction.select("ground_truth", "output")
    pnl.show(100)

    prediction_and_observations = pnl.rdd.map(lambda row: (row['ground_truth'], row['output']))
    metrics = RegressionMetrics(prediction_and_observations)
    print(metrics.r2)


def test_custom_objects(spark_context, boston_housing_dataset):
    def custom_activation(x):
        return 2 * relu(x)

    model = Sequential()
    model.add(Dense(64, input_shape=(13,)))
    model.add(Dense(64, activation=custom_activation))
    model.add(Dense(1, activation='linear'))
    x_train, y_train, x_test, y_test = boston_housing_dataset
    df = to_data_frame(spark_context, x_train, y_train)
    test_df = to_data_frame(spark_context, x_test, y_test)

    sgd = optimizers.SGD(lr=0.00001)
    sgd_conf = optimizers.serialize(sgd)
    estimator = ElephasEstimator()
    estimator.set_keras_model_config(model.to_json())
    estimator.set_optimizer_config(sgd_conf)
    estimator.set_mode("synchronous")
    estimator.set_loss("mae")
    estimator.set_metrics(['mae'])
    estimator.set_epochs(10)
    estimator.set_batch_size(32)
    estimator.set_validation_split(0.01)
    estimator.set_categorical_labels(False)
    estimator.set_custom_objects({'custom_activation': custom_activation})

    pipeline = Pipeline(stages=[estimator])
    fitted_pipeline = pipeline.fit(df)
    prediction = fitted_pipeline.transform(test_df)


def test_predict_classes_probability(spark_context, classification_model, mnist_data):
    batch_size = 64
    nb_classes = 10
    epochs = 1

    x_train, y_train, x_test, y_test = mnist_data
    x_train = x_train[:1000]
    y_train = y_train[:1000]
    df = to_data_frame(spark_context, x_train, y_train, categorical=True)
    test_df = to_data_frame(spark_context, x_test, y_test, categorical=True)

    sgd = optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    sgd_conf = optimizers.serialize(sgd)

    # Initialize Spark ML Estimator
    estimator = ElephasEstimator()
    estimator.set_keras_model_config(classification_model.to_json())
    estimator.set_optimizer_config(sgd_conf)
    estimator.set_mode("synchronous")
    estimator.set_loss("categorical_crossentropy")
    estimator.set_metrics(['acc'])
    estimator.set_epochs(epochs)
    estimator.set_batch_size(batch_size)
    estimator.set_validation_split(0.1)
    estimator.set_categorical_labels(True)
    estimator.set_nb_classes(nb_classes)

    # Fitting a model returns a Transformer
    pipeline = Pipeline(stages=[estimator])
    fitted_pipeline = pipeline.fit(df)

    results = fitted_pipeline.transform(test_df)
    # we should have an array of 10 elements in the prediction column, since we have 10 classes
    # and therefore 10 probabilities
    assert len(results.take(1)[0].prediction) == 10


def test_batch_predict_classes_probability(spark_context, classification_model, mnist_data):
    batch_size = 64
    nb_classes = 10
    epochs = 1

    x_train, y_train, x_test, y_test = mnist_data
    x_train = x_train[:1000]
    y_train = y_train[:1000]
    df = to_data_frame(spark_context, x_train, y_train, categorical=True)
    test_df = to_data_frame(spark_context, x_test, y_test, categorical=True)

    sgd = optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    sgd_conf = optimizers.serialize(sgd)

    # Initialize Spark ML Estimator
    estimator = ElephasEstimator()
    estimator.set_keras_model_config(classification_model.to_json())
    estimator.set_optimizer_config(sgd_conf)
    estimator.set_mode("synchronous")
    estimator.set_loss("categorical_crossentropy")
    estimator.set_metrics(['acc'])
    estimator.set_epochs(epochs)
    estimator.set_batch_size(batch_size)
    estimator.set_validation_split(0.1)
    estimator.set_categorical_labels(True)
    estimator.set_nb_classes(nb_classes)

    # Fitting a model returns a Transformer
    fitted_pipeline = estimator.fit(df)

    results = fitted_pipeline.transform(test_df)

    # Set inference batch size and do transform again on the same test_df
    inference_batch_size = int(len(y_test) / 10)
    fitted_pipeline.set_params(inference_batch_size=inference_batch_size)
    fitted_pipeline.set_params(outputCol="prediction_via_batch_inference")
    results_with_batch_prediction = fitted_pipeline.transform(results)
    # we should have an array of 10 elements in the prediction column, since we have 10 classes
    # and therefore 10 probabilities
    results_np = results_with_batch_prediction.take(1)[0]
    assert len(results_np.prediction) == 10
    assert len(results_np.prediction_via_batch_inference) == 10
    assert np.array_equal(results_np.prediction, results_np.prediction_via_batch_inference)


def test_save_pipeline(spark_context, classification_model):
    sgd = optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    sgd_conf = optimizers.serialize(sgd)

    # Initialize Spark ML Estimator
    estimator = ElephasEstimator()
    estimator.set_keras_model_config(classification_model.to_json())
    estimator.set_optimizer_config(sgd_conf)
    estimator.set_mode("synchronous")
    estimator.set_loss("categorical_crossentropy")
    estimator.set_metrics(['acc'])
    estimator.set_epochs(10)
    estimator.set_batch_size(10)
    estimator.set_validation_split(0.1)
    estimator.set_categorical_labels(True)
    estimator.set_nb_classes(10)

    # Fitting a model returns a Transformer
    pipeline = Pipeline(stages=[estimator])
    pipeline.save('tmp')
