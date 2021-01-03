import pytest
from keras import optimizers

from elephas.ml_model import ElephasEstimator, load_ml_estimator, ElephasTransformer, load_ml_transformer, \
    ModelType, LossModelTypeMapper
from elephas.ml.adapter import to_data_frame

from pyspark.mllib.evaluation import MulticlassMetrics
from pyspark.ml import Pipeline


def test_serialization_transformer(classification_model):
    transformer = ElephasTransformer()
    transformer.set_keras_model_config(classification_model.to_yaml())
    transformer.save("test.h5")
    loaded_model = load_ml_transformer("test.h5")
    assert loaded_model.get_model().to_yaml() == classification_model.to_yaml()


def test_serialization_estimator(classification_model):
    estimator = ElephasEstimator()
    estimator.set_keras_model_config(classification_model.to_yaml())
    estimator.set_loss("categorical_crossentropy")

    estimator.save("test.h5")
    loaded_model = load_ml_estimator("test.h5")
    assert loaded_model.get_model().to_yaml() == classification_model.to_yaml()


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
    estimator.set_keras_model_config(classification_model.to_yaml())
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

    prediction_and_label = pnl.rdd.map(lambda row: (row.label, row.prediction))
    metrics = MulticlassMetrics(prediction_and_label)
    print(metrics.precision())
    print(metrics.recall())


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
    estimator.set_keras_model_config(classification_model_functional.to_yaml())
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
    pnl.show(100)

    prediction_and_label = pnl.rdd.map(lambda row: (row.label, row.prediction))
    metrics = MulticlassMetrics(prediction_and_label)
    print(metrics.precision())
    print(metrics.recall())


def test_regression_model(spark_context, regression_model, boston_housing_dataset):
    batch_size = 64
    epochs = 100

    x_train, y_train, x_test, y_test = boston_housing_dataset
    df = to_data_frame(spark_context, x_train, y_train)
    test_df = to_data_frame(spark_context, x_test, y_test)

    sgd = optimizers.SGD(lr=0.00001)
    sgd_conf = optimizers.serialize(sgd)
    estimator = ElephasEstimator()
    estimator.set_keras_model_config(regression_model.to_yaml())
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

    prediction_and_label = pnl.rdd.map(lambda row: (row.label, row.prediction))
    metrics = MulticlassMetrics(prediction_and_label)
    print(metrics.precision())
    print(metrics.recall())


@pytest.mark.parametrize('loss, model_type', [('binary_crossentropy', ModelType.CLASSIFICATION),
                                              ('mean_squared_error', ModelType.REGRESSION),
                                              ('categorical_crossentropy', ModelType.CLASSIFICATION),
                                              ('mean_absolute_error', ModelType.REGRESSION)])
def test_model_type_mapper(loss, model_type):
    assert LossModelTypeMapper().get_model_type(loss) == model_type


def test_model_type_mapper_custom():
    LossModelTypeMapper().register_loss('test', ModelType.REGRESSION)
    assert LossModelTypeMapper().get_model_type('test') == ModelType.REGRESSION


def test_model_type_mapper_custom_callable():
    def custom_loss(y_true, y_pred):
        return y_true - y_pred
    LossModelTypeMapper().register_loss(custom_loss, ModelType.REGRESSION)
    assert LossModelTypeMapper().get_model_type('custom_loss') == ModelType.REGRESSION
