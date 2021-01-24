from tensorflow.keras.datasets import boston_housing
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation
from tensorflow.keras import optimizers

from elephas.ml_model import ElephasEstimator
from elephas.ml.adapter import to_data_frame

from pyspark import SparkContext, SparkConf
from pyspark.mllib.evaluation import RegressionMetrics
from pyspark.ml import Pipeline


# Define basic parameters
batch_size = 64
epochs = 1

# Load data
(x_train, y_train), (x_test, y_test) = boston_housing.load_data()

x_train = x_train.astype("float32")
x_test = x_test.astype("float32")
x_train /= 255
x_test /= 255
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')


model = Sequential()
model.add(Dense(64, input_shape=(x_train.shape[1],)))
model.add(Activation('relu'))
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dense(1))

# Create Spark context
conf = SparkConf().setAppName('BostonHousing_Spark_MLP').setMaster('local[*]')
sc = SparkContext(conf=conf)

# Build RDD from numpy features and labels
df = to_data_frame(sc, x_train, y_train)
test_df = to_data_frame(sc, x_test, y_test)

sgd = optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
sgd_conf = optimizers.serialize(sgd)

# Initialize Spark ML Estimator
estimator = ElephasEstimator()
estimator.set_keras_model_config(model.to_yaml())
estimator.set_optimizer_config(sgd_conf)
estimator.set_mode("synchronous")
estimator.set_loss("mae")
estimator.set_metrics(['mse'])
estimator.set_epochs(epochs)
estimator.set_batch_size(batch_size)
estimator.set_validation_split(0.1)
estimator.set_categorical_labels(False)

# Fitting a model returns a Transformer
pipeline = Pipeline(stages=[estimator])
fitted_pipeline = pipeline.fit(df)

# Evaluate Spark model by evaluating the underlying model
prediction = fitted_pipeline.transform(test_df)
pnl = prediction.select("label", "prediction")
pnl.show(100)

prediction_and_label = pnl.rdd.map(lambda row: (row.label, row.prediction))
metrics = RegressionMetrics(prediction_and_label)
print(metrics.r2)
print(metrics.meanAbsoluteError)
print(metrics.rootMeanSquaredError)