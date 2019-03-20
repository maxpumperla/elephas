from __future__ import print_function
from __future__ import absolute_import

from pyspark.ml.linalg import Vectors
import numpy as np
import random

from pyspark import SparkContext, SparkConf
from pyspark.sql import SQLContext
from pyspark.ml.feature import StringIndexer, StandardScaler
from pyspark.ml import Pipeline

from keras import optimizers
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation

from elephas.ml_model import ElephasEstimator


data_path = "../"

# Spark contexts
conf = SparkConf().setAppName('Otto_Spark_ML_Pipeline').setMaster('local[8]')
sc = SparkContext(conf=conf)
sql_context = SQLContext(sc)


# Data loader
def shuffle_csv(csv_file):
    lines = open(csv_file).readlines()
    random.shuffle(lines)
    open(csv_file, 'w').writelines(lines)


def load_data_rdd(csv_file, shuffle=True, train=True):
    if shuffle:
        shuffle_csv(data_path + csv_file)
    data = sc.textFile(data_path + csv_file)
    data = data.filter(lambda x: x.split(',')[0] != 'id').map(lambda line: line.split(','))
    if train:
        data = data.map(
            lambda line: (Vectors.dense(np.asarray(line[1:-1]).astype(np.float32)),
                          str(line[-1]).replace('Class_', '')))
    else:
        data = data.map(lambda line: (Vectors.dense(np.asarray(line[1:]).astype(np.float32)), "1"))
    return data


# Define Data frames
train_df = sql_context.createDataFrame(load_data_rdd("train.csv"), ['features', 'category'])
test_df = sql_context.createDataFrame(load_data_rdd("test.csv", shuffle=False, train=False), ['features', 'category'])

# Preprocessing steps
string_indexer = StringIndexer(inputCol="category", outputCol="index_category")
scaler = StandardScaler(inputCol="features", outputCol="scaled_features", withStd=True, withMean=True)

# Keras model
nb_classes = train_df.select("category").distinct().count()
input_dim = len(train_df.select("features").first()[0])

model = Sequential()
model.add(Dense(512, input_shape=(input_dim,)))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(nb_classes))
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam')

sgd = optimizers.SGD(lr=0.01)
sgd_conf = optimizers.serialize(sgd)

# Initialize Elephas Spark ML Estimator
estimator = ElephasEstimator()
estimator.set_keras_model_config(model.to_yaml())
estimator.set_optimizer_config(sgd_conf)
estimator.set_mode("synchronous")
estimator.set_loss("categorical_crossentropy")
estimator.set_metrics(['acc'])
estimator.setFeaturesCol("scaled_features")
estimator.setLabelCol("index_category")
estimator.set_epochs(10)
estimator.set_batch_size(128)
estimator.set_num_workers(1)
estimator.set_verbosity(0)
estimator.set_validation_split(0.15)
estimator.set_categorical_labels(True)
estimator.set_nb_classes(nb_classes)

# Fitting a model returns a Transformer
pipeline = Pipeline(stages=[string_indexer, scaler, estimator])
fitted_pipeline = pipeline.fit(train_df)

# Evaluate Spark model
prediction = fitted_pipeline.transform(train_df)
pnl = prediction.select("index_category", "prediction")
pnl.show(100)
