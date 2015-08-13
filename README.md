# Elephas: Keras Deep Learning on Apache Spark

## Introduction
Elephas brings deep learning with [Keras](http://keras.io) to [Apache Spark](http://spark.apache.org). ἐλέφας is Greek for _ivory_ and an accompanioning project to κέρας, meaning _horn_. If it seems weird to you mentioning this, like a bad dream, you should confirm it actually is, at the [Keras documentation](https://github.com/fchollet/keras/blob/master/README.md).

## Getting started
After installing both Keras and Spark, training a model is done as follows:

- Create a local pyspark context
```python
from pyspark import SparkContext, SparkConf
conf = SparkConf().setAppName('Elephas_App').setMaster('local[8]')
sc = SparkContext(conf=conf)
```

- Define a Keras model
```python
model = Sequential()
model.add(Dense(784, 128))
model.add(Activation('relu'))
model.add(Dropout(0.2))
model.add(Dense(128, 128))
model.add(Activation('relu'))
model.add(Dropout(0.2))
model.add(Dense(128, 10))
model.add(Activation('softmax'))
```

- Create an RDD from numpy arrays 
```python
from elephas.utils.rdd_utils import to_simple_rdd
rdd = to_simple_rdd(sc, X_train, Y_train)
```

- Define a SparkModel from Spark context and Keras model, then simply train it
```python
from elephas.spark_model import SparkModel
spark_model = SparkModel(sc,model)
spark_model.train(rdd, nb_epoch=20, batch_size=32, verbose=0, validation_split=0.1)
```

- Run your script using spark-submit
```
spark-submit --driver-memory 1G ./your_script.py
```

See the examples folder for a working example.
