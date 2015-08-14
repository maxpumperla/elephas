# Elephas: Keras Deep Learning on Apache Spark

## Introduction
Elephas brings deep learning with [Keras](http://keras.io) to [Apache Spark](http://spark.apache.org). Elephas intends to keep the simplicity and usability of Keras, allowing for fast prototyping of distributed models to run on large data sets.

ἐλέφας is Greek for _ivory_ and an accompanying project to κέρας, meaning _horn_. If this seems weird mentioning, like a bad dream, you should confirm it actually is at the [Keras documentation](https://github.com/fchollet/keras/blob/master/README.md). Elephas also means _elephant_, as in stuffed yellow elephant.

For now, elephas is a straight forward parallelization of Keras using Spark's RDDs. Models are initialized on the driver, then serialized and shipped to workers. Spark workers deserialize the model and train their chunk of data before broadcasting their parameters back to the driver. The "master" model is updated by averaging worker parameters. 


## Getting started
Currently Elephas is not available on PyPI, so you'll have to clone this repository and run
```
python setup.py install
```
from within that directory. As this is not the place to explain how to install Spark, you should simply follow the instructions at the (Spark download section)[http://spark.apache.org/downloads.html] for a local installation. After installing both Keras and Spark, training a model is done as follows:

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

## In the pipeline

- Integration with Spark MLLib:
  - Train and evaluate LabeledPoints RDDs
  - Make elephas models MLLib algorithms
- Integration with Spark ML:
  - Use DataFrames for training
  - Make models ML pipeline components
