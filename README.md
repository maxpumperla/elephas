# Elephas: Distributed Deep Learning with Keras & Spark 

![Elephas](https://github.com/maxpumperla/elephas/blob/master/elephas-logo.png)

## 

[![Build Status](https://travis-ci.org/maxpumperla/elephas.svg?branch=master)](https://travis-ci.org/maxpumperla/elephas)
[![license](https://img.shields.io/github/license/mashape/apistatus.svg?maxAge=2592000)](https://github.com/maxpumperla/elephas/blob/master/LICENSE)

Elephas is an extension of [Keras](http://keras.io), which allows you to run distributed deep learning models at 
scale with [Spark](http://spark.apache.org). Elephas currently supports a number of 
applications, including:

- [Data-parallel training of deep learning models](#basic-spark-integration)
- [Distributed hyper-parameter optimization](#distributed-hyper-parameter-optimization)
- [Distributed training of ensemble models](#distributed-training-of-ensemble-models)


Schematically, elephas works as follows.

![Elephas](https://github.com/maxpumperla/elephas/blob/master/elephas.gif)

Table of content:
* [Elephas: Distributed Deep Learning with Keras & Spark](#elephas-distributed-deep-learning-with-keras-&-spark-)
  * [Introduction](#introduction)
  * [Getting started](#getting-started)
  * [Basic Spark integration](#basic-spark-integration)
  * [Spark MLlib integration](#spark-mllib-integration)
  * [Spark ML integration](#spark-ml-integration)
  * [Distributed hyper-parameter optimization](#distributed-hyper-parameter-optimization)
  * [Distributed training of ensemble models](#distributed-training-of-ensemble-models)
  * [Discussion](#discussion)
  * [Literature](#literature)



## Introduction
Elephas brings deep learning with [Keras](http://keras.io) to [Spark](http://spark.apache.org). Elephas intends to 
keep the simplicity and high usability of Keras, thereby allowing for fast prototyping of distributed models, which 
can be run on massive data sets. For an introductory example, see the following 
[iPython notebook](https://github.com/maxpumperla/elephas/blob/master/examples/Spark_ML_Pipeline.ipynb).

ἐλέφας is Greek for _ivory_ and an accompanying project to κέρας, meaning _horn_. If this seems weird mentioning, like 
a bad dream, you should confirm it actually is at the 
[Keras documentation](https://github.com/fchollet/keras/blob/master/README.md). 
Elephas also means _elephant_, as in stuffed yellow elephant.

Elephas implements a class of data-parallel algorithms on top of Keras, using Spark's RDDs and data frames. 
Keras Models are initialized on the driver, then serialized and shipped to workers, alongside with data and broadcasted 
model parameters. Spark workers deserialize the model, train their chunk of data and send their gradients back to the 
driver. The "master" model on the driver is updated by an optimizer, which takes gradients either synchronously or
asynchronously.

## Getting started

Just install elephas from PyPI with, Spark will be installed through `pyspark` for you.

```
pip install elephas
```

That's it, you should now be able to run Elephas examples.

## Basic Spark integration

After installing both Elephas, you can train a model as follows. First, create a local pyspark context
```python
from pyspark import SparkContext, SparkConf
conf = SparkConf().setAppName('Elephas_App').setMaster('local[8]')
sc = SparkContext(conf=conf)
```

Next, you define and compile a Keras model
```python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation
from tensorflow.keras.optimizers import SGD
model = Sequential()
model.add(Dense(128, input_dim=784))
model.add(Activation('relu'))
model.add(Dropout(0.2))
model.add(Dense(128))
model.add(Activation('relu'))
model.add(Dropout(0.2))
model.add(Dense(10))
model.add(Activation('softmax'))
model.compile(loss='categorical_crossentropy', optimizer=SGD())
```

and create an RDD from numpy arrays (or however you want to create an RDD)
```python
from elephas.utils.rdd_utils import to_simple_rdd
rdd = to_simple_rdd(sc, x_train, y_train)
```

The basic model in Elephas is the `SparkModel`. You initialize a `SparkModel` by passing in a compiled Keras model, 
an update frequency and a parallelization mode. After that you can simply `fit` the model on your RDD. Elephas `fit`
has the same options as a Keras model, so you can pass `epochs`, `batch_size` etc. as you're used to from tensorflow.keras.

```python
from elephas.spark_model import SparkModel

spark_model = SparkModel(model, frequency='epoch', mode='asynchronous')
spark_model.fit(rdd, epochs=20, batch_size=32, verbose=0, validation_split=0.1)
```

Your script can now be run using spark-submit
```bash
spark-submit --driver-memory 1G ./your_script.py
```

Increasing the driver memory even further may be necessary, as the set of parameters in a network may be very large 
and collecting them on the driver eats up a lot of resources. See the examples folder for a few working examples.

## Distributed Inference / Evaluation

The `SparkModel` can also be used for distributed inference (prediction) and evaluation. Similar to the `fit` method,  the `predict` and `evaluate` methods
conform to the Keras Model API. 

```python
from elephas.spark_model import SparkModel

# create/train the model, similar to the previous section (Basic Spark Integration)
model = ...
spark_model = SparkModel(model, ...)
spark_model.fit(...)

x_test, y_test = ... # load test data

predictions = spark_model.predict(x_test) # perform inference
evaluation = spark_model.evaluate(x_test, y_test) # perform evaluation/scoring
```
The paradigm is identical to the data parallelism in training, as the model is serialized and shipped to the workers and used to evaluate a chunk of the testing data.

## Spark MLlib integration

Following up on the last example, to use Spark's MLlib library with Elephas, you create an RDD of LabeledPoints for 
supervised training as follows

```python
from elephas.utils.rdd_utils import to_labeled_point
lp_rdd = to_labeled_point(sc, x_train, y_train, categorical=True)
```

Training a given LabeledPoint-RDD is very similar to what we've seen already

```python
from elephas.spark_model import SparkMLlibModel
spark_model = SparkMLlibModel(model, frequency='batch', mode='hogwild')
spark_model.train(lp_rdd, epochs=20, batch_size=32, verbose=0, validation_split=0.1, 
                  categorical=True, nb_classes=nb_classes)
```


## Spark ML integration

To train a model with a SparkML estimator on a data frame, use the following syntax.
```python
df = to_data_frame(sc, x_train, y_train, categorical=True)
test_df = to_data_frame(sc, x_test, y_test, categorical=True)

estimator = ElephasEstimator(model, epochs=epochs, batch_size=batch_size, frequency='batch', mode='asynchronous',
                             categorical=True, nb_classes=nb_classes)
fitted_model = estimator.fit(df)
```

Fitting an estimator results in a SparkML transformer, which we can use for predictions and other evaluations by 
calling the transform method on it.

```python
prediction = fitted_model.transform(test_df)
pnl = prediction.select("label", "prediction")
pnl.show(100)

prediction_and_label= pnl.rdd.map(lambda row: (row.label, row.prediction))
metrics = MulticlassMetrics(prediction_and_label)
print(metrics.precision())
print(metrics.recall())
```


## Distributed hyper-parameter optimization

Hyper-parameter optimization with elephas is based on [hyperas](https://github.com/maxpumperla/hyperas), a convenience 
wrapper for hyperopt and keras. Each Spark worker executes a number of trials, the results get collected and the best 
model is returned. As the distributed mode in hyperopt (using MongoDB), is somewhat difficult to configure and error 
prone at the time of writing, we chose to implement parallelization ourselves. Right now, the only available 
optimization algorithm is random search.

The first part of this example is more or less directly taken from the hyperas documentation. We define data and model 
as functions, hyper-parameter ranges are defined through braces. See the hyperas documentation for more on how 
this works.

```python
from hyperopt import STATUS_OK
from hyperas.distributions import choice, uniform

def data():
    from tensorflow.keras.datasets import mnist
    from tensorflow.keras.utils import to_categorical
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = x_train.reshape(60000, 784)
    x_test = x_test.reshape(10000, 784)
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255
    x_test /= 255
    nb_classes = 10
    y_train = to_categorical(y_train, nb_classes)
    y_test = to_categorical(y_test, nb_classes)
    return x_train, y_train, x_test, y_test


def model(x_train, y_train, x_test, y_test):
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense, Dropout, Activation
    from tensorflow.keras.optimizers import RMSprop

    model = Sequential()
    model.add(Dense(512, input_shape=(784,)))
    model.add(Activation('relu'))
    model.add(Dropout({{uniform(0, 1)}}))
    model.add(Dense({{choice([256, 512, 1024])}}))
    model.add(Activation('relu'))
    model.add(Dropout({{uniform(0, 1)}}))
    model.add(Dense(10))
    model.add(Activation('softmax'))

    rms = RMSprop()
    model.compile(loss='categorical_crossentropy', optimizer=rms)

    model.fit(x_train, y_train,
              batch_size={{choice([64, 128])}},
              nb_epoch=1,
              show_accuracy=True,
              verbose=2,
              validation_data=(x_test, y_test))
    score, acc = model.evaluate(x_test, y_test, show_accuracy=True, verbose=0)
    print('Test accuracy:', acc)
    return {'loss': -acc, 'status': STATUS_OK, 'model': model.to_yaml()}
```

Once the basic setup is defined, running the minimization is done in just a few lines of code:

```python
from elephas.hyperparam import HyperParamModel
from pyspark import SparkContext, SparkConf

# Create Spark context
conf = SparkConf().setAppName('Elephas_Hyperparameter_Optimization').setMaster('local[8]')
sc = SparkContext(conf=conf)

# Define hyper-parameter model and run optimization
hyperparam_model = HyperParamModel(sc)
hyperparam_model.minimize(model=model, data=data, max_evals=5)
```

## Distributed training of ensemble models

Building on the last section, it is possible to train ensemble models with elephas by means of running hyper-parameter 
optimization on large search spaces and defining a resulting voting classifier on the top-n performing models. 
With ```data``` and ```model``` defined as above, this is a simple as running

```python
result = hyperparam_model.best_ensemble(nb_ensemble_models=10, model=model, data=data, max_evals=5)
```
In this example an ensemble of 10 models is built, based on optimization of at most 5 runs on each of the Spark workers.

## Discussion

Premature parallelization may not be the root of all evil, but it may not always be the best idea to do so. Keep in 
mind that more workers mean less data per worker and parallelizing a model is not an excuse for actual learning. 
So, if you can perfectly well fit your data into memory *and* you're happy with training speed of the model consider 
just using keras.

One exception to this rule may be that you're already working within the Spark ecosystem and want to leverage what's 
there. The above SparkML example shows how to use evaluation modules from Spark and maybe you wish to further process 
the outcome of an elephas model down the road. In this case, we recommend to use elephas as a simple wrapper by setting 
num_workers=1.

Note that right now elephas restricts itself to data-parallel algorithms for two reasons. First, Spark simply makes it 
very easy to distribute data. Second, neither Spark nor Theano make it particularly easy to split up the actual model 
in parts, thus making model-parallelism practically impossible to realize.

Having said all that, we hope you learn to appreciate elephas as a pretty easy to setup and use playground for 
data-parallel deep-learning algorithms.


## Literature
[1] J. Dean, G.S. Corrado, R. Monga, K. Chen, M. Devin, QV. Le, MZ. Mao, M’A. Ranzato, A. Senior, P. Tucker, K. Yang, and AY. Ng. [Large Scale Distributed Deep Networks](http://research.google.com/archive/large_deep_networks_nips2012.html).

[2] F. Niu, B. Recht, C. Re, S.J. Wright [HOGWILD!: A Lock-Free Approach to Parallelizing Stochastic Gradient Descent](http://arxiv.org/abs/1106.5730)

[3] C. Noel, S. Osindero. [Dogwild! — Distributed Hogwild for CPU & GPU](http://stanford.edu/~rezab/nips2014workshop/submits/dogwild.pdf)

## Maintainers / Contributions

This great project was started by Max Pumperla, and is currently maintained by Daniel Cahall (https://github.com/danielenricocahall). If you have any questions, please feel free to open up an issue or send an email to danielenricocahall@gmail.com. If you want to contribute, feel free to submit a PR, or start a conversation about how we can go about implementing something.