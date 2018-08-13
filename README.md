# Elephas: Distributed Deep Learning with Keras & Spark [![Build Status](https://travis-ci.org/maxpumperla/elephas.svg?branch=master)](https://travis-ci.org/maxpumperla/elephas)

Elephas is an extension of [Keras](http://keras.io), which allows you to run distributed deep learning models at scale with [Spark](http://spark.apache.org). Elephas currently supports a number of applications, including:

- [Data-parallel training of deep learning models](#usage-of-data-parallel-models)
- [Distributed hyper-parameter optimization](#distributed-hyper-parameter-optimization)
- [Distributed training of ensemble models](#distributed-training-of-ensemble-models)


Schematically, elephas works as follows.

![Elephas](elephas.gif)

Table of content:
- [Elephas: Distributed Deep Learning with Keras & Spark](#elephas-distributed-deep-learning-with-keras-&-spark-)
  - [Introduction](#introduction)
  - [Getting started](#getting-started)
    - [Installation](#installation)
    - [Basic example](#basic-example)
    - [Spark ML example](#spark-ml-example)
  - [Usage of data-parallel models](#usage-of-data-parallel-models)
    - [Model updates (optimizers)](#model-updates-optimizers)
    - [Update frequency](#update-frequency)
    - [Update mode](#update-mode)
      - [Asynchronous updates with read and write locks (`mode='asynchronous'`)](#asynchronous-updates-with-read-and-write-locks-modeasynchronous)
      - [Asynchronous updates without locks (`mode='hogwild'`)](#asynchronous-updates-without-locks-modehogwild)
      - [Synchronous updates (`mode='synchronous'`)](#synchronous-updates-modesynchronous)
    - [Degree of parallelization (number of workers)](#degree-of-parallelization-number-of-workers)
  - [Distributed hyper-parameter optimization](#distributed-hyper-parameter-optimization)
  - [Distributed training of ensemble models](#distributed-training-of-ensemble-models)
  - [Discussion](#discussion)
  - [Future work & contributions](#future-work-&-contributions)
  - [Literature](#literature)

## Introduction
Elephas brings deep learning with [Keras](http://keras.io) to [Spark](http://spark.apache.org). Elephas intends to keep the simplicity and high usability of Keras, thereby allowing for fast prototyping of distributed models, which can be run on massive data sets. For an introductory example, see the following [iPython notebook](https://github.com/maxpumperla/elephas/blob/master/examples/Spark_ML_Pipeline.ipynb).

ἐλέφας is Greek for _ivory_ and an accompanying project to κέρας, meaning _horn_. If this seems weird mentioning, like a bad dream, you should confirm it actually is at the [Keras documentation](https://github.com/fchollet/keras/blob/master/README.md). Elephas also means _elephant_, as in stuffed yellow elephant.

Elephas implements a class of data-parallel algorithms on top of Keras, using Spark's RDDs and data frames. Keras Models are initialized on the driver, then serialized and shipped to workers, alongside with data and broadcasted model parameters. Spark workers deserialize the model, train their chunk of data and send their gradients back to the driver. The "master" model on the driver is updated by an optimizer, which takes gradients either synchronously or asynchronously.

## Getting started

### Installation
Install elephas from PyPI with
```
pip install elephas
```
Depending on what OS you are using, you may need to install some prerequisite modules (LAPACK, BLAS, fortran compiler) first.

For example, on Ubuntu Linux:
```
sudo apt-get install liblapack-dev libblas-dev gfortran
```

A quick way to install Spark locally is to use homebrew on Mac
```
brew install spark
```
or linuxbrew on linux.
```
brew install apache-spark
```
The brew version of Spark may be outdated at times. To build from source, simply follow the instructions at the [Spark download section](http://spark.apache.org/downloads.html) or use the following commands.  
```
wget http://apache.mirrors.tds.net/spark/spark-1.5.2/spark-1.5.2-bin-hadoop2.6.tgz -P ~
sudo tar zxvf ~/spark-* -C /usr/local
sudo mv /usr/local/spark-* /usr/local/spark
```
After that, make sure to put these path variables to your shell profile (e.g. `~/.zshrc`):
```
export SPARK_HOME=/usr/local/spark
export PATH=$PATH:$SPARK_HOME/bin
```

### Using Docker

Install and get Docker running by following the instructions here (https://www.docker.com/).

#### Building 

The build takes quite a while to run the first time since many packages need to be downloaded and installed. In the same directory as the ```Dockerfile``` run the following commands

```
docker build . -t pyspark/elephas
```

#### Running

The following command starts a container with the Notebook server listening for HTTP connections on port 8899 (since local Jupyter notebooks use 8888) without authentication configured. 

```
docker run -d -p 8899:8888 pyspark/elephas
```

#### Settings

- Memory 
In the ```Dockerfile``` the following lines can be adjusted to configure memory settings.

```
ENV SPARK_OPTS --driver-java-options=-Xms1024M --driver-java-options=-Xmx4096M --driver-java-options=-Dlog4j.logLevel=info
```

- Other
Other settings / configurations can be examined here https://github.com/kmader/docker-stacks/tree/master/keras-spark-notebook

### Basic example
After installing both Elephas and Spark, training a model is done schematically as follows:

- Create a local pyspark context
```python
from pyspark import SparkContext, SparkConf
conf = SparkConf().setAppName('Elephas_App').setMaster('local[8]')
sc = SparkContext(conf=conf)
```

- Define and compile a Keras model
```python
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import SGD
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

- Create an RDD from numpy arrays
```python
from elephas.utils.rdd_utils import to_simple_rdd
rdd = to_simple_rdd(sc, X_train, Y_train)
```

- A SparkModel is defined by passing Spark context and Keras model. Additionally, one has choose an optimizer used for updating the elephas model, an update frequency, a parallelization mode and the degree of parallelism, i.e. the number of workers.
```python
from elephas.spark_model import SparkModel
from elephas import optimizers as elephas_optimizers

adagrad = elephas_optimizers.Adagrad()
spark_model = SparkModel(sc,model, optimizer=adagrad, frequency='epoch', mode='asynchronous', num_workers=2)
spark_model.train(rdd, nb_epoch=20, batch_size=32, verbose=0, validation_split=0.1)
```

- Run your script using spark-submit
```
spark-submit --driver-memory 1G ./your_script.py
```
Increasing the driver memory even further may be necessary, as the set of parameters in a network may be very large and collecting them on the driver eats up a lot of resources. See the examples folder for a few working examples.

### Spark MLlib example
Following up on the last example, to create an RDD of LabeledPoints for supervised training from pairs of numpy arrays, use
```python
from elephas.utils.rdd_utils import to_labeled_point
lp_rdd = to_labeled_point(sc, X_train, Y_train, categorical=True)
```
Training a given LabeledPoint-RDD is very similar to what we've seen already
```python
from elephas.spark_model import SparkMLlibModel
adadelta = elephas_optimizers.Adadelta()
spark_model = SparkMLlibModel(sc,model, optimizer=adadelta, frequency='batch', mode='hogwild', num_workers=2)
spark_model.train(lp_rdd, nb_epoch=20, batch_size=32, verbose=0, validation_split=0.1, categorical=True, nb_classes=nb_classes)
```

### Spark ML example
To train a model with a SparkML estimator on a data frame, use the following syntax.
```python
df = to_data_frame(sc, X_train, Y_train, categorical=True)
test_df = to_data_frame(sc, X_test, Y_test, categorical=True)

adadelta = elephas_optimizers.Adadelta()
estimator = ElephasEstimator(sc,model,
        nb_epoch=nb_epoch, batch_size=batch_size, optimizer=adadelta, frequency='batch', mode='asynchronous', num_workers=2,
        verbose=0, validation_split=0.1, categorical=True, nb_classes=nb_classes)

fitted_model = estimator.fit(df)
```

Fitting an estimator results in a SparkML transformer, which we can use for predictions and other evaluations by calling the transform method on it.

``` python
prediction = fitted_model.transform(test_df)
pnl = prediction.select("label", "prediction")
pnl.show(100)

prediction_and_label= pnl.map(lambda row: (row.label, row.prediction))
metrics = MulticlassMetrics(prediction_and_label)
print(metrics.precision())
print(metrics.recall())
```

## Usage of data-parallel models

In the first example above we have seen that an elephas model is instantiated like this

```python
spark_model = SparkModel(sc,model, optimizer=adagrad, frequency='epoch', mode='asynchronous', num_workers=2)
```
So, apart from the canonical Spark context and Keras model, Elephas models have four parameters to tune and we will describe each of them next.

### Model updates (optimizers)

`optimizer`: The optimizers module in elephas is an adaption of the same module in keras, i.e. it provides the user with the following list of optimizers:

- `SGD`
- `RMSprop`
- `Adagrad`
- `Adadelta`
- `Adam`

Once constructed, each of these can be passed to the *optimizer* parameter of the model. Updates in keras are computed with the help of theano, so most of the data structures in keras optimizers stem from theano. In elephas, gradients have already been computed by the respective workers, so it makes sense to entirely work with numpy arrays internally.

Note that in order to set up an elephas model, you have to specify two optimizers, one for elephas and one for the underlying keras model. Individual workers produce updates according to keras optimizers and the "master" model on the driver uses elephas optimizers to aggregate them. For starters, we recommend keras models with SGD and elephas models with Adagrad or Adadelta.

### Update frequency

`frequency`: The user can decide how often updates are passed to the master model by controlling the *frequency* parameter. To update every batch, choose 'batch' and to update only after every epoch, choose 'epoch'.

### Update mode

`mode`: Currently, there's three different modes available in elephas, each corresponding to a different heuristic or parallelization scheme adopted, which is controlled by the *mode* parameter. The default property is 'asynchronous'.

#### Asynchronous updates with read and write locks (`mode='asynchronous'`)

This mode implements the algorithm described as *downpour* in [1], i.e. each worker can send updates whenever they are ready. The master model makes sure that no update gets lost, i.e. multiple updates get applied at the "same" time,  by locking the master parameters while reading and writing parameters. This idea has been used in Google's DistBelief framework.

#### Asynchronous updates without locks (`mode='hogwild'`)
Essentially the same procedure as above, but without requiring the locks. This heuristic assumes that we still fare well enough, even if we loose an update here or there. Updating parameters lock-free in a non-distributed setting for SGD goes by the name 'Hogwild!' [2], it's distributed extension is called 'Dogwild!' [3].  

#### Synchronous updates (`mode='synchronous'`)

In this mode each worker sends a new batch of parameter updates at the same time, which are then processed on the master. Accordingly, this algorithm is sometimes called *batch synchronous parallel* or just BSP.

### Degree of parallelization (number of workers)

`num_workers`: Lastly, the degree to which we parallelize our training data is controlled by the parameter *num_workers*.

## Distributed hyper-parameter optimization

Hyper-parameter optimization with elephas is based on [hyperas](https://github.com/maxpumperla/hyperas), a convenience wrapper for hyperopt and keras. Make sure to have at least version ```0.1.2``` of hyperas installed. Each Spark worker executes a number of trials, the results get collected and the best model is returned. As the distributed mode in hyperopt (using MongoDB), is somewhat difficult to configure and error prone at the time of writing, we chose to implement parallelization ourselves. Right now, the only available optimization algorithm is random search.

The first part of this example is more or less directly taken from the hyperas documentation. We define data and model as functions, hyper-parameter ranges are defined through braces. See the hyperas documentation for more on how this works.

```python
from __future__ import print_function
from hyperopt import Trials, STATUS_OK, tpe
from hyperas.distributions import choice, uniform

def data():
    '''
    Data providing function:

    Make sure to have every relevant import statement included here and return data as
    used in model function below. This function is separated from model() so that hyperopt
    won't reload data for each evaluation run.
    '''
    from keras.datasets import mnist
    from keras.utils import np_utils
    (X_train, y_train), (X_test, y_test) = mnist.load_data()
    X_train = X_train.reshape(60000, 784)
    X_test = X_test.reshape(10000, 784)
    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')
    X_train /= 255
    X_test /= 255
    nb_classes = 10
    Y_train = np_utils.to_categorical(y_train, nb_classes)
    Y_test = np_utils.to_categorical(y_test, nb_classes)
    return X_train, Y_train, X_test, Y_test


def model(X_train, Y_train, X_test, Y_test):
    '''
    Model providing function:

    Create Keras model with double curly brackets dropped-in as needed.
    Return value has to be a valid python dictionary with two customary keys:
        - loss: Specify a numeric evaluation metric to be minimized
        - status: Just use STATUS_OK and see hyperopt documentation if not feasible
    The last one is optional, though recommended, namely:
        - model: specify the model just created so that we can later use it again.
    '''
    from keras.models import Sequential
    from keras.layers.core import Dense, Dropout, Activation
    from keras.optimizers import RMSprop

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

    model.fit(X_train, Y_train,
              batch_size={{choice([64, 128])}},
              nb_epoch=1,
              show_accuracy=True,
              verbose=2,
              validation_data=(X_test, Y_test))
    score, acc = model.evaluate(X_test, Y_test, show_accuracy=True, verbose=0)
    print('Test accuracy:', acc)
    return {'loss': -acc, 'status': STATUS_OK, 'model': model.to_yaml(), 'weights': pickle.dumps(model.get_weights())}
```

Once the basic setup is defined, running the minimization is done in just a few lines of code:

```python
from hyperas import optim
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

Building on the last section, it is possible to train ensemble models with elephas by means of running hyper-parameter optimization on large search spaces and defining a resulting voting classifier on the top-n performing models. With ```data``` and ```model``` defined as above, this is a simple as running

```python
result = hyperparam_model.best_ensemble(nb_ensemble_models=10, model=model, data=data, max_evals=5)
```
In this example an ensemble of 10 models is built, based on optimization of at most 5 runs on each of the Spark workers.

## Discussion

Premature parallelization may not be the root of all evil, but it may not always be the best idea to do so. Keep in mind that more workers mean less data per worker and parallelizing a model is not an excuse for actual learning. So, if you can perfectly well fit your data into memory *and* you're happy with training speed of the model consider just using keras.

One exception to this rule may be that you're already working within the Spark ecosystem and want to leverage what's there. The above SparkML example shows how to use evaluation modules from Spark and maybe you wish to further process the outcome of an elephas model down the road. In this case, we recommend to use elephas as a simple wrapper by setting num_workers=1.

Note that right now elephas restricts itself to data-parallel algorithms for two reasons. First, Spark simply makes it very easy to distribute data. Second, neither Spark nor Theano make it particularly easy to split up the actual model in parts, thus making model-parallelism practically impossible to realize.

Having said all that, we hope you learn to appreciate elephas as a pretty easy to setup and use playground for data-parallel deep-learning algorithms.


## Future work & contributions

Constructive feedback and pull requests for elephas are very welcome. Here's a few things we're having in mind for future development

- Benchmarks for training speed and accuracy.
- Some real-world tests on EC2 instances with large data sets like imagenet.

## Literature
[1] J. Dean, G.S. Corrado, R. Monga, K. Chen, M. Devin, QV. Le, MZ. Mao, M’A. Ranzato, A. Senior, P. Tucker, K. Yang, and AY. Ng. [Large Scale Distributed Deep Networks](http://research.google.com/archive/large_deep_networks_nips2012.html).

[2] F. Niu, B. Recht, C. Re, S.J. Wright [HOGWILD!: A Lock-Free Approach to Parallelizing Stochastic Gradient Descent](http://arxiv.org/abs/1106.5730)

[3] C. Noel, S. Osindero. [Dogwild! — Distributed Hogwild for CPU & GPU](http://stanford.edu/~rezab/nips2014workshop/submits/dogwild.pdf)
