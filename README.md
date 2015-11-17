# Elephas: Keras Deep Learning on Apache Spark

## Introduction
Elephas brings deep learning with [Keras](http://keras.io) to [Apache Spark](http://spark.apache.org). Elephas intends to keep the simplicity and usability of Keras, allowing for fast prototyping of distributed models to run on large data sets.

ἐλέφας is Greek for _ivory_ and an accompanying project to κέρας, meaning _horn_. If this seems weird mentioning, like a bad dream, you should confirm it actually is at the [Keras documentation](https://github.com/fchollet/keras/blob/master/README.md). Elephas also means _elephant_, as in stuffed yellow elephant.

Elephas implements a class of data-parallel algorithms on top of Keras, using Spark's RDDs and data frames. Keras Models are initialized on the driver, then serialized and shipped to workers, alongside with data and broadcasted model parameters. Spark workers deserialize the model, train their chunk of data and send their gradients back to the driver. The "master" model on the driver is updated by an optimizer, which takes gradients either synchronously or asynchronously.

![](elephas.gif)


## Getting started

### Installation
Install elephas from PyPI with 
```
pip install elephas
```
A quick way to install Spark locally is to use homebrew on Mac 
```
brew install spark
```
or linuxbrew on linux
```
brew install apache-spark
```
If this is not an option, you should simply follow the instructions at the [Spark download section](http://spark.apache.org/downloads.html). 

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
spark_model.train(rdd, nb_epoch=20, batch_size=32, verbose=0, validation_split=0.1, num_workers=8)
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

## Usage

In the first example above we have seen that an elephas model is instantiated like this

```python
spark_model = SparkModel(sc,model, optimizer=adagrad, frequency='epoch', mode='asynchronous', num_workers=2)
```
So, apart from the canonical Spark context and Keras model, Elephas models have four parameters to tune and we will describe each of them next.

### Model updates (optimizers)

The optimizers module in elephas is an adaption of the same module in keras, i.e. it provides the user with the following list of optimizers:
- SGD
- RMSprop
- Adagrad
- Adadelta
- Adam
Once constructed, each of these can be passed to the *optimizer* parameter of the model. Updates in keras are computed with the help of theano, so most of the data structures in keras optimizers stem from theano. In elephas, gradients have already been computed by the respective workers, so it makes sense to entirely work with numpy arrays internally. 

Note that in order to set up an elephas model, you have to specify two optimizers, one for elephas and one for the underlying keras model. Individual workers produce updates according to keras optimizers and the "master" model on the driver uses elephas optimizers to aggregate them. For starters, we recommend keras models with SGD and elephas models with Adagrad or Adadelta.

### Update frequency

The user can decide how often updates are passed to the master model by controlling the *frequency* parameter. To update every batch, choose 'batch' and to update only after every epoch, choose 'epoch'.

### Update mode

Currently, there's three different modes available in elephas, each corresponding to a different heuristic or parallelization scheme adopted, which is controlled by the *mode* parameter. The default property is 'asynchronous'.

#### Asynchronous updates with read and write locks (mode='asynchronous')

This mode implements the algorithm described as *downpour* in [1], i.e. each worker can send updates whenever they are ready. The master model makes sure that no update gets lost, i.e. multiple updates get applied at the "same" time,  by locking the master parameters while reading and writing parameters. This idea has been used in Google's DistBelief framework. 

#### Asynchronous updates without locks (mode='hogwild')
Essentially the same procedure as above, but without requiring the locks. This heuristic assumes that we still fare well enough, even if we loose an update here or there. Updating parameters lock-free in a non-distributed setting for SGD goes by the name 'Hogwild!' [2], it's distributed extension is called 'Dogwild!' [3].  

#### Synchronous updates (mode='synchronous')

In this mode each worker sends a new batch of parameter updates at the same time, which are then processed on the master. Accordingly, this algorithm is sometimes called *batch synchronous parallel*.

### Degree of parallelization (number of workers)

Lastly, the degree to which we parallelize our training data is controlled by the parameter *num_workers*.

## Discussion

Premature parallelization may not be the root of all evil, but it may not always be the best idea to do so. Keep in mind that more workers mean less data per worker and parallelizing a model is not an excuse for actual learning. So, if you can perfectly well fit your data into memory *and* you're happy with training speed of the model consider just using keras. 

One exception to this rule may be that you're already working within the Spark ecosystem and want to leverage what's there. The above SparkML example shows how to use evaluation modules from Spark and maybe you wish to further process the outcome of an elephas model down the road. In this case, we recommend to use elephas as a simple wrapper by setting num_workers=1.

Note that right now elephas restricts itself to data-parallel algorithms for two reasons. First, Spark simply makes it very easy to distribute data. Second, neither Spark nor Theano make it particularly easy to split up the actual model in parts, thus making model-parallelism practically impossible to realize.

Having said all that, we hope you learn to appreciate elephas as a pretty easy to setup and use playground for data-parallel deep-learning algorithms.


## Future work & contributions

Constructive feedback and pull requests for elephas are very welcome. Here's a few things we're having in mind for future development 

- Tighter Spark ML integration. Pipelines do not work yet.
- Benchmarks for training speed and accuracy.
- Some real-world tests on EC2 instances with large data sets like imagenet.

## Literature
[1] J. Dean, G.S. Corrado, R. Monga, K. Chen, M. Devin, QV. Le, MZ. Mao, M’A. Ranzato, A. Senior, P. Tucker, K. Yang, and AY. Ng. [Large Scale Distributed Deep Networks](http://research.google.com/archive/large_deep_networks_nips2012.html).
[2] F. Niu, B. Recht, C. Re, S.J. Wright [HOGWILD!: A Lock-Free Approach to Parallelizing Stochastic Gradient Descent](http://arxiv.org/abs/1106.5730)
[3] C. Noel, S. Osindero. [Dogwild! — Distributed Hogwild for CPU & GPU](http://stanford.edu/~rezab/nips2014workshop/submits/dogwild.pdf)
