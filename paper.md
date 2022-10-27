---
title: 'Elephas: Distributed Deep Learning with Keras & Spark'
tags:
  - Python
  - Distributed Computing
  - Deep Learning
  - Keras
  - Tensorflow
  - Apache Spark
authors:
  - name: Max Pumperla
    affiliation: "1, 2"
  - name: Daniel Cahall
    affiliation: 3
affiliations:
  - name: IU Internationale Hochschule
    index: 1
  - name: Pathmind Inc.
    index: 2
  - name: Independent researcher
    index: 3
date: 18 November 2021
bibliography: paper.bib
    
---

# Summary

Elephas is an extension of [Keras](https://keras.io/), which allows you to run distributed deep learning models at scale with [Apache Spark](http://spark.apache.org/).
It was built to allow researchers and developers to distribute their deep learning experiments as easily as possible on a Spark computer cluster.
With elephas, researchers can currently run data-parallel training of deep learning models with distribution modes as suggested in [@NIPS2012_6aca9700], [@NIPS2011_218a0aef] and [@Noel2014DogwildD].
Additionally, elephas supports distributed training of ensemble models.
Until version 2.1., elephas also supported distributed hyper-parameter optimization of Keras models.

Elephas keeps the simplicity and high usability of Keras, thereby allowing for fast prototyping of distributed models.
When ready, researchers can then scale out their experiments on massive data sets.
Elephas comes with [full API documentation](http://maxpumperla.com/elephas/) and [examples](https://github.com/maxpumperla/elephas/blob/master/examples/Spark_ML_Pipeline.ipynb) to get you started.
Initiated in late 2015, elephas has been actively maintained since then and has reached maturity for distributed deep learning on Spark.

# Statement of need

Modern deep learning solutions require ever more data and computation power.
A study by [OpenAI](https://openai.com/blog/ai-and-compute) suggests that the number of operations needed for AI systems, by now measured in _petaflops_, shows exponential growth and has in fact been doubling every 3.4 months since 2012.
This vastly outweighs the computational gains to expect from single machines, even according to the most optimistic version of Moore's law.
There's a clear need for solutions that can scale to compute clusters.
While some large companies have such large deep learning models that they have to resort to distributing the models themselves (model-parallelism), for most researchers and the majority of companies, compute time and data volume are the predominant bottlenecks.

Apache Spark has established itself as one of the most popular platforms for distributed computing.
However, its native machine learning (ML) capabilities are limited by design.
Spark excels when transforming massive datasets and applying built-in ML algorithms with [Spark MLlib](http://spark.apache.org/mllib/), but does not support implementation of custom algorithms with deep learning frameworks such as [Google's TensorFlow](https://www.tensorflow.org/) and specifically its convenient Keras API.

Elephas was the first open-source framework to support distributed training with Keras on Spark.
It was followed by libraries such as Yahoo's [TensorFlowOnSpark](https://github.com/yahoo/TensorFlowOnSpark), which does not follow Keras API design principles, later on.
In recent years, other popular distributed deep learning frameworks have emerged, such as the powerful Horovod [@sergeev2018horovod], which initially did not have Spark support.
BigDL [@SOCC2019_BIGDL] is another such framework worth mentioning, especially in conjunction with [Intel's Analytics Zoo](https://github.com/intel-analytics/analytics-zoo).
Elephas is still in active use and has been leveraged by [millions of users](https://pypistats.org/packages/elephas) in the Python deep learning community.

# Design and API

Elephas has a tight integration with many of Spark's core abstraction layers. Apart from the basic integration for Spark's resilient distributed datasets (RDDs), elephas works with MLlib models, with Spark ML estimators, can train ensemble models and run distributed inference.

## Basic Spark training

Elephas' core abstraction to bring Keras models to Spark is called `SparkModel`.
To use it, you define a Keras model, load your data, define your `SparkModel` with the training mode and update frequency of your choice, and then fit your model on distributed data:

```python
from elephas.spark_model import SparkModel
from elephas.utils.rdd_utils import to_simple_rdd
from pyspark import SparkContext, SparkConf
from tensorflow.keras.models import Sequential

# Define Spark context
conf = SparkConf().setAppName('Elephas_App')\
                  .setMaster('local[8]')
sc = SparkContext(conf=conf)

# Define Keras model
model = Sequential()  
model.add(...)
model.compile(...)

# Load training data
x_train, y_train = ... 

# Convert to Spark RDD
rdd = to_simple_rdd(sc, x_train, y_train)

# Define Elephas model
spark_model = SparkModel(model, frequency='epoch', mode='asynchronous')

# Run distributed training
spark_model.fit(rdd, ...)
```

Afterwards your training job can be submitted to any Spark cluster like this:

```bash
spark-submit --driver-memory 1G ./your_script.py
```

## Distributed Inference

When your model `spark_model` has finished training, you can easily run distributed inference on your test data or compute evaluation metrics on it as follows.

```python
# Load test data
x_test, y_test = ...

# Perform  distributed inference
predictions = spark_model.predict(x_test)

# Run distributed evaluation/scoring
evaluation = spark_model.evaluate(x_test, y_test)
```

## Spark MLlib and ML integrations

To leverage Spark's `LabeledPoint` RDD to encode features and labels for a prediction task, you can use helper functions provided by elephas and then train a so-called `SparkMLlibModel` on your data for your Keras `model`.

```python
from elephas.utils.rdd_utils import to_labeled_point
from elephas.spark_model import SparkMLlibModel

# Create a LabeledPoint RDD
lp_rdd = to_labeled_point(sc, x_train, y_train, categorical=True)

# Define and train a SparkMLlib model
spark_model = SparkMLlibModel(model, frequency='batch', mode='hogwild')
spark_model.train(lp_rdd, ...)
```

Likewise, to create a Spark ML `Estimator` to fit it on a Spark `DataFrame`, you can use elephas' s `SparkEstimator`.

```python
from elephas.ml.adapter import to_data_frame
from elephas.ml_model import ElephasEstimator

# Create a Spark DataFrame
df = to_data_frame(sc, x_train, y_train, categorical=True)

# Define and fit an Elephas Estimator
estimator = ElephasEstimator(model, ...)
fitted_model = estimator.fit(df)
```

To summarize, elephas provides you with training and evaluation support for custom Keras models for many practical scenarios and data structures supported on a Spark cluster.

# Acknowledgements

We would like to thank all the open-source contributors that helped making `elephas` what it is today.
A special thanks goes to Daniel Cahill, who has taken over the maintainer role for this project since 2020.

# References
