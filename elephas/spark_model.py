from __future__ import absolute_import

import numpy as np
import sys
from itertools import tee
from keras.models import Sequential, model_from_yaml
from pyspark.rdd import RDD

from .utils.functional_utils import add_params, get_neutral, divide_by

class SparkModel(object):
    def __init__(self, sc, master_network, nb_epoch=10, batch_size=32, verbose=0, validation_split=0.1):
        self.spark_context = sc
        self.master_network = master_network
        self.nb_epoch = nb_epoch
        self.batch_size = batch_size
        self.verbose = verbose
        self.validation_split = 0.1

    def get_train_config(self):
        train_config = {}
        train_config['nb_epoch'] = self.nb_epoch
        train_config['batch_size'] = self.batch_size
        train_config['verbose'] = self.verbose
        train_config['validation_split'] = self.validation_split
        return train_config

    def get_network(self):
        return self.master_network

    def set_network(self, network):
        self.master_network = network

    def train(self, rdd):
        num_partitions = rdd.getNumPartitions()
        yaml = self.master_network.to_yaml()
        parameters = self.spark_context.broadcast(self.master_network.get_weights())
        train_config = self.get_train_config()

        worker = SparkWorker(yaml, parameters, train_config)
        results = rdd.mapPartitions(worker.train)

        null_element = get_neutral(results.first())
        new_parameters = divide_by(results.fold(null_element, add_params), num_partitions)

        self.master_network.set_weights(new_parameters)

class SparkWorker(object):
    def __init__(self, yaml, parameters, train_config):
        self.yaml = yaml
        self.parameters = parameters
        self.train_config = train_config

    def train(self, data_iterator):
        feature_iterator, label_iterator = tee(data_iterator,2)
        X_train = np.asarray([x for x,y in feature_iterator])
        y_train = np.asarray([y for x,y in label_iterator])

        model = model_from_yaml(self.yaml)
        model.set_weights(self.parameters.value)
        model.fit(X_train, y_train, show_accuracy=True, **self.train_config)
        yield model.get_weights()
