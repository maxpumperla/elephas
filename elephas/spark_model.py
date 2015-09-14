from __future__ import absolute_import

import numpy as np
import sys
from itertools import tee
from keras.models import Sequential, model_from_yaml
from pyspark.rdd import RDD

from .utils.functional_utils import add_params, get_neutral, divide_by

def get_train_config(nb_epoch, batch_size, verbose, validation_split):
    train_config = {}
    train_config['nb_epoch'] = nb_epoch
    train_config['batch_size'] = batch_size
    train_config['verbose'] = verbose
    train_config['validation_split'] = validation_split
    return train_config


class SparkModel(object):
    def __init__(self, sc, master_network):
        self.spark_context = sc
        self.master_network = master_network

    def get_network(self):
        return self.master_network

    def set_network(self, network):
        self.master_network = network

    def train(self, rdd, nb_epoch=10, batch_size=32, verbose=0, validation_split=0.1, num_workers=8):

        rdd = rdd.repartition(num_workers)

        yaml = self.master_network.to_yaml()
        parameters = self.spark_context.broadcast(self.master_network.get_weights())
        train_config = get_train_config(nb_epoch, batch_size, verbose, validation_split)

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



