from __future__ import absolute_import

import numpy as np
import sys
from itertools import tee
from keras.models import Sequential, model_from_yaml
from pyspark.rdd import RDD

from threading import Thread
from multiprocessing import Process
import six.moves.cPickle as pickle

from .utils.functional_utils import add_params, subtract_params, get_neutral, divide_by
from .utils.rwlock import RWLock

from flask import Flask, request

import urllib2
import urlparse

from six.moves import range

def get_train_config(nb_epoch, batch_size, verbose, validation_split):
    train_config = {}
    train_config['nb_epoch'] = nb_epoch
    train_config['batch_size'] = batch_size
    train_config['verbose'] = verbose
    train_config['validation_split'] = validation_split
    return train_config

def get_server_weights(master='localhost:5000'):
    request = urllib2.Request('http://%s/parameters' % master,
        headers={'Content-Type': 'application/elephas'})
    return pickle.loads(urllib2.urlopen(request).read())

def put_deltas_to_server(delta, master='localhost:5000'):
    request = urllib2.Request('http://%s/update' % master, pickle.dumps(delta, -1),
        headers={'Content-Type': 'application/elephas'})
    return urllib2.urlopen(request).read()


class SparkModel(object):
    def __init__(self, sc, master_network):
        self.spark_context = sc
        self.master_network = master_network
        self.weights = master_network.get_weights()
        self.pickled_weights = None
        self.lock = RWLock()

    def start_server(self):
        print("Starting parameter server...")
        self.server = Process(target=self.start_service)
        self.server.start()

    def stop_server(self):
        print("Terminating parameter server...")
        self.server.terminate()
        self.server.join()

    def get_network(self):
        return self.master_network

    def set_network(self, network):
        self.master_network = network

    def start_service(self):

        app = Flask(__name__)
        self.app = app

        @app.route('/')
        def home():
            return 'Elephas'

        @app.route('/parameters', methods=['GET'])
        def get_parameters():
            self.lock.acquire_read()
            self.pickled_weights = pickle.dumps(self.weights, -1)
            pickled_weights = self.pickled_weights
            self.lock.release()
            return pickled_weights


        @app.route('/update', methods=['POST'])
        def update_parameters():
            delta = pickle.loads(request.data)
            self.lock.acquire_write()
            self.weights = add_params(self.weights,delta)
            self.lock.release()
            return 'Update done'

        print 'Listening to localhost:5000...'
        self.app.run(host='0.0.0.0', debug=True, threaded=True, use_reloader=False)

    def train(self, rdd, nb_epoch=10, batch_size=32, verbose=0, validation_split=0.1, num_workers=8):

        self.start_server()
        #TODO: Incorporate proper usage of Spark master, see deepdist

        rdd = rdd.repartition(num_workers)
        yaml = self.master_network.to_yaml()
        parameters = self.spark_context.broadcast(self.master_network.get_weights())
        train_config = get_train_config(nb_epoch, batch_size, verbose, validation_split)

        worker = SparkWorker(yaml, parameters, train_config)
        results = rdd.mapPartitions(worker.train).collect()

        # TODO: Replace results.first by self.master_network.get_weights()
        #null_element = get_neutral(results.first())
        #new_parameters = divide_by(results.fold(null_element, add_params), num_workers)

        res = get_server_weights()
        self.master_network.set_weights(res)

        self.stop_server()

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
        initial_weights = get_server_weights()
        model.set_weights(initial_weights)

        # TODO: Have to drill into batches
        nb_epoch = self.train_config['nb_epoch']
        self.train_config['nb_epoch'] = 1
        for epoch in range(nb_epoch):
            weights_before_training = model.get_weights()
            model.fit(X_train, y_train, show_accuracy=True, **self.train_config)
            weights_after_training = model.get_weights()

            deltas = subtract_params(weights_after_training, weights_before_training)
            put_deltas_to_server(deltas)
        yield weights_after_training
