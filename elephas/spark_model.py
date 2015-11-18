from __future__ import absolute_import

import numpy as np
import sys
from itertools import tee
from keras.models import Sequential, model_from_yaml, slice_X
import theano
import theano.tensor as T
from pyspark.rdd import RDD

from threading import Thread
from multiprocessing import Process
import six.moves.cPickle as pickle
from six.moves import range
from flask import Flask, request
import urllib2
import urlparse

from .utils.rwlock import RWLock
from .utils.functional_utils import add_params, subtract_params, get_neutral
from .utils.rdd_utils import lp_to_simple_rdd
from .mllib.adapter import to_matrix, from_matrix, to_vector, from_vector


def get_server_weights(master='localhost:5000'):
    request = urllib2.Request('http://%s/parameters' % master,
        headers={'Content-Type': 'application/elephas'})
    return pickle.loads(urllib2.urlopen(request).read())

def put_deltas_to_server(delta, master='localhost:5000'):
    request = urllib2.Request('http://%s/update' % master, pickle.dumps(delta, -1),
        headers={'Content-Type': 'application/elephas'})
    return urllib2.urlopen(request).read()


class SparkModel(object):
    def __init__(self, sc, master_network, optimizer=None, mode='asynchronous', frequency='epoch', num_workers=4,  *args, **kwargs):
        self.spark_context = sc
        self.master_network = master_network
        if optimizer == None:
            self.optimizer = master_network.optimizer
        else:
            self.optimizer = optimizer
        self.mode = mode
        self.frequency = frequency
        self.num_workers = num_workers

        self.weights = master_network.get_weights()
        self.pickled_weights = None
        self.lock = RWLock()

    def get_train_config(self, nb_epoch, batch_size, verbose, validation_split):
        train_config = {}
        train_config['nb_epoch'] = nb_epoch
        train_config['batch_size'] = batch_size
        train_config['verbose'] = verbose
        train_config['validation_split'] = validation_split
        return train_config

    def get_config(self):
        model_config = {}
        model_config['model'] = self.master_network.get_config()
        model_config['optimizer'] = self.optimizer.get_config()
        model_config['mode'] = self.mode

    def get_network(self):
        return self.master_network

    def set_network(self, network):
        self.master_network = network

    def start_server(self):
        print("Starting parameter server...")
        self.server = Process(target=self.start_service)
        self.server.start()

    def stop_server(self):
        print("Terminating parameter server...")
        self.server.terminate()
        self.server.join()

    def start_service(self):

        app = Flask(__name__)
        self.app = app

        @app.route('/')
        def home():
            return 'Elephas'

        @app.route('/parameters', methods=['GET'])
        def get_parameters():
            if self.mode == 'asynchronous':
                self.lock.acquire_read()
            self.pickled_weights = pickle.dumps(self.weights, -1)
            pickled_weights = self.pickled_weights
            if self.mode == 'asynchronous':
                self.lock.release()
            return pickled_weights

        @app.route('/update', methods=['POST'])
        def update_parameters():
            delta = pickle.loads(request.data)
            if self.mode == 'asynchronous':
                self.lock.acquire_write()
            self.weights = self.optimizer.get_updates(self.weights, self.master_network.constraints, delta)
            if self.mode == 'asynchronous':
                self.lock.release()
            return 'Update done'

        print 'Listening to localhost:5000...'
        self.app.run(host='0.0.0.0', debug=True, threaded=True, use_reloader=False)

    def predict(self,data):
        return self.master_network.predict(data)

    def train(self, rdd, nb_epoch=10, batch_size=32, verbose=0, validation_split=0.1):

        rdd = rdd.repartition(self.num_workers)
        if self.mode in ['asynchronous', 'synchronous', 'hogwild']:
            self._train(rdd, nb_epoch, batch_size, verbose, validation_split)
        else:
            print 'Choose from one of the three available modes: asynchronous, synchronous or hogwild'

    def _train(self, rdd, nb_epoch=10, batch_size=32, verbose=0, validation_split=0.1):
        if self.mode in ['asynchronous', 'hogwild']:
            self.start_server()

        init = self.master_network.get_weights()

        yaml = self.master_network.to_yaml()
        train_config = self.get_train_config(nb_epoch, batch_size, verbose, validation_split)

        if self.mode in ['asynchronous', 'hogwild']:
            worker = AsynchronousSparkWorker(yaml, train_config, self.frequency)
            resultsi = rdd.mapPartitions(worker.train).collect()
            new_parameters = get_server_weights()
        elif self.mode == 'synchronous':
            parameters = self.spark_context.broadcast(self.master_network.get_weights())
            worker = SparkWorker(yaml, parameters, train_config)
            deltas = rdd.mapPartitions(worker.train).collect()
            new_parameters = self.master_network.get_weights()
            for delta in deltas:
                new_parameters = self.optimizer.get_updates(self.weights, self.master_network.constraints, delta)
        self.master_network.set_weights(new_parameters)

        if self.mode in ['asynchronous', 'hogwild']:
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
        model.set_weights(self.parameters.value)
        weights_before_training = model.get_weights()
        if X_train.shape[0] > self.train_config.get('batch_size'):
            model.fit(X_train, y_train, show_accuracy=True, **self.train_config)
        weights_after_training = model.get_weights()
        deltas = subtract_params(weights_before_training, weights_after_training)
        yield deltas


class AsynchronousSparkWorker(object):
    def __init__(self, yaml, train_config, frequency):
        self.yaml = yaml
        self.train_config = train_config
        self.frequency = frequency

    def train(self, data_iterator):
        feature_iterator, label_iterator = tee(data_iterator,2)
        X_train = np.asarray([x for x,y in feature_iterator])
        y_train = np.asarray([y for x,y in label_iterator])

        model = model_from_yaml(self.yaml)

        nb_epoch = self.train_config['nb_epoch']
        batch_size = self.train_config.get('batch_size')
        nb_train_sample = len(X_train[0])
        nb_batch = int(np.ceil(nb_train_sample/float(batch_size)))
        index_array = np.arange(nb_train_sample)
        batches = [(i*batch_size, min(nb_train_sample, (i+1)*batch_size)) for i in range(0, nb_batch)]

        if self.frequency == 'epoch':
            for epoch in range(nb_epoch):
                weights_before_training = get_server_weights()
                model.set_weights(weights_before_training)
                self.train_config['nb_epoch'] = 1
                if X_train.shape[0] > batch_size:
                    model.fit(X_train, y_train, show_accuracy=True, **self.train_config)
                weights_after_training = model.get_weights()
                deltas = subtract_params(weights_before_training, weights_after_training)
                put_deltas_to_server(deltas)
        elif self.frequency == 'batch':
            for epoch in range(nb_epoch):
                if X_train.shape[0] > batch_size:
                    for (batch_start, batch_end) in batches:
                        weights_before_training = get_server_weights()
                        model.set_weights(weights_before_training)
                        batch_ids = index_array[batch_start:batch_end]
                        X = slice_X(X_train, batch_ids)
                        y = slice_X(y_train, batch_ids)
                        model.train_on_batch(X,y)
                        weights_after_training = model.get_weights()
                        deltas = subtract_params(weights_before_training, weights_after_training)
                        put_deltas_to_server(deltas)
        else: 
            print 'Choose frequency to be either batch or epoch'
        yield []


class SparkMLlibModel(SparkModel):
    def train(self, labeled_points, nb_epoch=10, batch_size=32, verbose=0, validation_split=0.1, categorical=False, nb_classes=None):
        rdd = lp_to_simple_rdd(labeled_points, categorical, nb_classes)
        rdd = rdd.repartition(self.num_workers)
        self._train(rdd, nb_epoch, batch_size, verbose, validation_split)

    def predict(mllib_data):
        if isinstance(mllib_data, Matrix):
            return to_matrix(self.master_network.predict(from_matrix(mllib_data)))
        elif isinstance(mllib_data, Vector):
            return to_vector(self.master_network.predict(from_vector(mllib_data)))
        else:
            print 'Provide either an MLLib matrix or vector'
