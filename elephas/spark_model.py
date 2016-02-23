from __future__ import absolute_import
from __future__ import print_function

import numpy as np
from itertools import tee
import socket
from multiprocessing import Process
import six.moves.cPickle as pickle
from six.moves import range
from flask import Flask, request
try:
    import urllib.request as urllib2
except ImportError:
    import urllib2

from pyspark.mllib.linalg import Matrix, Vector

from keras.models import model_from_yaml, slice_X

from .utils.rwlock import RWLock
from .utils.functional_utils import subtract_params
from .utils.rdd_utils import lp_to_simple_rdd
from .mllib.adapter import to_matrix, from_matrix, to_vector, from_vector
from .optimizers import SGD as default_optimizer


def get_server_weights(master_url='localhost:5000'):
    '''
    Retrieve master weights from parameter server
    '''
    request = urllib2.Request('http://{0}/parameters'.format(master_url),
                              headers={'Content-Type': 'application/elephas'})
    return pickle.loads(urllib2.urlopen(request).read())


def put_deltas_to_server(delta, master_url='localhost:5000'):
    '''
    Update master parameters with deltas from training process
    '''
    request = urllib2.Request('http://{0}/update'.format(master_url),
                              pickle.dumps(delta, -1), headers={'Content-Type': 'application/elephas'})
    return urllib2.urlopen(request).read()


class SparkModel(object):
    '''
    SparkModel is the main abstraction of elephas. Every other model
    should inherit from it.
    '''
    def __init__(self, sc, master_network, optimizer=None, mode='asynchronous', frequency='epoch',
                 num_workers=4, *args, **kwargs):
        self.spark_context = sc
        self._master_network = master_network
        if optimizer is None:
            self.optimizer = default_optimizer()
        else:
            self.optimizer = optimizer
        self.mode = mode
        self.frequency = frequency
        self.num_workers = num_workers
        self.weights = master_network.get_weights()
        self.pickled_weights = None
        self.lock = RWLock()

    @staticmethod
    def determine_master():
        '''
        Get URL of parameter server, running on master
        '''
        master_url = socket.gethostbyname(socket.gethostname()) + ':5000'
        return master_url

    def get_train_config(self, nb_epoch, batch_size,
                         verbose, validation_split):
        '''
        Get configuration of training parameters
        '''
        train_config = {}
        train_config['nb_epoch'] = nb_epoch
        train_config['batch_size'] = batch_size
        train_config['verbose'] = verbose
        train_config['validation_split'] = validation_split
        return train_config

    def get_config(self):
        '''
        Get configuration of model parameters
        '''
        model_config = {}
        model_config['model'] = self.master_network.get_config()
        model_config['optimizer'] = self.optimizer.get_config()
        model_config['mode'] = self.mode
        return model_config

    @property
    def master_network(self):
        ''' Get master network '''
        return self._master_network

    @master_network.setter
    def master_network(self, network):
        ''' Set master network '''
        self._master_network = network

    def start_server(self):
        ''' Start parameter server'''
        self.server = Process(target=self.start_service)
        self.server.start()

    def stop_server(self):
        ''' Terminate parameter server'''
        self.server.terminate()
        self.server.join()

    def start_service(self):
        ''' Define service and run flask app'''
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
            constraints = self.master_network.constraints

            self.weights = self.optimizer.get_updates(self.weights, constraints, delta)
            if self.mode == 'asynchronous':
                self.lock.release()
            return 'Update done'

        self.app.run(host='0.0.0.0', debug=True,
                     threaded=True, use_reloader=False)

    def predict(self, data):
        '''
        Get prediction probabilities for a numpy array of features
        '''
        return self.master_network.predict(data)

    def predict_classes(self, data):
        '''
        Predict classes for a numpy array of features
        '''
        return self.master_network.predict_classes(data)

    def train(self, rdd, nb_epoch=10, batch_size=32,
              verbose=0, validation_split=0.1):
        '''
        Train an elephas model.
        '''
        rdd = rdd.repartition(self.num_workers)
        master_url = self.determine_master()

        if self.mode in ['asynchronous', 'synchronous', 'hogwild']:
            self._train(rdd, nb_epoch, batch_size, verbose, validation_split, master_url)
        else:
            print("""Choose from one of the modes: asynchronous, synchronous or hogwild""")

    def _train(self, rdd, nb_epoch=10, batch_size=32, verbose=0,
               validation_split=0.1, master_url='localhost:5000'):
        '''
        Protected train method to make wrapping of modes easier
        '''
        if self.mode in ['asynchronous', 'hogwild']:
            self.start_server()
        yaml = self.master_network.to_yaml()
        train_config = self.get_train_config(nb_epoch, batch_size,
                                             verbose, validation_split)
        if self.mode in ['asynchronous', 'hogwild']:
            worker = AsynchronousSparkWorker(yaml, train_config, self.frequency, master_url)
            rdd.mapPartitions(worker.train).collect()
            new_parameters = get_server_weights(master_url)
        elif self.mode == 'synchronous':
            init = self.master_network.get_weights()
            parameters = self.spark_context.broadcast(init)
            worker = SparkWorker(yaml, parameters, train_config)
            deltas = rdd.mapPartitions(worker.train).collect()
            new_parameters = self.master_network.get_weights()
            for delta in deltas:
                constraints = self.master_network.constraints
                new_parameters = self.optimizer.get_updates(self.weights, constraints, delta)
        self.master_network.set_weights(new_parameters)
        if self.mode in ['asynchronous', 'hogwild']:
            self.stop_server()


class SparkWorker(object):
    '''
    Synchronous Spark worker. This code will be executed on workers.
    '''
    def __init__(self, yaml, parameters, train_config):
        self.yaml = yaml
        self.parameters = parameters
        self.train_config = train_config

    def train(self, data_iterator):
        '''
        Train a keras model on a worker
        '''
        feature_iterator, label_iterator = tee(data_iterator, 2)
        x_train = np.asarray([x for x, y in feature_iterator])
        y_train = np.asarray([y for x, y in label_iterator])

        model = model_from_yaml(self.yaml)
        model.set_weights(self.parameters.value)
        weights_before_training = model.get_weights()
        if x_train.shape[0] > self.train_config.get('batch_size'):
            model.fit(x_train, y_train, show_accuracy=True, **self.train_config)
        weights_after_training = model.get_weights()
        deltas = subtract_params(weights_before_training, weights_after_training)
        yield deltas


class AsynchronousSparkWorker(object):
    '''
    Asynchronous Spark worker. This code will be executed on workers.
    '''
    def __init__(self, yaml, train_config, frequency, master_url):
        self.yaml = yaml
        self.train_config = train_config
        self.frequency = frequency
        self.master_url = master_url

    def train(self, data_iterator):
        '''
        Train a keras model on a worker and send asynchronous updates
        to parameter server
        '''
        feature_iterator, label_iterator = tee(data_iterator, 2)
        x_train = np.asarray([x for x, y in feature_iterator])
        y_train = np.asarray([y for x, y in label_iterator])

        if x_train.size == 0:
            return
        model = model_from_yaml(self.yaml)

        nb_epoch = self.train_config['nb_epoch']
        batch_size = self.train_config.get('batch_size')
        nb_train_sample = len(x_train[0])
        nb_batch = int(np.ceil(nb_train_sample/float(batch_size)))
        index_array = np.arange(nb_train_sample)
        batches = [(i*batch_size, min(nb_train_sample, (i+1)*batch_size)) for i in range(0, nb_batch)]

        if self.frequency == 'epoch':
            for epoch in range(nb_epoch):
                weights_before_training = get_server_weights(self.master_url)
                model.set_weights(weights_before_training)
                self.train_config['nb_epoch'] = 1
                if x_train.shape[0] > batch_size:
                    model.fit(x_train, y_train, show_accuracy=True, **self.train_config)
                weights_after_training = model.get_weights()
                deltas = subtract_params(weights_before_training, weights_after_training)
                put_deltas_to_server(deltas, self.master_url)
        elif self.frequency == 'batch':
            for epoch in range(nb_epoch):
                if x_train.shape[0] > batch_size:
                    for (batch_start, batch_end) in batches:
                        weights_before_training = get_server_weights(self.master_url)
                        model.set_weights(weights_before_training)
                        batch_ids = index_array[batch_start:batch_end]
                        X = slice_X(x_train, batch_ids)
                        y = slice_X(y_train, batch_ids)
                        model.train_on_batch(X, y)
                        weights_after_training = model.get_weights()
                        deltas = subtract_params(weights_before_training, weights_after_training)
                        put_deltas_to_server(deltas, self.master_url)
        else:
            print('Choose frequency to be either batch or epoch')
        yield []


class SparkMLlibModel(SparkModel):
    '''
    MLlib model takes RDDs of LabeledPoints. Internally we just convert
    back to plain old pair RDDs and continue as in SparkModel
    '''
    def __init__(self, sc, master_network, optimizer=None, mode='asynchronous', frequency='epoch', num_workers=4):
        SparkModel.__init__(self, sc, master_network, optimizer, mode, frequency, num_workers)

    def train(self, labeled_points, nb_epoch=10, batch_size=32, verbose=0, validation_split=0.1,
              categorical=False, nb_classes=None):
        '''
        Train an elephas model on an RDD of LabeledPoints
        '''
        rdd = lp_to_simple_rdd(labeled_points, categorical, nb_classes)
        rdd = rdd.repartition(self.num_workers)
        self._train(rdd, nb_epoch, batch_size, verbose, validation_split)

    def predict(self, mllib_data):
        '''
        Predict probabilities for an RDD of features
        '''
        if isinstance(mllib_data, Matrix):
            return to_matrix(self.master_network.predict(from_matrix(mllib_data)))
        elif isinstance(mllib_data, Vector):
            return to_vector(self.master_network.predict(from_vector(mllib_data)))
        else:
            print('Provide either an MLLib matrix or vector')
