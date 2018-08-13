from __future__ import absolute_import
from __future__ import print_function

import pyspark

from .utils import lp_to_simple_rdd
from .utils import model_to_dict
from .mllib import to_matrix, from_matrix, to_vector, from_vector
from .optimizers import SGD as default_optimizer
from .worker import AsynchronousSparkWorker, SparkWorker
from .parameter import HttpServer, SocketServer
from .parameter import HttpClient, SocketClient


class SparkModel(object):
    """SparkModel is the main abstraction of elephas. Every other model
    should inherit from it.
    """
    # TODO: Eliminate Spark context (only used for first broadcast, can be extracted)
    def __init__(self, master_network, optimizer=None,
                 mode='asynchronous', frequency='epoch',
                 num_workers=4,
                 master_optimizer="sgd",  # TODO: other default
                 master_loss="categorical_crossentropy",
                 master_metrics=None,
                 custom_objects=None,
                 parameter_server='http',
                 *args, **kwargs):

        self._master_network = master_network
        if custom_objects is None:
            custom_objects = {}
        if master_metrics is None:
            master_metrics = ["accuracy"]
        if optimizer is None:
            self.optimizer = default_optimizer()
        else:
            self.optimizer = optimizer
        self.mode = mode
        self.frequency = frequency
        self.num_workers = num_workers
        self.weights = master_network.get_weights()
        self.pickled_weights = None
        self.master_optimizer = master_optimizer
        self.master_loss = master_loss
        self.master_metrics = master_metrics
        self.custom_objects = custom_objects

        # TODO: clients have to be initialized on workers. Only init servers here, clients on workers
        if parameter_server == 'http':
            self.parameter_server = HttpServer(self.master_network, self.optimizer, self.mode)
            self.connector = HttpClient()
        else:
            self.parameter_server = SocketServer(model_to_dict(self.master_network))
            self.connector = SocketClient()

    def get_train_config(self, nb_epoch, batch_size,
                         verbose, validation_split):
        """Get configuration of training parameters
        """
        train_config = {}
        train_config['nb_epoch'] = nb_epoch
        train_config['batch_size'] = batch_size
        train_config['verbose'] = verbose
        train_config['validation_split'] = validation_split
        return train_config

    def get_config(self):
        """Get configuration of model parameters
        """
        model_config = {}
        model_config['model'] = self.master_network.get_config()
        model_config['optimizer'] = self.optimizer.get_config()
        model_config['mode'] = self.mode
        model_config['frequency'] = self.frequency
        model_config['num_workers'] = self.num_workers
        return model_config

    @property
    def master_network(self):
        return self._master_network

    @master_network.setter
    def master_network(self, network):
        self._master_network = network

    def start_server(self):
        self.parameter_server.start()

    def stop_server(self):
        self.parameter_server.stop()

    def predict(self, data):
        """Get prediction probabilities for a numpy array of features
        """
        return self.master_network.predict(data)

    def predict_classes(self, data):
        """ Predict classes for a numpy array of features
        """
        return self.master_network.predict_classes(data)

    def train(self, rdd, nb_epoch=10, batch_size=32,
              verbose=0, validation_split=0.1):
        # TODO: Make dataframe the standard, but support RDDs as well
        """Train an elephas model.
        """
        rdd = rdd.repartition(self.num_workers)

        if self.mode in ['asynchronous', 'synchronous', 'hogwild']:
            self._train(rdd, nb_epoch, batch_size, verbose, validation_split)
        else:
            raise Exception("""Choose from one of the modes: asynchronous, synchronous or hogwild""")

    def _train(self, rdd, nb_epoch=10, batch_size=32, verbose=0,
               validation_split=0.1):
        """Protected train method to make wrapping of modes easier
        """
        self.master_network.compile(optimizer=self.master_optimizer,
                                    loss=self.master_loss,
                                    metrics=self.master_metrics)
        if self.mode in ['asynchronous', 'hogwild']:
            self.start_server()
        yaml = self.master_network.to_yaml()
        train_config = self.get_train_config(nb_epoch, batch_size,
                                             verbose, validation_split)
        if self.mode in ['asynchronous', 'hogwild']:
            worker = AsynchronousSparkWorker(
                yaml, self.connector, train_config, self.frequency,
                self.master_optimizer, self.master_loss, self.master_metrics, self.custom_objects
            )
            rdd.mapPartitions(worker.train).collect()
            new_parameters = self.connector.get_parameters()
        elif self.mode == 'synchronous':
            parameters = self.master_network.get_weights()
            worker = SparkWorker(
                yaml, parameters, train_config,
                self.master_optimizer, self.master_loss, self.master_metrics, self.custom_objects
            )
            deltas = rdd.mapPartitions(worker.train).collect()
            new_parameters = self.master_network.get_weights()
            for delta in deltas:
                constraints = self.master_network.constraints
                new_parameters = self.optimizer.get_updates(self.weights, constraints, delta)
        else:
            raise ValueError("Unsupported mode {}".format(self.mode))
        self.master_network.set_weights(new_parameters)
        if self.mode in ['asynchronous', 'hogwild']:
            self.stop_server()


class SparkMLlibModel(SparkModel):
    """MLlib model takes RDDs of LabeledPoints. Internally we just convert
    back to plain old pair RDDs and continue as in SparkModel
    """
    def __init__(self, sc, master_network, optimizer=None, mode='asynchronous', frequency='epoch', num_workers=4,
                 master_optimizer="adam",
                 master_loss="categorical_crossentropy",
                 master_metrics=None,
                 custom_objects=None):
        SparkModel.__init__(self, sc, master_network, optimizer, mode, frequency, num_workers,
                            master_optimizer=master_optimizer, master_loss=master_loss, master_metrics=master_metrics,
                            custom_objects=custom_objects)

    def train(self, labeled_points, nb_epoch=10, batch_size=32, verbose=0, validation_split=0.1,
              categorical=False, nb_classes=None):
        """Train an elephas model on an RDD of LabeledPoints
        """
        rdd = lp_to_simple_rdd(labeled_points, categorical, nb_classes)
        rdd = rdd.repartition(self.num_workers)
        self._train(rdd, nb_epoch, batch_size, verbose, validation_split)

    def predict(self, mllib_data):
        """Predict probabilities for an RDD of features
        """
        if isinstance(mllib_data, pyspark.mllib.linalg.Matrix):
            return to_matrix(self.master_network.predict(from_matrix(mllib_data)))
        elif isinstance(mllib_data, pyspark.mllib.linalg.Vector):
            return to_vector(self.master_network.predict(from_vector(mllib_data)))
        else:
            raise ValueError('Provide either an MLLib matrix or vector, got {}'.format(mllib_data.__name__))
