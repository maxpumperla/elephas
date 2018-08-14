from __future__ import absolute_import
from __future__ import print_function

import pyspark

from .utils import lp_to_simple_rdd
from .utils import model_to_dict
from .mllib import to_matrix, from_matrix, to_vector, from_vector
from .optimizers import SGD
from .worker import AsynchronousSparkWorker, SparkWorker
from .parameter import HttpServer, SocketServer
from .parameter import HttpClient, SocketClient


class SparkModel(object):

    def __init__(self, master_network, optimizer=None,
                 mode='asynchronous', frequency='epoch',
                 num_workers=None,
                 master_optimizer="sgd",
                 master_loss="categorical_crossentropy",
                 master_metrics=None,
                 custom_objects=None,
                 parameter_server_mode='http',
                 *args, **kwargs):
        """SparkModel

        Base class for distributed training on RDDs. Spark model takes a Keras
        model as master network, an optimization scheme, a parallelisation mode
        and an averaging frequency.

        :param master_network: Keras model (not compiled)
        :param optimizer: Elephas optimizer
        :param mode: String, choose from `asynchronous`, `synchronous` and `hogwild`
        :param frequency: String, either `epoch` or `batch`
        :param num_workers: int, number of workers used for training (defaults to None)
        :param master_optimizer: Keras optimizer for master network
        :param master_loss: Keras loss function for master network
        :param master_metrics: Keras metrics used for master network
        :param custom_objects: Keras custom objects
        :param parameter_server_mode: String, either `http` or `socket`
        """

        self._master_network = master_network
        if custom_objects is None:
            custom_objects = {}
        if master_metrics is None:
            master_metrics = ["accuracy"]
        if optimizer is None:
            self.optimizer = SGD()
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
        self.parameter_server_mode = parameter_server_mode

        self.serialized_model = model_to_dict(self.master_network)
        if self.parameter_server_mode == 'http':
            self.parameter_server = HttpServer(self.serialized_model, self.optimizer, self.mode)
            self.client = HttpClient()
        elif self.parameter_server_mode == 'socket':
            self.parameter_server = SocketServer(self.serialized_model)
            self.client = SocketClient()
        else:
            raise ValueError("Parameter server mode has to be either `http` or `socket`, "
                             "got {}".format(self.parameter_server_mode))

    @staticmethod
    def get_train_config(nb_epoch, batch_size,
                         verbose, validation_split):
        """Get configuration of training parameters
        """
        train_config = {'nb_epoch': nb_epoch,
                        'batch_size': batch_size,
                        'verbose': verbose,
                        'validation_split': validation_split}
        return train_config

    def get_config(self):
        """Get configuration of model parameters
        """
        model_config = {'model': self.master_network.get_config(),
                        'optimizer': self.optimizer.get_config(),
                        'mode': self.mode,
                        'frequency': self.frequency,
                        'num_workers': self.num_workers}
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

    def fit(self, rdd, epochs=10, batch_size=32,
            verbose=0, validation_split=0.1):
        """
        Train an elephas model on an RDD. The Keras model configuration as specified
        in the elephas model is sent to Spark workers, abd each worker will be trained
        on their data partition.

        :param rdd: RDD with features and labels
        :param epochs: number of epochs used for training
        :param batch_size: batch size used for training
        :param verbose: logging verbosity level (0, 1 or 2)
        :param validation_split: percentage of data set aside for validation
        """
        if self.num_workers:
            rdd = rdd.repartition(self.num_workers)

        if self.mode in ['asynchronous', 'synchronous', 'hogwild']:
            self._fit(rdd, epochs, batch_size, verbose, validation_split)
        else:
            raise ValueError("Choose from one of the modes: asynchronous, synchronous or hogwild")

    def _fit(self, rdd, epochs, batch_size, verbose, validation_split):
        """Protected train method to make wrapping of modes easier
        """
        self.master_network.compile(optimizer=self.master_optimizer,
                                    loss=self.master_loss,
                                    metrics=self.master_metrics)
        if self.mode in ['asynchronous', 'hogwild']:
            self.start_server()
        train_config = self.get_train_config(epochs, batch_size, verbose, validation_split)

        if self.mode in ['asynchronous', 'hogwild']:
            worker = AsynchronousSparkWorker(self.parameter_server_mode, train_config, self.frequency,
                self.master_optimizer, self.master_loss, self.master_metrics, self.custom_objects)
            rdd.mapPartitions(worker.train).collect()
            new_parameters = self.client.get_parameters()
        elif self.mode == 'synchronous':
            worker = SparkWorker(self.serialized_model, train_config, self.master_optimizer, self.master_loss,
                                 self.master_metrics, self.custom_objects)
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

    def __init__(self, master_network, optimizer=None, mode='asynchronous', frequency='epoch', num_workers=4,
                 master_optimizer="adam", master_loss="categorical_crossentropy",
                 master_metrics=None, custom_objects=None, parameter_server_mode='http',
                 *args, **kwargs):
        """SparkMLlibModel

        The Spark MLlib model takes RDDs of LabeledPoints for training.

        :param master_network: Keras model (not compiled)
        :param optimizer: Elephas optimizer
        :param mode: String, choose from `asynchronous`, `synchronous` and `hogwild`
        :param frequency: String, either `epoch` or `batch`
        :param num_workers: int, number of workers used for training (defaults to None)
        :param master_optimizer: Keras optimizer for master network
        :param master_loss: Keras loss function for master network
        :param master_metrics: Keras metrics used for master network
        :param custom_objects: Keras custom objects
        :param parameter_server_mode: String, either `http` or `socket
        """
        SparkModel.__init__(self, master_network=master_network, optimizer=optimizer, mode=mode, frequency=frequency,
                            num_workers=num_workers, master_optimizer=master_optimizer, master_loss=master_loss,
                            master_metrics=master_metrics, custom_objects=custom_objects,
                            parameter_server_mode=parameter_server_mode, *args, **kwargs)

    def train(self, labeled_points, nb_epoch=10, batch_size=32, verbose=0, validation_split=0.1,
              categorical=False, nb_classes=None):
        """Train an elephas model on an RDD of LabeledPoints
        """
        rdd = lp_to_simple_rdd(labeled_points, categorical, nb_classes)
        rdd = rdd.repartition(self.num_workers)
        self._fit(rdd, nb_epoch, batch_size, verbose, validation_split)

    def predict(self, mllib_data):
        """Predict probabilities for an RDD of features
        """
        if isinstance(mllib_data, pyspark.mllib.linalg.Matrix):
            return to_matrix(self.master_network.predict(from_matrix(mllib_data)))
        elif isinstance(mllib_data, pyspark.mllib.linalg.Vector):
            return to_vector(self.master_network.predict(from_vector(mllib_data)))
        else:
            raise ValueError('Provide either an MLLib matrix or vector, got {}'.format(mllib_data.__name__))
