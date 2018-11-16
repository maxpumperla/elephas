from __future__ import absolute_import
from __future__ import print_function

import pyspark
import h5py
import json
from keras.optimizers import serialize as serialize_optimizer
from keras.models import load_model

from .utils import lp_to_simple_rdd
from .utils import model_to_dict
from .mllib import to_matrix, from_matrix, to_vector, from_vector
from .optimizers import SGD
from .worker import AsynchronousSparkWorker, SparkWorker
from .parameter import HttpServer, SocketServer
from .parameter import HttpClient, SocketClient


class SparkModel(object):

    def __init__(self, model, mode='asynchronous', frequency='epoch',  parameter_server_mode='http', num_workers=None,
                 elephas_optimizer=None, custom_objects=None, batch_size=32, *args, **kwargs):
        """SparkModel

        Base class for distributed training on RDDs. Spark model takes a Keras
        model as master network, an optimization scheme, a parallelisation mode
        and an averaging frequency.

        :param model: Compiled Keras model
        :param mode: String, choose from `asynchronous`, `synchronous` and `hogwild`
        :param frequency: String, either `epoch` or `batch`
        :param parameter_server_mode: String, either `http` or `socket`
        :param num_workers: int, number of workers used for training (defaults to None)
        :param elephas_optimizer: Elephas optimizer
        :param custom_objects: Keras custom objects
        """

        self._master_network = model
        if not hasattr(model, "loss"):
            raise Exception("Compile your Keras model before initializing an Elephas model with it")
        metrics = model.metrics
        loss = model.loss
        optimizer = serialize_optimizer(model.optimizer)

        if custom_objects is None:
            custom_objects = {}
        if metrics is None:
            metrics = ["accuracy"]
        if elephas_optimizer is None:
            self.optimizer = SGD()
        else:
            self.optimizer = elephas_optimizer
        self.mode = mode
        self.frequency = frequency
        self.num_workers = num_workers
        self.weights = self._master_network.get_weights()
        self.pickled_weights = None
        self.master_optimizer = optimizer
        self.master_loss = loss
        self.master_metrics = metrics
        self.custom_objects = custom_objects
        self.parameter_server_mode = parameter_server_mode
        self.batch_size = batch_size
        self.kwargs = kwargs

        self.serialized_model = model_to_dict(self.master_network)
        # TODO only set this for async/hogwild mode
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
    def get_train_config(epochs, batch_size, verbose, validation_split):
        return {'epochs': epochs,
                'batch_size': batch_size,
                'verbose': verbose,
                'validation_split': validation_split}

    def get_config(self):
        base_config = {
            'parameter_server_mode': self.parameter_server_mode,
            'elephas_optimizer': self.optimizer.get_config(),
            'mode': self.mode,
            'frequency': self.frequency,
            'num_workers': self.num_workers,
            'batch_size': self.batch_size}
        config = base_config.copy()
        config.update(self.kwargs)
        return config

    def save(self, file_name):
        model = self.master_network
        model.save(file_name)
        f = h5py.File(file_name, mode='a')

        f.attrs['distributed_config'] = json.dumps({
            'class_name': self.__class__.__name__,
            'config': self.get_config()
        }).encode('utf8')

        f.flush()
        f.close()

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
        mode = self.parameter_server_mode
        freq = self.frequency
        optimizer = self.master_optimizer
        loss = self.master_loss
        metrics = self.master_metrics
        custom = self.custom_objects

        yaml = self.master_network.to_yaml()
        init = self.master_network.get_weights()
        parameters = rdd.context.broadcast(init)

        if self.mode in ['asynchronous', 'hogwild']:
            worker = AsynchronousSparkWorker(yaml, parameters, mode, train_config, freq, optimizer, loss, metrics, custom)
            rdd.mapPartitions(worker.train).collect()
            new_parameters = self.client.get_parameters()
        elif self.mode == 'synchronous':

            worker = SparkWorker(yaml, parameters, train_config, optimizer, loss, metrics, custom)
            deltas = rdd.mapPartitions(worker.train).collect()
            new_parameters = self.master_network.get_weights()
            for delta in deltas:
                base_constraint = lambda a: a
                constraints = [base_constraint for _ in self.weights]
                new_parameters = self.optimizer.get_updates(self.weights, constraints, delta)
        else:
            raise ValueError("Unsupported mode {}".format(self.mode))
        self.master_network.set_weights(new_parameters)
        if self.mode in ['asynchronous', 'hogwild']:
            self.stop_server()


def load_spark_model(file_name):
    model = load_model(file_name)
    f = h5py.File(file_name, mode='r')

    elephas_conf = json.loads(f.attrs.get('distributed_config'))
    class_name = elephas_conf.get('class_name')
    config = elephas_conf.get('config')
    if class_name == "SparkModel":
        return SparkModel(model=model, **config)
    elif class_name == "SparkMLlibModel":
        return SparkMLlibModel(model=model, **config)


class SparkMLlibModel(SparkModel):

    def __init__(self, model, mode='asynchronous', frequency='epoch', parameter_server_mode='http',
                 num_workers=4, elephas_optimizer=None, custom_objects=None, batch_size=32, *args, **kwargs):
        """SparkMLlibModel

        The Spark MLlib model takes RDDs of LabeledPoints for training.

        :param model: Compiled Keras model
        :param mode: String, choose from `asynchronous`, `synchronous` and `hogwild`
        :param frequency: String, either `epoch` or `batch`
        :param parameter_server_mode: String, either `http` or `socket`
        :param num_workers: int, number of workers used for training (defaults to None)
        :param elephas_optimizer: Elephas optimizer
        :param custom_objects: Keras custom objects
        """
        SparkModel.__init__(self, model=model, mode=mode, frequency=frequency,
                            parameter_server_mode=parameter_server_mode, num_workers=num_workers,
                            elephas_optimizer=elephas_optimizer, custom_objects=custom_objects,
                            batch_size=batch_size, *args, **kwargs)

    def fit(self, labeled_points, epochs=10, batch_size=32, verbose=0, validation_split=0.1,
              categorical=False, nb_classes=None):
        """Train an elephas model on an RDD of LabeledPoints
        """
        rdd = lp_to_simple_rdd(labeled_points, categorical, nb_classes)
        rdd = rdd.repartition(self.num_workers)
        self._fit(rdd=rdd, epochs=epochs, batch_size=batch_size, verbose=verbose, validation_split=validation_split)

    def predict(self, mllib_data):
        """Predict probabilities for an RDD of features
        """
        if isinstance(mllib_data, pyspark.mllib.linalg.Matrix):
            return to_matrix(self.master_network.predict(from_matrix(mllib_data)))
        elif isinstance(mllib_data, pyspark.mllib.linalg.Vector):
            return to_vector(self.master_network.predict(from_vector(mllib_data)))
        else:
            raise ValueError('Provide either an MLLib matrix or vector, got {}'.format(mllib_data.__name__))

