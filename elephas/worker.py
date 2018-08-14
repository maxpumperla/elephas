import numpy as np
from itertools import tee

from .utils.serialization import dict_to_model
from .utils import subtract_params
from .parameter import SocketClient, HttpClient


class SparkWorker(object):
    """Synchronous Spark worker. This code will be executed on workers.
    """
    def __init__(self, serialized_model, train_config, master_optimizer,
                 master_loss, master_metrics, custom_objects):
        # TODO handle custom_objects
        self.model = dict_to_model(serialized_model)
        self.train_config = train_config
        self.master_optimizer = master_optimizer
        self.master_loss = master_loss
        self.master_metrics = master_metrics

    def train(self, data_iterator):
        """Train a keras model on a worker
        """
        feature_iterator, label_iterator = tee(data_iterator, 2)
        x_train = np.asarray([x for x, y in feature_iterator])
        y_train = np.asarray([y for x, y in label_iterator])

        self.model.compile(optimizer=self.master_optimizer, loss=self.master_loss, metrics=self.master_metrics)
        weights_before_training = self.model.get_weights()
        if x_train.shape[0] > self.train_config.get('batch_size'):
            self.model.fit(x_train, y_train, **self.train_config)
        weights_after_training = self.model.get_weights()
        deltas = subtract_params(weights_before_training, weights_after_training)
        yield deltas


class AsynchronousSparkWorker(object):
    """Asynchronous Spark worker. This code will be executed on workers.
    """
    def __init__(self, serialized_model, parameter_server_mode, train_config, frequency,
                 master_optimizer, master_loss, master_metrics, custom_objects):
        # TODO handle custom_objects
        self.model = dict_to_model(serialized_model)
        if parameter_server_mode == 'http':
            self.client = HttpClient()
        elif parameter_server_mode == 'socket':
            self.client = SocketClient()
        else:
            raise ValueError("Parameter server mode has to be either `http` or `socket`, "
                             "got {}".format(parameter_server_mode))

        self.client = parameter_server_mode
        self.train_config = train_config
        self.frequency = frequency
        self.master_optimizer = master_optimizer
        self.master_loss = master_loss
        self.master_metrics = master_metrics

    def train(self, data_iterator):
        """Train a keras model on a worker and send asynchronous updates
        to parameter server
        """
        feature_iterator, label_iterator = tee(data_iterator, 2)
        x_train = np.asarray([x for x, y in feature_iterator])
        y_train = np.asarray([y for x, y in label_iterator])

        if x_train.size == 0:
            return

        self.model.compile(optimizer=self.master_optimizer, loss=self.master_loss, metrics=self.master_metrics)

        nb_epoch = self.train_config['nb_epoch']
        batch_size = self.train_config.get('batch_size')
        nb_train_sample = x_train.shape[0]
        nb_batch = int(np.ceil(nb_train_sample / float(batch_size)))
        index_array = np.arange(nb_train_sample)
        batches = [
            (i * batch_size, min(nb_train_sample, (i + 1) * batch_size))
            for i in range(0, nb_batch)
        ]

        if self.frequency == 'epoch':
            for epoch in range(nb_epoch):
                weights_before_training = self.client.get_parameters()
                self.model.set_weights(weights_before_training)
                self.train_config['nb_epoch'] = 1
                if x_train.shape[0] > batch_size:
                    self.model.fit(x_train, y_train, **self.train_config)
                weights_after_training = self.model.get_weights()
                deltas = subtract_params(weights_before_training, weights_after_training)
                self.client.update_parameters(deltas)
        elif self.frequency == 'batch':
            from keras.engine.training import slice_X
            for epoch in range(nb_epoch):
                if x_train.shape[0] > batch_size:
                    for (batch_start, batch_end) in batches:
                        weights_before_training = self.client.get_parameters()
                        self.model.set_weights(weights_before_training)
                        batch_ids = index_array[batch_start:batch_end]
                        X = slice_X(x_train, batch_ids)
                        y = slice_X(y_train, batch_ids)
                        self.model.train_on_batch(X, y)
                        weights_after_training = self.model.get_weights()
                        deltas = subtract_params(weights_before_training, weights_after_training)
                        self.client.update_parameters(deltas)
        else:
            raise ValueError('frequency parameter can be `epoch` or `batch, got {}'.format(self.frequency))
        yield []
