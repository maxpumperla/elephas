from pyspark.ml.param.shared import Param, Params


class HasKerasModelConfig(Params):
    """Mandatory field:

    Parameter mixin for Keras model yaml
    """

    def __init__(self):
        super(HasKerasModelConfig, self).__init__()
        self.keras_model_config = Param(
            self, "keras_model_config", "Serialized Keras model as yaml string")

    def set_keras_model_config(self, keras_model_config):
        self._paramMap[self.keras_model_config] = keras_model_config
        return self

    def get_keras_model_config(self):
        return self.getOrDefault(self.keras_model_config)


class HasMode(Params):
    """Parameter mixin for Elephas mode
    """

    def __init__(self):
        super(HasMode, self).__init__()
        self.mode = Param(self, "mode", "Elephas mode")
        self._setDefault(mode='asynchronous')

    def set_mode(self, mode):
        self._paramMap[self.mode] = mode
        return self

    def get_mode(self):
        return self.getOrDefault(self.mode)


class HasFrequency(Params):
    """Parameter mixin for Elephas frequency
    """

    def __init__(self):
        super(HasFrequency, self).__init__()
        self.frequency = Param(self, "frequency", "Elephas frequency")
        self._setDefault(frequency='epoch')

    def set_frequency(self, frequency):
        self._paramMap[self.frequency] = frequency
        return self

    def get_frequency(self):
        return self.getOrDefault(self.frequency)


class HasNumberOfClasses(Params):
    """Mandatory:

    Parameter mixin for number of classes
    """

    def __init__(self):
        super(HasNumberOfClasses, self).__init__()
        self.nb_classes = Param(self, "nb_classes", "number of classes")
        self._setDefault(nb_classes=10)

    def set_nb_classes(self, nb_classes):
        self._paramMap[self.nb_classes] = nb_classes
        return self

    def get_nb_classes(self):
        return self.getOrDefault(self.nb_classes)


class HasCategoricalLabels(Params):
    """Mandatory:

    Parameter mixin for setting categorical features
    """

    def __init__(self):
        super(HasCategoricalLabels, self).__init__()
        self.categorical = Param(
            self, "categorical", "Boolean to indicate if labels are categorical")
        self._setDefault(categorical=True)

    def set_categorical_labels(self, categorical):
        self._paramMap[self.categorical] = categorical
        return self

    def get_categorical_labels(self):
        return self.getOrDefault(self.categorical)


class HasEpochs(Params):
    """Parameter mixin for number of epochs
    """

    def __init__(self):
        super(HasEpochs, self).__init__()
        self.epochs = Param(self, "epochs", "Number of epochs to train")
        self._setDefault(epochs=10)

    def set_epochs(self, epochs):
        self._paramMap[self.epochs] = epochs
        return self

    def get_epochs(self):
        return self.getOrDefault(self.epochs)


class HasBatchSize(Params):
    """Parameter mixin for batch size
    """

    def __init__(self):
        super(HasBatchSize, self).__init__()
        self.batch_size = Param(self, "batch_size", "Batch size")
        self._setDefault(batch_size=32)

    def set_batch_size(self, batch_size):
        self._paramMap[self.batch_size] = batch_size
        return self

    def get_batch_size(self):
        return self.getOrDefault(self.batch_size)


class HasVerbosity(Params):
    """Parameter mixin for output verbosity
    """

    def __init__(self):
        super(HasVerbosity, self).__init__()
        self.verbose = Param(self, "verbose", "Stdout verbosity")
        self._setDefault(verbose=0)

    def set_verbosity(self, verbose):
        self._paramMap[self.verbose] = verbose
        return self

    def get_verbosity(self):
        return self.getOrDefault(self.verbose)


class HasValidationSplit(Params):
    """Parameter mixin for validation split percentage
    """

    def __init__(self):
        super(HasValidationSplit, self).__init__()
        self.validation_split = Param(
            self, "validation_split", "validation split percentage")
        self._setDefault(validation_split=0.1)

    def set_validation_split(self, validation_split):
        self._paramMap[self.validation_split] = validation_split
        return self

    def get_validation_split(self):
        return self.getOrDefault(self.validation_split)


class HasNumberOfWorkers(Params):
    """Parameter mixin for number of workers
    """

    def __init__(self):
        super(HasNumberOfWorkers, self).__init__()
        self.num_workers = Param(self, "num_workers", "number of workers")
        self._setDefault(num_workers=8)

    def set_num_workers(self, num_workers):
        self._paramMap[self.num_workers] = num_workers
        return self

    def get_num_workers(self):
        return self.getOrDefault(self.num_workers)


class HasKerasOptimizerConfig(Params):
    """Parameter mixin for Keras optimizer config
    """

    def __init__(self):
        super(HasKerasOptimizerConfig, self).__init__()
        self.optimizer_config = Param(
            self, "optimizer_config", "Serialized Keras optimizer properties")
        self._setDefault(optimizer_config=None)

    def set_optimizer_config(self, optimizer_config):
        self._paramMap[self.optimizer_config] = optimizer_config
        return self

    def get_optimizer_config(self):
        return self.getOrDefault(self.optimizer_config)


class HasMetrics(Params):
    """Parameter mixin for Keras metrics
    """

    def __init__(self):
        super(HasMetrics, self).__init__()
        self.metrics = Param(self, "metrics", "Keras metrics")
        self._setDefault(metrics=['acc'])

    def set_metrics(self, metrics):
        self._paramMap[self.metrics] = metrics
        return self

    def get_metrics(self):
        return self.getOrDefault(self.metrics)


class HasLoss(Params):
    """Parameter mixin for Keras metrics
    """

    def __init__(self):
        super(HasLoss, self).__init__()
        self.loss = Param(self, "loss", "Keras loss")

    def set_loss(self, loss):
        self._paramMap[self.loss] = loss
        return self

    def get_loss(self):
        return self.getOrDefault(self.loss)


class HasCustomObjects(Params):
    def __init__(self):
        super(HasCustomObjects, self).__init__()
        self.custom_objects = Param(self, "custom_objects",
                                    "Custom objects (i.e; custom activation functions, custom layers) in a model")
        self._setDefault(custom_objects={})

    def set_custom_objects(self, custom_objects):
        self._paramMap[self.custom_objects] = custom_objects
        return self

    def get_custom_objects(self):
        return self.getOrDefault(self.custom_objects)
