from pyspark.ml.param.shared import Param, Params


class HasKerasModelConfig(Params):
    '''
    Mandatory:
    
    Parameter mixin for Keras model yaml
    '''
    def __init__(self):
        super(HasKerasModelConfig, self).__init__()
        self.keras_model = Param(self, "keras_model", "Serialized Keras model as yaml string")

    def set_keras_model(self, model):
        self._paramMap[self.keras_model] = model
        return self

    def get_keras_model(self):
        return self.getOrDefault(self.keras_model)

class HasNumberOfClasses(Param):
    '''
    Mandatory:

    Parameter mixin for number of classes
    '''
    def __init__(self):
        super(HasNumberOfClasses, self).__init__()
        self.nb_classes = Param(self, "nb_classes", "number of classes")

    def set_nb_classes(self, nb_classes):
        self._paramMap[self.nb_classes] = nb_classes
        return self

    def get_nb_classes(self):
        return self.getOrDefault(self.nb_classes)

class HasCategoricalFeatures(Param):
    '''
    Mandatory:

    Parameter mixin for setting categorical features
    '''
    def __init__(self):
        super(HasCategoricalFeatures, self).__init__()
        self.categorical = Param(self, "categorical", "Boolean to indicate if labels are categorical")
        #self._setDefault(categorical=False)

    def set_categorical_features(self, categorical):
        self._paramMap[self.categorical] = categorical
        return self

    def get_categorical_features(self):
        return self.getOrDefault(self.categorical)


class HasEpochs(Param):
    '''
    Parameter mixin for number of epochs
    '''
    def __init__(self):
        super(HasEpochs, self).__init__()
        self.epochs = Param(self, "epochs", "Number of epochs to train")
        self._setDefault(epochs=10)

    def set_num_epochs(self, epochs):
        self._paramMap[self.epochs] = epochs
        return self

    def get_num_epochs(self):
        return self.getOrDefault(self.epochs)


class HasBatchSize(Param):
    '''
    Parameter mixin for batch size
    '''
    def __init__(self):
        super(HasBatchSize, self).__init__()
        self.batch_size = Param(self, "batch_size", "Batch size")
        self._setDefault(batch_size=32)

    def set_batch_size(self, batch_size):
        self._paramMap[self.batch_size] = batch_size
        return self

    def get_batch_size(self):
        return self.getOrDefault(self.batch_size)


class HasVerbosity(Param):
    '''
    Parameter mixin for output verbosity
    '''
    def __init__(self):
        super(HasVerbosity, self).__init__()
        self.verbose = Param(self, "verbosity", "Stdout verbosity")
        self._setDefault(verbose=0)

    def set_verbosity(self, verbose):
        self._paramMap[self.verbose] = verbose
        return self

    def get_verbosity(self):
        return self.getOrDefault(self.verbose)


class HasValidationSplit(Param):
    '''
    Parameter mixin for validation split percentage
    '''
    def __init__(self):
        super(HasValidationSplit, self).__init__()
        self.validation_split = Param(self, "validation_split", "validation split percentage")
        self._setDefault(validation_split=0.1)

    def set_validation_split(self, validation_split):
        self._paramMap[self.validation_split] = validation_split
        return self

    def get_verbosity(self):
        return self.getOrDefault(self.validation_split)


class HasNumberOfWorkers(Param):
    '''
    Parameter mixin for number of workers
    '''
    def __init__(self):
        super(HasNumberOfWorkers, self).__init__()
        self.num_workers = Param(self, "num_workers", "number of workers")
        self._setDefault(num_workers=8)

    def set_num_workers(self, num_workers):
        self._paramMap[self.num_workers] = num_workers
        return self

    def get_num_workers(self):
        return self.getOrDefault(self.num_workers)
