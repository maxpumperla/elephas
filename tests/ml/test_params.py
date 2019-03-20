from elephas.ml.params import *


def test_has_keras_model_config():
    param = HasKerasModelConfig()
    conf = {"foo": "bar"}
    param.set_keras_model_config(conf)
    assert conf == param.get_keras_model_config()


def test_has_optimizer_config():
    param = HasKerasOptimizerConfig()
    conf = {"foo": "bar"}
    param.set_optimizer_config(conf)
    assert conf == param.get_optimizer_config()


def test_has_mode():
    param = HasMode()
    assert param.get_mode() == "asynchronous"
    mode = "foobar"
    param.set_mode(mode)
    assert param.get_mode() == mode


def test_has_frequency():
    param = HasFrequency()
    assert param.get_frequency() == "epoch"
    freq = "foobar"
    param.set_frequency(freq)
    assert param.get_frequency() == freq


def test_has_number_of_classes():
    param = HasNumberOfClasses()
    assert param.get_nb_classes() == 10
    classes = 42
    param.set_nb_classes(classes)
    assert param.get_nb_classes() == classes


def test_has_categorical_labels():
    param = HasCategoricalLabels()
    assert param.get_categorical_labels()
    has_labels = False
    param.set_categorical_labels(has_labels)
    assert param.get_categorical_labels() == has_labels


def test_has_epochs():
    param = HasEpochs()
    assert param.get_epochs() == 10
    epochs = 42
    param.set_epochs(epochs)
    assert param.get_epochs() == epochs


def test_has_batch_size():
    param = HasBatchSize()
    assert param.get_batch_size() == 32
    bs = 42
    param.set_batch_size(bs)
    assert param.get_batch_size() == bs


def test_has_verbosity():
    param = HasVerbosity()
    assert param.get_verbosity() == 0
    verbosity = 2
    param.set_verbosity(verbosity)
    assert param.get_verbosity() == verbosity


def test_has_validation_split():
    param = HasValidationSplit()
    assert param.get_validation_split() == 0.1
    split = 0.5
    param.set_validation_split(split)
    assert param.get_validation_split() == split


def test_has_number_of_workers():
    param = HasNumberOfWorkers()
    assert param.get_num_workers() == 8
    workers = 12
    param.set_num_workers(workers)
    assert param.get_num_workers() == workers
