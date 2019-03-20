from .spark_model import SparkModel
try:
    from elephas.java import java_classes, adapter
except:
    raise Exception("Warning: java classes couldn't be loaded.")


class ParameterAveragingModel(SparkModel):
    def __init__(self, java_spark_context, model, num_workers, batch_size, averaging_frequency=5,
                 num_batches_prefetch=0, collect_stats=False, save_file='temp.h5', *args, **kwargs):
        """ParameterAveragingModel

         :param java_spark_context JavaSparkContext, initialized through pyjnius
         :param model: compiled Keras model
         :param num_workers: number of Spark workers/executors.
         :param batch_size: batch size used for model training
         :param averaging_frequency: int, after how many batches of training averaging takes place
         :param num_batches_prefetch: int, how many batches to pre-fetch, deactivated if 0.
         :param collect_stats: boolean, if statistics get collected during training
         :param save_file: where to store elephas model temporarily.
         """
        SparkModel.__init__(self, model=model, batch_size=batch_size, mode='synchronous',
                            averaging_frequency=averaging_frequency, num_batches_prefetch=num_batches_prefetch,
                            num_workers=num_workers, collect_stats=collect_stats, *args, **kwargs)

        self.save(save_file)
        model_file = java_classes.File(save_file)
        keras_model_type = model.__class__.__name__
        self.java_spark_model = dl4j_import(
            java_spark_context, model_file, keras_model_type)

    def fit_rdd(self, data_set_rdd, epochs):
        for _ in range(epochs):
            self.java_spark_model.fit(data_set_rdd)

    def get_keras_model(self):
        model = self.master_network
        java_model = self.java_spark_model.getNetwork()
        weights = adapter.retrieve_keras_weights(java_model)
        model.set_weights(weights)
        return model


class ParameterSharingModel(SparkModel):
    def __init__(self, java_spark_context, model, num_workers, batch_size,
                 shake_frequency=0, min_threshold=1e-5, update_threshold=1e-3, workers_per_node=-1,
                 num_batches_prefetch=0, step_delay=50, step_trigger=0.05, threshold_step=1e-5,
                 collect_stats=False, save_file='temp.h5', *args, **kwargs):
        """ParameterSharingModel

        :param java_spark_context JavaSparkContext, initialized through pyjnius
        :param model: compiled Keras model
        :param num_workers: number of Spark workers/executors.
        :param batch_size: batch size used for model training
        :param shake_frequency:
        :param min_threshold:
        :param update_threshold:
        :param workers_per_node:
        :param num_batches_prefetch:
        :param step_delay:
        :param step_trigger:
        :param threshold_step:
        :param collect_stats:
        :param save_file:
        :param args:
        :param kwargs:
        """
        SparkModel.__init__(self, model=model, num_workers=num_workers, batch_size=batch_size, mode='asynchronous',
                            shake_frequency=shake_frequency, min_threshold=min_threshold,
                            update_threshold=update_threshold, workers_per_node=workers_per_node,
                            num_batches_prefetch=num_batches_prefetch, step_delay=step_delay, step_trigger=step_trigger,
                            threshold_step=threshold_step, collect_stats=collect_stats, *args, **kwargs)

        self.save(save_file)
        model_file = java_classes.File(save_file)
        keras_model_type = model.__class__.__name__
        self.java_spark_model = dl4j_import(
            java_spark_context, model_file, keras_model_type)

    def fit_rdd(self, data_set_rdd, epochs):
        for _ in range(epochs):
            self.java_spark_model.fit(data_set_rdd)

    def get_keras_model(self):
        model = self.master_network
        java_model = self.java_spark_model.getNetwork()
        weights = adapter.retrieve_keras_weights(java_model)
        model.set_weights(weights)
        return model


def dl4j_import(jsc, model_file, keras_model_type):
    emi = java_classes.ElephasModelImport
    if keras_model_type == "Sequential":
        try:
            return emi.importElephasSequentialModelAndWeights(
                jsc, model_file.absolutePath)
        except:
            print("Couldn't load Keras model into DL4J")
    elif keras_model_type == "Model":
        try:
            return emi.importElephasModelAndWeights(jsc, model_file.absolutePath)
        except:
            print("Couldn't load Keras model into DL4J")
    else:
        raise Exception(
            "Keras model not understood, got: {}".format(keras_model_type))
