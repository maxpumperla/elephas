from jnius import autoclass

# Java
File = autoclass('java.io.File')
ClassLoader = autoclass('java.lang.ClassLoader')
ArrayList = autoclass('java.util.ArrayList')

System = autoclass('java.lang.System')
Integer = autoclass('java.lang.Integer')
Float = autoclass('java.lang.Float')
Double = autoclass('java.lang.Double')

# ND4J
Nd4j = autoclass('org.nd4j.linalg.factory.Nd4j')
INDArray = autoclass('org.nd4j.linalg.api.ndarray.INDArray')
Transforms = autoclass('org.nd4j.linalg.ops.transforms.Transforms')
NDArrayIndex = autoclass('org.nd4j.linalg.indexing.NDArrayIndex')
DataBuffer = autoclass('org.nd4j.linalg.api.buffer.DataBuffer')
Shape = autoclass('org.nd4j.linalg.api.shape.Shape')
BinarySerde = autoclass('org.nd4j.serde.binary.BinarySerde')
DataTypeUtil = autoclass('org.nd4j.linalg.api.buffer.util.DataTypeUtil')


KerasModelImport = autoclass('org.deeplearning4j.nn.modelimport.keras.KerasModelImport')
ElephasModelImport = autoclass('org.deeplearning4j.spark.parameterserver.modelimport.elephas.ElephasModelImport')