import pydl4j
import os

pydl4j.validate_jars()
pydl4j.add_classpath(os.getcwd())

# -------------JVM starts here-------------
from jnius import autoclass


# Java
File = autoclass('java.io.File')
ClassLoader = autoclass('java.lang.ClassLoader')
ArrayList = autoclass('java.util.ArrayList')
Arrays = autoclass('java.util.Arrays')
String = autoclass('java.lang.String')

System = autoclass('java.lang.System')
Integer = autoclass('java.lang.Integer')
Float = autoclass('java.lang.Float')
Double = autoclass('java.lang.Double')

# JavaCPP
DoublePointer = autoclass('org.bytedeco.javacpp.DoublePointer')
FloatPointer = autoclass('org.bytedeco.javacpp.FloatPointer')
IntPointer = autoclass('org.bytedeco.javacpp.IntPointer')

# Spark
SparkContext = autoclass('org.apache.spark.SparkContext')
JavaSparkContext = autoclass('org.apache.spark.api.java.JavaSparkContext')
SparkConf = autoclass('org.apache.spark.SparkConf')

# ND4J
Nd4j = autoclass('org.nd4j.linalg.factory.Nd4j')
INDArray = autoclass('org.nd4j.linalg.api.ndarray.INDArray')
Transforms = autoclass('org.nd4j.linalg.ops.transforms.Transforms')
NDArrayIndex = autoclass('org.nd4j.linalg.indexing.NDArrayIndex')
DataBuffer = autoclass('org.nd4j.linalg.api.buffer.DataBuffer')
Shape = autoclass('org.nd4j.linalg.api.shape.Shape')
BinarySerde = autoclass('org.nd4j.serde.binary.BinarySerde')
DataTypeUtil = autoclass('org.nd4j.linalg.api.buffer.util.DataTypeUtil')
NativeOpsHolder = autoclass('org.nd4j.nativeblas.NativeOpsHolder')
DataSet = autoclass('org.nd4j.linalg.dataset.DataSet')


# Import
KerasModelImport = autoclass(
    'org.deeplearning4j.nn.modelimport.keras.KerasModelImport')
ElephasModelImport = autoclass(
    'org.deeplearning4j.spark.parameterserver.modelimport.elephas.ElephasModelImport')
