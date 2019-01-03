from pyspark import SparkContext, SparkConf
from pyspark.sql import SQLContext
import pytest
import logging


def quiet_py4j():
    """ turn down spark logging for the test context """
    logger = logging.getLogger('py4j')
    logger.setLevel(logging.WARN)


@pytest.fixture(scope="session")
def spark_context(request):
    """ fixture for creating a SparkContext
    Args:
        request: pytest.FixtureRequest object
    """
    conf = (SparkConf().setMaster("local[2]").setAppName(
        "pytest-pyspark-local-testing"))
    sc = SparkContext(conf=conf)
    request.addfinalizer(lambda: sc.stop())

    quiet_py4j()
    return sc


@pytest.fixture(scope="session")
def sql_context(request):
    """ fixture for creating a Spark SQLContext
    Args:
        request: pytest.FixtureRequest object
    """
    conf = (SparkConf().setMaster("local[2]").setAppName(
        "pytest-pyspark-local-testing"))
    sc = SparkContext(conf=conf)
    sql_context = SQLContext(sc)
    request.addfinalizer(lambda: sc.stop())

    quiet_py4j()
    return sql_context
